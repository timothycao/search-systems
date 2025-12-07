# Dynamic Tiering – Detailed Design and Decisions

This document records the full technical rationale, assumptions, and step-by-step flow for the dynamic tiering pipeline built on top of the BM25 inverted index. It is meant to be a deep reference for collaborators.

## Objectives and Constraints
- **Objective**: Partition documents into Tier-1 (frequently consulted at query time) and Tier-2 (optional/fallback) using only document-side signals at ingestion.
- **Signal choice**: Use static BM25 scores weighted by query-term frequency (QTF) as the relevance proxy (no Monte-Carlo refinement).
- **Model goal**: Train a classifier that predicts Tier-1/Tier-2 from document-side features only; choose a probability threshold to meet a target Tier-1 share.
- **Index compatibility**: Reuse the existing BM25 index (lexicon, postings, page_table, avg_len) built on `collection.tsv`; use the same tokenizer.

## Preprocessing and Tokenization
- **Tokenizer**: `search_system.shared.utils.tokenize` (lowercase, replace non-alphanumeric with space, whitespace split). Matches the BM25 indexer and avoids feature/query mismatch.
- **Stopwords/stemming**: None (as per the provided tokenizer).

## Query-Term Frequency (QTF)
- **Input**: `data/queries/queries.all.tsv` (~200k queries, tab-separated: `query_id<TAB>text`), produced by concatenating dev+eval queries via `scripts/prepare_queries_all.py`.
- **Process**: Tokenize each query with the index tokenizer; accumulate `qtf[token] += 1`.
- **Option**: Dropping singletons is supported (`--drop-singletons`), but we kept all terms to retain tail coverage.
- **Output**: `artifacts/tiering/qtf.json`.

Preparation command:
```bash
python -m scripts.prepare_queries_all \
  --dev data/queries/queries.dev.tsv \
  --eval data/queries/queries.eval.tsv \
  --out data/queries/queries.all.tsv \
  --map data/queries/queries.all.map.tsv
```

Command:
```bash
python -m scripts.tiering compute-qtf
```

## Static BM25 Score Computation
- **Formula**: `StaticScore(d) = Σ_t QTF(t) * BM25(tf_td, doc_len)` using BM25 params from the index (k1=1.2, b=0.75, avg_len from collection_stats).
- **Inputs**: `qtf.json`, BM25 index files (`lexicon.json`, `page_table.json`, `inverted_index.bin`), `collection_stats.json`.
- **Traversal**: For each term with QTF>0, iterate its postings (term’s inverted list) and accumulate weighted BM25 into `doc_scores[d]`.
- **Output**: `artifacts/tiering/static_scores.pkl` (doc_id → score).

Command:
```bash
python -m scripts.tiering static-scores --qtf artifacts/tiering/qtf.json --index artifacts/bm25/index
```

## Tier Assignment (Labeling)
- **Normalization**: Min-max normalize static scores to [0,1] to avoid scale issues.
- **Ranking/Cutoff**: Sort by normalized score descending; choose Tier-1 as top X% (we used 40%).
- **Outputs**: `artifacts/tiering/labels.json` (doc_id → {1|0}), optional `artifacts/tiering/static_norm.json`.

Command:
```bash
python -m scripts.tiering assign-tiers --scores artifacts/tiering/static_scores.pkl --tier-ratio 0.4 --normalized-output artifacts/tiering/static_norm.json
```

## Document-Side Features
Computed for each document (requires BM25 index + static scores):
- `static_score`, `log1p(static_score)`
- `doc_len`, `log1p(doc_len)` (from `page_table.json`)
- IDF stats across unique terms in the doc: `mean_idf`, `max_idf`, `std_idf` (using `idf(df,N)=log((N-df+0.5)/(df+0.5)+1)`)
- `unique_term_count`
- `entropy_tf` = `(doc_len * log(doc_len) - Σ tf_i * log(tf_i)) / doc_len` (TF distribution entropy)

Output: `artifacts/tiering/features.pkl` (doc_id → feature dict).

Command:
```bash
python -m scripts.tiering features --index artifacts/bm25/index --scores artifacts/tiering/static_scores.pkl --output artifacts/tiering/features.pkl
```

## Dataset Assembly
- **Join**: Intersect `features.pkl` and `labels.json` on doc_id.
- **Structure**: `doc_ids`, `feature_names`, `X` (feature vectors), `y` (labels).
- **Split**: Stratified 80/20 train/val to preserve Tier-1/Tier-2 balance.
- **Outputs**: `artifacts/tiering/train.pkl`, `artifacts/tiering/val.pkl`.

Command:
```bash
python -m scripts.tiering dataset --features artifacts/tiering/features.pkl --labels artifacts/tiering/labels.json --val-ratio 0.2 --seed 42 --train-output artifacts/tiering/train.pkl --val-output artifacts/tiering/val.pkl
```

## Model Training (XGBoost) and Threshold Selection
- **Model**: Gradient-boosted trees (XGBoost `binary:logistic`), AUC eval, early stopping.
- **Inputs**: Train/val splits from the dataset step.
- **Threshold**: Select probability cutoff so predicted Tier-1 share ≈ target ratio (default 0.4). Evaluate precision/recall on val at this threshold.
- **Outputs**:
  - `artifacts/tiering/model.json` (XGBoost model)
  - `artifacts/tiering/metrics.json` (AUC, best iteration, thresholded precision/recall, pred ratio)
  - `artifacts/tiering/threshold.json` (chosen threshold + target ratio)

### Training on Kaggle GPUs (current workflow)
- Upload a Kaggle dataset containing `artifacts/tiering/train.pkl` and `artifacts/tiering/val.pkl` (e.g., named `tiering-artifacts`).
- In a Kaggle notebook, attach that dataset and select a GPU runtime (A100/P100).
- Run `kaggle_train.ipynb` (from this repo) which:
  - Installs minimal deps (`xgboost`, `search_system`)
  - Clones the repo to access tiering utilities
  - Loads train/val splits, trains XGBoost with early stopping (`tree_method=hist`, `device=cuda`), selects a threshold to hit target_ratio, and saves `model.json`, `metrics.json`, `threshold.json` under `/kaggle/working/artifacts/tiering`.

## Tiered BM25 Index Build (Tier-1 / Tier-2)
- **Inputs**: `artifacts/tiering/labels.json` (doc_id -> {1|0}), `data/collection/collection.tsv`.
- **Process**: Write subset ID files for Tier-1 and Tier-2, then run parser/indexer for each tier.
- **Outputs**: `artifacts/bm25_T1/index` and `artifacts/bm25_T2/index` (plus postings/collection_stats/page_table/lexicon).

Command:
```bash
python -m scripts.build_tiers \
  --labels artifacts/tiering/labels.json \
  --dataset data/collection/collection.tsv \
  --out-root artifacts
```

## Delta ingestion and rebuild policy
- **Thresholds**: Tier-1 delta > 1,000 docs triggers rebuild of bm25_T1; Tier-2 delta > 100,000 triggers rebuild of bm25_T2.
- **Inference-based routing (scripts/ingest_infer_tiered.py + systems.tiering.infer/ingest)**:
  - Input TSV: `doc_id<TAB>text`. For each doc: compute features (static score via qtf+BM25 stats, length/idf stats/entropy), run the tiering model (`model.json`, `threshold.json`), assign tier.
  - Append to `artifacts/tiering/delta_t1.tsv` or `delta_t2.tsv` based on predicted tier.
  - Always rebuild small delta indexes (`artifacts/bm25_T1_delta/index`, `bm25_T2_delta/index`) so new docs are searchable; delta postings/index dirs are cleaned before each rebuild to avoid stale files.
  - If a threshold is exceeded, rebuild the corresponding base index (bm25_T1 or bm25_T2) from original tier docs + delta, then clear the delta TSV, delta index dir, and the temporary rebuild dataset.
- **Config knobs**:
  - Thresholds and paths live in `utils/config.py` (e.g., `DELTA_T1_THRESHOLD=1000`, `DELTA_T2_THRESHOLD=100000`, `TIERING_QTF_PATH`, `TIERING_MODEL_PATH`, `TIERING_THRESHOLD_PATH`, `TIERING_FEATURE_NAMES_PATH`, `TIER1_IDS_PATH`, `TIER2_IDS_PATH`, `DELTA_DIR`). CLI flags override these if provided.
- Direct routing with pre-labeled docs is not used in this flow; inference-based routing is the default.
- **Query-time**: planned to overfetch base + delta, compute merged stats (N_total, avgdl_total, df_total), rescore candidates from both shards with BM25, and merge to top_k (RRF as fallback).

## HNSW Tiering (Dense)
- **Encoder**: `sentence-transformers/msmarco-bert-base-dot-v5` (dot-product optimized).
- **Embeddings**: Generate doc and query embeddings (`scripts/hnsw_generate_embeddings.py`); stored as HDF5 (id, embedding). Use raw outputs (no L2) and normalize during scoring/index build to mirror HNSW flow.
- **Static scores for labeling**: Build a Faiss inner-product index over query embeddings; for each doc embedding, search topK=25 queries, aggregate scores (avg of topK) as the dense static relevance score. Normalize/rank and assign Tier-1 (top 40%) / Tier-2.
- **Outputs**: Dense labels (`artifacts/tiering_dense/labels.json`), static scores (`static_scores.npy`), tier1_ids/tier2_ids, and tier-specific embedding files (`doc_embeddings_t1.h5`, `doc_embeddings_t2.h5`).
- **Index build**: Use the existing HNSW system to build `artifacts/hnsw_T1` and `artifacts/hnsw_T2` from the tier-specific embeddings (inner product, same params as current HNSW).

```

## Platform note (macOS)
- XGBoost needs the OpenMP runtime (`libomp`) on macOS. Install once before running inference locally:
  ```bash
  brew install libomp
  ```
```

## Assumptions and Design Choices
- **Full corpus index**: The BM25 index must be built on `collection.tsv` (not subset) to cover all doc IDs; otherwise static scores and features will be limited.
- **No Monte-Carlo refinement**: Static BM25 + QTF is the sole relevance proxy.
- **Feature-only inference**: Model uses document-side features; at ingestion we may approximate/omit static score (or precompute if feasible).
- **Tier size target**: Set by `tier_ratio` and mirrored in threshold selection (here 40%).
- **Tokenizer consistency**: Must match the index; otherwise QTF/static scores/features misalign.
- **Evaluation focus**: AUC plus precision/recall for Tier-1 at the chosen threshold.

## Pending Integration
- Load model + threshold in the ingestion/build pipeline; compute features for new docs and assign tier at ingest.
- Query-time flow: search Tier-1 first; optionally query Tier-2 and merge; evaluate recall/precision vs. baseline.

## Updating This Document
When adding new steps (integration, evaluation results, alternative models), append sections with rationale, commands, and artifacts.***
