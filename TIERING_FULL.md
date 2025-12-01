# Dynamic Tiering – Full Technical Walkthrough

This document is a comprehensive, step-by-step explanation of the tiering pipeline: data preparation, scoring, labeling, feature extraction, dataset assembly, model training (Kaggle), tiered index builds, and how the pieces fit together. It is intended for detailed review and reproducibility.

---

## Preliminaries and Objectives
- **Goal**: Partition documents into Tier-1 (hot) and Tier-2 (cold) using document-side signals. Train a model that predicts a doc’s tier at ingestion based on features, not query-time signals.
- **Signal choice**: Static BM25 scores weighted by query-term frequency (QTF) serve as the relevance proxy (no Monte-Carlo refinement).
- **Index compatibility**: Reuse the existing BM25 index (lexicon, postings, page_table, avg_len) built on the full `collection.tsv`. Use the same tokenizer to avoid feature/query mismatch.

---

## Tokenization
- **Function**: `search_system.shared.utils.tokenize`
- **Behavior**: Lowercase → replace non-alphanumeric with space → whitespace split.
- **Stopwords/stemming**: None.
- **Rationale**: Matches the BM25 indexer, ensuring consistency across QTF, static scores, and feature extraction.

---

## Query-Term Frequency (QTF)
- **Input**: `data/queries/queries.all.tsv` (~200k queries, tab-separated `query_id<TAB>text`), produced by `scripts/prepare_queries_all.py` (concatenates dev+eval and renumbers).
- **Process**: Tokenize each query; accumulate `qtf[token] += 1`.
- **Output**: `artifacts/tiering/qtf.json`.
- **Command**:
  ```bash
  python -m scripts.prepare_queries_all \
    --dev data/queries/queries.dev.tsv \
    --eval data/queries/queries.eval.tsv \
    --out data/queries/queries.all.tsv \
    --map data/queries/queries.all.map.tsv

  python -m scripts.tiering compute-qtf
  ```

---

## Static BM25 Score Computation
- **Formula**: `StaticScore(d) = Σ_t QTF(t) * BM25(tf_td, doc_len)`, BM25 params k1=1.2, b=0.75, avg_len from `collection_stats.json`.
- **Inputs**: `qtf.json`, BM25 index (`lexicon.json`, `page_table.json`, `inverted_index.bin`), `collection_stats.json`.
- **Traversal**: For each term with QTF>0, iterate its postings list, compute term BM25 with doc_len, accumulate weighted score.
- **Output**: `artifacts/tiering/static_scores.pkl` (doc_id → score), optional normalized scores (`static_norm.json`).
- **Command**:
  ```bash
  python -m scripts.tiering static-scores --qtf artifacts/tiering/qtf.json --index artifacts/bm25/index
  ```

---

## Tier Assignment (Labeling)
- **Normalization**: Min-max normalize static scores to [0,1].
- **Cutoff**: Sort by normalized score; Tier-1 = top X% (we used 40%), Tier-2 = remainder.
- **Outputs**: `artifacts/tiering/labels.json` (doc_id → {1|0}), optional `static_norm.json`.
- **Command**:
  ```bash
  python -m scripts.tiering assign-tiers --scores artifacts/tiering/static_scores.pkl --tier-ratio 0.4 --normalized-output artifacts/tiering/static_norm.json
  ```

---

## Document-Side Features
Computed per doc using the BM25 index and static scores:
- `static_score`, `log1p(static_score)`
- `doc_len`, `log1p(doc_len)` (from `page_table.json`)
- IDF stats over unique terms: `mean_idf`, `max_idf`, `std_idf` (`idf(df,N)=log((N-df+0.5)/(df+0.5)+1)`)
- `unique_term_count`
- `entropy_tf` = `(doc_len * log(doc_len) - Σ tf_i * log(tf_i)) / doc_len`
- **Output**: `artifacts/tiering/features.pkl` (doc_id → feature dict).
- **Command**:
  ```bash
  python -m scripts.tiering features --index artifacts/bm25/index --scores artifacts/tiering/static_scores.pkl --output artifacts/tiering/features.pkl
  ```

---

## Dataset Assembly
- **Join**: Intersect `features.pkl` and `labels.json` on doc_id.
- **Structure**: `doc_ids`, `feature_names` (sorted keys), `X` (feature vectors), `y` (labels).
- **Split**: Stratified 80/20 train/val to preserve label balance.
- **Outputs**: `artifacts/tiering/train.pkl`, `artifacts/tiering/val.pkl`.
- **Command**:
  ```bash
  python -m scripts.tiering dataset --features artifacts/tiering/features.pkl --labels artifacts/tiering/labels.json --val-ratio 0.2 --seed 42 --train-output artifacts/tiering/train.pkl --val-output artifacts/tiering/val.pkl
  ```

---

## Model Training (Kaggle GPU Workflow)
- **Model**: XGBoost `binary:logistic`, AUC eval, early stopping.
- **Params**: `eta=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method=hist`, `device=cuda` (GPU), `seed=42`, `num_rounds=500`, `early_stopping=50`.
- **Threshold**: Chosen so predicted Tier-1 share ≈ target_ratio=0.4 on validation.
- **Outputs**: `artifacts/tiering/model.json`, `metrics.json`, `threshold.json`.
- **Kaggle steps**:
  - Upload dataset with `artifacts/tiering/train.pkl` and `artifacts/tiering/val.pkl` (e.g., `tiering-artifacts`).
  - Attach dataset in a Kaggle notebook; select GPU runtime (A100/P100).
  - Run `kaggle_train.ipynb` (this repo) to train and save outputs.

**Training results (from artifacts)**:
- Best iteration: 64
- Val AUC: ~0.999992
- Threshold: ~0.4058 (yields ~40% Tier-1)
- Val precision/recall @ threshold: ~0.9977 / ~0.9977
- Interpretation: Labels come from static score; static score is a feature → near-perfect separation and balanced precision/recall at the target share.

---

## Tiered BM25 Index Builds (bm25_T1 / bm25_T2)
- **Purpose**: Split the corpus into two physical BM25 indexes for Tier-1 and Tier-2 based on labels.
- **Inputs**: `artifacts/tiering/labels.json`, `data/collection/collection.tsv`.
- **Process (scripts/build_tiers.py)**:
  1) Write subset ID files:
     - `artifacts/tiering/tier1_ids.txt` (label==1)
     - `artifacts/tiering/tier2_ids.txt` (label==0)
  2) For each tier:
     - Run `run_parser` with `subset_ids_path` → postings in `artifacts/bm25_T{1,2}/postings`
     - Run `run_indexer` → `artifacts/bm25_T{1,2}/index` (inverted_index.bin, lexicon.json, page_table.json, collection_stats.json)
- **Command**:
  ```bash
  python -m scripts.build_tiers \
    --labels artifacts/tiering/labels.json \
    --dataset data/collection/collection.tsv \
    --out-root artifacts
  ```
- **Outputs**: Two BM25 indexes: `artifacts/bm25_T1/index`, `artifacts/bm25_T2/index`.

---

## Delta Ingestion, Thresholds, and Rebuilds
- **Delta buffers**: Small per-tier TSVs for new docs: `artifacts/tiering/delta_t1.tsv`, `delta_t2.tsv`.
- **Threshold policy**: Tier-1 delta > 1,000 docs triggers rebuild of `bm25_T1`; Tier-2 delta > 100,000 triggers rebuild of `bm25_T2`.
- **Routing with inference (scripts/ingest_infer_tiered.py + systems.tiering.infer/ingest)**:
  - Input TSV: `doc_id<TAB>text`. For each doc:
    - Tokenize with the index tokenizer; compute features in training order: `doc_len/log1p`, `static_score/log1p` (via qtf + BM25 stats: k1/b/avgdl/N/idf from lexicon), `mean/max/std idf`, `unique_term_count`, `entropy_tf`.
    - Build the feature vector, load `model.json` + `threshold.json`, predict Tier-1/Tier-2.
    - Append `doc_id<TAB>text` to the corresponding delta TSV (`artifacts/tiering/delta_t1.tsv` or `delta_t2.tsv`).
  - Delta index refresh: after ingestion, rebuild small delta indexes (`artifacts/bm25_T1_delta/index`, `artifacts/bm25_T2_delta/index`) from the delta TSVs so new docs are immediately searchable. These live under `artifacts/` alongside the base tier indexes; the TSV buffers live under `artifacts/tiering/`. Delta postings/index dirs are cleaned before each rebuild to avoid stale files.
  - Threshold-triggered rebuilds: if T1 delta > 1,000 or T2 delta > 100,000, materialize a temp dataset of original tier docs (from `tier1_ids.txt`/`tier2_ids.txt`) plus delta docs, run parser/indexer into `artifacts/bm25_T{1,2}`, then clear that tier’s delta TSV and delta index dir, and remove the temp dataset.
- **Config knobs** (in `utils/config.py`, overridable via CLI flags where available):
  - `DELTA_T1_THRESHOLD`, `DELTA_T2_THRESHOLD`
  - `TIERING_QTF_PATH`, `TIERING_MODEL_PATH`, `TIERING_THRESHOLD_PATH`, `TIERING_FEATURE_NAMES_PATH`
  - `TIER1_IDS_PATH`, `TIER2_IDS_PATH`, `DELTA_DIR`
- **Rebuild mechanics**:
  - Load tier subset IDs (`tier1_ids.txt`, `tier2_ids.txt`) from the label split.
  - Materialize a temp dataset: original tier docs from `collection.tsv` plus delta docs.
  - Run `run_parser` and `run_indexer` into `artifacts/bm25_T1` or `artifacts/bm25_T2`.
  - Remove/clear the delta TSV after rebuild.
- **Command (ingest routing + auto-rebuild)**:
  - Use `scripts/ingest_infer_tiered.py` to infer tier and route (see Commands in TIERING.md).
- **Query-time (planned)**: Overfetch from base + delta, compute merged stats (N_total, avgdl_total, df_total), rescore candidates from both shards with BM25, and merge to top_k (RRF as a fallback).

---

## Inference Considerations
- **Feature computation**: For new docs, compute the same features in the same order. If feasible, recompute static_score using QTF + BM25 params; otherwise approximate (expect some calibration drift).
- **Model usage**: Load `model.json`, `threshold.json`; build feature vector aligned to `feature_names`, run XGBoost to get a probability, compare to threshold → assign Tier-1/Tier-2.
- **Routing**: Insert into bm25_T1 or bm25_T2 indexes accordingly.

---

## Platform note (macOS)
- XGBoost needs the OpenMP runtime (`libomp`) on macOS. Install once before running inference locally:
  ```bash
  brew install libomp
  ```

---

## Pending Integration Work
- Integrate model + feature computation into ingestion/build so new docs are routed to the correct tier.
- Query-time flow: search Tier-1 first; optionally search Tier-2 and merge; evaluate recall/precision vs. the single-index baseline.

---

## Key Assumptions and Design Choices
- Full corpus index (`collection.tsv`) to align labels/features.
- Static BM25 + QTF is the only relevance proxy (no Monte-Carlo refinement).
- Tokenizer consistency with the index is mandatory.
- Tier size controlled by `tier_ratio` and matched in threshold selection (here 40%).
