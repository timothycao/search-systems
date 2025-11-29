# Dynamic Tiering Artifacts (Static BM25 + QTF)

This document records the data artifacts and steps used to produce Tier-1/Tier-2 labels from the static BM25 signals and query term frequencies (QTF).

## Approach
- **Tokenization/preprocessing**: Reuse `search_system.shared.utils.tokenize` (lowercase, strip non-alphanumeric, whitespace split), matching the BM25 indexer.
- **QTF (query-term frequency)**: Count term occurrences across `data/queries/queries.all.tsv` (≈200k queries). Optional singleton-drop was left disabled.
- **Static BM25 score**: For each term with QTF weight `QTF(t)`, traverse its inverted list in the BM25 index and accumulate `QTF(t) * BM25(tf_td, doc_len)` for every posting. BM25 parameters come from the index (`k1=1.2`, `b=0.75`, `avg_len` from `collection_stats.json`).
- **Tier assignment**: Min-max normalize scores, rank documents, and assign Tier-1 to the top share (here 40%); remaining documents are Tier-2.

## Commands Run
Executed from the repository root:
```bash
# 1) Prepare combined query log (dev+eval -> queries.all.tsv)
python -m scripts.prepare_queries_all

# 2) Compute query term frequencies
python -m scripts.tiering compute-qtf

# 3) Compute static BM25-based document scores (requires BM25 index from full collection.tsv)
python -m scripts.tiering static-scores --qtf artifacts/tiering/qtf.json --index artifacts/bm25/index

# 4) Assign tiers (Tier-1 = top 40%) and emit normalized scores
python -m scripts.tiering assign-tiers --scores artifacts/tiering/static_scores.pkl --tier-ratio 0.4 --normalized-output artifacts/tiering/static_norm.json

# 5) Extract document-side features
python -m scripts.tiering features --index artifacts/bm25/index --scores artifacts/tiering/static_scores.pkl --output artifacts/tiering/features.pkl

# 6) Assemble dataset and stratified train/val splits (80/20)
python -m scripts.tiering dataset --features artifacts/tiering/features.pkl --labels artifacts/tiering/labels.json --val-ratio 0.2 --seed 42 --train-output artifacts/tiering/train.pkl --val-output artifacts/tiering/val.pkl

# 7) Train XGBoost classifier with early stopping and select threshold (use --use-gpu on A100)
python -m scripts.tiering train --train artifacts/tiering/train.pkl --val artifacts/tiering/val.pkl --model-output artifacts/tiering/model.json --metrics-output artifacts/tiering/metrics.json --threshold-output artifacts/tiering/threshold.json --target-ratio 0.4
```

## Generated Artifacts
- `artifacts/tiering/qtf.json` — term → frequency counts from the query log.
- `artifacts/tiering/static_scores.pkl` — `doc_id -> static_score` (BM25 weighted by QTF).
- `artifacts/tiering/static_norm.json` — normalized scores in `[0,1]` (optional).
- `artifacts/tiering/labels.json` — `doc_id -> {1|0}` Tier-1/Tier-2 labels (Tier-1 at 40% cutoff in the run above).
- `artifacts/tiering/features.pkl` — `doc_id -> feature dict` with static score/log1p, doc length/log1p, IDF stats (mean/max/std), unique term count, and TF entropy.
- `artifacts/tiering/train.pkl`, `artifacts/tiering/val.pkl` — stratified splits for model training and validation.
- `artifacts/tiering/model.json` — trained XGBoost classifier.
- `artifacts/tiering/metrics.json` — training/validation metrics including thresholded precision/recall.
- `artifacts/tiering/threshold.json` — selected Tier-1 probability threshold with target ratio metadata.

## Status and Next Steps
- **Completed**: QTF computation; static BM25 score pass over the full BM25 index; tier labeling (40% Tier-1); document feature extraction; dataset assembly with stratified train/val splits; XGBoost training with early stopping; threshold selection to hit target Tier-1 share.
- **Remaining**: Integrate model + threshold into ingestion/build; update query-time flow to search Tier-1 first (optional Tier-2 fallback) and evaluate recall vs. baseline.
