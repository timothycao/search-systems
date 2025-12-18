# Tiered HNSW (Train/Work Split) — Full Rationale and Technical Details

This document captures the detailed rationale, design choices, and math for the HNSW tiered workflow. It parallels the BM25 tiering docs but focuses on dense retrieval (dot-product) with HNSW. All artifacts and paths are suffixed with `hnsw` to avoid collisions with BM25 outputs.

## 1) Embeddings and Splits
- Sharded doc embeddings (from Kaggle): `data/collection/doc_embeddings_part{0..4}.h5` (descending order as produced in the sharded notebook).
- Query embeddings: `data/collection/query_embeddings.h5`.
- Split doc embeddings into train/work (script: `hnsw_split_embeddings.py`):
  - Inputs: `collection_train.tsv`, `collection_work.tsv` (work contains all eval docIds).
  - Outputs: `data/collection/collection_train_hnsw.h5`, `data/collection/collection_work_hnsw.h5`.
  - Rationale: Prevent train/eval leakage; eval docs only appear in the work split and are ingested via inference.

## 2) Scoring and Labels (HNSW)
- Normalize doc/query embeddings (L2) to match HNSW retrieval.
- Build a Faiss IP index over queries; search each doc embedding against topK queries (e.g., K=25).
- Aggregates saved per doc (npz): `score` (avg topK), `sim_max`, `sim_std`, `sim_p90`.
- Tiering: sort by `score`, assign Tier-1 to the top tier_ratio (e.g., 0.4), Tier-2 to the rest.
- Outputs: `static_scores_hnsw.npz`, `labels_hnsw.json`, tier ID lists (`tier1_ids_hnsw_train.txt`, `tier2_ids_hnsw_train.txt`), and tier-specific train embeddings (`doc_embeddings_t1_hnsw.h5`, `_t2_...`), all under `artifacts/tiering_dense/`.

## 3) Tiering Model (HNSW)
- Features (built by `hnsw_build_dataset.py`):
  - Static score, log1p_static_score, sim_max, sim_std, sim_p90
  - Embedding norm, log1p_embedding_norm
  - Doc_len, log1p_doc_len, unique_term_count, tf_entropy (from doc text)
- Train/val splits from HNSW train artifacts; stratified to preserve class balance.
- Train XGBoost in `kaggle_train_xgboost_hnsw.ipynb` (GPU-enabled), select threshold on val to hit target ratio (e.g., 0.4).
- Outputs: `model_hnsw.json`, `threshold_hnsw.json`, `metrics_hnsw.json`.

## 4) Tiered HNSW Indexes (Work ingestion)
- We do not build tiered indexes from train labels. Instead, we infer tiers on the work split and build base/delta indexes from those predictions, mirroring the BM25 flow.
- Work is split deterministically: `collection_work_hnsw_init.tsv` (work minus 500k) and `collection_work_hnsw_delta.tsv` (500k).
- Base indexes: `hnsw_T1`, `hnsw_T2` (built from the large work-init ingest after rollover).
- Delta indexes: `hnsw_T1_delta`, `hnsw_T2_delta` (built from the smaller work-delta ingest; remain live unless thresholds are exceeded).
- Base/delta membership files (doc_id and embeddings) are maintained to support rebuilds.

## 5) Work Ingestion and Deltas
- Split work TSV into `collection_work_hnsw_init.tsv` (work minus 500k) and `collection_work_hnsw_delta.tsv` (500k).
- Ingest init with `model_hnsw.json`/`threshold_hnsw.json`: route to deltas, thresholds exceeded → merge into base (`hnsw_T1/T2`), clear deltas, rebuild bases.
- Ingest delta: route to `hnsw_T1_delta/T2_delta`; thresholds not exceeded → keep deltas to test base+delta merge.

## 6) Query-Time Merge
- Search base and delta HNSW separately; overfetch (e.g., 2× topK) from each.
- Merge by dot-product score (same embedding model makes scores directly comparable); optionally use RRF if needed for stability.
- Return final topK per query; multiprocessing runner (`run_tiered_hnsw_multi.py`) mirrors the BM25 tiered runner.

## 7) Evaluation
- Evaluate tiered HNSW vs non-tiered HNSW on the working/eval queries (MRR@10, Recall@100, MAP, NDCG@10, NDCG@100).
- Compare to BM25 tiered baselines when combined later in fusion/reranking.

## 8) Next Steps
- Fill in detailed commands, parameter choices (M, efConstruction, efSearch), scoring specifics, and empirical observations as we implement each step and the HNSW ingestion/merge runner.
