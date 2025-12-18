# Tiered HNSW Flow (Train/Work Split)

This document tracks the HNSW tiering workflow. All artifacts are suffixed with `hnsw` to avoid clashes with BM25.

## 1) Split doc embeddings into train/work
- Inputs:
  - Sharded doc embeddings from Kaggle: `data/collection/doc_embeddings_part{0..4}.h5`
  - Train/work TSVs: `data/collection/collection_train.tsv`, `data/collection/collection_work.tsv`
- Command:
  ```bash
  python -m scripts.hnsw_split_embeddings \
    --shards "data/collection/doc_embeddings_part*.h5" \
    --train-tsv data/collection/collection_train.tsv \
    --work-tsv data/collection/collection_work.tsv \
    --train-out data/collection/collection_train_hnsw.h5 \
    --work-out data/collection/collection_work_hnsw.h5
  ```
- Output: split HNSW doc embeddings for train/work, ready for scoring/labeling.

## 2) Train-side scoring and labels (HNSW)
- Compute dense static scores using query embeddings (dot-product, avg topK), assign tiers (e.g., ratio 0.4), and produce HNSW-labeled artifacts (static scores, labels, tier id lists):
  ```bash
  python -m scripts.hnsw_dense_labels \
    --doc-emb data/collection/collection_train_hnsw.h5 \
    --query-emb data/collection/query_embeddings.h5 \
    --topk 25 \
    --tier-ratio 0.4 \
    --scores-out artifacts/tiering_dense/static_scores_hnsw.npy \
    --labels-out artifacts/tiering_dense/labels_hnsw.json \
    --tier-dir artifacts/tiering_dense \
    --tier1-ids-out artifacts/tiering_dense/tier1_ids_hnsw_train.txt \
    --tier2-ids-out artifacts/tiering_dense/tier2_ids_hnsw_train.txt \
    --t1-emb-out artifacts/tiering_dense/doc_embeddings_t1_hnsw.h5 \
    --t2-emb-out artifacts/tiering_dense/doc_embeddings_t2_hnsw.h5 \
    --batch-size 50000
  ```
- Output: HNSW train static scores, labels, tier ID lists, and tier-specific embeddings (train-only).

## 3) Train-side features and datasets (HNSW)
- Build features (static score + aggregates + embedding/text stats) and train/val splits:
  ```bash
  python -m scripts.hnsw_build_dataset \
    --scores artifacts/tiering_dense/static_scores_hnsw.npz \
    --labels artifacts/tiering_dense/labels_hnsw.json \
    --embeddings data/collection/collection_train_hnsw.h5 \
    --doc-collection data/collection/collection_train.tsv \
    --features-out artifacts/tiering_dense/features_hnsw.pkl \
    --train-out artifacts/tiering_dense/train_hnsw.pkl \
    --val-out artifacts/tiering_dense/val_hnsw.pkl \
    --val-ratio 0.2 --seed 42
  ```
- Output: HNSW feature dict and train/val datasets for model training.

## 4) Train-side model (HNSW)
- Train XGBoost on `train_hnsw.pkl` / `val_hnsw.pkl` (GPU-enabled notebook):
  - Notebook: `kaggle_train_xgboost_hnsw.ipynb`
  - Inputs: upload `train_hnsw.pkl`, `val_hnsw.pkl` (from step 3)
  - Outputs: `model_hnsw.json`, `threshold_hnsw.json`, `metrics_hnsw.json` under `artifacts/tiering_dense/`

## 5) Work split into init/delta (HNSW)
- Deterministic split of work TSV (500k docs to delta):
  ```bash
  python -m scripts.split_work_hnsw_init_delta \
    --work data/collection/collection_work.tsv \
    --delta-size 500000 --seed 42 \
    --init-out data/collection/collection_work_hnsw_init.tsv \
    --delta-out data/collection/collection_work_hnsw_delta.tsv
  ```
- Output: work-init and work-delta TSVs for HNSW ingestion mirroring BM25 flow.

## 6) Ingest work splits (inference; build base/delta HNSW)
- Ingest init (expected to roll into base):
  ```bash
  python -m scripts.ingest_infer_tiered_hnsw \
    --input data/collection/collection_work_hnsw_init.tsv \
    --work-emb data/collection/collection_work_hnsw.h5 \
    --query-emb data/collection/query_embeddings_remapped.h5 \
    --model artifacts/tiering_dense/model_hnsw.json \
    --threshold artifacts/tiering_dense/threshold_hnsw.json \
    --feature-names artifacts/tiering_dense/train_hnsw.pkl \
    --topk 25 --batch-size 4096 --faiss-threads 8 \
    --base-t1 artifacts/tiering_dense/base_T1_hnsw.tsv \
    --base-t2 artifacts/tiering_dense/base_T2_hnsw.tsv \
    --delta-t1 artifacts/tiering_dense/delta_T1_hnsw.tsv \
    --delta-t2 artifacts/tiering_dense/delta_T2_hnsw.tsv \
    --index-t1 artifacts/hnsw_T1 \
    --index-t2 artifacts/hnsw_T2 \
    --index-t1-delta artifacts/hnsw_T1_delta \
    --index-t2-delta artifacts/hnsw_T2_delta \
    --m 24 --ef-construction 200
  ```
- Ingest delta (expected to remain in deltas):
  ```bash
  python -m scripts.ingest_infer_tiered_hnsw \
    --input data/collection/collection_work_hnsw_delta.tsv \
    --work-emb data/collection/collection_work_hnsw.h5 \
    --query-emb data/collection/query_embeddings_remapped.h5 \
    --model artifacts/tiering_dense/model_hnsw.json \
    --threshold artifacts/tiering_dense/threshold_hnsw.json \
    --feature-names artifacts/tiering_dense/train_hnsw.pkl \
    --topk 25 --batch-size 4096 --faiss-threads 8 \
    --base-t1 artifacts/tiering_dense/base_T1_hnsw.tsv \
    --base-t2 artifacts/tiering_dense/base_T2_hnsw.tsv \
    --delta-t1 artifacts/tiering_dense/delta_T1_hnsw.tsv \
    --delta-t2 artifacts/tiering_dense/delta_T2_hnsw.tsv \
    --index-t1 artifacts/hnsw_T1 \
    --index-t2 artifacts/hnsw_T2 \
    --index-t1-delta artifacts/hnsw_T1_delta \
    --index-t2-delta artifacts/hnsw_T2_delta \
    --m 24 --ef-construction 200
  ```
- Behavior: routes docs to delta TSVs + indexes; if thresholds exceeded (T1>400k, T2>1M), deltas are merged into base and bases rebuilt; otherwise deltas are rebuilt for queryability.

## 7) Tiered HNSW Retrieval (base + delta, multiprocessing)
- Run tiered HNSW retrieval with merge over base+delta:
  ```bash
  python -m scripts.run_tiered_hnsw_multi_norescore \
    --qrels dev \
    --save hnsw_dev_working_FT_multi_norescore_m24_ef200_es200_of2 \
    --topk 100 \
    --overfetch-factor 2 \
    --ef-search 200 \
    --workers 4 \
    --query-emb data/collection/query_embeddings_remapped.h5
  ```
- Searches `hnsw_T1`, `hnsw_T2`, `hnsw_T1_delta`, `hnsw_T2_delta`, overfetches per index, merges by dot-product, outputs topK in the same run format as `run.py`.
