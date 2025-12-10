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
