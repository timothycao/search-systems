# Tiered BM25 Flow (Train/Work Split)

This document captures the revised BM25 tiering workflow that separates training data (for the XGBoost tiering model and base tiered indexes) from the working set (held out for inference-only ingress and evaluation).

## 1) Split corpus into train vs work
- Extract working doc IDs: include all docIds from qrels (dev + eval1 + eval2) and add a random sample of the remaining docs to reach ~30% of the corpus.
- Outputs:
  - `data/collection/docids_working.txt` (working IDs)
  - `data/collection/collection_train.tsv` (~70% of docs)
  - `data/collection/collection_work.tsv` (~30% of docs, contains all qrels docs)
- Integrity: union of train/work equals original `collection.tsv`; no overlap between train and work; docIds preserved.
- Script:  
  ```bash
  python -m scripts.split_train_work \
    --collection data/collection/collection.tsv \
    --qrels-dev data/qrels/qrels.dev.tsv \
    --qrels-eval1 data/qrels/qrels.eval.one.tsv \
    --qrels-eval2 data/qrels/qrels.eval.two.tsv \
    --work-frac 0.3 --seed 42 \
    --docids-working-out data/collection/docids_working.txt \
    --train-out data/collection/collection_train.tsv \
    --work-out data/collection/collection_work.tsv
  ```

## 2) Build BM25 on train-only corpus
- Command (example):  
  ```bash
  python -m scripts.build --system bm25 \
    --dataset-path data/collection/collection_train.tsv \
    --artifacts-dir artifacts/bm25_train
  ```
- Note: build script now accepts dataset/dir overrides; lazy imports avoid faiss when building BM25.

## 3) Tiering artifacts on train-only index
- Reuse existing `qtf.json` (query-log–based, unchanged), or recompute if desired.
- Static scores:  
  ```bash
  python -m scripts.tiering static-scores \
    --qtf artifacts/tiering/qtf.json \
    --index artifacts/bm25_train/index \
    --output artifacts/tiering/static_scores_train.pkl
  ```
- Labels (tier ratio e.g., 0.4):  
  ```bash
  python -m scripts.tiering assign-tiers \
    --scores artifacts/tiering/static_scores_train.pkl \
    --tier-ratio 0.4 \
    --normalized-output artifacts/tiering/static_norm_train.json \
    --labels-output artifacts/tiering/labels_train.json \
    --tier1-ids artifacts/tiering/tier1_ids_train.txt \
    --tier2-ids artifacts/tiering/tier2_ids_train.txt
  ```
- Features:  
  ```bash
  python -m scripts.tiering features \
    --index artifacts/bm25_train/index \
    --scores artifacts/tiering/static_scores_train.pkl \
    --output artifacts/tiering/features_train.pkl
  ```
- Dataset split:  
  ```bash
  python -m scripts.tiering dataset \
    --features artifacts/tiering/features_train.pkl \
    --labels artifacts/tiering/labels_train.json \
    --val-ratio 0.2 --seed 42 \
    --train-output artifacts/tiering/train_train.pkl \
    --val-output artifacts/tiering/val_train.pkl
  ```

## 4) Train XGBoost externally (Kaggle)
- Use `train_train.pkl` / `val_train.pkl` to train XGBoost.
- Save `model.json` and `threshold.json` (train-only derived) back into `artifacts/tiering/`.

## 5) Build base tiered BM25 indexes from train labels
- Build `bm25_T1` and `bm25_T2` base indexes using `collection_train.tsv` filtered by `tier1_ids_train.txt` / `tier2_ids_train.txt`.
- Store tier ID lists with a train suffix to avoid confusion.

## 6) Ingest working set via inference
- Run inference ingestion on `collection_work.tsv` (or any new docs) using the trained model/threshold and train-based BM25 stats:
  ```bash
  python -m scripts.ingest_infer_tiered \
    --input data/collection/collection_work.tsv \
    --index artifacts/bm25_train/index \
    --model artifacts/tiering/model.json \
    --threshold artifacts/tiering/threshold.json \
    --qtf artifacts/tiering/qtf.json \
    --feature-names artifacts/tiering/train_train.pkl \
    --collection data/collection/collection_train.tsv \
    --tier1-ids artifacts/tiering/tier1_ids_train.txt \
    --tier2-ids artifacts/tiering/tier2_ids_train.txt \
    --out-root artifacts \
    --delta-dir artifacts/tiering
  ```
- Docs are routed to `delta_t1.tsv` / `delta_t2.tsv`, delta indexes are rebuilt, and when thresholds are exceeded the base tier indexes are rebuilt and deltas cleared.

## 7) Evaluation readiness
- Working/eval docs live in the work split and enter via inference. Ensure query-time search merges base + deltas so eval docs are searchable.
- No leakage: eval docIds are excluded from train artifacts and only appear through inference.

## 8) Notes
- QTF remains global (query-log–derived) and is reused across splits.
- Keep paths distinct (e.g., `bm25_train`, `tier*_train.txt`) to avoid mixing with previous full-corpus artifacts.
