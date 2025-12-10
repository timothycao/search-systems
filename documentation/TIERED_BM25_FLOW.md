# Tiered BM25 Flow (Train/Work Split)

This document captures the BM25 tiering workflow that separates training data for the XGBoost tiering model from the working set (held out for inference-only ingress and evaluation). We ensure all docs that appear in the three eval sets land in the working split so that we can systematically evaluate at varying topKs (MRR@10, Recall@100, MAP, NDCG@10, NDCG@100) and compare tiered/delta vs. non-tiered baselines.

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
- Compute or reuse `qtf.json` (query-log–based, unchanged):  
  ```bash
  python -m scripts.tiering compute-qtf \
    --queries data/queries/queries.all.tsv \
    --output artifacts/tiering/qtf.json
  ```
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

## 4) Train BM25 focused XGBoost model externally (Kaggle)
- Use `train_train.pkl` / `val_train.pkl` to train XGBoost.
- Save `model.json` and `threshold.json` (train-only derived) back into `artifacts/tiering/`.

## 5) Work-only pipeline (no train leakage)
- Split work corpus into init/delta subsets (500k docs in delta):  
  ```bash
  python -m scripts.split_work_init_delta \
    --work data/collection/collection_work.tsv \
    --delta-size 500000 --seed 42 \
    --init-out data/collection/collection_work_init.tsv \
    --delta-out data/collection/collection_work_delta.tsv
  ```
  _Why:_ deterministic split so we can roll most work docs into base tiers and keep a smaller delta for testing rollover behavior.

- Build vanilla BM25 over the full work set (for stats/idf/avgdl and baseline):  
  ```bash
  python -m scripts.build --system bm25 \
    --dataset-path data/collection/collection_work.tsv \
    --artifacts-dir artifacts/bm25
  ```
  _Why:_ provides consistent BM25 stats for feature computation during work-set inference.

- Ingest work init split (expected to exceed thresholds and roll into base tiers):  
  ```bash
  python -m scripts.ingest_infer_tiered \
    --input data/collection/collection_work_init.tsv \
    --index artifacts/bm25/index \
    --model artifacts/tiering/model.json \
    --threshold artifacts/tiering/threshold.json \
    --qtf artifacts/tiering/qtf.json \
    --feature-names artifacts/tiering/train_train.pkl \
    --out-root artifacts \
    --base-t1 artifacts/tiering/base_T1.tsv \
    --base-t2 artifacts/tiering/base_T2.tsv \
    --delta-t1 artifacts/tiering/delta_T1.tsv \
    --delta-t2 artifacts/tiering/delta_T2.tsv
  ```
  _Why:_ routes most work docs into Tier-1/Tier-2; deltas exceed thresholds and are rolled into base `bm25_T1`/`bm25_T2`.

- Ingest work delta split (expected to stay in deltas for testing):  
  ```bash
  python -m scripts.ingest_infer_tiered \
    --input data/collection/collection_work_delta.tsv \
    --index artifacts/bm25/index \
    --model artifacts/tiering/model.json \
    --threshold artifacts/tiering/threshold.json \
    --qtf artifacts/tiering/qtf.json \
    --feature-names artifacts/tiering/train_train.pkl \
    --out-root artifacts \
    --base-t1 artifacts/tiering/base_T1.tsv \
    --base-t2 artifacts/tiering/base_T2.tsv \
    --delta-t1 artifacts/tiering/delta_T1.tsv \
    --delta-t2 artifacts/tiering/delta_T2.tsv
  ```
  _Why:_ keeps a smaller subset in `bm25_T1_delta` / `bm25_T2_delta` (no rollover), useful for testing base+delta query merging.

## 6) Tiered BM25 Retrieval (base + delta, multiprocessing)
- Run tiered retrieval with merge/rescore over base+delta (multiprocessing):
  ```bash
  python -m scripts.run_tiered_multi \
    --system bm25 \
    --qrels <dev | eval1 | eval2> \
    --save <output_run_file> \
    --topk 100 \
    --overfetch-factor 2 \
    --workers <num_workers>
  ```
- Searches `bm25_T1`, `bm25_T2`, `bm25_T1_delta`, `bm25_T2_delta`, overfetches per index, rescoring with global BM25 stats, merges, and outputs topK in the same run format as `scripts/run.py`.

## 7) Evaluation Readiness
- The work split excludes all docs used for training.
- All eval docs live in the work split, ensuring alignment with existing eval1/eval2/dev evaluations.

## 8) Query Readiness
- All docs in the work split enter our base/delta tiered indexes purely through tiered inference (Fully learned approach).
- The init working split is large which triggers a roll and build from the delta tiered index to the base tiered index.
- The delta working split is small which avoids a roll and build ensuring we have documents in both the base tiered indexes and the delta tiered index at query time.


