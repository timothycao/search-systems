# Tiered BM25 (Train/Work Split) — Full Rationale and Technical Details

This document explains the motivation, design decisions, and mathematical underpinnings of the revised tiered BM25 pipeline. The key goals are: (1) avoid train/eval leakage by separating training and working corpora, (2) learn a tiering model on train-only data, (3) apply tier inference to the held-out working set with delta/base maintenance, and (4) enable systematic evaluation (MRR@10, Recall@100, MAP, NDCG@10, NDCG@100) comparing tiered/delta vs. non-tiered baselines.

## 1) Motivation for the Train/Work Split
- **Leakage avoidance:** We exclude all documents appearing in qrels (dev/eval sets) from the training corpus to ensure downstream evaluation is apples-to-apples. Eval docs live only in the working split and enter via inference.
- **Realistic ingestion:** The working set is treated as “new” documents; we infer their tiers instead of using ground-truth labels, mirroring a production ingestion flow.
- **Fair comparison:** By keeping eval docs in the working split, we can compare a tiered/delta system to a non-tiered baseline on the same documents.

## 2) Constructing Train vs. Work
- **Selection logic:** Include all qrels docIds (dev + eval1 + eval2) in the working set. Sample additional docs to reach ~30% of the corpus; the remaining ~70% becomes the training set.
- **Scripted split:** `scripts.split_train_work` produces:
  - `collection_train.tsv` (~70%)
  - `collection_work.tsv` (~30%, guaranteed to contain all qrels docs)
  - `docids_working.txt` (working docIds)
- **Integrity:** Union(train, work) = original corpus; intersection = ∅; docIds preserved.

## 3) QTF and Static Scores (Train Only)
- **QTF:** Query Term Frequency is computed over the full query log (200k queries) with the same preprocessing as the index. We retain a single `qtf.json` (can be reused).
- **Static score math:** For doc `d`,
  ```
  StaticScore(d) = Σ_{t∈d} QTF(t) * w(t, d)
  w(t, d) = idf(t) * (tf_td * (k1+1)) / (tf_td + k1 * (1 - b + b * len_d / avgdl))
  ```
  where `tf_td` is term frequency in doc, `len_d` doc length, `avgdl` corpus avg length, `idf(t)`, and BM25 `k1`, `b`.
- **Rationale:** QTF weights BM25 term contributions by query popularity, yielding a query-log–aligned prior relevance signal.

## 4) Labeling and Feature Extraction (Train Only)
- **Labels:** Min–max normalize static scores, rank, and assign Tier-1 to the top ratio (e.g., 40%), Tier-2 to the rest.
- **Features:** Static score + log variant; length + log length; IDF stats (mean/max/std); lexical entropy; unique term count. Extracted against the train BM25 index to ensure consistent stats (idf/avgdl).
- **Datasets:** Assemble train/val splits (e.g., 80/20) from train-only docs; no eval/work docs involved.

## 5) Kaggle External XGBoost Model Training (Train Only, External)
- **Why XGBoost:** Gradient-boosted decision trees handle heterogeneous, non-linear feature interactions (static scores, lengths, IDF stats, entropy) without heavy feature scaling. They are robust to mixed magnitudes, capture thresholds and interactions naturally, and train quickly on GPU with strong tabular performance.
- **How it works:** Boosting builds an ensemble of shallow trees sequentially, each new tree fitting the residuals (errors) of the current ensemble. With shrinkage (learning rate), subsampling, and depth limits, the model learns complex decision boundaries while controlling overfitting. Early stopping on a held-out validation set halts training when generalization stops improving.
- **Why ideal here:** Our feature space is tabular and moderate-sized; relationships between static score, IDF stats, and length are non-linear and threshold-like. XGBoost can model such interactions without extensive feature engineering. It also provides probabilistic outputs for flexible thresholding to hit a target Tier-1 ratio.
- **Training recipe:** XGBoost classifier on train/validation features, with early stopping on validation. Hyperparameters tuned for stability (learning rate, max_depth, subsample, colsample_bytree). Use GPU if available for speed.
- **Thresholding:** Select a probability threshold on the validation set to match the desired Tier-1 prevalence (e.g., 0.4), rather than defaulting to 0.5. This aligns model outputs with target tier size.
- **Artifacts:** `model.json`, `threshold.json`, trained solely on the train split.

## 6) Work-Only Index and Inference Ingestion
- **Vanilla work BM25:** Build BM25 over `collection_work.tsv` (artifacts/bm25) to provide idf/avgdl/k1/b/lexicon for feature computation on work docs.
- **Init/Delta split for work:** Deterministic split (seed=42):
  - `collection_work_init.tsv` = ~2.1M docs
  - `collection_work_delta.tsv` = 500k docs
- **Inference flow:** For each input TSV:
  - Compute features using work BM25 stats + global `qtf.json`.
  - Build feature vectors with train feature names.
  - Predict tier with the train-derived model/threshold.
  - Append to `delta_T1.tsv` or `delta_T2.tsv`.
  - If delta exceeds thresholds (T1 > 400k, T2 > 1M): merge delta into base TSV (`base_T1.tsv`/`base_T2.tsv`), clear delta TSV + delta index, rebuild base `bm25_T1`/`bm25_T2` from the updated base TSV.
  - If below threshold: rebuild delta indexes `bm25_T1_delta`/`bm25_T2_delta` so new docs are queryable.
- **Result:** Most work docs end up in base tier indexes (after init ingest); a smaller subset remains in deltas (after delta ingest) to support base+delta query merging tests.

## 7) Evaluation Readiness
- **Eval docs in work:** All qrels docIds are in the working split and were ingested via inference (no training leakage).
- **Comparisons:** Evaluate tiered+delta vs. non-tiered BM25 at multiple topKs: MRR@10, Recall@100, MAP, NDCG@10, NDCG@100.
- **Query-time merging (next step):** Overfetch base+delta and merge (simple merge, RRF, or rescoring) so deltas contribute to retrieved results.

## 8) Retrieval (base + delta)
- Search base and delta BM25 separately, overfetch, merge/rescore with global stats, and return topK. Multiprocessing runner mirrors the non-tiered run format. Example:
  ```bash
  python -m scripts.run_tiered_multi \
    --system bm25 \
    --qrels <dev | eval1 | eval2> \
    --save <output_run_file> \
    --topk 100 \
    --overfetch-factor 2 \
    --workers <num_workers>
  ```

## 9) Why This Design
- **No leakage:** Train artifacts (scores, labels, features, model) are derived only from train docs; working set docs which include eval docs never influence training.
- **Query-log alignment:** QTF-weighted BM25 static scores bias tiering toward terms users actually search.
- **Operational realism:** Inference-driven tiering for new docs mirrors production ingestion; delta thresholds manage rebuild cadence.
- **Traceability:** Base/delta TSVs explicitly track membership; rebuilds are deterministic and reproducible.

## 10) Remaining Work
- Add query-time merge/rescoring, logging, and regression tests for ingest + rebuild.
