# XGBoost Overview and Tiering Results

## XGBoost in Brief
XGBoost (Extreme Gradient Boosting) is a gradient-boosted decision tree library that builds an ensemble of trees to minimize a specified loss. For binary classification, each boosting round adds trees that improve the model’s fit to the residuals of the logistic loss. Key advantages are strong performance on tabular data, regularization to combat overfitting, and support for GPU acceleration.

### Core Parameters (what they control)
- `objective`: Loss to optimize. For binary classification, `binary:logistic` outputs probabilities in [0,1].
- `eval_metric`: Metric reported during training (e.g., `auc` for ranking quality).
- `eta` (learning rate): Shrinks each tree’s contribution. Lower values slow learning but can improve generalization.
- `max_depth`: Maximum depth of each tree. Higher depth increases model complexity/capacity.
- `subsample`: Fraction of training rows sampled per tree. <1.0 adds stochasticity and regularization.
- `colsample_bytree`: Fraction of features sampled per tree. <1.0 adds stochasticity and regularization.
- `num_boost_round`: Maximum number of boosting rounds (trees).
- `early_stopping_rounds`: Stop if the validation metric doesn’t improve for this many rounds.
- `tree_method`: Algorithm for building trees. We use `hist` (fast histogram-based). When GPU is available, pair with `device="cuda"` for acceleration.
- `seed`: Controls randomness (row/feature sampling, initialization).
- `device`: Set to `"cuda"` to run on GPU; omit or set `"cpu"` for CPU.

### How training proceeds
1) Convert data to `DMatrix` (dense/sparse matrix with labels and optional feature names).
2) Train with parameters + train/val sets; watch metrics on both.
3) Early stopping picks the best iteration on the validation metric.
4) For inference, the booster outputs probabilities; a threshold converts probabilities to labels.

## Our Tiering Training Setup
- Objective: `binary:logistic`
- Eval metric: `auc`
- Params: `eta=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method=hist`, `device=cuda` (on GPU), `seed=42`
- Early stopping: 50 rounds
- Max rounds: 500
- Data: Stratified splits from tiering artifacts (`train.pkl`, `val.pkl`)
- Target Tier-1 share: 40% (threshold chosen to match this on validation)

## Results (from artifacts/tiering/metrics.json and threshold.json)
- Best iteration: 64 (early stopping selected this)
- Validation AUC: ~0.999992 (best_score)
- Threshold: 0.4058 (probability cutoff to get ~40% Tier-1)
- Validation precision @ threshold: ~0.9977
- Validation recall @ threshold: ~0.9977
- Validation predicted Tier-1 ratio: ~0.4000 (matches target_ratio=0.4)

## Interpretation of the Results
- **Near-perfect AUC**: The model almost perfectly ranks Tier-1 above Tier-2 on validation. This is expected because labels come from a deterministic static BM25×QTF score, and that score (and its log) is included as a feature.
- **Threshold alignment**: The chosen threshold yields ~40% predicted Tier-1 on validation, matching the labeling policy. Precision and recall are both ~0.998 at this cutoff, meaning almost every predicted Tier-1 is correct and almost every true Tier-1 is captured.
- **Convergence**: Early stopping at 64 rounds indicates the model fit quickly; additional trees didn’t improve validation AUC.
- **Practical takeaway**: As long as the ingestion-time features (especially static_score or its approximation) align with the labeling scheme, the model will reproduce the Tier-1 cutoff reliably. If static_score is unavailable or approximated at ingest, expect some degradation; other features (length, IDF stats, entropy) may not fully substitute the deterministic signal.
