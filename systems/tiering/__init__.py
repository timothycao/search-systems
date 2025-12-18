"""Dynamic document tiering utilities.

Note: Heavy dependencies (e.g., xgboost for inference) are intentionally not imported
at package import time. Import the specific modules you need (e.g., systems.tiering.infer)
to avoid pulling optional deps when running lightweight CLI commands.
"""

from .tiering import (
    compute_qtf,
    load_qtf,
    compute_static_scores,
    normalize_scores,
    assign_tiers,
    compute_doc_features,
    save_features,
    load_features,
    assemble_dataset,
    stratified_split,
    save_dataset,
    load_dataset,
    select_threshold,
    evaluate_at_threshold,
)
from .infer import predict_tier, build_feature_vector, load_feature_names

__all__ = [
    "compute_qtf",
    "load_qtf",
    "compute_static_scores",
    "normalize_scores",
    "assign_tiers",
    "compute_doc_features",
    "save_features",
    "load_features",
    "assemble_dataset",
    "stratified_split",
    "save_dataset",
    "load_dataset",
    "select_threshold",
    "evaluate_at_threshold",
    "predict_tier",
    "build_feature_vector",
    "load_feature_names",
]
