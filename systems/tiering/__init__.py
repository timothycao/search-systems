"""Dynamic document tiering utilities."""

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
)
from .infer import (
    compute_features as infer_compute_features,
    build_feature_vector as infer_build_feature_vector,
    predict_tier,
)
from .ingest import (
    append_delta,
    delta_count,
    rebuild_delta_index,
    rebuild_tier,
    route_and_maybe_rebuild,
)

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
    "infer_compute_features",
    "infer_build_feature_vector",
    "predict_tier",
    "append_delta",
    "delta_count",
    "rebuild_delta_index",
    "rebuild_tier",
    "route_and_maybe_rebuild",
]
