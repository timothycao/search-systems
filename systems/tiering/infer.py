"""
Inference utilities to assign tiers to new documents and route them into delta buffers.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xgboost as xgb

from search_system.query.query_startup_context import QueryStartupContext
from search_system.shared.utils import tokenize


def load_qtf(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_idf(df: int, N: int) -> float:
    numerator = N - df + 0.5
    denominator = df + 0.5
    return math.log((numerator / denominator) + 1.0)


def compute_static_score(text: str, qtf: Dict[str, int], ctx: QueryStartupContext, k1: float = 1.2, b: float = 0.75) -> float:
    tokens = tokenize(text)
    tf: Dict[str, int] = {}
    for tok in tokens:
        tf[tok] = tf.get(tok, 0) + 1

    doc_len = len(tokens) if tokens else 1
    score = 0.0
    N = ctx.total_docs
    avg_len = ctx.avg_len
    lexicon = ctx.lexicon

    for term, freq in tf.items():
        q_weight = qtf.get(term, 0)
        if q_weight == 0:
            continue
        term_meta = lexicon.get(term)
        if term_meta is None:
            continue
        df = term_meta.get("df", 0)
        idf = compute_idf(df, N)
        denom = freq + k1 * (1 - b + b * (doc_len / avg_len))
        bm25 = idf * (freq * (k1 + 1.0) / denom) if denom else 0.0
        score += q_weight * bm25
    return score


def compute_features(text: str, qtf: Dict[str, int], ctx: QueryStartupContext) -> Dict[str, float]:
    tokens = tokenize(text)
    doc_len = len(tokens)

    # term frequencies
    tf: Dict[str, int] = {}
    for tok in tokens:
        tf[tok] = tf.get(tok, 0) + 1

    unique_terms = len(tf)
    sum_idf = 0.0
    sum_idf_sq = 0.0
    max_idf = 0.0
    for term, freq in tf.items():
        term_meta = ctx.lexicon.get(term)
        if term_meta is None:
            continue
        idf = compute_idf(term_meta.get("df", 0), ctx.total_docs)
        sum_idf += idf
        sum_idf_sq += idf * idf
        if idf > max_idf:
            max_idf = idf

    mean_idf = sum_idf / unique_terms if unique_terms > 0 else 0.0
    var_idf = (sum_idf_sq / unique_terms - mean_idf * mean_idf) if unique_terms > 0 else 0.0
    std_idf = math.sqrt(var_idf) if var_idf > 0 else 0.0

    entropy = 0.0
    if doc_len > 0:
        sum_tf_log_tf = sum(freq * math.log(freq) for freq in tf.values() if freq > 0)
        entropy = (doc_len * math.log(doc_len) - sum_tf_log_tf) / doc_len

    static_score = compute_static_score(text, qtf, ctx)

    return {
        "doc_len": doc_len,
        "doc_len_log1p": math.log1p(doc_len),
        "static_score": static_score,
        "static_score_log1p": math.log1p(static_score),
        "unique_term_count": unique_terms,
        "mean_idf": mean_idf,
        "max_idf": max_idf,
        "std_idf": std_idf,
        "entropy_tf": entropy,
    }


def load_feature_names(path: Path) -> List[str]:
    import pickle
    with path.open("rb") as f:
        ds = pickle.load(f)
    return ds.get("feature_names", [])


def build_feature_vector(features: Dict[str, float], feature_names: List[str]) -> np.ndarray:
    return np.array([features.get(name, 0.0) for name in feature_names], dtype=np.float32).reshape(1, -1)


def predict_tier(
    model_path: Path,
    threshold_path: Path,
    feature_vector: np.ndarray,
    feature_names: List[str],
) -> Tuple[int, float]:
    model = xgb.Booster()
    model.load_model(model_path)
    threshold = json.loads(threshold_path.read_text())["threshold"]
    dm = xgb.DMatrix(feature_vector, feature_names=feature_names if feature_names else None)
    prob = float(model.predict(dm, iteration_range=(0, model.best_iteration + 1))[0])
    tier = 1 if prob >= threshold else 0
    return tier, prob
