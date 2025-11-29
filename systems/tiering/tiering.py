"""
Utilities for computing query-term frequencies (QTF), static BM25 scores,
and tier assignments for dynamic document tiering.
"""

import json
import math
import os
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional

from tqdm import tqdm

from search_system.query.inverted_list import InvertedList
from search_system.query.query_startup_context import QueryStartupContext
from search_system.shared.utils import tokenize
import numpy as np

try:
    import xgboost as xgb
except Exception:
    xgb = None


@dataclass
class TieringArtifacts:
    """Paths for tiering-related artifacts."""

    qtf_path: str
    static_scores_path: str
    labels_path: str


def compute_qtf(queries_path: str, drop_singletons: bool = False) -> Dict[str, int]:
    """
    Compute term frequencies across a query log using the same tokenizer as the indexer.
    """
    qtf: Dict[str, int] = defaultdict(int)

    with open(queries_path, "r", encoding="utf-8") as f, tqdm(
        desc="Computing QTF", unit="query"
    ) as progress:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            _, text = parts
            for token in tokenize(text):
                qtf[token] += 1
            progress.update(1)

    if drop_singletons:
        qtf = {t: c for t, c in qtf.items() if c > 1}

    return dict(qtf)


def save_qtf(qtf: Dict[str, int], output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qtf, f)


def load_qtf(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_postings(ilist: InvertedList) -> Iterable[Tuple[int, int]]:
    """
    Iterate (doc_id, term_freq) pairs for a single inverted list.
    """
    for block_idx in range(ilist.block_count):
        ilist.load_block(block_idx)
        for doc_id, freq in zip(ilist.curr_block_docIDs, ilist.curr_block_freqs):
            yield doc_id, freq


def compute_static_scores(
    index_dir: str,
    qtf: Dict[str, int],
    k1: float = 1.2,
    b: float = 0.75,
) -> Dict[int, float]:
    """
    Compute static BM25-based scores per document:
        score(d) = sum_t QTF(t) * BM25(tf_td, dl_d)
    """
    ctx = QueryStartupContext(index_dir)
    page_table = ctx.page_table

    doc_scores: Dict[int, float] = {int(doc_id): 0.0 for doc_id in page_table.keys()}

    terms = [(t, w) for t, w in qtf.items() if w > 0]
    with tqdm(total=len(terms), desc="Static score pass", unit="term") as pbar:
        for term, weight in terms:
            term_meta = ctx.lexicon.get(term)
            if term_meta is None:
                pbar.update(1)
                continue

            ilist = InvertedList(
                term=term,
                term_meta=term_meta,
                index_path=ctx.index_path,
                page_table=page_table,
                N=ctx.total_docs,
                avg_len=ctx.avg_len,
                k1=k1,
                b=b,
            )

            for doc_id, freq in _iter_postings(ilist):
                doc_meta = page_table.get(str(doc_id), {})
                doc_len = doc_meta.get("length", 1)
                bm25 = ilist.getBM25(freq, doc_len)
                if bm25 > 0.0:
                    doc_scores[doc_id] += weight * bm25

            ilist.closeList()
            pbar.update(1)

    return doc_scores


def save_scores(scores: Dict[int, float], path: str) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scores, f)


def load_scores(path: str) -> Dict[int, float]:
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
    values = list(scores.values())
    if not values:
        return {}

    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        return {doc_id: 0.0 for doc_id in scores}

    scale = max_score - min_score
    return {doc_id: (score - min_score) / scale for doc_id, score in scores.items()}


def assign_tiers(
    scores: Dict[int, float],
    tier_ratio: float = 0.3,
) -> Dict[int, int]:
    """
    Assign Tier-1 (1) / Tier-2 (0) labels based on normalized scores.
    """
    normalized = normalize_scores(scores)
    sorted_docs = sorted(normalized.items(), key=lambda kv: kv[1], reverse=True)

    cutoff = max(1, int(len(sorted_docs) * tier_ratio))
    tier1_docs = set(doc_id for doc_id, _ in sorted_docs[:cutoff])

    return {doc_id: (1 if doc_id in tier1_docs else 0) for doc_id in scores.keys()}


def compute_doc_features(
    index_dir: str,
    static_scores: Dict[int, float],
) -> Dict[int, Dict[str, float]]:
    """
    Compute document-side features:
      - static_score, log1p(static_score)
      - doc_len, log1p(doc_len)
      - IDF stats: mean, max, std
      - unique_term_count
      - tf entropy
    """
    ctx = QueryStartupContext(index_dir)
    page_table = ctx.page_table
    N = ctx.total_docs

    sum_idf: Dict[int, float] = defaultdict(float)
    sum_idf_sq: Dict[int, float] = defaultdict(float)
    max_idf: Dict[int, float] = defaultdict(float)
    unique_terms: Dict[int, int] = defaultdict(int)
    sum_tf_log_tf: Dict[int, float] = defaultdict(float)

    def compute_idf(df: int) -> float:
        numerator = N - df + 0.5
        denominator = df + 0.5
        return math.log((numerator / denominator) + 1.0)

    terms = list(ctx.lexicon.items())
    with tqdm(total=len(terms), desc="Doc feature pass", unit="term") as pbar:
        for term, term_meta in terms:
            df = term_meta.get("df", 0)
            idf = compute_idf(df)

            ilist = InvertedList(
                term=term,
                term_meta=term_meta,
                index_path=ctx.index_path,
                page_table=page_table,
                N=N,
                avg_len=ctx.avg_len,
                k1=1.2,
                b=0.75,
            )

            for doc_id, freq in _iter_postings(ilist):
                sum_idf[doc_id] += idf
                sum_idf_sq[doc_id] += idf * idf
                if idf > max_idf[doc_id]:
                    max_idf[doc_id] = idf
                unique_terms[doc_id] += 1
                if freq > 0:
                    sum_tf_log_tf[doc_id] += freq * math.log(freq)

            ilist.closeList()
            pbar.update(1)

    features: Dict[int, Dict[str, float]] = {}
    for doc_id_str, meta in page_table.items():
        doc_id = int(doc_id_str)
        doc_len = meta.get("length", 0)
        ut = unique_terms.get(doc_id, 0)

        mean_idf = sum_idf[doc_id] / ut if ut > 0 else 0.0
        variance = (sum_idf_sq[doc_id] / ut - mean_idf * mean_idf) if ut > 0 else 0.0
        std_idf = math.sqrt(variance) if variance > 0 else 0.0

        entropy = 0.0
        if doc_len > 0:
            entropy = (doc_len * math.log(doc_len) - sum_tf_log_tf[doc_id]) / doc_len

        static_score = static_scores.get(doc_id, 0.0)

        features[doc_id] = {
            "doc_len": doc_len,
            "doc_len_log1p": math.log1p(doc_len),
            "static_score": static_score,
            "static_score_log1p": math.log1p(static_score),
            "unique_term_count": ut,
            "mean_idf": mean_idf,
            "max_idf": max_idf.get(doc_id, 0.0),
            "std_idf": std_idf,
            "entropy_tf": entropy,
        }

    return features


def save_features(features: Dict[int, Dict[str, float]], path: str) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(features, f)


def load_features(path: str) -> Dict[int, Dict[str, float]]:
    with open(path, "rb") as f:
        return pickle.load(f)


def assemble_dataset(
    features: Dict[int, Dict[str, float]],
    labels: Dict[str, int],
) -> Dict[str, object]:
    """
    Join features and labels on doc_id, returning a dataset dict with:
      - doc_ids: list[int]
      - feature_names: list[str] (ordered)
      - X: list[list[float]]
      - y: list[int]
    """
    # Use intersection to avoid missing entries
    label_int = {int(k): int(v) for k, v in labels.items()}
    common_ids = sorted(set(features.keys()) & set(label_int.keys()))
    if not common_ids:
        return {"doc_ids": [], "feature_names": [], "X": [], "y": []}

    feature_names = sorted(next(iter(features.values())).keys())
    X: List[List[float]] = []
    y: List[int] = []

    for doc_id in common_ids:
        feats = features.get(doc_id)
        if feats is None:
            continue
        X.append([feats.get(name, 0.0) for name in feature_names])
        y.append(label_int[doc_id])

    return {
        "doc_ids": common_ids,
        "feature_names": feature_names,
        "X": X,
        "y": y,
    }


def stratified_split(
    dataset: Dict[str, object],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """
    Stratified train/val split on labels.
    """
    rng = random.Random(seed)
    y = dataset.get("y", [])
    doc_ids = dataset.get("doc_ids", [])
    X = dataset.get("X", [])
    feature_names = dataset.get("feature_names", [])

    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(y):
        label_to_indices[label].append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []
    for indices in label_to_indices.values():
        rng.shuffle(indices)
        val_count = max(1, int(len(indices) * val_ratio)) if len(indices) > 0 else 0
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    def subset(indices: List[int]) -> Dict[str, object]:
        return {
            "doc_ids": [doc_ids[i] for i in indices],
            "feature_names": feature_names,
            "X": [X[i] for i in indices],
            "y": [y[i] for i in indices],
        }

    return subset(train_indices), subset(val_indices)


def save_dataset(dataset: Dict[str, object], path: str) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dataset, f)


def load_dataset(path: str) -> Dict[str, object]:
    with open(path, "rb") as f:
        return pickle.load(f)


def train_xgboost_model(
    train_dataset: Dict[str, object],
    val_dataset: Dict[str, object],
    num_rounds: int = 500,
    early_stopping_rounds: int = 50,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    use_gpu: bool = False,
    seed: int = 42,
) -> Tuple[Optional["xgb.Booster"], Dict[str, object]]:
    """
    Train an XGBoost binary classifier with early stopping. Returns (model, metrics dict).
    """
    if xgb is None:
        raise ImportError("xgboost is not installed; install it to train the model.")

    if not train_dataset.get("X") or not train_dataset.get("y"):
        return None, {"error": "empty training data"}
    if not val_dataset.get("X") or not val_dataset.get("y"):
        return None, {"error": "empty validation data"}

    feature_names = train_dataset.get("feature_names", [])
    X_train = np.array(train_dataset["X"], dtype=np.float32)
    y_train = np.array(train_dataset["y"], dtype=np.float32)
    X_val = np.array(val_dataset["X"], dtype=np.float32)
    y_val = np.array(val_dataset["y"], dtype=np.float32)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names if feature_names else None)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names if feature_names else None)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": learning_rate,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "seed": seed,
        "tree_method": "gpu_hist" if use_gpu else "hist",
    }

    evals_result: Dict[str, List[float]] = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )

    metrics = {
        "best_iteration": booster.best_iteration,
        "best_score": booster.best_score,
        "eval_metric": params["eval_metric"],
        "evals_result": evals_result,
    }

    return booster, metrics


def select_threshold(probs: List[float], target_ratio: float = 0.4) -> float:
    """
    Choose a threshold so that approximately target_ratio of items are predicted as Tier-1.
    """
    if not probs:
        return 0.5
    sorted_probs = sorted(probs, reverse=True)
    cutoff_idx = max(0, min(len(sorted_probs) - 1, int(len(sorted_probs) * target_ratio) - 1))
    return sorted_probs[cutoff_idx]


def evaluate_at_threshold(
    probs: List[float],
    labels: List[int],
    threshold: float,
) -> Dict[str, float]:
    """
    Compute precision/recall for Tier-1 at the given threshold.
    """
    if not probs or not labels:
        return {"precision": 0.0, "recall": 0.0, "pred_ratio": 0.0}

    preds = [1 if p >= threshold else 0 for p in probs]
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pred_ratio = sum(preds) / len(preds) if preds else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "pred_ratio": pred_ratio,
    }
