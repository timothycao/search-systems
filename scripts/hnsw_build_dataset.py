"""
Build HNSW tiering features and train/val datasets from static scores, labels, and embeddings.

Features (per doc):
- static_score
- log1p_static_score
- embedding_norm

Outputs:
- features pickle: doc_id -> feature vector (list[float])
- train/val pickles with X, y, feature_names

Usage (example):
  python -m scripts.hnsw_build_dataset \
    --scores artifacts/tiering_dense/static_scores_hnsw.npy \
    --labels artifacts/tiering_dense/labels_hnsw.json \
    --embeddings data/collection/collection_train_hnsw.h5 \
    --features-out artifacts/tiering_dense/features_hnsw.pkl \
    --train-out artifacts/tiering_dense/train_hnsw.pkl \
    --val-out artifacts/tiering_dense/val_hnsw.pkl \
    --val-ratio 0.2 --seed 42
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from search_system.shared.utils import tokenize


def load_embeddings(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        ids = np.array(f["id"]).astype(str)
        emb = np.array(f["embedding"]).astype(np.float32)
    return ids, emb


def main() -> None:
    ap = argparse.ArgumentParser(description="Build HNSW tiering features and train/val datasets.")
    ap.add_argument("--scores", required=True, help="Numpy file of static_scores_hnsw.npz (with aggregates)")
    ap.add_argument("--labels", required=True, help="JSON file of labels_hnsw.json")
    ap.add_argument("--embeddings", required=True, help="HDF5 with train embeddings (id/embedding)")
    ap.add_argument("--doc-collection", required=True, help="TSV of doc_id<TAB>text for the same split")
    ap.add_argument("--features-out", required=True, help="Output pickle for features dict")
    ap.add_argument("--train-out", required=True, help="Output pickle for train split")
    ap.add_argument("--val-out", required=True, help="Output pickle for val split")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Loading labels and scores...")
    with open(args.labels, "r", encoding="utf-8") as f:
        labels: Dict[str, int] = json.load(f)
    scores = np.load(args.scores)

    print("Loading embeddings...")
    ids, emb = load_embeddings(Path(args.embeddings))
    id_to_idx = {doc_id: i for i, doc_id in enumerate(ids)}

    # Load scores and aggregates
    scores_np = np.load(args.scores)
    if isinstance(scores_np, np.lib.npyio.NpzFile):
        static_scores = scores_np["score"]
        sim_max = scores_np["sim_max"]
        sim_std = scores_np["sim_std"]
        sim_p90 = scores_np["sim_p90"]
    else:
        static_scores = scores_np
        sim_max = np.zeros_like(static_scores)
        sim_std = np.zeros_like(static_scores)
        sim_p90 = np.zeros_like(static_scores)

    # Precompute doc stats
    print("Computing doc length/entropy stats...")
    doc_len: Dict[str, int] = {}
    uniq_count: Dict[str, int] = {}
    entropy: Dict[str, float] = {}
    with Path(args.doc_collection).open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Docs", unit="doc"):
            if not line.strip():
                continue
            doc_id, text = line.rstrip("\n").split("\t", 1)
            tokens = tokenize(text)
            length = len(tokens)
            doc_len[doc_id] = length
            if length == 0:
                uniq_count[doc_id] = 0
                entropy[doc_id] = 0.0
                continue
            freq: Dict[str, int] = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            uniq_count[doc_id] = len(freq)
            probs = np.array(list(freq.values()), dtype=np.float32) / float(length)
            entropy[doc_id] = float(-np.sum(probs * np.log(probs + 1e-12)))

    feature_names = [
        "static_score",
        "log1p_static_score",
        "sim_max",
        "sim_std",
        "sim_p90",
        "embedding_norm",
        "log1p_embedding_norm",
        "doc_len",
        "log1p_doc_len",
        "unique_term_count",
        "tf_entropy",
    ]
    features: Dict[str, List[float]] = {}
    X = []
    y = []

    print("Building features...")
    for doc_id, label in tqdm(labels.items(), desc="Docs", unit="doc"):
        idx = id_to_idx.get(doc_id)
        if idx is None:
            continue
        static = float(static_scores[idx])
        norm = float(np.linalg.norm(emb[idx]))
        fv = [
            static,
            np.log1p(static),
            float(sim_max[idx]),
            float(sim_std[idx]),
            float(sim_p90[idx]),
            norm,
            np.log1p(norm),
            float(doc_len.get(doc_id, 0)),
            np.log1p(float(doc_len.get(doc_id, 0))),
            float(uniq_count.get(doc_id, 0)),
            float(entropy.get(doc_id, 0.0)),
        ]
        features[doc_id] = fv
        X.append(fv)
        y.append(int(label))

    # Save features dict
    Path(args.features_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.features_out, "wb") as f:
        pickle.dump(features, f)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_ratio, random_state=args.seed, stratify=y
    )

    train_payload = {"X": X_train, "y": y_train, "feature_names": feature_names}
    val_payload = {"X": X_val, "y": y_val, "feature_names": feature_names}

    with open(args.train_out, "wb") as f:
        pickle.dump(train_payload, f)
    with open(args.val_out, "wb") as f:
        pickle.dump(val_payload, f)

    print(f"Saved features: {len(features)} docs -> {args.features_out}")
    print(f"Train split: {len(y_train)} docs -> {args.train_out}")
    print(f"Val split: {len(y_val)} docs -> {args.val_out}")


if __name__ == "__main__":
    main()
