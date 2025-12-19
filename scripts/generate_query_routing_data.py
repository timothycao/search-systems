"""
Generate supervised query-routing data (labels + simple features) using judged qrels.

Label rule:
- 0 if any judged relevant doc is retrieved in T1-only top-k
- 1 if not in T1-only top-k, but is in FULL (T1+T2) top-k
- drop if not even in FULL top-k

Features (derived from runs at top-k):
- t1_hit:          1 if any relevant doc in T1 top-k else 0
- full_hit:        1 if any relevant doc in FULL top-k else 0
- t1_rel_hits:     count of relevant docs in T1 top-k
- full_rel_hits:   count of relevant docs in FULL top-k
- t1_overlap_full: |T1 ∩ FULL| / |T1|   (0 if T1 empty)
- full_overlap_t1: |T1 ∩ FULL| / |FULL| (0 if FULL empty)

Usage:
  python -m scripts.generate_query_routing_data \
    --qrels qrels.train.filtered.tsv \
    --run-t1 bm25_train_filtered_t1.run \
    --run-full bm25_train_filtered_full.run \
    --topk 100 \
    --out-labels labels.train_filtered.json \
    --out-features features.train_filtered.json
"""

import json
import os
from argparse import ArgumentParser
from typing import Dict, Set, Any

from utils.io import load_qrels, load_run_topk
from utils.config import QRELS_DIR, RUNS_DIR, ARTIFACTS_DIR


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--run-t1", required=True)
    parser.add_argument("--run-full", required=True)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--out-labels", required=True)
    parser.add_argument("--out-features", required=True)
    args = parser.parse_args()

    qrels_path = os.path.join(QRELS_DIR, args.qrels)

    run_dir = os.path.join(RUNS_DIR, "bm25_tiered")
    run_t1_path = os.path.join(run_dir, args.run_t1)
    run_full_path = os.path.join(run_dir, args.run_full)

    out_dir = os.path.join(ARTIFACTS_DIR, "query_routing")
    out_labels_path = os.path.join(out_dir, args.out_labels)
    out_features_path = os.path.join(out_dir, args.out_features)
    os.makedirs(out_dir, exist_ok=True)

    qrels: Dict[str, Dict[str, int]] = load_qrels(qrels_path)
    t1_topk = load_run_topk(run_t1_path, args.topk)
    full_topk = load_run_topk(run_full_path, args.topk)

    labels: Dict[str, int] = {}
    features: Dict[str, Dict[str, Any]] = {}
    dropped_no_full_hit = 0
    dropped_missing_run = 0

    for qid, rels in qrels.items():
        # Extract set of judged relevant docids for this query
        rel_docs: Set[str] = {docid for docid, rel in rels.items() if rel > 0}
        if not rel_docs: continue   # skip if none

        # Retrieve top-k doc sets from T1-only and FULL runs
        t1_docs = set(t1_topk.get(qid, []))
        full_docs = set(full_topk.get(qid, []))

        # Drop queries missing from the FULL run
        if not full_docs:
            dropped_missing_run += 1
            continue

        # Compute hit/overlap stats from runs (top-k only)
        t1_rel_hits = len(rel_docs & t1_docs)
        full_rel_hits = len(rel_docs & full_docs)
        inter = len(t1_docs & full_docs)

        feats = {
            "t1_hit": 1 if t1_rel_hits > 0 else 0,
            "full_hit": 1 if full_rel_hits > 0 else 0,
            "t1_rel_hits": t1_rel_hits,
            "full_rel_hits": full_rel_hits,
            "t1_overlap_full": (inter / len(t1_docs)) if len(t1_docs) > 0 else 0.0,
            "full_overlap_t1": (inter / len(full_docs)) if len(full_docs) > 0 else 0.0,
        }

        # Assign label 0 if any relevant doc appears in T1-only top-k
        if t1_rel_hits > 0:
            labels[qid] = 0
            features[qid] = feats

        # Assign label 1 if relevant doc appears only after falling through to FULL
        elif full_rel_hits > 0:
            labels[qid] = 1
            features[qid] = feats

        # Drop queries where even FULL retrieval misses all relevant docs
        else:
            dropped_no_full_hit += 1

    with open(out_labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    with open(out_features_path, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)

    print(f"[Data] Kept: {len(labels):,}")
    print(f"[Data] Dropped (no FULL hit): {dropped_no_full_hit:,}")
    print(f"[Data] Dropped (missing run): {dropped_missing_run:,}")
    print(f"[Data] Wrote labels to {out_labels_path}")
    print(f"[Data] Wrote features to {out_features_path}")


if __name__ == "__main__":
    main()