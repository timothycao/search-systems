"""
Generate query-routing labels using judged qrels.

Label rule:
- 0 if any judged relevant doc is retrieved in T1-only top-k
- 1 if not in T1-only top-k, but is in FULL (T1+T2) top-k
- drop if not even in FULL top-k

Usage:
  python -m scripts.generate_query_routing_labels \
    --qrels qrels.train.filtered.tsv \
    --run-t1 bm25_train_filtered_t1.run \
    --run-full bm25_train_filtered_full.run \
    --topk 100 \
    --out labels.train_filtered.json
"""

import json
import os
from argparse import ArgumentParser
from typing import Dict, Set

from utils.io import load_qrels, load_run_topk
from utils.config import QRELS_DIR, RUNS_DIR, ARTIFACTS_DIR


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--run-t1", required=True)
    parser.add_argument("--run-full", required=True)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    qrels_path = os.path.join(QRELS_DIR, args.qrels)

    run_dir = os.path.join(RUNS_DIR, "bm25_tiered")
    run_t1_path = os.path.join(run_dir, args.run_t1)
    run_full_path = os.path.join(run_dir, args.run_full)

    out_dir = os.path.join(ARTIFACTS_DIR, "query_routing")
    out_path = os.path.join(out_dir, args.out)
    os.makedirs(out_dir, exist_ok=True)

    qrels: Dict[str, Dict[str, int]] = load_qrels(qrels_path)
    t1_topk = load_run_topk(run_t1_path, args.topk)
    full_topk = load_run_topk(run_full_path, args.topk)

    labels: Dict[str, int] = {}
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

        # Assign label 0 if any relevant doc appears in T1-only top-k
        if rel_docs & t1_docs: labels[qid] = 0
        
        # Assign label 1 if relevant doc appears only after falling through to FULL
        elif rel_docs & full_docs: labels[qid] = 1
        
        # Drop queries where even FULL retrieval misses all relevant docs
        else: dropped_no_full_hit += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    print(f"[Labels] Kept: {len(labels):,}")
    print(f"[Labels] Dropped (no FULL hit): {dropped_no_full_hit:,}")
    print(f"[Labels] Dropped (missing run): {dropped_missing_run:,}")
    print(f"[Labels] Wrote labels to {out_path}")


if __name__ == "__main__":
    main()