"""
Sample a fixed number of queries (and matching qrels) for faster experiments.

Usage:
  python -m scripts.sample_queries \
    --queries queries.train.filtered.tsv \
    --qrels qrels.train.filtered.tsv \
    --n 5000 \
    --seed 42 \
    --out-queries queries.train.sample.tsv \
    --out-qrels qrels.train.sample.tsv
"""

import os
import random
from argparse import ArgumentParser
from typing import Dict, List

from utils.io import load_queries, load_qrels
from utils.config import QUERIES_DIR, QRELS_DIR


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-queries", required=True)
    parser.add_argument("--out-qrels", required=True)
    args = parser.parse_args()

    queries_path = os.path.join(QUERIES_DIR, args.queries)
    qrels_path = os.path.join(QRELS_DIR, args.qrels)

    out_queries_path = os.path.join(QUERIES_DIR, args.out_queries)
    out_qrels_path = os.path.join(QRELS_DIR, args.out_qrels)
    
    os.makedirs(os.path.dirname(out_queries_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_qrels_path), exist_ok=True)

    queries: Dict[str, str] = load_queries(queries_path)
    qrels: Dict[str, Dict[str, int]] = load_qrels(qrels_path)

    candidate_qids: List[str] = [qid for qid in qrels.keys() if qid in queries]

    # Deterministic sampling
    rng = random.Random(args.seed)
    rng.shuffle(candidate_qids)

    n = min(args.n, len(candidate_qids))
    sampled_qids = candidate_qids[:n]

    # Write sampled queries
    with open(out_queries_path, "w", encoding="utf-8") as f:
        for qid in sampled_qids:
            f.write(f"{qid}\t{queries[qid]}\n")

    # Write sampled qrels (4-col format)
    with open(out_qrels_path, "w", encoding="utf-8") as f:
        for qid in sampled_qids:
            for docid, rel in qrels[qid].items():
                f.write(f"{qid}\t0\t{docid}\t{rel}\n")

    print(f"[Sample] Candidates: {len(candidate_qids):,}")
    print(f"[Sample] Sampled: {len(sampled_qids):,} (n={args.n}, seed={args.seed})")
    print(f"[Sample] Wrote queries to {out_queries_path}")
    print(f"[Sample] Wrote qrels to {out_qrels_path}")


if __name__ == "__main__":
    main()