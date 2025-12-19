"""
Filter MS MARCO queries/qrels down to only queries whose judged relevant docs
exist in the given corpus.

Usage:
  python -m scripts.filter_queries \
    --collection collection_work.tsv \
    --queries queries.train.tsv \
    --qrels qrels.train.tsv \
    --out-queries queries.train.filtered.tsv \
    --out-qrels qrels.train.filtered.tsv
"""

import os
from argparse import ArgumentParser
from typing import Set, Dict, List

from utils.io import load_docids, load_queries, load_qrels
from utils.config import COLLECTION_DIR, QUERIES_DIR, QRELS_DIR


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--out-queries", required=True)
    parser.add_argument("--out-qrels", required=True)
    args = parser.parse_args()

    collection_path = os.path.join(COLLECTION_DIR, args.collection)
    queries_path = os.path.join(QUERIES_DIR, args.queries)
    qrels_path = os.path.join(QRELS_DIR, args.qrels)

    out_queries_path = os.path.join(QUERIES_DIR, args.out_queries)
    out_qrels_path = os.path.join(QRELS_DIR, args.out_qrels)

    os.makedirs(os.path.dirname(out_queries_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_qrels_path), exist_ok=True)

    docids: Set[str] = load_docids(collection_path)
    queries: Dict[str, str] = load_queries(queries_path)
    qrels: Dict[str, Dict[str, int]] = load_qrels(qrels_path)

    kept_qids: List[str] = []
    dropped = 0

    for qid, rels in qrels.items():
        if qid not in queries:
            dropped += 1
            continue

        if any(docid in docids and rel > 0 for docid, rel in rels.items()):
            kept_qids.append(qid)
        else:
            dropped += 1

    with open(out_queries_path, "w", encoding="utf-8") as f:
        for qid in kept_qids:
            f.write(f"{qid}\t{queries[qid]}\n")

    with open(out_qrels_path, "w", encoding="utf-8") as f:
        for qid in kept_qids:
            for docid, rel in qrels[qid].items():
                if rel > 0:
                    f.write(f"{qid}\t0\t{docid}\t{rel}\n")

    print(f"[Filter] Kept: {len(kept_qids):,}")
    print(f"[Filter] Dropped: {dropped:,}")
    print(f"[Filter] Wrote queries to {out_queries_path}")
    print(f"[Filter] Wrote qrels to {out_qrels_path}")


if __name__ == "__main__":
    main()