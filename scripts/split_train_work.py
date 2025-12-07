"""
Split the corpus into train and work sets, ensuring all qrels docIds land in the work set.

Usage (defaults to config paths):
    python -m scripts.split_train_work

With overrides:
    python -m scripts.split_train_work \
      --collection data/collection/collection.tsv \
      --qrels-dev data/qrels/qrels.dev.tsv \
      --qrels-eval1 data/qrels/qrels.eval.one.tsv \
      --qrels-eval2 data/qrels/qrels.eval.two.tsv \
      --work-frac 0.3 \
      --seed 42 \
      --docids-working-out data/collection/docids_working.txt \
      --train-out data/collection/collection_train.tsv \
      --work-out data/collection/collection_work.tsv
"""

import argparse
import math
import random
from pathlib import Path
from typing import Iterable, Set

from utils.config import (
    DATASET_PATH,
    QRELS_DEV_PATH,
    QRELS_EVAL1_PATH,
    QRELS_EVAL2_PATH,
)


def load_qrels_docids(paths: Iterable[Path]) -> Set[str]:
    docids: Set[str] = set()
    for path in paths:
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    # dev: qid docid rel
                    docid = parts[1]
                else:
                    # eval: qid _ docid rel
                    docid = parts[2]
                docids.add(docid)
    return docids


def main() -> None:
    ap = argparse.ArgumentParser(description="Split corpus into train/work ensuring qrels docIds are in work set.")
    ap.add_argument("--collection", default=DATASET_PATH, help="Path to collection.tsv")
    ap.add_argument("--qrels-dev", default=QRELS_DEV_PATH, help="Path to qrels.dev.tsv")
    ap.add_argument("--qrels-eval1", default=QRELS_EVAL1_PATH, help="Path to qrels.eval.one.tsv")
    ap.add_argument("--qrels-eval2", default=QRELS_EVAL2_PATH, help="Path to qrels.eval.two.tsv")
    ap.add_argument("--work-frac", type=float, default=0.3, help="Target fraction of docs in work set (includes all qrels docIds)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for sampling non-qrels docs")
    ap.add_argument("--docids-working-out", default="data/collection/docids_working.txt", help="Output path for working docIds")
    ap.add_argument("--train-out", default="data/collection/collection_train.tsv", help="Output path for train subset TSV")
    ap.add_argument("--work-out", default="data/collection/collection_work.tsv", help="Output path for work subset TSV")
    args = ap.parse_args()

    collection_path = Path(args.collection)
    qrels_paths = [Path(args.qrels_dev), Path(args.qrels_eval1), Path(args.qrels_eval2)]
    docids_working_out = Path(args.docids_working_out)
    train_out = Path(args.train_out)
    work_out = Path(args.work_out)

    qrels_docids = load_qrels_docids(qrels_paths)
    print(f"Loaded {len(qrels_docids)} unique qrels docIds.")

    # First pass: count total docs and qrels present
    total = 0
    qrels_in_coll = 0
    with collection_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            docid = line.split("\t", 1)[0]
            if docid in qrels_docids:
                qrels_in_coll += 1

    work_target = math.ceil(total * args.work_frac)
    remaining_needed = max(work_target - qrels_in_coll, 0)
    remaining_pool = total - qrels_in_coll
    prob = remaining_needed / remaining_pool if remaining_pool else 0

    print(f"Total docs: {total}")
    print(f"Qrels docIds found in collection: {qrels_in_coll}")
    print(f"Work target (~{args.work_frac*100:.0f}%): {work_target}")
    print(f"Sampling probability for non-qrels: {prob:.4f}")

    rng = random.Random(args.seed)
    working_ids = []

    # Second pass: split and write outputs
    with collection_path.open() as fin, train_out.open("w") as ftrain, work_out.open("w") as fwork:
        for line in fin:
            if not line.strip():
                continue
            docid, rest = line.split("\t", 1)
            if docid in qrels_docids:
                fwork.write(line)
                working_ids.append(docid)
            else:
                if rng.random() < prob:
                    fwork.write(line)
                    working_ids.append(docid)
                else:
                    ftrain.write(line)

    docids_working_out.write_text("\n".join(working_ids) + "\n")
    print(f"Wrote working docIds: {len(working_ids)} -> {docids_working_out}")
    print(f"Train TSV: {train_out}")
    print(f"Work TSV:  {work_out}")


if __name__ == "__main__":
    main()
