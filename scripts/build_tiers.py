"""
Build Tier-1 and Tier-2 BM25 indexes using precomputed tier labels.

Steps:
- Read doc_id->label from artifacts/tiering/labels.json
- Write subset ID files for Tier-1 and Tier-2
- Run parser/indexer for each tier into artifacts/bm25_T1 and artifacts/bm25_T2

Usage:
  python -m scripts.build_tiers \
    [--labels artifacts/tiering/labels.json] \
    [--dataset data/collection/collection.tsv] \
    [--out-root artifacts]
"""

import argparse
import json
import os
from typing import Dict

from search_system.parser import run_parser
from search_system.indexer import run_indexer

from utils.config import DATASET_PATH, ARTIFACTS_DIR


def write_subset_ids(labels: Dict[str, int], label_value: int, path: str) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for doc_id, lbl in labels.items():
            if int(lbl) == label_value:
                f.write(f"{doc_id}\n")
                count += 1
    return count


def build_tier(
    dataset_path: str,
    subset_path: str,
    out_dir: str,
) -> None:
    postings_dir = os.path.join(out_dir, "postings")
    index_dir = os.path.join(out_dir, "index")
    os.makedirs(postings_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)

    print(f"[Tier build] Parsing subset {subset_path} -> {postings_dir}")
    run_parser(dataset_path=dataset_path, subset_ids_path=subset_path, output_dir=postings_dir)

    print(f"[Tier build] Indexing postings -> {index_dir}")
    run_indexer(input_dir=postings_dir, output_dir=index_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Tier-1 and Tier-2 BM25 indexes.")
    parser.add_argument("--labels", default=os.path.join(ARTIFACTS_DIR, "tiering", "labels.json"))
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--out-root", default=ARTIFACTS_DIR)
    args = parser.parse_args()

    with open(args.labels, "r", encoding="utf-8") as f:
        labels = json.load(f)

    tier_dir = os.path.join(args.out_root, "tiering")
    t1_ids_path = os.path.join(tier_dir, "tier1_ids.txt")
    t2_ids_path = os.path.join(tier_dir, "tier2_ids.txt")

    t1_count = write_subset_ids(labels, 1, t1_ids_path)
    t2_count = write_subset_ids(labels, 0, t2_ids_path)
    print(f"[Tier build] Wrote {t1_count} Tier-1 IDs -> {t1_ids_path}")
    print(f"[Tier build] Wrote {t2_count} Tier-2 IDs -> {t2_ids_path}")

    # Tier-1
    build_tier(
        dataset_path=args.dataset,
        subset_path=t1_ids_path,
        out_dir=os.path.join(args.out_root, "bm25_T1"),
    )

    # Tier-2
    build_tier(
        dataset_path=args.dataset,
        subset_path=t2_ids_path,
        out_dir=os.path.join(args.out_root, "bm25_T2"),
    )


if __name__ == "__main__":
    main()
