"""
Split the work TSV into init/delta subsets for HNSW ingestion.

Defaults:
- Work TSV: data/collection/collection_work.tsv
- Delta size: 500_000 docs
- Outputs: data/collection/collection_work_hnsw_init.tsv, data/collection/collection_work_hnsw_delta.tsv

Usage:
  python -m scripts.split_work_hnsw_init_delta \
    --work data/collection/collection_work.tsv \
    --delta-size 500000 \
    --seed 42 \
    --init-out data/collection/collection_work_hnsw_init.tsv \
    --delta-out data/collection/collection_work_hnsw_delta.tsv
"""

import argparse
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Split work TSV into init/delta subsets for HNSW.")
    ap.add_argument("--work", default="data/collection/collection_work.tsv", help="Work TSV (doc_id<TAB>text)")
    ap.add_argument("--delta-size", type=int, default=500_000, help="Number of docs for delta subset")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--init-out", default="data/collection/collection_work_hnsw_init.tsv", help="Output TSV for init subset")
    ap.add_argument("--delta-out", default="data/collection/collection_work_hnsw_delta.tsv", help="Output TSV for delta subset")
    args = ap.parse_args()

    work_path = Path(args.work)
    init_path = Path(args.init_out)
    delta_path = Path(args.delta_out)

    with work_path.open() as f:
        lines = [line for line in f if line.strip()]

    total = len(lines)
    if args.delta_size > total:
        raise ValueError(f"delta_size {args.delta_size} exceeds total docs {total}")

    rng = random.Random(args.seed)
    idxs = list(range(total))
    rng.shuffle(idxs)
    delta_idxs = set(idxs[: args.delta_size])

    init_lines, delta_lines = [], []
    for i, line in enumerate(lines):
        if i in delta_idxs:
            delta_lines.append(line)
        else:
            init_lines.append(line)

    init_path.write_text("".join(init_lines))
    delta_path.write_text("".join(delta_lines))

    print(f"Total docs: {total}")
    print(f"Init docs: {len(init_lines)} -> {init_path}")
    print(f"Delta docs: {len(delta_lines)} -> {delta_path}")


if __name__ == "__main__":
    main()
