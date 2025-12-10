"""
Split the work corpus into an initial chunk and a delta chunk.

Default behavior (seed=42):
- Reads data/collection/collection_work.tsv
- Samples 500_000 docs into collection_work_delta.tsv
- Writes the rest to collection_work_init.tsv

Usage:
  python -m scripts.split_work_init_delta \
    --work data/collection/collection_work.tsv \
    --delta-size 500000 \
    --seed 42 \
    --init-out data/collection/collection_work_init.tsv \
    --delta-out data/collection/collection_work_delta.tsv
"""

import argparse
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Split work corpus into init and delta subsets.")
    ap.add_argument("--work", default="data/collection/collection_work.tsv", help="Path to work TSV.")
    ap.add_argument("--delta-size", type=int, default=500_000, help="Number of docs for delta subset.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for deterministic shuffle.")
    ap.add_argument("--init-out", default="data/collection/collection_work_init.tsv", help="Output TSV for init subset.")
    ap.add_argument("--delta-out", default="data/collection/collection_work_delta.tsv", help="Output TSV for delta subset.")
    args = ap.parse_args()

    work_path = Path(args.work)
    init_path = Path(args.init_out)
    delta_path = Path(args.delta_out)

    print(f"Reading work corpus from {work_path}")
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
