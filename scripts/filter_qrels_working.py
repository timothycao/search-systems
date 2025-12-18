"""
Filter qrels.train.tsv to only those docids present in the working collection.

Usage:
  python -m scripts.filter_qrels_working \
    --qrels data/qrels/qrels.train.tsv \
    --working-docids data/collection/docids_working.txt \
    --out data/qrels/qrels.train.working.tsv
"""

import argparse
from pathlib import Path


def load_working_ids(path: Path) -> set:
    with path.open("r", encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter qrels to working docids.")
    ap.add_argument("--qrels", required=True, help="Input qrels file (train).")
    ap.add_argument("--working-docids", required=True, help="Path to docids_working.txt")
    ap.add_argument("--out", required=True, help="Output filtered qrels file.")
    args = ap.parse_args()

    working_ids = load_working_ids(Path(args.working_docids))

    total = 0
    kept = 0
    out_lines = []
    with Path(args.qrels).open("r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            total += 1
            parts = ln.strip().split()
            if len(parts) == 3:
                qid, docid, rel = parts
            elif len(parts) == 4:
                qid, _, docid, rel = parts
            else:
                continue  # unexpected format
            if docid in working_ids:
                kept += 1
                out_lines.append(f"{qid}\t{docid}\t{rel}\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(out_lines), encoding="utf-8")

    print(f"Total lines processed: {total}")
    print(f"Kept lines: {kept}")
    print(f"Output written to: {out_path}")


if __name__ == "__main__":
    main()
