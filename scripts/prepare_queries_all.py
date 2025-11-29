"""
Concatenate dev + eval queries into a single renumbered file.

Outputs (new files only):
- data/queries/queries.all.tsv        (new_id \\t query_text)
- data/queries/queries.all.map.tsv    (new_id \\t original_id)
"""

import argparse
import os
from typing import Iterable, Tuple


def read_queries(path: str) -> Iterable[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            qid, text = line.rstrip("\n").split("\t", 1)
            yield qid, text


def write_all(
    dev_path: str,
    eval_path: str,
    out_path: str,
    map_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    next_id = 1
    with open(out_path, "w", encoding="utf-8") as out_f, \
         open(map_path, "w", encoding="utf-8") as map_f:

        for source_path in (dev_path, eval_path):
            for orig_id, text in read_queries(source_path):
                out_f.write(f"{next_id}\t{text}\n")
                map_f.write(f"{next_id}\t{orig_id}\n")
                next_id += 1

    print(f"Wrote {next_id - 1} queries to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine dev+eval queries and renumber sequentially.")
    parser.add_argument("--dev", default="data/queries/queries.dev.tsv")
    parser.add_argument("--eval", default="data/queries/queries.eval.tsv")
    parser.add_argument("--out", default="data/queries/queries.all.tsv")
    parser.add_argument("--map", default="data/queries/queries.all.map.tsv")
    args = parser.parse_args()

    write_all(args.dev, args.eval, args.out, args.map)


if __name__ == "__main__":
    main()
