"""
Remap query embedding IDs using a mapping file (sequential_id -> original_id).

Use when embeddings were built from queries.all.tsv (sequential IDs) but you need
the original MS MARCO query IDs (as in qrels).

Example:
  python -m scripts.remap_query_embeddings \
    --emb data/collection/query_embeddings.h5 \
    --map data/queries/queries.all.map.tsv \
    --out data/collection/query_embeddings_remapped.h5
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def load_map(path: Path):
    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            seq, orig = line.rstrip("\n").split("\t")
            mapping[seq] = orig
    return mapping


def main():
    ap = argparse.ArgumentParser(description="Remap query embedding IDs using queries.all.map.tsv.")
    ap.add_argument("--emb", required=True, help="Input query embeddings HDF5 (id, embedding)")
    ap.add_argument("--map", required=True, help="Mapping TSV: seq_id<TAB>orig_id")
    ap.add_argument("--out", required=True, help="Output HDF5 with remapped ids")
    args = ap.parse_args()

    mapping = load_map(Path(args.map))
    with h5py.File(args.emb, "r") as f:
        ids = np.array(f["id"]).astype(str)
        embs = np.array(f["embedding"]).astype(np.float32)

    remapped_ids = []
    missing = 0
    for i in tqdm(ids, desc="Remapping", unit="id"):
        if i in mapping:
            remapped_ids.append(mapping[i])
        else:
            remapped_ids.append(i)
            missing += 1

    if missing:
        print(f"Warning: {missing} ids not found in map; left unchanged.")

    remapped_ids = np.array(remapped_ids, dtype=object)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("id", data=remapped_ids, dtype=h5py.string_dtype(encoding="utf-8"), compression="gzip")
        f.create_dataset("embedding", data=embs, compression="gzip")
    print(f"Saved remapped query embeddings to {args.out}")


if __name__ == "__main__":
    main()
