"""
Split sharded doc embeddings into train/work HDF5 files based on doc IDs.

Defaults assume:
- Sharded embeddings: data/collection/doc_embeddings_part*.h5
- Train/work TSVs: data/collection/collection_train.tsv, data/collection/collection_work.tsv
- Outputs: artifacts/hnsw_embeddings/collection_hnsw_train.h5 and collection_hnsw_work.h5

Usage (example):
  python -m scripts.hnsw_split_embeddings.py \
    --shards "data/collection/doc_embeddings_part*.h5" \
    --train-tsv data/collection/collection_train.tsv \
    --work-tsv data/collection/collection_work.tsv \
    --train-out artifacts/hnsw_embeddings/collection_hnsw_train.h5 \
    --work-out artifacts/hnsw_embeddings/collection_hnsw_work.h5
"""

import argparse
import glob
from pathlib import Path
from typing import Set

import h5py
import numpy as np
from tqdm import tqdm


def load_ids_from_tsv(path: Path) -> Set[str]:
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc_id = line.split("\t", 1)[0]
            ids.add(doc_id)
    return ids


def append_batch(out_h5: h5py.File, ids: np.ndarray, embs: np.ndarray, str_dtype, emb_dim: int) -> None:
    ids = np.asarray(ids, dtype=object)  # ensure variable-length string dtype
    if "id" not in out_h5:
        out_h5.create_dataset("id", data=ids, maxshape=(None,), dtype=str_dtype, compression="gzip")
        out_h5.create_dataset("embedding", data=embs, maxshape=(None, emb_dim), dtype="float32", compression="gzip")
    else:
        id_ds = out_h5["id"]
        emb_ds = out_h5["embedding"]
        new_size = id_ds.shape[0] + ids.shape[0]
        id_ds.resize((new_size,))
        emb_ds.resize((new_size, emb_dim))
        id_ds[-ids.shape[0]:] = ids
        emb_ds[-embs.shape[0]:] = embs


def split_embeddings(shards, train_ids: Set[str], work_ids: Set[str], train_out: Path, work_out: Path, batch_size: int = 50000):
    train_out.parent.mkdir(parents=True, exist_ok=True)
    work_out.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(train_out, "w") as f_train, h5py.File(work_out, "w") as f_work:
        wrote_train = 0
        wrote_work = 0
        for shard in tqdm(shards, desc="Shards", unit="shard"):
            with h5py.File(shard, "r") as h5f:
                ids = np.array(h5f["id"]).astype(str)
                embs = np.array(h5f["embedding"]).astype(np.float32)
                emb_dim = embs.shape[1]
                total = ids.shape[0]
                for start in tqdm(range(0, total, batch_size), desc="Batches", unit="batch", leave=False):
                    end = min(total, start + batch_size)
                    batch_ids = ids[start:end]
                    batch_embs = embs[start:end]
                    # Boolean masks
                    mask_train = np.isin(batch_ids, list(train_ids))
                    mask_work = np.isin(batch_ids, list(work_ids))
                    if mask_train.any():
                        append_batch(f_train, batch_ids[mask_train], batch_embs[mask_train], str_dtype, emb_dim)
                        wrote_train += mask_train.sum()
                    if mask_work.any():
                        append_batch(f_work, batch_ids[mask_work], batch_embs[mask_work], str_dtype, emb_dim)
                        wrote_work += mask_work.sum()
            tqdm.write(f"[Shard done] {shard}")
        print(f"Wrote train embeddings: {wrote_train} -> {train_out}")
        print(f"Wrote work embeddings: {wrote_work} -> {work_out}")


def main():
    ap = argparse.ArgumentParser(description="Split sharded doc embeddings into train/work HDF5s.")
    ap.add_argument("--shards", default="data/collection/doc_embeddings_part*.h5", help="Glob pattern for sharded doc embeddings.")
    ap.add_argument("--train-tsv", default="data/collection/collection_train.tsv", help="Train TSV with doc_id<TAB>text.")
    ap.add_argument("--work-tsv", default="data/collection/collection_work.tsv", help="Work TSV with doc_id<TAB>text.")
    ap.add_argument("--train-out", default="data/collection/collection_train_hnsw.h5", help="Output H5 for train embeddings.")
    ap.add_argument("--work-out", default="data/collection/collection_work_hnsw.h5", help="Output H5 for work embeddings.")
    ap.add_argument("--batch-size", type=int, default=50000, help="Batch size for processing shards.")
    args = ap.parse_args()

    shards = sorted(glob.glob(args.shards))
    if not shards:
        raise FileNotFoundError(f"No shards matched pattern: {args.shards}")

    print("Loading train/work doc IDs...")
    train_ids = load_ids_from_tsv(Path(args.train_tsv))
    work_ids = load_ids_from_tsv(Path(args.work_tsv))
    print(f"Train IDs: {len(train_ids)} | Work IDs: {len(work_ids)}")

    split_embeddings(shards, train_ids, work_ids, Path(args.train_out), Path(args.work_out), batch_size=args.batch_size)


if __name__ == "__main__":
    main()
