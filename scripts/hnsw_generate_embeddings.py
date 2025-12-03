"""
Generate dense embeddings for passages and queries using a dot-product model.

Outputs HDF5 files with datasets:
- id: string doc_id or query_id
- embedding: float32 embedding vectors
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import Iterable, Tuple

from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from utils.config import (
    DATASET_PATH,
    QUERIES_DIR,
    HNSW_MODEL_NAME,
    HNSW_EMBED_DIR,
    HNSW_DOC_EMB_PATH,
    HNSW_QUERY_EMB_PATH,
)


def read_tsv(path: Path) -> Iterable[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            yield parts[0], parts[1]


def encode_and_write(pairs: Iterable[Tuple[str, str]], model: SentenceTransformer, out_path: Path, batch_size: int = 512) -> None:
    ids = []
    texts = []
    for pid, text in pairs:
        ids.append(pid)
        texts.append(text)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # raw dot space
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5f:
        h5f.create_dataset("id", data=np.array(ids, dtype="S"), compression="gzip")
        h5f.create_dataset("embedding", data=embeddings.astype(np.float32), compression="gzip")
    print(f"Wrote {len(ids)} embeddings to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dense embeddings for corpus and queries.")
    parser.add_argument("--model", default=HNSW_MODEL_NAME)
    parser.add_argument("--docs", default=DATASET_PATH, help="TSV: doc_id<TAB>text")
    parser.add_argument("--queries", default=str(Path(QUERIES_DIR) / "queries.all.tsv"))
    parser.add_argument("--doc-out", default=HNSW_DOC_EMB_PATH)
    parser.add_argument("--query-out", default=HNSW_QUERY_EMB_PATH)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    model = SentenceTransformer(args.model)
    encode_and_write(read_tsv(Path(args.docs)), model, Path(args.doc_out), batch_size=args.batch_size)
    encode_and_write(read_tsv(Path(args.queries)), model, Path(args.query_out), batch_size=args.batch_size)


if __name__ == "__main__":
    main()
