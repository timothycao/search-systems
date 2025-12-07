"""
Build search system indices.
Usage:
    python -m scripts.build \
        --system <bm25 | hnsw> \
        [--track <time | memory>]
"""

from argparse import ArgumentParser

from utils.performance import track_performance
from utils.config import (
    DATASET_PATH,
    SUBSET_PATH,
    ARTIFACTS_DIR,
    SUBSET_EMBEDDINGS_PATH,
    QUERIES_EMBEDDINGS_PATH,
)


def main() -> None:
    parser = ArgumentParser(description="Build search system indices.")
    parser.add_argument("--system", choices=["bm25", "hnsw"], required=True)
    parser.add_argument("--track", choices=["time", "memory"], required=False)
    parser.add_argument("--dataset-path", default=DATASET_PATH, help="Path to collection TSV (BM25)")
    parser.add_argument("--subset-path", default=SUBSET_PATH, help="Optional subset IDs file (BM25)")
    parser.add_argument("--subset-embeddings-path", default=SUBSET_EMBEDDINGS_PATH, help="Doc embeddings H5 (HNSW)")
    parser.add_argument("--queries-embeddings-path", default=QUERIES_EMBEDDINGS_PATH, help="Query embeddings H5 (HNSW)")
    parser.add_argument("--artifacts-dir", default=ARTIFACTS_DIR, help="Output directory for artifacts")
    args = parser.parse_args()

    if args.system == "bm25":
        # Lazy import to avoid faiss dependency when building BM25 only
        from systems.retrieval.sparse.bm25 import BM25System

        system = BM25System(args.dataset_path, args.subset_path, args.artifacts_dir)
    else:
        # Lazy import to avoid loading faiss unless requested
        from systems.retrieval.dense.hnsw import HNSWSystem

        system = HNSWSystem(args.subset_embeddings_path, args.queries_embeddings_path, args.artifacts_dir)

    track_performance(system.build, track=args.track)


if __name__ == "__main__":
    main()
