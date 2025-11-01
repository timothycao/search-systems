"""
Build search system indices.
Usage:
    python -m scripts.build \
        --system <bm25 | hnsw> \
        [--track <time | memory>]
"""

from argparse import ArgumentParser
from typing import Dict, Tuple, Type

from systems.retrieval.sparse.bm25 import BM25System
from systems.retrieval.dense.hnsw import HNSWSystem
from utils.performance import track_performance
from utils.config import DATASET_PATH, SUBSET_PATH, ARTIFACTS_DIR, SUBSET_EMBEDDINGS_PATH, QUERIES_EMBEDDINGS_PATH

# System registry: maps system name to (class, init args)
SYSTEM_CONFIG: Dict[str, Tuple[Type, Tuple[str, ...]]] = {
    "bm25": (BM25System, (DATASET_PATH, SUBSET_PATH, ARTIFACTS_DIR)),
    "hnsw": (HNSWSystem, (SUBSET_EMBEDDINGS_PATH, QUERIES_EMBEDDINGS_PATH, ARTIFACTS_DIR)),
}

def main() -> None:
    # Parse command line arguments
    parser = ArgumentParser(description="Build search system indices.")
    parser.add_argument("--system", choices=list(SYSTEM_CONFIG.keys()), required=True)
    parser.add_argument("--track", choices=["time", "memory"], required=False)
    args = parser.parse_args()

    # Initialize system
    system_cls, init_args = SYSTEM_CONFIG[args.system]
    system = system_cls(*init_args)
    
    # Build (optionally track time or memory)
    track_performance(system.build, track=args.track)

if __name__ == "__main__":
    main()