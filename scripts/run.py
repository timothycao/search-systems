"""
Run retrieval or rerank systems on MS MARCO queries.
Usage:
    python -m scripts.run \
        --system <bm25 | hnsw | rrf | lsf> \
        --save <filename> \
        [--qrels <dev | eval1 | eval2>] \
        [--targets <run1 run2>] \
        [--track <time | memory>]
Example:
    # Retrieval
    python -m scripts.run --system bm25 --qrels eval1 --save eval1.txt

    # Rerank (fusion)
    python -m scripts.run --system rrf --targets eval1.txt eval1.txt --save eval1.txt
"""

import os
from argparse import ArgumentParser
from typing import Dict, List, Tuple, Type

from systems.retrieval.sparse.bm25 import BM25System
from systems.retrieval.dense.hnsw import HNSWSystem
from systems.rerank.fusion.reciprocal import ReciprocalFusionSystem
from systems.rerank.fusion.linear import LinearFusionSystem
from utils.io import load_queries, load_qrels, save_run
from utils.performance import track_performance
from utils.config import (
    ARTIFACTS_DIR,
    RUNS_DIR,
    DATASET_PATH,
    SUBSET_PATH,
    SUBSET_EMBEDDINGS_PATH,
    QUERIES_EMBEDDINGS_PATH,
    QRELS_DEV_PATH,
    QRELS_EVAL1_PATH,
    QRELS_EVAL2_PATH,
    QUERIES_DEV_PATH,
    QUERIES_EVAL_PATH,
)

# System registry: maps system name to (class, init args)
SYSTEM_CONFIG: Dict[str, Tuple[Type, Tuple[str, ...]]] = {
    # Retrieval systems
    "bm25": (BM25System, (DATASET_PATH, SUBSET_PATH, ARTIFACTS_DIR)),
    "hnsw": (HNSWSystem, (SUBSET_EMBEDDINGS_PATH, QUERIES_EMBEDDINGS_PATH, ARTIFACTS_DIR)),
    # Rerank systems
    "rrf": (ReciprocalFusionSystem, ()),
    "lsf": (LinearFusionSystem, ()),
}

# Dataset registry
DATASETS: Dict[str, Dict[str, str]] = {
    "dev": {"qrels": QRELS_DEV_PATH, "queries": QUERIES_DEV_PATH},
    "eval1": {"qrels": QRELS_EVAL1_PATH, "queries": QUERIES_EVAL_PATH},
    "eval2": {"qrels": QRELS_EVAL2_PATH, "queries": QUERIES_EVAL_PATH},
}

def get_queries_subset(qrels_path: str, query_path: str) -> List[Tuple[str, str]]:
    """Join qrels and query text into (query_id, query_text) tuples."""
    qrels: Dict[str, Dict[str, int]] = load_qrels(qrels_path)
    queries: Dict[str, str] = load_queries(query_path)
    query_ids: List[str] = qrels.keys()
    
    return [
        (query_id, queries[query_id]) # (query_id, query_text)
        for query_id in query_ids
        if query_id in queries
    ]

def main() -> None:
    # Parse command line arguments
    parser = ArgumentParser(description="Run retrieval or rerank systems on MS MARCO queries.")
    parser.add_argument("--system", choices=list(SYSTEM_CONFIG.keys()), required=True)
    parser.add_argument("--save", required=True)
    parser.add_argument("--qrels", choices=list(DATASETS.keys()), required=False)
    parser.add_argument("--targets", nargs="+", required=False)
    parser.add_argument("--track", choices=["time", "memory"], required=False)
    args = parser.parse_args()

    # Initialize system
    system_cls, init_args = SYSTEM_CONFIG[args.system]
    system = system_cls(*init_args)

    # Retrieval systems
    if args.system in ["bm25", "hnsw"]:
        if not args.qrels: raise ValueError("--qrels flag is required for retrieval systems.")

        # Resolve dataset paths
        dataset: Dict[str, str] = DATASETS[args.qrels]
        qrels_path: str = dataset["qrels"]
        query_path: str = dataset["queries"]

        # Get (query_id, query_text) pairs based on the subset
        queries: List[Tuple[str, str]] = get_queries_subset(qrels_path, query_path)

        # Run retrieval (optionally track time or memory)
        results: List = track_performance(system.retrieve, queries, top_k=100, track=args.track)

    # Rerank systems
    elif args.system in ["rrf", "lsf"]:
        if not args.targets: raise ValueError("--targets flag is required for rerank systems.")

        # Resolve BM25 and HNSW paths
        bm25_file, hnsw_file = args.targets
        bm25_path = os.path.join(RUNS_DIR, "bm25", bm25_file)
        hnsw_path = os.path.join(RUNS_DIR, "hnsw", hnsw_file)
        
        # Run fusion or reranking
        results: List = track_performance(
            system.rerank,
            {"bm25": bm25_path, "hnsw": hnsw_path},
            top_k=100,
            track=args.track
        )

    # Save results
    save_path = os.path.join(RUNS_DIR, args.system, args.save)
    save_run(results, save_path)

if __name__ == "__main__":
    main()