"""
Run search systems on MS MARCO queries.
Usage:
    python -m scripts.run \
        --system <bm25 | hnsw | rerank> \
        --qrels <dev | eval1 | eval2> \
        --save <filename>
"""

from argparse import ArgumentParser
from typing import Dict, List, Tuple, Type

from systems.bm25 import BM25System
from systems.hnsw import HNSWSystem
from systems.rerank import ReRankSystem
from utils.loaders import load_queries, load_qrels
from utils.config import QUERIES_DEV_PATH, QUERIES_EVAL_PATH, QRELS_DEV_PATH, QRELS_EVAL1_PATH, QRELS_EVAL2_PATH

# Available systems
SYSTEMS: Dict[str, Type] = {
    "bm25": BM25System,
    "hnsw": HNSWSystem,
    "rerank": ReRankSystem,
}

# Qrels datasets mapping
DATASETS: Dict[str, Dict[str, str]] = {
    "dev": {"qrels": QRELS_DEV_PATH, "queries": QUERIES_DEV_PATH},
    "eval1": {"qrels": QRELS_EVAL1_PATH, "queries": QUERIES_EVAL_PATH},
    "eval2": {"qrels": QRELS_EVAL2_PATH, "queries": QUERIES_EVAL_PATH},
}

def main() -> None:
    # Parse command line arguments
    parser = ArgumentParser(description="Run search systems on MS MARCO queries.")
    parser.add_argument("--system", choices=list(SYSTEMS.keys()), required=True)
    parser.add_argument("--qrels", choices=list(DATASETS.keys()), required=True)
    parser.add_argument("--save", required=True)
    args = parser.parse_args()

    # Initialize system
    system_cls = SYSTEMS[args.system]
    system = system_cls()

    # Resolve dataset paths
    dataset: Dict[str, str] = DATASETS[args.qrels]
    qrels_path: str = dataset["qrels"]
    query_path: str = dataset["queries"]

    # Load query IDs and queries dataset
    query_ids: set[str] = set(load_qrels(qrels_path).keys())
    queries_dataset: Dict[str, str] = load_queries(query_path)

    # Filter for qrels queries
    queries: List[Tuple[str, str]] = [
        (query_id, queries_dataset[query_id]) # (query_id, query_text)
        for query_id in query_ids
        if query_id in queries_dataset
    ]

    # Run retrieval
    results: List = system.search(queries, top_k=100)

    # Save results
    if args.save: system.save_run(results, args.save)

if __name__ == "__main__":
    main()