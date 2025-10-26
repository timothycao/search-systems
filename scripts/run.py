"""
Run search systems on MS MARCO queries.
Usage:
    python -m scripts.run \
        --system <bm25 | hnsw | rerank> \
        --query  <dev | eval1 | eval2> \
        [--save <filename>]
"""

from argparse import ArgumentParser
from typing import Dict, List, Tuple, Type

from systems.bm25 import BM25System
from systems.hnsw import HNSWSystem
from systems.rerank import ReRankSystem
from utils.loaders import load_queries, load_qrel_query_ids
from utils.config import QUERIES_DEV_PATH, QUERIES_EVAL_PATH, QRELS_DEV_PATH, QRELS_EVAL1_PATH, QRELS_EVAL2_PATH

# Available systems
SYSTEMS: Dict[str, Type] = {
    "bm25": BM25System,
    "hnsw": HNSWSystem,
    "rerank": ReRankSystem,
}

# Query/qrels mapping
DATASETS: Dict[str, Dict[str, str]] = {
    "dev": {"queries": QUERIES_DEV_PATH, "qrels": QRELS_DEV_PATH},
    "eval1": {"queries": QUERIES_EVAL_PATH, "qrels": QRELS_EVAL1_PATH},
    "eval2": {"queries": QUERIES_EVAL_PATH, "qrels": QRELS_EVAL2_PATH},
}

def main() -> None:
    parser = ArgumentParser(description="Run search systems on MS MARCO queries.")
    parser.add_argument(
        "--system",
        type=str,
        required=True,
        choices=list(SYSTEMS.keys()),
        help="Which system to run."
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        choices=list(DATASETS.keys()),
        help="Which query/qrels set to run."
    )
    parser.add_argument(
        "--save",
        type=str,
        required=False,
        help="Filename to save search results."
    )
    args = parser.parse_args()

    # Initialize system
    system_cls = SYSTEMS[args.system]
    system = system_cls()

    # Resolve dataset paths
    dataset: Dict[str, str] = DATASETS[args.query]
    query_path: str = dataset["queries"]
    qrels_path: str = dataset["qrels"]

    # Load queries and qrels
    all_queries: Dict[str, str] = load_queries(query_path)
    qrel_query_ids: set[str] = load_qrel_query_ids(qrels_path)

    # Filter queries to those with qrels
    queries: List[Tuple[str, str]] = [
        (query_id, all_queries[query_id])
        for query_id in qrel_query_ids
        if query_id in all_queries
    ]

    # Run retrieval
    results: List = system.search(queries)

    # Save results
    if args.save: system.save_run(results, args.save)

if __name__ == "__main__":
    main()