"""
Run search systems.
Usage:
    python -m scripts.run --system <bm25 | hnsw | rerank> [--save <output filename>]
"""

from argparse import ArgumentParser
from typing import Dict, List, Type

from systems.bm25 import BM25System
from systems.hnsw import HNSWSystem
from systems.rerank import ReRankSystem

SYSTEMS: Dict[str, Type] = {
    "bm25": BM25System,
    "hnsw": HNSWSystem,
    "rerank": ReRankSystem,
}

def main() -> None:
    parser = ArgumentParser(description="Run search systems.")
    parser.add_argument(
        "--system",
        type=str,
        required=True,
        choices=list(SYSTEMS.keys()),
        help="Which system to run."
    )
    parser.add_argument(
        "--save",
        type=str,
        required=False,
        help="Filename to save search results."
    )
    args = parser.parse_args()

    system_cls = SYSTEMS[args.system]
    system = system_cls()

    query: str = input("Enter query: ").strip()
    results: List = system.search([query])

    if args.save: system.save_run(results, args.save)

if __name__ == "__main__":
    main()