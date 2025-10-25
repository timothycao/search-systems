"""
Build search system indices.
Usage:
    python -m scripts.build --system <bm25, hnsw, rerank>
"""

from argparse import ArgumentParser
from typing import Dict

from systems.bm25 import BM25System
from systems.hnsw import HNSWSystem
from systems.rerank import ReRankSystem

def main() -> None:
    parser = ArgumentParser(description="Build search system indices.")
    parser.add_argument(
        "--system",
        type=str,
        required=True,
        choices=["bm25", "hnsw", "rerank"],
        help="Which system to build."
    )
    args = parser.parse_args()

    systems: Dict[str, object] = {
        "bm25": BM25System,
        "hnsw": HNSWSystem,
        "rerank": ReRankSystem,
    }

    system_cls = systems[args.system]
    system = system_cls()
    
    system.build()

if __name__ == "__main__":
    main()