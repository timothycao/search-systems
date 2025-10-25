"""
Run search systems.
Usage:
    python -m scripts.run --system <bm25, hnsw, rerank>
"""

from argparse import ArgumentParser
from typing import Dict, List

from systems.bm25 import BM25System
from systems.hnsw import HNSWSystem
from systems.rerank import ReRankSystem

def main() -> None:
    parser = ArgumentParser(description="Run search systems.")
    parser.add_argument(
        "--system",
        type=str,
        required=True,
        choices=["bm25", "hnsw", "rerank"],
        help="Which system to run."
    )
    args = parser.parse_args()

    systems: Dict[str, object] = {
        "bm25": BM25System,
        "hnsw": HNSWSystem,
        "rerank": ReRankSystem,
    }

    system_cls = systems[args.system]
    system = system_cls()

    query: str = input("Enter query: ").strip()
    queries: List[str] = [query] if query else []
    results: List = system.search(queries)
    print(f"[{system.name}] Results:", results)

if __name__ == "__main__":
    main()