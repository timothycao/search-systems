"""
Build search system indices.
Usage:
    python -m scripts.build --system <bm25 | hnsw | rerank>
"""

from argparse import ArgumentParser
from typing import Dict, Type

from systems.bm25 import BM25System
from systems.hnsw import HNSWSystem
from systems.rerank_rrf import RecipricalRankFusion
from systems.rerank_linear import LinearScoreFusion

SYSTEMS: Dict[str, Type] = {
    "bm25": BM25System,
    "hnsw": HNSWSystem,
    "rerank-rrf": RecipricalRankFusion,
    "rerank-lsf": LinearScoreFusion,
}

def main() -> None:
    parser = ArgumentParser(description="Build search system indices.")
    parser.add_argument(
        "--system",
        type=str,
        required=True,
        choices=list(SYSTEMS.keys()),
        help="Which system to build."
    )
    args = parser.parse_args()

    system_cls = SYSTEMS[args.system]
    system = system_cls()
    
    system.build()

if __name__ == "__main__":
    main()