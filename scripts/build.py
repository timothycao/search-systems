"""
Build search system indices.
Usage:
    python -m scripts.build \
        --system <bm25 | hnsw | rerank> \
        [--track <time | memory>]
"""

from argparse import ArgumentParser
from typing import Dict, Type

from systems.bm25 import BM25System
from systems.hnsw import HNSWSystem
from systems.rerank_rrf import RecipricalRankFusion
from systems.rerank_linear import LinearScoreFusion
from utils.performance import track_performance

# Available systems
SYSTEMS: Dict[str, Type] = {
    "bm25": BM25System,
    "hnsw": HNSWSystem,
    "rerank-rrf": RecipricalRankFusion,
    "rerank-lsf": LinearScoreFusion,
}

def main() -> None:
    # Parse command line arguments
    parser = ArgumentParser(description="Build search system indices.")
    parser.add_argument("--system", choices=list(SYSTEMS.keys()), required=True)
    parser.add_argument("--track", choices=["time", "memory"], required=False)
    args = parser.parse_args()

    # Initialize system
    system_cls = SYSTEMS[args.system]
    system = system_cls()
    
    # Build (optionally track time or memory)
    track_performance(system.build, track=args.track)

if __name__ == "__main__":
    main()