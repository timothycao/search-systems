"""
Build tiered HNSW indexes (Tier-1 / Tier-2) using precomputed dense labels and embeddings.
"""

import argparse
from systems.retrieval.dense.hnsw import HNSWSystem

from utils.config import (
    HNSW_T1_EMB_PATH,
    HNSW_T2_EMB_PATH,
    HNSW_QUERY_EMB_PATH,
    HNSW_T1_DIR,
    HNSW_T2_DIR,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tiered HNSW indexes.")
    parser.add_argument("--t1-emb", default=HNSW_T1_EMB_PATH)
    parser.add_argument("--t2-emb", default=HNSW_T2_EMB_PATH)
    parser.add_argument("--query-emb", default=HNSW_QUERY_EMB_PATH)
    parser.add_argument("--t1-out", default=HNSW_T1_DIR)
    parser.add_argument("--t2-out", default=HNSW_T2_DIR)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=200)
    args = parser.parse_args()

    # Tier-1
    t1_system = HNSWSystem(
        subset_embeddings_path=args.t1_emb,
        query_embeddings_path=args.query_emb,
        artifacts_dir=args.t1_out,
        m=args.m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )
    t1_system.build()

    # Tier-2
    t2_system = HNSWSystem(
        subset_embeddings_path=args.t2_emb,
        query_embeddings_path=args.query_emb,
        artifacts_dir=args.t2_out,
        m=args.m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )
    t2_system.build()


if __name__ == "__main__":
    main()
