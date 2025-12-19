"""
Fast BM25 top-k retrieval on a single index.

Usage:
  python -m scripts.run_bm25_multi \
    --index bm25_T1 \
    --qrels qrels.train.filtered.tsv \
    --queries queries.train.filtered.tsv \
    --save bm25_train_filtered_t1.run \
    --topk 100 \
    --workers 8
"""

import os
import multiprocessing as mp
from argparse import ArgumentParser
from typing import Dict, List, Tuple
from contextlib import redirect_stdout
from io import StringIO

from tqdm import tqdm

from search_system.query.query_startup_context import QueryStartupContext
from search_system.query import run_query
from search_system.query.query import LIST_CACHE

from utils.io import load_qrels, load_queries, save_run
from utils.config import ARTIFACTS_DIR, QRELS_DIR, QUERIES_DIR, RUNS_DIR


def get_queries_subset(qrels_path: str, query_path: str) -> List[Tuple[str, str]]:
    """Join qrels and query text into (query_id, query_text) tuples."""
    qrels: Dict[str, Dict[str, int]] = load_qrels(qrels_path)
    queries: Dict[str, str] = load_queries(query_path)
    return [(qid, queries[qid]) for qid in qrels.keys() if qid in queries]


CTX = None
TOPK = 0
MODE = "bwand-or"

def init_worker(index_dir: str, topk: int, mode: str, cache_capacity: int):
    global CTX, TOPK, MODE
    CTX = QueryStartupContext(index_dir)
    TOPK = topk
    MODE = mode

    LIST_CACHE.cache.clear()
    LIST_CACHE.capacity = cache_capacity


def process_query(q: Tuple[str, str]) -> Tuple[str, List[Tuple[int, float]]]:
    qid, qtext = q

    LIST_CACHE.cache.clear()
    with redirect_stdout(StringIO()):
        hits = run_query(
            startup_context=CTX,
            query=qtext,
            mode=MODE,
            top_k=TOPK,
        )
    
    return qid, hits


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--save", required=True)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--mode", default="bwand-or", choices=["and", "or", "bwand-or"])
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--chunksize", type=int, default=64, help="Bigger chunks => less IPC overhead")
    parser.add_argument("--cache-capacity", type=int, default=512, help="Avoid FD explosion from huge cache sizes")
    args = parser.parse_args()

    index_dir = os.path.join(ARTIFACTS_DIR, args.index, "index")
    qrels_path = os.path.join(QRELS_DIR, args.qrels)
    queries_path = os.path.join(QUERIES_DIR, args.queries)

    queries = get_queries_subset(qrels_path, queries_path)

    with mp.Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(index_dir, args.topk, args.mode, args.cache_capacity),
        maxtasksperchild=5000,  # helps long runs stay stable
    ) as pool:
        results = list(
            tqdm(
                pool.imap(process_query, queries, chunksize=args.chunksize),
                total=len(queries),
                desc="Queries",
            )
        )

    save_path = os.path.join(RUNS_DIR, args.index, args.save)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_run(results, save_path)
    print(f"[BM25] Saved run to {save_path}")


if __name__ == "__main__":
    main()