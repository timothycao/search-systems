"""
Run tiered retrieval (base + delta) with merge/rescore for BM25 using multiprocessing.

Usage:
  python -m scripts.run_tiered_multi \
    --system bm25 \
    --qrels <dev | eval1 | eval2> \
    --save <output_run_file> \
    --topk 100 \
    [--overfetch-factor 2] \
    [--workers <int>] \
    [--track <time | memory>]

Notes:
- Currently supports BM25 tiered/delta; HNSW tiered merge is not implemented yet.
- Each worker builds its own QueryStartupContext and rescoring contexts to avoid shared state issues.
"""

import json
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
from search_system.query.inverted_list import InvertedList
from search_system.shared.utils import tokenize

from utils.io import load_queries, load_qrels, save_run
from utils.performance import track_performance
from utils.config import (
    ARTIFACTS_DIR,
    RUNS_DIR,
    QRELS_DEV_PATH,
    QRELS_EVAL1_PATH,
    QRELS_EVAL2_PATH,
    QUERIES_DEV_PATH,
    QUERIES_EVAL_PATH,
)

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
    return [(qid, queries[qid]) for qid in qrels.keys() if qid in queries]


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class BM25IndexCtx:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.lexicon = load_json(os.path.join(index_dir, "lexicon.json"))
        self.page_table = load_json(os.path.join(index_dir, "page_table.json"))
        stats = load_json(os.path.join(index_dir, "collection_stats.json"))
        self.N = stats.get("num_docs") or stats.get("total_docs")
        self.avgdl = stats.get("avg_doc_len") or stats.get("avg_len")
        self.k1 = stats.get("k1", 1.2)
        self.b = stats.get("b", 0.75)
        self.inverted_path = os.path.join(index_dir, "inverted_index.bin")
        self.ctx = QueryStartupContext(index_dir)


# Globals for workers
GLOBAL_STATS = {}
GLOBAL_LEXICON = {}
IDX_CTXS: List[BM25IndexCtx] = []
FETCH_K = 0
TOP_K = 0


def init_worker(global_index_dir: str, tier_dirs: List[str], fetch_k: int, top_k: int):
    global GLOBAL_STATS, GLOBAL_LEXICON, IDX_CTXS, FETCH_K, TOP_K
    stats = load_json(os.path.join(global_index_dir, "collection_stats.json"))
    GLOBAL_STATS = {
        "num_docs": stats.get("num_docs") or stats.get("total_docs"),
        "avg_doc_len": stats.get("avg_doc_len") or stats.get("avg_len"),
        "k1": stats.get("k1", 1.2),
        "b": stats.get("b", 0.75),
    }
    GLOBAL_LEXICON = load_json(os.path.join(global_index_dir, "lexicon.json"))
    IDX_CTXS = []
    for d in tier_dirs:
        index_dir = os.path.join(d, "index")
        if os.path.isdir(index_dir):
            IDX_CTXS.append(BM25IndexCtx(index_dir))
    FETCH_K = fetch_k
    TOP_K = top_k


def rescore_bm25(doc_id: int, query_tokens: List[str], index_ctx: BM25IndexCtx) -> float:
    score = 0.0
    for term in query_tokens:
        term_meta = index_ctx.lexicon.get(term)
        if not term_meta:
            continue
        ilist = InvertedList(
            term,
            term_meta,
            index_ctx.inverted_path,
            index_ctx.page_table,
            N=index_ctx.N,
            avg_len=index_ctx.avgdl,
            k1=index_ctx.k1,
            b=index_ctx.b,
        )
        # Override with global stats/df if available
        if term in GLOBAL_LEXICON:
            ilist.N = GLOBAL_STATS["num_docs"]
            ilist.avg_len = GLOBAL_STATS["avg_doc_len"]
            ilist.k1 = GLOBAL_STATS["k1"]
            ilist.b = GLOBAL_STATS["b"]
            ilist.df = GLOBAL_LEXICON[term].get("df", ilist.df)
            ilist.idf = ilist.compute_idf()
        found = ilist.nextGEQ(doc_id)
        if found == doc_id:
            score += ilist.getScore(doc_id)
    return score


def process_query(query: Tuple[str, str]) -> Tuple[str, List[Tuple[int, float]]]:
    qid, qtext = query
    query_tokens = tokenize(qtext)
    candidates: Dict[int, float] = {}

    for ctx in IDX_CTXS:
        LIST_CACHE.cache.clear()
        with redirect_stdout(StringIO()):
            hits = run_query(
                startup_context=ctx.ctx,
                query=qtext,
                mode="bwand-or",
                top_k=FETCH_K,
            )
        for docid, _ in hits:
            if docid in candidates:
                continue
            score = rescore_bm25(docid, query_tokens, ctx)
            candidates[docid] = score

    ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    return qid, ranked


def bm25_search_and_merge_mp(
    queries: List[Tuple[str, str]],
    top_k: int,
    overfetch_factor: int,
    global_index_dir: str,
    tier_dirs: List[str],
    workers: int,
) -> List[Tuple[str, List[Tuple[int, float]]]]:
    fetch_k = overfetch_factor * top_k
    with mp.Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(global_index_dir, tier_dirs, fetch_k, top_k),
    ) as pool:
        results = list(tqdm(pool.imap(process_query, queries), total=len(queries), desc="Queries"))
    return results


def main() -> None:
    parser = ArgumentParser(description="Run tiered retrieval (base+delta) with merge/rescore using multiprocessing.")
    parser.add_argument("--system", choices=["bm25", "hnsw"], required=True)
    parser.add_argument("--qrels", choices=list(DATASETS.keys()), required=True)
    parser.add_argument("--save", required=True)
    parser.add_argument("--topk", type=int, default=100, help="Final topK to output")
    parser.add_argument("--overfetch-factor", type=int, default=2, help="Overfetch multiplier before merge/rescore")
    parser.add_argument("--workers", type=int, default=mp.cpu_count(), help="Number of worker processes")
    parser.add_argument("--track", choices=["time", "memory"], required=False)
    args = parser.parse_args()

    dataset: Dict[str, str] = DATASETS[args.qrels]
    qrels_path: str = dataset["qrels"]
    query_path: str = dataset["queries"]
    queries: List[Tuple[str, str]] = get_queries_subset(qrels_path, query_path)

    if args.system == "bm25":
        tier_dirs = [
            os.path.join(ARTIFACTS_DIR, "bm25_T1"),
            os.path.join(ARTIFACTS_DIR, "bm25_T2"),
            os.path.join(ARTIFACTS_DIR, "bm25_T1_delta"),
            os.path.join(ARTIFACTS_DIR, "bm25_T2_delta"),
        ]
        global_index_dir = os.path.join(ARTIFACTS_DIR, "bm25", "index")
        results = track_performance(
            bm25_search_and_merge_mp,
            queries,
            args.topk,
            args.overfetch_factor,
            global_index_dir,
            tier_dirs,
            args.workers,
            track=args.track,
        )
    else:
        raise NotImplementedError("HNSW tiered retrieval merge is not implemented yet.")

    save_path = os.path.join(RUNS_DIR, f"{args.system}_tiered", args.save)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_run(results, save_path)


if __name__ == "__main__":
    main()
