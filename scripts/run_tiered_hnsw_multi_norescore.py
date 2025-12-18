"""
Run tiered HNSW retrieval (base + delta) with merge by dot-product, multiprocessing.
This version does NOT do a second-stage rescore; it relies on FAISS scores per shard.

Usage:
  python -m scripts.run_tiered_hnsw_multi_norescore \
    --qrels <dev | eval1 | eval2> \
    --save <output_run_file> \
    --topk 100 \
    --overfetch-factor 2 \
    --workers <int> \
    --query-emb data/collection/query_embeddings_remapped.h5 \
    [--ef-search 200]
"""

import os
import json
import multiprocessing as mp
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import h5py
import numpy as np
from tqdm import tqdm

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

DATASETS: Dict[str, Dict[str, str]] = {
    "dev": {"qrels": QRELS_DEV_PATH, "queries": QUERIES_DEV_PATH},
    "eval1": {"qrels": QRELS_EVAL1_PATH, "queries": QUERIES_EVAL_PATH},
    "eval2": {"qrels": QRELS_EVAL2_PATH, "queries": QUERIES_EVAL_PATH},
}


def get_queries_subset(qrels_path: str, query_path: str) -> List[Tuple[str, str]]:
    qrels: Dict[str, Dict[str, int]] = load_qrels(qrels_path)
    queries: Dict[str, str] = load_queries(query_path)
    return [(qid, queries[qid]) for qid in qrels.keys() if qid in queries]


def load_query_embeddings(path: Path) -> Dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        ids = np.array(f["id"]).astype(str)
        embs = np.array(f["embedding"]).astype(np.float32)
    embs = embs.copy()
    faiss.normalize_L2(embs)
    return {qid: emb for qid, emb in zip(ids, embs)}


# Globals for workers
HNSW_IDXS = []
FETCH_K = 0
TOP_K = 0
QEMB: Dict[str, np.ndarray] = {}
EF_SEARCH = 200


def init_worker(index_dirs: List[str], fetch_k: int, top_k: int, query_emb_path: str, ef_search: int):
    global HNSW_IDXS, FETCH_K, TOP_K, QEMB, EF_SEARCH
    HNSW_IDXS = []
    for d in index_dirs:
        idx_path = Path(d) / "index.faiss"
        if not idx_path.exists():
            continue
        idx = faiss.read_index(str(idx_path))
        if hasattr(idx, "hnsw"):
            idx.hnsw.efSearch = ef_search
        HNSW_IDXS.append(idx)
    FETCH_K = fetch_k
    TOP_K = top_k
    QEMB = load_query_embeddings(Path(query_emb_path))
    EF_SEARCH = ef_search


def process_query(q: Tuple[str, str]) -> Tuple[str, List[Tuple[int, float]]]:
    qid, _ = q
    if qid not in QEMB:
        return qid, []
    qvec = QEMB[qid].reshape(1, -1)
    candidates: Dict[int, float] = {}
    for idx in HNSW_IDXS:
        if idx is None or idx.ntotal == 0:
            continue
        scores, ids = idx.search(qvec, FETCH_K)
        for docid, score in zip(ids[0], scores[0]):
            if docid < 0:
                continue
            # keep max score if duplicate (unlikely)
            if docid in candidates:
                if score > candidates[docid]:
                    candidates[docid] = score
            else:
                candidates[docid] = score
    ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    return qid, ranked


def run_hnsw(queries: List[Tuple[str, str]], index_dirs: List[str], fetch_k: int, top_k: int, workers: int, query_emb_path: str, ef_search: int):
    with mp.Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(index_dirs, fetch_k, top_k, query_emb_path, ef_search),
    ) as pool:
        results = list(tqdm(pool.imap(process_query, queries), total=len(queries), desc="Queries"))
    return results


def main():
    ap = ArgumentParser(description="Run tiered HNSW retrieval with merge (multiprocessing), no rescore.")
    ap.add_argument("--qrels", choices=list(DATASETS.keys()), required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--overfetch-factor", type=int, default=2)
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    ap.add_argument("--query-emb", default="data/collection/query_embeddings_remapped.h5")
    ap.add_argument("--ef-search", type=int, default=200, help="HNSW efSearch for all indexes")
    ap.add_argument("--track", choices=["time", "memory"], required=False)
    args = ap.parse_args()

    dataset = DATASETS[args.qrels]
    qrels_path = dataset["qrels"]
    queries_path = dataset["queries"]
    queries = get_queries_subset(qrels_path, queries_path)

    index_dirs = [
        os.path.join(ARTIFACTS_DIR, "hnsw_T1"),
        os.path.join(ARTIFACTS_DIR, "hnsw_T2"),
        os.path.join(ARTIFACTS_DIR, "hnsw_T1_delta"),
        os.path.join(ARTIFACTS_DIR, "hnsw_T2_delta"),
    ]
    fetch_k = args.overfetch_factor * args.topk * len(index_dirs)

    results = track_performance(
        run_hnsw,
        queries,
        index_dirs,
        fetch_k,
        args.topk,
        args.workers,
        args.query_emb,
        args.ef_search,
        track=args.track,
    )

    save_path = os.path.join(RUNS_DIR, "hnsw_tiered", args.save)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_run(results, save_path)


if __name__ == "__main__":
    main()
