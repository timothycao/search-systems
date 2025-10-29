"""
Linear Combination Reranker System
"""

import os
from collections import defaultdict
from typing import List, Tuple
from tqdm import tqdm

from systems.base import SearchSystem
from utils.loaders import load_run
from utils.config import RUNS_BM25_DIR, RUNS_HNSW_DIR, RUNS_RERANK_LCF_DIR

class LinearScoreFusion(SearchSystem):
    def __init__(self, alpha: float = 0.6):
        super().__init__("ReRankTwo")
        self.alpha = alpha

    def build(self):
        print("[ReRankTwo] No build step required.")

    def search(self, bm25_filename: str, hnsw_filename: str, top_k: int = 100):
        bm25_path = os.path.join(RUNS_BM25_DIR, bm25_filename)
        hnsw_path = os.path.join(RUNS_HNSW_DIR, hnsw_filename)

        print(f"[ReRankTwo] Loading BM25: {bm25_path}")
        print(f"[ReRankTwo] Loading HNSW: {hnsw_path}")

        bm25 = load_run(bm25_path)
        hnsw = load_run(hnsw_path)

        fused_results = []
        all_queries = set(bm25.keys()) | set(hnsw.keys())

        for qid in tqdm(all_queries, desc="[ReRankTwo] Fusing", unit="query"):
            scores = defaultdict(float)
            all_docs = set(bm25.get(qid, {})) | set(hnsw.get(qid, {}))

            for pid in all_docs:
                b = bm25[qid].get(pid, 0.0)
                h = hnsw[qid].get(pid, 0.0)
                scores[pid] = self.alpha * b + (1 - self.alpha) * h

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            fused_results.append((qid, ranked))

        print("[ReRankTwo] Fusion complete.")
        return fused_results

    def save_run(self, results: List[Tuple[str, List[Tuple[str, float]]]], output_filename: str):
        os.makedirs(RUNS_RERANK_LCF_DIR, exist_ok=True)
        output_path = os.path.join(RUNS_RERANK_LCF_DIR, output_filename)

        with open(output_path, "w", encoding="utf-8") as f:
            with tqdm(total=len(results), desc=f"[{self.name}] Saving", unit="query") as pbar:
                for qid, ranked_docs in results:
                    for rank, (pid, score) in enumerate(ranked_docs, start=1):
                        f.write(f"{qid}\t{pid}\t{rank}\t{score:.6f}\n")
                    pbar.update(1)

        print(f"[ReRankTwo] Saved run to {output_path}")
