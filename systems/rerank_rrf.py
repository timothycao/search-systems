"""
Reciprocal Rank Fusion (RRF) Reranker System
"""

import os
from collections import defaultdict
from typing import List, Tuple

from tqdm import tqdm
from systems.base import SearchSystem
from utils.loaders import load_ranked_run
from utils.config import RUNS_BM25_DIR, RUNS_HNSW_DIR, RUNS_RERANK_RRF_DIR

class RecipricalRankFusion(SearchSystem):
    def __init__(self, k: int = 60) -> None:
        super().__init__("ReRank")
        self.k = k  # Number of BM25 candidates to re-rank

    def build(self) -> None:
        print("[ReRank] Preparing re-ranking pipeline...")
        

    def search(
        self,
        bm25_filename: str = "bm25_eval1.tsv",
        hnsw_filename: str = "hnsw_eval1.tsv",
        top_k: int = 100,
    ) -> List[Tuple[str, List[Tuple[str, float]]]]:
        """
        Perform Reciprocal Rank Fusion (RRF) between BM25 and HNSW runs.

        Returns:
            List of (query_id, ranked_results)
            where ranked_results = List[(doc_id, rrf_score)]
        """
        print(f"[ReRank] Re-ranking BM25 candidates...")

        bm25_path = os.path.join(RUNS_BM25_DIR, bm25_filename)
        hnsw_path = os.path.join(RUNS_HNSW_DIR, hnsw_filename)

        print(f"[ReRank] Loading BM25 run: {bm25_path}")
        bm25 = load_ranked_run(bm25_path)

        print(f"[ReRank] Loading HNSW run: {hnsw_path}")
        hnsw = load_ranked_run(hnsw_path)

        print("[ReRank] Computing Reciprocal Rank Fusion (RRF)...")

        fused_results: List[Tuple[str, List[Tuple[str, float]]]] = []
        all_queries = set(bm25.keys()) | set(hnsw.keys())

        for qid in tqdm(all_queries, desc="[ReRank] Fusing queries", unit="query"):
            rrf_scores: dict[str, float] = defaultdict(float)

            # Build rank mappings based on descending scores
            bm25_docs = bm25.get(qid, {})
            hnsw_docs = hnsw.get(qid, {})
            all_docs = set(bm25_docs.keys()) | set(hnsw_docs.keys())

            # Compute RRF score for each candidate doc
            for pid in all_docs:
                if pid in bm25_docs:
                    rrf_scores[pid] += 1 / (self.k + bm25_docs[pid])
                if pid in hnsw_docs:
                    rrf_scores[pid] += 1 / (self.k + hnsw_docs[pid])

            ranked_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            fused_results.append((qid, ranked_docs))

        print(f"[ReRank] Fusion complete for {len(all_queries)} queries.")
        return fused_results

    def save_run(self, results: List[Tuple[str, List[Tuple[str, float]]]], output_filename: str) -> None:
        """
        Save ranked retrieval results in plain tab-separated format.

        Args:
            results: List of (query_id, ranked_results) pairs.
            output_filename: Name of the output file (saved under runs/rerank/).
        """
        os.makedirs(RUNS_RERANK_RRF_DIR, exist_ok=True)
        output_path = os.path.join(RUNS_RERANK_RRF_DIR, output_filename)

        with open(output_path, "w", encoding="utf-8") as output_file:
            with tqdm(total=len(results), desc=f"[{self.name}] Saving results", unit="query") as progress:
                for query_id, ranked_docs in results:
                    for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
                        output_file.write(f"{query_id}\t{doc_id}\t{rank}\t{score:.6f}\n")
                    progress.update(1)

        print(f"[ReRank] Saved run to {output_path}")