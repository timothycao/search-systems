"""
Reciprocal Rank Fusion (RRF) rerank system.
"""

from collections import defaultdict
from typing import Dict, List

from tqdm import tqdm

from systems.rerank.base import RerankSystem, QueryResult

class ReciprocalFusionSystem(RerankSystem):
    """Combine retrieval runs using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, k: int = 60) -> None:
        super().__init__("RRF")
        self.k = k  # Rank smoothing constant

    def build(self) -> None:
        """No setup required for RRF fusion."""
        pass

    def rerank(self, runs: Dict[str, str], top_k: int = 100) -> List[QueryResult]:
        """Fuse retrieval results using Reciprocal Rank Fusion."""
        run_a_path, run_b_path = runs.values()
        run_a = self._load_run(run_a_path)
        run_b = self._load_run(run_b_path)

        fused_results: List[QueryResult] = []
        all_queries = set(run_a.keys()) | set(run_b.keys())

        print(f"[{self.name}] Performing Reciprocal Rank Fusion...")

        for query_id in tqdm(all_queries, desc=f"[{self.name}] Fusing queries", unit="query"):
            rrf_scores: Dict[str, float] = defaultdict(float)
            docs_a = run_a.get(query_id, {})
            docs_b = run_b.get(query_id, {})
            all_docs = set(docs_a) | set(docs_b)

            for doc_id in all_docs:
                if doc_id in docs_a:
                    rank_a, _ = docs_a[doc_id]
                    rrf_scores[doc_id] += 1 / (self.k + rank_a)
                if doc_id in docs_b:
                    rank_b, _ = docs_b[doc_id]
                    rrf_scores[doc_id] += 1 / (self.k + rank_b)

            ranked_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            fused_results.append((query_id, ranked_docs))

        print(f"[{self.name}] Fusion complete for {len(all_queries)} queries.")
        return fused_results