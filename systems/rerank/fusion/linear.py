"""
Linear Score Fusion (LSF) rerank system.
"""

from collections import defaultdict
from typing import Dict, List

from tqdm import tqdm

from systems.rerank.base import RerankSystem, QueryResult

class LinearFusionSystem(RerankSystem):
    """Combine retrieval runs using Linear Score Fusion (LSF)."""

    def __init__(self, alpha: float = 0.6) -> None:
        super().__init__("LSF")
        self.alpha = alpha  # Weight for the first run’s contribution

    def build(self) -> None:
        """No setup required for LSF fusion."""
        pass

    def rerank(self, runs: Dict[str, str], top_k: int = 100) -> List[QueryResult]:
        """Fuse retrieval results using weighted linear score combination."""
        run_a_path, run_b_path = runs.values()
        run_a = self._load_run(run_a_path)
        run_b = self._load_run(run_b_path)

        fused_results: List[QueryResult] = []
        all_queries = set(run_a.keys()) | set(run_b.keys())

        print(f"[{self.name}] Performing Linear Score Fusion...")

        for query_id in tqdm(all_queries, desc=f"[{self.name}] Fusing queries", unit="query"):
            lsf_scores: Dict[str, float] = defaultdict(float)
            docs_a = run_a.get(query_id, {})
            docs_b = run_b.get(query_id, {})
            all_docs = set(docs_a) | set(docs_b)

            for doc_id in all_docs:
                _, score_a = run_a[query_id].get(doc_id, (0, 0.0))
                _, score_b = run_b[query_id].get(doc_id, (0, 0.0))
                lsf_scores[doc_id] = self.alpha * score_a + (1 - self.alpha) * score_b

            ranked_docs = sorted(lsf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            fused_results.append((query_id, ranked_docs))

        print(f"[{self.name}] Fusion complete for {len(all_queries)} queries.")
        return fused_results
