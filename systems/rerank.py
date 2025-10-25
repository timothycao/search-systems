"""
BM25 → HNSW re-ranking system.
"""

from typing import List

from systems.base import SearchSystem

class ReRankSystem(SearchSystem):
    def __init__(self) -> None:
        super().__init__("ReRank")

    def build(self) -> None:
        print("[ReRank] Preparing re-ranking pipeline...")

    def search(self, queries: List[str], top_k: int = 10) -> List:
        print(f"[ReRank] Re-ranking BM25 candidates...")
        return []

    def save_run(self, results: List, output_path: str) -> None:
        print(f"[ReRank] Saving results to {output_path}...")