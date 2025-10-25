"""
HNSW search system.
"""

from typing import List

from systems.base import SearchSystem

class HNSWSystem(SearchSystem):
    def __init__(self) -> None:
        super().__init__("HNSW")

    def build(self) -> None:
        print("[HNSW] Building dense vector index...")

    def search(self, queries: List[str], top_k: int = 10) -> List:
        print("[HNSW] Searching using index...")
        return []

    def save_run(self, results: List, output_path: str) -> None:
        print(f"[HNSW] Saving results to {output_path}...")