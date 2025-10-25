"""
BM25 search system.
"""

from typing import List

from systems.base import SearchSystem

class BM25System(SearchSystem):
    def __init__(self) -> None:
        super().__init__("BM25")

    def build(self) -> None:
        print("[BM25] Building inverted index...")

    def search(self, queries: List[str], top_k: int = 10) -> List:
        print("[BM25] Searching using ranking...")
        return []

    def save_run(self, results: List, output_path: str) -> None:
        print(f"[BM25] Saving results to {output_path}...")