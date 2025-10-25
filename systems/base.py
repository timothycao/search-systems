"""
Base class for all search systems (BM25, HNSW, ReRank).
"""

from abc import ABC, abstractmethod
from typing import List

class SearchSystem(ABC):
    """Abstract base class representing a generic search system."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def build(self) -> None:
        """Build or load resources required by the system."""
        pass

    @abstractmethod
    def search(self, queries: List[str], top_k: int = 10) -> List:
        """Execute retrieval for given queries."""
        pass

    @abstractmethod
    def save_run(self, results: List, output_path: str) -> None:
        """Save results to disk in TREC format."""
        pass