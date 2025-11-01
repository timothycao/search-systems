"""
Base class for retrieval systems (e.g., BM25, HNSW).
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import h5py
import numpy as np

# Types
RankedResults = List[Tuple[int, float]]
QueryResult = Tuple[str, RankedResults] # (query_id, [(doc_id, score), ...])

class RetrievalSystem(ABC):
    """Abstract base class for document retrieval systems."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def build(self) -> None:
        """Build or load resources required for retrieval."""
        pass

    @abstractmethod
    def retrieve(self, queries: List, top_k: int = 100) -> List[QueryResult]:
        """Retrieve top documents for given queries."""
        pass
    
    def _load_embeddings(self, file_path: str, id_key: str = 'id', embedding_key: str = 'embedding') -> Tuple[np.ndarray, np.ndarray]:
        """Load IDs and embeddings from an HDF5 file."""
        with h5py.File(file_path, 'r') as file:
            ids: np.ndarray = np.array(file[id_key]).astype(str)
            embeddings: np.ndarray = np.array(file[embedding_key]).astype(np.float32)  

        return ids, embeddings