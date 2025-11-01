"""
Base class rerank systems (fusion and neural).
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple

# Types
RankedResults = List[Tuple[int, float]]
QueryResult = Tuple[str, RankedResults] # (query_id, [(doc_id, score), ...])

class RerankSystem(ABC):
    """Abstract base class for rerankers and fusion systems."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def build(self, **kwargs) -> None:
        """Load model weights or other resources."""
        pass

    @abstractmethod
    def rerank(self, runs: Dict[str, str], top_k: int = 100) -> List:
        """Combine or rerank runs from one or more retrieval systems."""
        pass

    def _load_run(self, file_path: str) -> Dict[str, Dict[str, Tuple[int, float]]]:
        """Load run file into {query_id: {doc_id: (rank, score)}}."""
        run: Dict[str, Dict[str, Tuple[int, float]]] = defaultdict(dict)
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                if not line.strip(): continue
                
                query_id, doc_id, rank, score = line.strip().split("\t")
                run[query_id][doc_id] = (int(rank), float(score))

        return dict(run)