"""
HNSW dense retrieval system using FAISS.
"""

import os
from typing import List, Tuple, Optional

import faiss
import numpy as np
from tqdm import tqdm

from systems.retrieval.base import RetrievalSystem, QueryResult

class HNSWSystem(RetrievalSystem):
    """Dense vector retrieval using FAISS HNSW index."""

    def __init__(
        self,
        subset_embeddings_path: str,
        query_embeddings_path: str,
        artifacts_dir: str,
        m: int = 8,
        ef_construction: int = 200,
        ef_search: int = 200,
    ) -> None:
        super().__init__("HNSW")
        
        # Input resources
        self.subset_embeddings_path = subset_embeddings_path
        self.query_embeddings_path = query_embeddings_path
        
        # Output locations
        self.build_dir = os.path.join(artifacts_dir, self.name.lower())
        self.index_path = os.path.join(self.build_dir, "index.faiss")
        self.doc_ids_path = os.path.join(self.build_dir, "doc_ids.npy")
        
        # HNSW tuning parameters (higher = better accuracy, slower/more memory)
        self.m = m                              # Graph degree: average edges per node (suggested 4-8)
        self.ef_construction = ef_construction  # Build-time beam width: candidates explored per insert (suggested 50-200)
        self.ef_search = ef_search              # Search-time beam width: candidates explored per search (suggested 50-200)

        # Runtime state
        self.index: Optional[faiss.IndexHNSWFlat] = None
        self.doc_ids: Optional[np.ndarray] = None
    
    def build(self) -> None:
        """Build HNSW index from document embeddings."""
        os.makedirs(self.build_dir, exist_ok=True)

        # Load document embeddings (doc_id -> doc_embedding)
        print(f"[{self.name}] Loading document embeddings...")
        doc_ids, doc_embeddings = self._load_embeddings(self.subset_embeddings_path)

        # Normalize so inner product behaves like cosine similarity
        faiss.normalize_L2(doc_embeddings)
        
        # Initialize HNSW index
        index = faiss.IndexHNSWFlat(doc_embeddings.shape[1], self.m, faiss.METRIC_INNER_PRODUCT)
        
        # Set build-time beam width
        index.hnsw.efConstruction = self.ef_construction

        # Add embeddings in batches (for progress display)
        batch_size = 10000
        with tqdm(total=len(doc_embeddings), desc=f"[{self.name}] Building index", unit="embedding") as progress:
            for start in range(0, len(doc_embeddings), batch_size):
                end = min(start + batch_size, len(doc_embeddings))
                index.add(doc_embeddings[start:end])
                progress.update(end - start)

        # Save index and corresponding doc IDs
        faiss.write_index(index, self.index_path)
        np.save(self.doc_ids_path, doc_ids)

    def retrieve(self, queries: List[Tuple[str, str]], top_k: int = 100) -> List[QueryResult]:
        """Run approximate nearest-neighbor retrieval for a list of queries."""
        # Load index and doc IDs if not already in memory
        if self.index is None or self.doc_ids is None:
            print(f"[{self.name}] Loading index...")
            self.index = faiss.read_index(self.index_path)
            self.doc_ids = np.load(self.doc_ids_path, allow_pickle=True)

        # Load and normalize query embeddings (must match index normalization)
        print(f"[{self.name}] Loading query embeddings...")
        query_ids, query_embeddings = self._load_embeddings(self.query_embeddings_path)
        faiss.normalize_L2(query_embeddings)
        query_map = dict(zip(query_ids, query_embeddings))

        # Set search-time beam width
        self.index.hnsw.efSearch = self.ef_search

        # Perform ANN search for each query
        all_results: List[QueryResult] = []
        with tqdm(total=len(queries), desc=f"[{self.name}] Searching queries", unit="query") as progress:
            for query_id, _ in queries:
                query_embedding = query_map.get(query_id)
                if query_embedding is None:
                    progress.update(1)
                    continue

                scores, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
                ranked = [
                    (int(self.doc_ids[i]), float(scores[0][j]))
                    for j, i in enumerate(indices[0])
                ]
                all_results.append((query_id, ranked))
                progress.update(1)

        return all_results