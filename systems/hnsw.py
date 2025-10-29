"""
HNSW search system using FAISS.
"""

import os
from typing import List, Tuple

import faiss
import numpy as np
from tqdm import tqdm

from systems.base import SearchSystem
from utils.loaders import load_h5_embeddings
from utils.config import SUBSET_EMBEDDINGS_PATH, QUERIES_EMBEDDINGS_PATH, ARTIFACTS_DIR, RUNS_DIR

# HNSW tuning parameters (higher = better accuracy, slower/more memory)
M: int = 8                  # Graph degree: average edges per node (suggested 4-8)
EF_CONSTRUCTION: int = 200  # Build-time beam width: candidates explored per insert (suggested 50-200)
EF_SEARCH: int = 200        # Search-time beam width: candidates explored per search (suggested 50-200)

# Types
RankedResults = List[Tuple[int, float]]
QueryResult = Tuple[str, RankedResults] # (query_id, [(doc_id, score), ...])

class HNSWSystem(SearchSystem):
    """Implements dense vector retrieval using FAISS HNSW index."""

    def __init__(self) -> None:
        super().__init__("HNSW")
        self.index: faiss.IndexHNSWFlat | None = None
        self.doc_ids: np.ndarray | None = None
    
    def build(self) -> None:
        """
        Build FAISS HNSW index from document embeddings.
        Outputs are stored under artifacts/hnsw/.
        """
        build_dir = os.path.join(ARTIFACTS_DIR, self.name.lower())
        os.makedirs(build_dir, exist_ok=True)
        index_path = os.path.join(build_dir, "index.faiss")
        doc_ids_path = os.path.join(build_dir, "doc_ids.npy")

        # Load document embeddings (doc_id -> doc_embedding)
        print(f"[{self.name}] Loading document embeddings...")
        doc_ids, doc_embeddings = load_h5_embeddings(SUBSET_EMBEDDINGS_PATH)

        # Normalize so inner product behaves like cosine similarity
        faiss.normalize_L2(doc_embeddings)
        
        # Initialize HNSW index
        index = faiss.IndexHNSWFlat(
            doc_embeddings.shape[1],
            M,
            faiss.METRIC_INNER_PRODUCT
        )
        
        # Set build-time beam width
        index.hnsw.efConstruction = EF_CONSTRUCTION

        # Add embeddings in batches (for progress display)
        batch_size = 10000
        with tqdm(total=len(doc_embeddings), desc=f"[{self.name}] Building index", unit="embedding") as progress:
            for start in range(0, len(doc_embeddings), batch_size):
                end = min(start + batch_size, len(doc_embeddings))
                index.add(doc_embeddings[start:end])
                progress.update(end - start)

        # Save index and corresponding doc IDs
        self.index = index
        self.doc_ids = doc_ids
        faiss.write_index(index, index_path)
        np.save(doc_ids_path, doc_ids)

    def search(self, queries: List[Tuple[str, str]], top_k: int = 10) -> List[QueryResult]:
        """
        Execute ANN retrieval for a list of queries.

        Args:
            queries: List of (query_id, query_text) pairs.
            top_k: Number of top documents to retrieve per query.

        Returns:
            A list of (query_id, ranked_results) pairs.
        """
        build_dir = os.path.join(ARTIFACTS_DIR, self.name.lower())
        index_path = os.path.join(build_dir, "index.faiss")
        doc_ids_path = os.path.join(build_dir, "doc_ids.npy")

        # Load index and doc IDs if not already in memory
        if self.index is None or self.doc_ids is None:
            print(f"[{self.name}] Loading index...")
            self.index = faiss.read_index(index_path)
            self.doc_ids = np.load(doc_ids_path, allow_pickle=True)

        # Load and normalize query embeddings (must match index normalization)
        print(f"[{self.name}] Loading query embeddings...")
        query_ids, query_embeddings = load_h5_embeddings(QUERIES_EMBEDDINGS_PATH)
        faiss.normalize_L2(query_embeddings)
        query_map = dict(zip(query_ids, query_embeddings))

        # Set search-time beam width
        self.index.hnsw.efSearch = EF_SEARCH

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

    def save_run(self, results: List[QueryResult], output_filename: str) -> None:
        """
        Save ranked retrieval results in plain tab-separated format.

        Args:
            results: List of (query_id, ranked_results) pairs.
            output_filename: Name of the output file (saved under runs/bm25/).
        """
        output_dir = os.path.join(RUNS_DIR, self.name.lower())
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, "w", encoding="utf-8") as output_file:
            with tqdm(total=len(results), desc=f"[{self.name}] Saving results", unit="query") as progress:
                for query_id, ranked_docs in results:
                    for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
                        # Columns: query_id, doc_id, rank, score
                        output_file.write(f"{query_id}\t{doc_id}\t{rank}\t{score:.6f}\n")
                    
                    progress.update(1)