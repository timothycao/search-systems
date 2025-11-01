"""
BM25 sparse retrieval system using Assignment 2 search_system package.
"""

import os
from contextlib import redirect_stdout
from io import StringIO
from typing import List, Tuple, Optional

from tqdm import tqdm
# Assignment 2 search_system package imports
from search_system.parser import run_parser
from search_system.indexer import run_indexer
from search_system.query import run_query, QueryStartupContext
from search_system.query.query import LIST_CACHE

from systems.retrieval.base import RetrievalSystem, QueryResult

class BM25System(RetrievalSystem):
    """BM25 retrieval based on custom inverted index implementation."""

    def __init__(self, dataset_path: str, subset_path: str, artifacts_dir: str) -> None:
        super().__init__("BM25")
        
        # Input resources
        self.dataset_path = dataset_path
        self.subset_path = subset_path
        
        # Output locations
        self.postings_dir = os.path.join(artifacts_dir, self.name.lower(), "postings")
        self.index_dir = os.path.join(artifacts_dir, self.name.lower(), "index")
        
        # Runtime state
        self.context: Optional[QueryStartupContext] = None

    def build(self) -> None:
        """Parse dataset and build BM25 index."""
        os.makedirs(self.postings_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)
        
        print(f"[{self.name}] Starting build pipeline...")
        run_parser(dataset_path=self.dataset_path, subset_ids_path=self.subset_path, output_dir=self.postings_dir)
        run_indexer(input_dir=self.postings_dir, output_dir=self.index_dir)

    def retrieve(self, queries: List[Tuple[str, str]], top_k: int = 100) -> List[QueryResult]:
        """Run BM25 retrieval for a list of queries."""
        # Keep all postings open to avoid file-handle eviction
        LIST_CACHE.cache.clear()
        LIST_CACHE.capacity = 1000000
        
        if self.context is None:
            print(f"[{self.name}] Loading index...")
            self.context = QueryStartupContext(self.index_dir)
        
        all_results: List[QueryResult] = []
        with tqdm(total=len(queries), desc=f"[{self.name}] Searching queries", unit="query") as progress:
            for query_id, query_text in queries:
                # Suppress prints from run_query (timing info)
                with redirect_stdout(StringIO()):
                    results = run_query(startup_context=self.context, query=query_text, mode="bwand-or", top_k=top_k)
                
                all_results.append((query_id, results))
                progress.update(1)
        
        return all_results