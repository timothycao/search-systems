"""
BM25 search system using Assignment 2 search_system package.
"""

import os
from contextlib import redirect_stdout
from io import StringIO
from typing import List, Tuple

from tqdm import tqdm
# Assignment 2 search_system package imports
from search_system.parser import run_parser
from search_system.indexer import run_indexer
from search_system.query import run_query, QueryStartupContext
from search_system.query.query import LIST_CACHE

from systems.base import SearchSystem
from utils.config import DATASET_PATH, SUBSET_PATH, ARTIFACTS_DIR, RUNS_DIR

RankedResults = List[Tuple[int, float]]
QueryResult = Tuple[str, RankedResults] # (query_id, [(doc_id, score), ...])

class BM25System(SearchSystem):
    """Implements the BM25 retrieval system using the search_system package."""

    def __init__(self) -> None:
        super().__init__("BM25")
        self.context: QueryStartupContext | None = None # loaded once before querying

    def build(self) -> None:
        """
        Build BM25 index by parsing the raw dataset (filtered to subset IDs)
        and indexing the resulting posting chunks.
        Outputs are stored under artifacts/bm25/.
        """
        postings_dir = os.path.join(ARTIFACTS_DIR, self.name.lower(), "postings")
        index_dir = os.path.join(ARTIFACTS_DIR, self.name.lower(), "index")
        os.makedirs(postings_dir, exist_ok=True)
        os.makedirs(index_dir, exist_ok=True)
        
        print(f"[{self.name}] Starting build pipeline...")
        run_parser(
            dataset_path=DATASET_PATH,
            subset_ids_path=SUBSET_PATH,
            output_dir=postings_dir
        )
        run_indexer(
            input_dir=postings_dir,
            output_dir=index_dir
        )

    def search(self, queries: List[Tuple[str, str]], top_k: int = 10) -> List[QueryResult]:
        """
        Execute BM25 retrieval for a list of queries.

        Args:
            queries: List of (query_id, query_text) pairs.
            top_k: Number of top documents to retrieve per query.

        Returns:
            A list of (query_id, ranked_results) pairs.
        """
        # Prevent mid-query eviction from closing file handles
        LIST_CACHE.cache.clear()
        LIST_CACHE.capacity = 1000000 # big enough to avoid eviction
        
        index_dir = os.path.join(ARTIFACTS_DIR, self.name.lower(), "index")
        
        if self.context is None:
            print(f"[{self.name}] Loading index...")
            self.context = QueryStartupContext(index_dir)
        
        all_results: List[QueryResult] = []
        with tqdm(total=len(queries), desc=f"[{self.name}] Searching queries", unit="query") as progress:
            for query_id, query_text in queries:
                # Suppress prints from run_query (timing info)
                with redirect_stdout(StringIO()):
                    results = run_query(
                        startup_context=self.context,
                        query=query_text,
                        mode="bwand-or",
                        top_k=top_k
                    )
                
                all_results.append((query_id, results))
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