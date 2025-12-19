"""
I/O utilities for loading and saving MS MARCO dataset files and system runs.
Includes helpers for reading queries, qrels, and runs, and writing ranked outputs.
"""

import os
from collections import defaultdict
from typing import Set, Dict, List, Tuple

def load_docids(file_path: str) -> Set[str]:
    docids: Set[str] = set()
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip(): continue
            
            docid, _ = line.strip().split("\t", 1)
            docids.add(docid)
    
    return docids

def load_queries(file_path: str) -> Dict[str, str]:
    """Load queries file into {query_id: query_text}."""
    queries: Dict[str, str] = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip(): continue
            
            query_id, text = line.strip().split("\t", 1)
            queries[query_id] = text
    
    return queries

def load_qrels(file_path: str) -> Dict[str, Dict[str, int]]:
    """
    Load qrels file into {query_id: {doc_id: relevance}}.

    Handles both formats:
      - 3 columns: query_id, doc_id, relevance  (dev set)
      - 4 columns: query_id, <ignored>, doc_id, relevance  (eval sets)
    """
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip(): continue
            
            parts = line.strip().split("\t")
            if len(parts) == 3: query_id, doc_id, rel = parts
            elif len(parts) == 4: query_id, _, doc_id, rel = parts
            else: continue

            qrels[query_id][doc_id] = int(rel)

    return dict(qrels)

def load_run(file_path: str) -> Dict[str, Dict[str, float]]:
    """Load run file into {query_id: {doc_id: score}}."""
    run: Dict[str, Dict[str, float]] = defaultdict(dict)
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip(): continue
            
            query_id, doc_id, _, score = line.strip().split("\t")
            run[query_id][doc_id] = float(score)

    return dict(run)

def load_passages_from_subset(dataset_path: str, subset_path: str) -> Dict[str, str]:
    """
    Load passage texts for a subset of passage IDs.
    Uses the subset file to filter the main dataset.
    """
    # Load allowed PIDs
    with open(subset_path, "r", encoding="utf-8") as subset_file:
        subset_pids = {line.strip() for line in subset_file if line.strip()}

    passages: Dict[str, str] = {}
    with open(dataset_path, "r", encoding="utf-8") as dataset_file:
        for line in dataset_file:
            pid, text = line.strip().split("\t", 1)
            if pid in subset_pids:
                passages[pid] = text

    return passages

def save_run(results: List[Tuple[str, List[Tuple[str, float]]]], output_path: str) -> None:
    """Save ranked retrieval results in plain tab-separated format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as output_file:
        for query_id, ranked_docs in results:
            for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
                # Columns: query_id, doc_id, rank, score
                output_file.write(f"{query_id}\t{doc_id}\t{rank}\t{score:.6f}\n")