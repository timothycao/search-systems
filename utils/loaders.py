"""
Utility functions for loading MSMARCO input files (queries, qrels, runs, etc.).
"""

from collections import defaultdict
from typing import Dict, Tuple

import h5py
import numpy as np

def load_queries(file_path: str) -> Dict[str, str]:
    """
    Load queries file into {query_id: query_text}.
    """
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
    """
    Load run file into {query_id: {doc_id: score}}.
    """
    run: Dict[str, Dict[str, float]] = defaultdict(dict)
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip(): continue
            
            query_id, doc_id, _, score = line.strip().split("\t")
            run[query_id][doc_id] = float(score)

    return dict(run)

# TODO: Refactor needed components to use load_run and avoid duplication

def load_ranked_run(file_path: str) -> Dict[str, Dict[str, int]]:
    """
    Load run file into {query_id: {doc_id: rank}}.
    """
    run: Dict[str, Dict[str, int]] = defaultdict(dict)
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip(): continue
            
            query_id, doc_id, rank, _ = line.strip().split("\t")
            run[query_id][doc_id] = int(rank)

    return dict(run)

def load_h5_embeddings(file_path: str, id_key: str = 'id', embedding_key: str = 'embedding') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load IDs and embeddings from an HDF5 file.

    Args:
    - id_key: Dataset name for the IDs inside the HDF5 file.
    - embedding_key: Dataset name for the embeddings inside the HDF5 file.

    Returns:
    - ids: Numpy array of IDs (as strings).
    - embeddings: Numpy array of embeddings (as float32).
    """
    with h5py.File(file_path, 'r') as file:
        ids: np.ndarray = np.array(file[id_key]).astype(str)
        embeddings: np.ndarray = np.array(file[embedding_key]).astype(np.float32)  

    return ids, embeddings