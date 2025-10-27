"""
Utility functions for loading MSMARCO input files (queries, qrels, runs, etc.).
"""

from collections import defaultdict
from typing import Dict

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