"""
Utility functions for loading MSMARCO input files (queries, qrels, embeddings, etc.).
"""

from typing import Dict, Set

def load_queries(file_path: str) -> Dict[str, str]:
    """
    Load queries from a MSMARCO queries file (TSV format).
    Each line: query_id <TAB> query_text

    Returns:
        A dictionary mapping query_id -> query_text.
    """
    queries: Dict[str, str] = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip(): continue
            
            parts = line.strip().split("\t", 1)
            if len(parts) < 2: continue
            
            query_id, text = parts
            queries[query_id] = text
    
    return queries

def load_qrel_query_ids(file_path: str) -> Set[str]:
    """
    Load unique query IDs from a qrels file (first column only).

    Returns:
        A set of query IDs present in the qrels.
    """
    query_ids: Set[str] = set()
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip(): continue
            
            query_id = line.split()[0]
            query_ids.add(query_id)
    
    return query_ids