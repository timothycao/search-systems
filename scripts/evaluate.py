"""
Evaluate search system results on MS MARCO datasets using pytrec_eval.
Usage:
    python -m scripts.evaluate \
        --system <bm25 | hnsw | rerank> \
        --qrels <dev | eval1 | eval2> \
        --run <filename>
"""

import os
from argparse import ArgumentParser
from typing import Dict, List

from pytrec_eval import RelevanceEvaluator

from utils.loaders import load_qrels, load_run
from utils.config import RUNS_DIR, QRELS_DEV_PATH, QRELS_EVAL1_PATH, QRELS_EVAL2_PATH

# Available systems
SYSTEMS: List[str] = ["bm25", "hnsw", "rerank-rrf", "rerank-lsf"]

# Available qrels
QRELS: Dict[str, str] = {
    "dev": QRELS_DEV_PATH,
    "eval1": QRELS_EVAL1_PATH,
    "eval2": QRELS_EVAL2_PATH,
}

# Metric labels mapped to official trec_eval keys
METRICS: Dict[str, str] = {
    "MRR@10": "recip_rank",
    "Recall@100": "recall_100",
    "NDCG@10": "ndcg_cut_10",
    "NDCG@100": "ndcg_cut_100",
    "MAP": "map",
}

def evaluate(run: Dict, qrels: Dict, is_binary: bool) -> None:
    """Compute MRR, Recall, and NDCG/MAP metrics using pytrec_eval."""
    evaluator = RelevanceEvaluator(qrels, METRICS.values())
    results = evaluator.evaluate(run)
    # print(results)

    # Aggregate scores across all queries for each metric
    aggregated = {
        key: [res[key] for res in results.values() if key in res]
        for key in METRICS.values()
    }

    # Print metrics (skip NDCG for binary qrels and MAP for graded qrels)
    for label, key in METRICS.items():
        if is_binary and key in ["ndcg_cut_10", "ndcg_cut_100"]: continue
        if not is_binary and key in ["map"]: continue
        
        values = aggregated.get(key, [])
        if values: print(f"{label:<10}: {sum(values) / len(values):.4f}")

def main() -> None:
    # Parse command line arguments
    parser = ArgumentParser(description="Evaluate results using pytrec_eval.")
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--qrels", choices=list(QRELS.keys()), required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    # Resolve paths
    run_path = os.path.join(RUNS_DIR, args.system, args.run)
    qrels_path = QRELS[args.qrels]

    # Load data
    run = load_run(run_path)
    qrels = load_qrels(qrels_path)

    # Evaluate (binary for dev, graded for eval)
    is_binary = args.qrels == "dev"
    evaluate(run, qrels, is_binary)

if __name__ == "__main__":
    main()