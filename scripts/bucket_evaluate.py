"""
Comprehensively evaluate search system results on MS MARCO datasets using pytrec_eval 
on a aggregate and query length basis.
Usage:
    python -m scripts.bucket_evaluate \
        --system <bm25 | hnsw | rrf | lsf | biencoder> \
        --qrels <dev | eval1 | eval2> \
        --run <filename>
        --save <output_filename>
"""

import os
import statistics
from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict, List

import pytrec_eval

from utils.io import load_qrels, load_run, load_queries
from utils.config import (
    RUNS_DIR,
    QRELS_DEV_PATH,
    QRELS_EVAL1_PATH,
    QRELS_EVAL2_PATH,
    QUERIES_EVAL_PATH,
    QUERIES_DEV_PATH
)

# Systems must match evaluate.py
SYSTEMS: List[str] = ["bm25", "hnsw", "rrf", "lsf", "biencoder"]

# Qrels mapping must match evaluate.py
QRELS = {
    "dev": QRELS_DEV_PATH,
    "eval1": QRELS_EVAL1_PATH,
    "eval2": QRELS_EVAL2_PATH,
}

# Metrics must match evaluate.py
METRICS = {
    "MRR@10": "recip_rank",
    "Recall@100": "recall_100",
    "NDCG@10": "ndcg_cut_10",
    "NDCG@100": "ndcg_cut_100",
    "MAP": "map",
}


# -----------------------------
# Query Length Bucketing
# -----------------------------
def query_length_bin(n_tokens: int) -> str:
    if n_tokens <= 3:
        return "short"
    elif n_tokens <= 6:
        return "medium"
    else:
        return "long"


# -----------------------------
# Filter Metrics (binary vs graded)
# -----------------------------
def filter_metrics_for_qrels(is_binary: bool) -> Dict[str, str]:
    filtered = {}
    for label, key in METRICS.items():
        if is_binary and key.startswith("ndcg"):
            continue
        if not is_binary and key == "map":
            continue
        filtered[label] = key
    return filtered


# -----------------------------
# Aggregated Metrics
# -----------------------------
def compute_aggregated_metrics(run, qrels, is_binary):
    metrics = filter_metrics_for_qrels(is_binary)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, set(metrics.values()))
    results = evaluator.evaluate(run)

    aggregated = {
        label: statistics.mean([metric_values[key] for metric_values in results.values()])
        for label, key in metrics.items()
    }

    return aggregated, results, metrics


# -----------------------------
# Bucketed Metrics
# -----------------------------
def compute_bucketed_metrics(results, queries, is_binary):
    metrics = filter_metrics_for_qrels(is_binary)

    # Compute query lengths
    qlens = {qid: len(text.split()) for qid, text in queries.items()}
    bins = {qid: query_length_bin(n) for qid, n in qlens.items()}

    # Query count statistics
    bucket_counts = {
        "short": 0,
        "medium": 0,
        "long": 0,
    }

    bucket_scores = {
        "short": defaultdict(list),
        "medium": defaultdict(list),
        "long": defaultdict(list),
    }

    for qid, metric_values in results.items():
        if qid not in bins:
            continue

        bucket = bins[qid]
        bucket_counts[bucket] += 1

        for label, key in metrics.items():
            if key in metric_values:
                bucket_scores[bucket][label].append(metric_values[key])

    # Average bucket metrics
    bucket_avgs = {
        bucket: {
            label: (statistics.mean(values) if values else 0.0)
            for label, values in bucket_scores[bucket].items()
        }
        for bucket in ["short", "medium", "long"]
    }

    return bucket_avgs, bucket_counts


# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = ArgumentParser(description="Analyze performance by query length.")
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--qrels", choices=list(QRELS.keys()), required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--save", required=True, help="Filename (without extension) for saving results")
    args = parser.parse_args()

    # Resolve paths
    run_path = os.path.join(RUNS_DIR, args.system, args.run)
    qrels_path = QRELS[args.qrels]

    run = load_run(run_path)
    qrels = load_qrels(qrels_path)

    if args.qrels == "dev":
        queries = load_queries(QUERIES_DEV_PATH)
    else:
        queries = load_queries(QUERIES_EVAL_PATH)
        
    is_binary = (args.qrels == "dev")

    # 1. Overall (aggregated) metrics
    overall, results, metrics_used = compute_aggregated_metrics(run, qrels, is_binary)

    # 2. Bucketed metrics + counts
    bucketed, bucket_counts = compute_bucketed_metrics(results, queries, is_binary)

    total_queries = sum(bucket_counts.values())

    # -----------------------------
    # Build Output Text
    # -----------------------------
    out = []
    out.append("===================== OVERALL PERFORMANCE =====================\n")
    for label, value in overall.items():
        out.append(f"{label:<12}: {value:.4f}")
    out.append(f"\nMetrics used: {', '.join(metrics_used.keys())}\n")

    # Query count statistics
    out.append("===================== QUERY LENGTH STATISTICS =====================\n")
    out.append(f"Total Queries: {total_queries}")
    out.append(f"Short Queries (1–3 tokens):  {bucket_counts['short']}")
    out.append(f"Medium Queries (4–6 tokens): {bucket_counts['medium']}")
    out.append(f"Long Queries (7+ tokens):    {bucket_counts['long']}\n")

    # Bucketed metrics
    out.append("================= PERFORMANCE BY QUERY LENGTH =================\n")
    for bucket in ["short", "medium", "long"]:
        out.append(f"\n--- {bucket.upper()} QUERIES ---")
        for label, value in bucketed[bucket].items():
            out.append(f"{label:<12}: {value:.4f}")

    output_text = "\n".join(out)

    # Print
    print(output_text)

    # Save to results/<system>/<filename>.txt
    output_dir = os.path.join("results", args.system)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.save)

    with open(output_path, "w") as f:
        f.write(output_text)

    print(f"\nSaved results to {output_path}\n")


if __name__ == "__main__":
    main()
