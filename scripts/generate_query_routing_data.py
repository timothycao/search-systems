"""
Generate supervised query-routing labels and features using judged qrels.

Label rule:
- 0 if judged hit in T1 AND pseudo_recall ≥ threshold
- 1 if not in T1-only top-k, but is in FULL (T1+T2) top-k
- drop if not even in FULL top-k

Query features:
- q_len_terms, q_len_chars, q_unique_terms, q_frac_unique, q_avg_term_len
- idf_max, idf_min, idf_mean, idf_std  (needs index lexicon + N)

Output:
- JSONL file: one example per line: {"qid": <str>, "y": <int>, "x": <List[float]>}
- Sidecar JSON: {"feature_names": [...]}

Usage:
  python -m scripts.generate_query_routing_data \
    --qrels qrels.train.sample.tsv \
    --queries queries.train.sample.tsv \
    --run-t1 queries_train_sample.run \
    --run-full queries_train_sample.run \
    --topk 100 \
    --index bm25_full \
    --out train_query_routing.jsonl
"""

import json
import os
import math
from argparse import ArgumentParser
from typing import Dict, Set, List

from search_system.query.query_startup_context import QueryStartupContext
from search_system.shared.utils import tokenize

from utils.io import load_qrels, load_run_topk, load_queries
from utils.config import QRELS_DIR, QUERIES_DIR, RUNS_DIR, ARTIFACTS_DIR


FEATURE_NAMES: List[str] = [
    "q_len_terms",
    "q_len_chars",
    "q_unique_terms",
    "q_frac_unique",
    "q_avg_term_len",
    "idf_max",
    "idf_min",
    "idf_mean",
    "idf_std",
]


def compute_idf(df: int, N: int) -> float:
    return math.log(((N - df + 0.5) / (df + 0.5)) + 1.0)


def compute_query_features_vec(qtext: str, lexicon: Dict[str, Dict], N: int) -> List[float]:
    tokens = tokenize(qtext)
    q_len = len(tokens)
    chars = len(qtext)

    uniq_terms = set(tokens)
    unique = len(uniq_terms)

    frac_unique = (unique / q_len) if q_len > 0 else 0.0
    avg_term_len = (sum(len(t) for t in tokens) / q_len) if q_len > 0 else 0.0

    idfs: List[float] = []
    for t in uniq_terms:
        meta = lexicon.get(t)
        if not meta: continue
        df = int(meta.get("df", 0))
        if df > 0: idfs.append(compute_idf(df, N))

    if idfs:
        idf_max = max(idfs)
        idf_min = min(idfs)
        idf_mean = sum(idfs) / len(idfs)
        var = sum((x - idf_mean) ** 2 for x in idfs) / len(idfs)
        idf_std = math.sqrt(var)
    else:
        idf_max = idf_min = idf_mean = idf_std = 0.0

    return [
        float(q_len),
        float(chars),
        float(unique),
        float(frac_unique),
        float(avg_term_len),
        float(idf_max),
        float(idf_min),
        float(idf_mean),
        float(idf_std),
    ]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--run-t1", required=True)
    parser.add_argument("--run-full", required=True)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()

    qrels_path = os.path.join(QRELS_DIR, args.qrels)
    queries_path = os.path.join(QUERIES_DIR, args.queries)

    run_t1_path = os.path.join(RUNS_DIR, "bm25_T1", args.run_t1)
    run_full_path = os.path.join(RUNS_DIR, "bm25_full", args.run_full)

    index_dir = os.path.join(ARTIFACTS_DIR, args.index, "index")

    out_path = os.path.join(ARTIFACTS_DIR, "query_routing", args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Write feature names
    features_meta_path = os.path.splitext(out_path)[0] + "_features.json"
    with open(features_meta_path, "w", encoding="utf-8") as fmeta:
        json.dump({"feature_names": FEATURE_NAMES}, fmeta, indent=2)

    # Load index context for IDF stats
    context = QueryStartupContext(index_dir)
    lexicon = context.lexicon
    N = context.total_docs

    qrels: Dict[str, Dict[str, int]] = load_qrels(qrels_path)
    queries: Dict[str, str] = load_queries(queries_path)
    t1_topk = load_run_topk(run_t1_path, args.topk)
    full_topk = load_run_topk(run_full_path, args.topk)

    kept = 0
    dropped_no_full_hit = 0
    dropped_missing_run = 0
    dropped_missing_query = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for qid, rels in qrels.items():
            # Fetch query text (skip if missing)
            qtext = queries.get(qid)
            if not qtext:
                dropped_missing_query += 1
                continue

            # Extract set of judged relevant docids for this query
            rel_docs: Set[str] = {docid for docid, rel in rels.items() if rel > 0}
            if not rel_docs: continue   # skip if none

            # Retrieve top-k doc sets from T1-only and FULL runs
            t1_docs = set(t1_topk.get(qid, []))
            full_docs = set(full_topk.get(qid, []))

            # Drop queries missing from the FULL run
            if not full_docs:
                dropped_missing_run += 1
                continue

            # Compute label from run-vs-qrels overlap
            t1_rel_hits = len(rel_docs & t1_docs)
            full_rel_hits = len(rel_docs & full_docs)

            # Assign label 0 if any relevant doc appears in T1-only top-k
            if t1_rel_hits > 0:
                # Additional check: "pseudo-recall" where FULL top-k is treated as ground truth
                pseudo_recall = (len(t1_docs & full_docs) / len(full_docs)) if len(full_docs) > 0 else 0.0
                if pseudo_recall >= args.threshold: y = 0
                else: y = 1 # judged hit exists, but T1 differs too much from FULL -> fall through
            
            # Assign label 1 if relevant doc appears only after falling through to FULL
            elif full_rel_hits > 0: y = 1
            
            # Drop queries where even FULL retrieval misses all relevant docs
            else:
                dropped_no_full_hit += 1
                continue

            # Compute query-time features (vector form)
            x = compute_query_features_vec(qtext, lexicon, N)

            fout.write(json.dumps({"qid": qid, "y": y, "x": x}) + "\n")
            kept += 1

    print(f"[Data] Kept: {kept:,}")
    print(f"[Data] Dropped (no FULL hit): {dropped_no_full_hit:,}")
    print(f"[Data] Dropped (missing run): {dropped_missing_run:,}")
    print(f"[Data] Dropped (missing query text): {dropped_missing_query:,}")
    print(f"[Data] Wrote JSONL to {out_path}")
    print(f"[Data] Wrote feature names to {features_meta_path}")


if __name__ == "__main__":
    main()