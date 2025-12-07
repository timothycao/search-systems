"""
Tiering utilities CLI.

Usage examples:
  python -m scripts.tiering compute-qtf --queries data/queries/queries.all.tsv
  python -m scripts.tiering static-scores --qtf artifacts/tiering/qtf.json
  python -m scripts.tiering assign-tiers --scores artifacts/tiering/static_scores.pkl
"""

import argparse
import json
import os

from systems.tiering.tiering import (
    assign_tiers,
    assemble_dataset,
    compute_qtf,
    compute_static_scores,
    compute_doc_features,
    stratified_split,
    save_features,
    load_qtf,
    load_scores,
    normalize_scores,
    save_qtf,
    save_scores,
    save_dataset,
    load_features,
)
from utils.config import ARTIFACTS_DIR, QUERIES_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic tiering utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # compute-qtf
    qtf_parser = subparsers.add_parser("compute-qtf", help="Compute query term frequencies.")
    qtf_parser.add_argument(
        "--queries",
        default=os.path.join(QUERIES_DIR, "queries.all.tsv"),
        help="TSV file with <query_id> <query_text> columns.",
    )
    qtf_parser.add_argument(
        "--output",
        default=os.path.join(ARTIFACTS_DIR, "tiering", "qtf.json"),
        help="Output path for QTF JSON.",
    )
    qtf_parser.add_argument(
        "--drop-singletons",
        action="store_true",
        help="Drop terms that appear only once in the query log.",
    )

    # static-scores
    static_parser = subparsers.add_parser(
        "static-scores", help="Compute static BM25-based document scores."
    )
    static_parser.add_argument(
        "--index",
        default=os.path.join(ARTIFACTS_DIR, "bm25", "index"),
        help="Path to BM25 index directory (lexicon/page_table/inverted_index.bin).",
    )
    static_parser.add_argument(
        "--qtf",
        default=os.path.join(ARTIFACTS_DIR, "tiering", "qtf.json"),
        help="Path to QTF JSON produced by compute-qtf.",
    )
    static_parser.add_argument(
        "--output",
        default=os.path.join(ARTIFACTS_DIR, "tiering", "static_scores.pkl"),
        help="Output path for pickled static scores dict.",
    )

    # assign-tiers
    tiers_parser = subparsers.add_parser("assign-tiers", help="Assign Tier-1/Tier-2 labels.")
    tiers_parser.add_argument(
        "--scores",
        default=os.path.join(ARTIFACTS_DIR, "tiering", "static_scores.pkl"),
        help="Pickle file of doc_id->static_score.",
    )
    tiers_parser.add_argument(
        "--tier-ratio",
        type=float,
        default=0.3,
        help="Fraction of documents to assign to Tier-1.",
    )
    tiers_parser.add_argument(
        "--labels-output",
        default=os.path.join(ARTIFACTS_DIR, "tiering", "labels.json"),
        help="Output path for doc_id->label JSON.",
    )
    tiers_parser.add_argument(
        "--normalized-output",
        default=None,
        help="Optional output path for normalized scores JSON.",
    )
    tiers_parser.add_argument(
        "--tier1-ids",
        default=None,
        help="Optional output path to write Tier-1 doc_ids (one per line).",
    )
    tiers_parser.add_argument(
        "--tier2-ids",
        default=None,
        help="Optional output path to write Tier-2 doc_ids (one per line).",
    )

    # features
    features_parser = subparsers.add_parser(
        "features", help="Extract document-side features for tiering."
    )
    features_parser.add_argument(
        "--index",
        default=os.path.join(ARTIFACTS_DIR, "bm25", "index"),
        help="Path to BM25 index directory.",
    )
    features_parser.add_argument(
        "--scores",
        default=os.path.join(ARTIFACTS_DIR, "tiering", "static_scores.pkl"),
        help="Pickled doc_id->static_score file.",
    )
    features_parser.add_argument(
        "--output",
        default=os.path.join(ARTIFACTS_DIR, "tiering", "features.pkl"),
        help="Output path for pickled feature dict.",
    )

    # dataset
    dataset_parser = subparsers.add_parser(
        "dataset", help="Assemble dataset and stratified train/val splits."
    )
    dataset_parser.add_argument(
        "--features",
        default=os.path.join(ARTIFACTS_DIR, "tiering", "features.pkl"),
        help="Pickled doc_id->feature dict.",
    )
    dataset_parser.add_argument(
        "--labels",
        default=os.path.join(ARTIFACTS_DIR, "tiering", "labels.json"),
        help="JSON file of doc_id->label.",
    )
    dataset_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    dataset_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling.",
    )
    dataset_parser.add_argument(
        "--train-output",
        default=os.path.join(ARTIFACTS_DIR, "tiering", "train.pkl"),
        help="Output path for train split pickle.",
    )
    dataset_parser.add_argument(
        "--val-output",
        default=os.path.join(ARTIFACTS_DIR, "tiering", "val.pkl"),
        help="Output path for validation split pickle.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "compute-qtf":
        qtf = compute_qtf(args.queries, drop_singletons=args.drop_singletons)
        save_qtf(qtf, args.output)
        print(f"Saved QTF with {len(qtf)} terms to {args.output}")
    elif args.command == "static-scores":
        qtf = load_qtf(args.qtf)
        scores = compute_static_scores(args.index, qtf)
        save_scores(scores, args.output)
        print(f"Saved static scores for {len(scores)} docs to {args.output}")
    elif args.command == "assign-tiers":
        scores = load_scores(args.scores)
        labels = assign_tiers(scores, tier_ratio=args.tier_ratio)
        labels_dir = os.path.dirname(args.labels_output)
        if labels_dir:
            os.makedirs(labels_dir, exist_ok=True)
        with open(args.labels_output, "w", encoding="utf-8") as f:
            json.dump(labels, f)
        print(f"Saved tier labels for {len(labels)} docs to {args.labels_output}")

        # Optional: write tier1/tier2 ID lists
        if args.tier1_ids:
            os.makedirs(os.path.dirname(args.tier1_ids) or ".", exist_ok=True)
            with open(args.tier1_ids, "w", encoding="utf-8") as f:
                for doc_id, lbl in labels.items():
                    if int(lbl) == 1:
                        f.write(f"{doc_id}\n")
            print(f"Wrote Tier-1 IDs to {args.tier1_ids}")
        if args.tier2_ids:
            os.makedirs(os.path.dirname(args.tier2_ids) or ".", exist_ok=True)
            with open(args.tier2_ids, "w", encoding="utf-8") as f:
                for doc_id, lbl in labels.items():
                    if int(lbl) == 0:
                        f.write(f"{doc_id}\n")
            print(f"Wrote Tier-2 IDs to {args.tier2_ids}")

        if args.normalized_output:
            normalized = normalize_scores(scores)
            norm_dir = os.path.dirname(args.normalized_output)
            if norm_dir:
                os.makedirs(norm_dir, exist_ok=True)
            with open(args.normalized_output, "w", encoding="utf-8") as f:
                json.dump(normalized, f)
            print(f"Saved normalized scores to {args.normalized_output}")
    elif args.command == "features":
        scores = load_scores(args.scores)
        features = compute_doc_features(args.index, scores)
        save_features(features, args.output)
        print(f"Saved features for {len(features)} docs to {args.output}")
    elif args.command == "dataset":
        features = load_features(args.features)
        with open(args.labels, "r", encoding="utf-8") as f:
            labels = json.load(f)
        dataset = assemble_dataset(features, labels)
        train_split, val_split = stratified_split(
            dataset, val_ratio=args.val_ratio, seed=args.seed
        )
        save_dataset(train_split, args.train_output)
        save_dataset(val_split, args.val_output)
        print(
            f"Saved train ({len(train_split['y'])}) to {args.train_output} and "
            f"val ({len(val_split['y'])}) to {args.val_output}"
        )
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
