"""
Ingest new documents (doc_id<TAB>text), infer tier using the trained model, append to tiered deltas,
rebuild delta indexes, and trigger base rebuilds when thresholds are exceeded.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from systems.tiering.infer import (
    compute_features,
    build_feature_vector,
    load_qtf,
    load_feature_names,
    predict_tier,
)
from systems.tiering.ingest import route_and_maybe_rebuild
from search_system.query.query_startup_context import QueryStartupContext
from utils.config import (
    ARTIFACTS_DIR,
    TIERING_QTF_PATH,
    TIERING_MODEL_PATH,
    TIERING_THRESHOLD_PATH,
    TIERING_FEATURE_NAMES_PATH,
    DELTA_DIR,
)


def load_input(path: Path) -> List[Tuple[int, str]]:
    docs: List[Tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            doc_id, text = parts
            docs.append((int(doc_id), text))
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer tiers for new docs and route to deltas.")
    parser.add_argument("--input", required=True, help="TSV: doc_id<TAB>text")
    parser.add_argument("--model", default=TIERING_MODEL_PATH)
    parser.add_argument("--threshold", default=TIERING_THRESHOLD_PATH)
    parser.add_argument("--qtf", default=TIERING_QTF_PATH)
    parser.add_argument("--feature-names", default=TIERING_FEATURE_NAMES_PATH, help="Pickle containing feature_names")
    parser.add_argument("--index", default=f"{ARTIFACTS_DIR}/bm25/index", help="Path to BM25 index (for stats/lexicon)")
    parser.add_argument("--out-root", default=ARTIFACTS_DIR)
    parser.add_argument("--base-t1", default=f"{DELTA_DIR}/base_T1.tsv", help="Doc list for base Tier-1 (doc_id\\ttext)")
    parser.add_argument("--base-t2", default=f"{DELTA_DIR}/base_T2.tsv", help="Doc list for base Tier-2 (doc_id\\ttext)")
    parser.add_argument("--delta-t1", default=f"{DELTA_DIR}/delta_T1.tsv", help="Doc list for delta Tier-1 (doc_id\\ttext)")
    parser.add_argument("--delta-t2", default=f"{DELTA_DIR}/delta_T2.tsv", help="Doc list for delta Tier-2 (doc_id\\ttext)")
    args = parser.parse_args()

    docs = load_input(Path(args.input))

    qtf = load_qtf(Path(args.qtf))
    feature_names = load_feature_names(Path(args.feature_names))
    ctx = QueryStartupContext(args.index)

    routed = []
    for doc_id, text in tqdm(docs, desc="Inferring tiers", unit="doc"):
        feats = compute_features(text, qtf, ctx)
        fv = build_feature_vector(feats, feature_names)
        tier, prob = predict_tier(Path(args.model), Path(args.threshold), fv, feature_names)
        routed.append((doc_id, text, tier))

    route_and_maybe_rebuild(
        docs=routed,
        base_t1=Path(args.base_t1),
        base_t2=Path(args.base_t2),
        delta_t1=Path(args.delta_t1),
        delta_t2=Path(args.delta_t2),
        out_root=Path(args.out_root),
    )


if __name__ == "__main__":
    main()
