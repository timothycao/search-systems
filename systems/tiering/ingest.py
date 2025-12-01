"""
Utilities for routing new documents into tiered BM25 delta buffers and triggering rebuilds.

Assumptions:
- Tier labels for existing corpus already exist (artifacts/tiering/labels.json).
- Base tiered indexes live at artifacts/bm25_T1 and artifacts/bm25_T2.
- Delta buffers are small TSV files with new docs per tier (doc_id<TAB>text).
- Rebuilds are triggered when delta sizes exceed thresholds.
"""

import json
import os
from pathlib import Path
from typing import Iterable, Tuple, Dict, Set

from search_system.parser import run_parser
from search_system.indexer import run_indexer
import shutil

from utils.config import (
    DELTA_T1_THRESHOLD,
    DELTA_T2_THRESHOLD,
)


def load_subset_ids(path: Path) -> Set[int]:
    ids: Set[int] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.add(int(line.strip()))
    return ids


def append_delta(delta_path: Path, doc_id: int, text: str) -> None:
    delta_path.parent.mkdir(parents=True, exist_ok=True)
    with delta_path.open("a", encoding="utf-8") as f:
        f.write(f"{doc_id}\t{text}\n")


def delta_count(delta_path: Path) -> int:
    if not delta_path.exists():
        return 0
    with delta_path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def rebuild_tier(
    tier_name: str,
    collection_path: Path,
    subset_ids_path: Path,
    delta_path: Path,
    out_root: Path,
) -> None:
    """
    Rebuild a tier index by materializing the tier's original docs plus delta docs into a temp dataset,
    then running parser and indexer.
    """
    if not delta_path.exists():
        print(f"[Rebuild:{tier_name}] No delta found at {delta_path}")
        return

    tier_out = out_root / tier_name
    postings_dir = tier_out / "postings"
    index_dir = tier_out / "index"
    postings_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    subset_ids = load_subset_ids(subset_ids_path)
    temp_dataset = tier_out / "rebuild_dataset.tsv"

    # Materialize original tier docs
    with collection_path.open("r", encoding="utf-8") as coll_f, temp_dataset.open(
        "w", encoding="utf-8"
    ) as tmp_f:
        for line in coll_f:
            if not line.strip():
                continue
            doc_id_str, text = line.rstrip("\n").split("\t", 1)
            doc_id = int(doc_id_str)
            if doc_id in subset_ids:
                tmp_f.write(f"{doc_id}\t{text}\n")
        # Append delta docs
        with delta_path.open("r", encoding="utf-8") as delta_f:
            for line in delta_f:
                if line.strip():
                    tmp_f.write(line)

    print(f"[Rebuild:{tier_name}] Temp dataset written to {temp_dataset}")

    # Run parser/indexer on the materialized dataset
    run_parser(dataset_path=str(temp_dataset), output_dir=str(postings_dir))
    run_indexer(input_dir=str(postings_dir), output_dir=str(index_dir))

    # Clear delta TSV and any delta index dirs
    delta_path.unlink(missing_ok=True)
    delta_index_dir = out_root / f"{tier_name}_delta"
    shutil.rmtree(delta_index_dir, ignore_errors=True)

    # Remove temp dataset
    temp_dataset.unlink(missing_ok=True)
    print(f"[Rebuild:{tier_name}] Rebuild complete; delta cleared and temp removed")


def rebuild_delta_index(tier_name: str, delta_path: Path, out_root: Path) -> None:
    """
    Build a small delta index for a tier (base remains untouched).
    """
    if not delta_path.exists():
        return
    # empty delta -> skip
    if delta_count(delta_path) == 0:
        return
    tier_out = out_root / f"{tier_name}_delta"
    postings_dir = tier_out / "postings"
    index_dir = tier_out / "index"
    # Clean out old delta dirs to avoid stale files
    shutil.rmtree(postings_dir, ignore_errors=True)
    shutil.rmtree(index_dir, ignore_errors=True)
    postings_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    run_parser(dataset_path=str(delta_path), output_dir=str(postings_dir))
    run_indexer(input_dir=str(postings_dir), output_dir=str(index_dir))
    print(f"[Delta build:{tier_name}] Built delta index at {index_dir}")


def route_and_maybe_rebuild(
    docs: Iterable[Tuple[int, str, int]],
    collection_path: Path,
    subset_t1: Path,
    subset_t2: Path,
    out_root: Path,
    delta_dir: Path,
) -> None:
    """
    Route docs into tiered deltas and trigger rebuilds if thresholds exceeded.
    docs: iterable of (doc_id, text, tier_label)
    """
    delta_t1 = delta_dir / "delta_t1.tsv"
    delta_t2 = delta_dir / "delta_t2.tsv"

    for doc_id, text, tier in docs:
        if tier == 1:
            append_delta(delta_t1, doc_id, text)
        else:
            append_delta(delta_t2, doc_id, text)

    t1_count = delta_count(delta_t1)
    t2_count = delta_count(delta_t2)
    print(f"[Ingest] Delta sizes: T1={t1_count}, T2={t2_count}")

    t1_rebuilt = False
    t2_rebuilt = False
    if t1_count > T1_DELTA_THRESHOLD:
        rebuild_tier("bm25_T1", collection_path, subset_t1, delta_t1, out_root)
        t1_rebuilt = True
    if t2_count > T2_DELTA_THRESHOLD:
        rebuild_tier("bm25_T2", collection_path, subset_t2, delta_t2, out_root)
        t2_rebuilt = True

    # Rebuild delta indexes to reflect new docs (unless they were just consumed)
    if not t1_rebuilt:
        rebuild_delta_index("bm25_T1", delta_t1, out_root)
    if not t2_rebuilt:
        rebuild_delta_index("bm25_T2", delta_t2, out_root)
