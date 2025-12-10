"""
Utilities for routing new documents into tiered BM25 buffers and triggering rebuilds.

Flow:
- Append inferred Tier-1/Tier-2 docs into delta TSVs (doc_id<TAB>text).
- If a delta exceeds its threshold, merge delta into the base TSV, clear delta TSV and delta index, and rebuild the base tier index from the updated base TSV.
- If below threshold, rebuild the delta index so new docs are queryable.
"""

import json
import os
from pathlib import Path
from typing import Iterable, Tuple

from search_system.parser import run_parser
from search_system.indexer import run_indexer
import shutil

from utils.config import DELTA_T1_THRESHOLD, DELTA_T2_THRESHOLD


def append_delta(delta_path: Path, doc_id: int, text: str) -> None:
    delta_path.parent.mkdir(parents=True, exist_ok=True)
    with delta_path.open("a", encoding="utf-8") as f:
        f.write(f"{doc_id}\t{text}\n")


def delta_count(delta_path: Path) -> int:
    if not delta_path.exists():
        return 0
    with delta_path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def rebuild_base_from_tsv(tier_name: str, base_tsv: Path, out_root: Path) -> None:
    """
    Rebuild a base tier index from its TSV (doc_id<TAB>text).
    Clears any existing postings/index dirs for that tier.
    """
    if not base_tsv.exists():
        print(f"[Rebuild:{tier_name}] Base TSV not found at {base_tsv}, skipping.")
        return

    tier_out = out_root / tier_name
    postings_dir = tier_out / "postings"
    index_dir = tier_out / "index"
    shutil.rmtree(postings_dir, ignore_errors=True)
    shutil.rmtree(index_dir, ignore_errors=True)
    postings_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    run_parser(dataset_path=str(base_tsv), output_dir=str(postings_dir))
    run_indexer(input_dir=str(postings_dir), output_dir=str(index_dir))
    print(f"[Rebuild:{tier_name}] Rebuilt base index from {base_tsv}")


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
    base_t1: Path,
    base_t2: Path,
    delta_t1: Path,
    delta_t2: Path,
    out_root: Path,
) -> None:
    """
    Route docs into tiered deltas and trigger rebuilds if thresholds exceeded.
    docs: iterable of (doc_id, text, tier_label)
    """
    for doc_id, text, tier in docs:
        if tier == 1:
            append_delta(delta_t1, doc_id, text)
        else:
            append_delta(delta_t2, doc_id, text)

    t1_count = delta_count(delta_t1)
    t2_count = delta_count(delta_t2)
    print(f"[Ingest] Delta sizes: T1={t1_count}, T2={t2_count}")

    # Thresholds: roll into base if exceeded; otherwise build delta indexes.
    if t1_count > DELTA_T1_THRESHOLD:
        # merge delta into base, clear delta, rebuild base index
        base_t1.parent.mkdir(parents=True, exist_ok=True)
        if delta_t1.exists():
            with base_t1.open("a", encoding="utf-8") as fout, delta_t1.open("r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
            delta_t1.unlink(missing_ok=True)
        shutil.rmtree(out_root / "bm25_T1_delta", ignore_errors=True)
        rebuild_base_from_tsv("bm25_T1", base_t1, out_root)
    else:
        rebuild_delta_index("bm25_T1", delta_t1, out_root)

    if t2_count > DELTA_T2_THRESHOLD:
        base_t2.parent.mkdir(parents=True, exist_ok=True)
        if delta_t2.exists():
            with base_t2.open("a", encoding="utf-8") as fout, delta_t2.open("r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
            delta_t2.unlink(missing_ok=True)
        shutil.rmtree(out_root / "bm25_T2_delta", ignore_errors=True)
        rebuild_base_from_tsv("bm25_T2", base_t2, out_root)
    else:
        rebuild_delta_index("bm25_T2", delta_t2, out_root)

    t1_size = base_t1.stat().st_size if base_t1.exists() else 0
    t2_size = base_t2.stat().st_size if base_t2.exists() else 0
    print(f"[Ingest] Base file sizes: T1={t1_size} bytes, T2={t2_size} bytes")
