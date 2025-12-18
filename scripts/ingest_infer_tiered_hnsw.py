"""
Ingest new documents for HNSW tiering:
- Infer tier with trained XGBoost (HNSW features)
- Append to delta TSVs (doc_id<TAB>text)
- Rebuild delta HNSW indexes
- Roll delta into base when thresholds exceeded, then rebuild base indexes

Features (must match training order):
1) static_score (avg topK query sim)
2) log1p_static_score
3) sim_max
4) sim_std
5) sim_p90
6) embedding_norm
7) log1p_embedding_norm
8) doc_len
9) log1p_doc_len
10) unique_term_count
11) tf_entropy
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import h5py
import numpy as np
import xgboost as xgb
from tqdm import tqdm

from utils.config import DELTA_T1_THRESHOLD, DELTA_T2_THRESHOLD
from search_system.shared.utils import tokenize
from systems.tiering import load_feature_names


def load_h5_embeddings(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    with h5py.File(path, "r") as f:
        ids = np.array(f["id"]).astype(str)
        emb = np.array(f["embedding"]).astype(np.float32)
    id_to_idx = {doc_id: i for i, doc_id in enumerate(ids)}
    return ids, emb, id_to_idx


def load_query_index(query_emb_path: Path) -> faiss.IndexFlatIP:
    with h5py.File(query_emb_path, "r") as f:
        q_emb = np.array(f["embedding"]).astype(np.float32)
    faiss.normalize_L2(q_emb)
    dim = q_emb.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(q_emb)
    return idx


def append_tsv(path: Path, rows: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(r)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def rebuild_hnsw(index_path: Path, ids: List[int], emb: np.ndarray, m: int, ef_construction: int) -> None:
    """
    Rebuild an HNSW index with explicit ids using IndexIDMap2 wrapper.
    """
    index_path.mkdir(parents=True, exist_ok=True)
    dim = emb.shape[1]
    base = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
    base.hnsw.efConstruction = ef_construction
    idx = faiss.IndexIDMap2(base)
    emb_copy = emb.copy()
    faiss.normalize_L2(emb_copy)
    ids_arr = np.array(ids, dtype=np.int64)
    # add in chunks with progress
    chunk = 50000
    for start in tqdm(range(0, len(ids_arr), chunk), desc=f"Building {index_path.name}", unit="vec", leave=False):
        end = min(len(ids_arr), start + chunk)
        idx.add_with_ids(emb_copy[start:end], ids_arr[start:end])
    faiss.write_index(idx, str(index_path / "index.faiss"))


def gather_embeddings(id_list: List[str], emb_table: np.ndarray, id_to_idx: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    vecs = []
    numeric_ids = []
    for doc_id in id_list:
        idx = id_to_idx.get(doc_id)
        if idx is None:
            continue
        vecs.append(emb_table[idx])
        numeric_ids.append(int(doc_id))
    if not vecs:
        return np.zeros((0, emb_table.shape[1]), dtype=np.float32), np.array([], dtype=np.int64)
    return np.stack(vecs, axis=0), np.array(numeric_ids, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description="Infer tiers for HNSW, manage base/delta indexes.")
    ap.add_argument("--input", required=True, help="TSV doc_id<TAB>text (work init or delta split)")
    ap.add_argument("--work-emb", default="data/collection/collection_work_hnsw.h5")
    ap.add_argument("--query-emb", default="data/collection/query_embeddings.h5")
    ap.add_argument("--model", default="artifacts/tiering_dense/model_hnsw.json")
    ap.add_argument("--threshold", default="artifacts/tiering_dense/threshold_hnsw.json")
    ap.add_argument("--feature-names", default="artifacts/tiering_dense/train_hnsw.pkl", help="Pickle containing feature_names")
    ap.add_argument("--topk", type=int, default=25, help="TopK queries for scoring")
    ap.add_argument("--batch-size", type=int, default=4096, help="Batch size for ingestion")
    ap.add_argument("--faiss-threads", type=int, default=8, help="FAISS OMP threads")
    ap.add_argument("--base-t1", default="artifacts/tiering_dense/base_T1_hnsw.tsv")
    ap.add_argument("--base-t2", default="artifacts/tiering_dense/base_T2_hnsw.tsv")
    ap.add_argument("--delta-t1", default="artifacts/tiering_dense/delta_T1_hnsw.tsv")
    ap.add_argument("--delta-t2", default="artifacts/tiering_dense/delta_T2_hnsw.tsv")
    ap.add_argument("--index-t1", default="artifacts/hnsw_T1")
    ap.add_argument("--index-t2", default="artifacts/hnsw_T2")
    ap.add_argument("--index-t1-delta", default="artifacts/hnsw_T1_delta")
    ap.add_argument("--index-t2-delta", default="artifacts/hnsw_T2_delta")
    ap.add_argument("--m", type=int, default=8, help="HNSW M (match systems/retrieval/dense/hnsw.py)")
    ap.add_argument("--ef-construction", type=int, default=200, help="HNSW efConstruction (match hnsw.py)")
    args = ap.parse_args()

    feature_names = load_feature_names(Path(args.feature_names))
    # Load embeddings
    ids, emb_table, id_to_idx = load_h5_embeddings(Path(args.work_emb))
    faiss.omp_set_num_threads(args.faiss_threads)

    # Load query index
    q_index = load_query_index(Path(args.query_emb))

    # Load model and threshold once
    model = xgb.Booster()
    model.load_model(args.model)
    threshold = json.loads(Path(args.threshold).read_text())["threshold"]
    routed_rows_t1: List[str] = []
    routed_rows_t2: List[str] = []

    def predict_batch(X: np.ndarray) -> np.ndarray:
        dm = xgb.DMatrix(X, feature_names=feature_names if feature_names else None)
        # best_iteration may be None if not set; fall back to full model
        it_end = getattr(model, "best_iteration", None)
        if it_end is not None:
            probs = model.predict(dm, iteration_range=(0, it_end + 1))
        else:
            probs = model.predict(dm)
        return probs

    # Pre-count lines for progress
    total_docs = sum(1 for _ in open(args.input, "r", encoding="utf-8") if _.strip())

    # Process docs in batches with progress
    with open(args.input, "r", encoding="utf-8") as f, tqdm(total=total_docs, desc="Docs", unit="doc") as pbar:
        batch_lines = []
        for line in f:
            if not line.strip():
                continue
            batch_lines.append(line.rstrip("\n"))
            if len(batch_lines) >= args.batch_size:
                pbar.update(len(batch_lines))
                doc_ids = []
                texts = []
                for ln in batch_lines:
                    did, txt = ln.split("\t", 1)
                    doc_ids.append(did)
                    texts.append(txt)
                batch_lines = []

                doc_lens = []
                uniq_counts = []
                entropies = []
                valid_mask = []
                for did, txt in zip(doc_ids, texts):
                    tokens = tokenize(txt)
                    dl = len(tokens)
                    if dl == 0:
                        uniq_counts.append(0)
                        entropies.append(0.0)
                        doc_lens.append(0)
                    else:
                        freq: Dict[str, int] = {}
                        for t in tokens:
                            freq[t] = freq.get(t, 0) + 1
                        uniq_counts.append(len(freq))
                        probs = np.array(list(freq.values()), dtype=np.float32) / float(dl)
                        entropies.append(float(-np.sum(probs * np.log(probs + 1e-12))))
                        doc_lens.append(dl)
                    valid_mask.append(did in id_to_idx)

                vecs = []
                kept_ids = []
                kept_texts = []
                kept_lens = []
                kept_uniq = []
                kept_ent = []
                for did, txt, dl, uq, ent, ok in zip(doc_ids, texts, doc_lens, uniq_counts, entropies, valid_mask):
                    if not ok:
                        continue
                    idx = id_to_idx[did]
                    vecs.append(emb_table[idx])
                    kept_ids.append(did)
                    kept_texts.append(txt)
                    kept_lens.append(dl)
                    kept_uniq.append(uq)
                    kept_ent.append(ent)

                if vecs:
                    emb_batch = np.stack(vecs, axis=0)
                    norms = np.linalg.norm(emb_batch, axis=1)
                    emb_normed = emb_batch.copy()
                    faiss.normalize_L2(emb_normed)
                    scores, _ = q_index.search(emb_normed, args.topk)
                    static = scores.mean(axis=1)
                    sim_max = scores.max(axis=1)
                    sim_std = scores.std(axis=1)
                    sim_p90 = np.percentile(scores, 90, axis=1)

                    feats = np.column_stack(
                        [
                            static,
                            np.log1p(static),
                            sim_max,
                            sim_std,
                            sim_p90,
                            norms,
                            np.log1p(norms),
                            np.array(kept_lens, dtype=np.float32),
                            np.log1p(np.array(kept_lens, dtype=np.float32)),
                            np.array(kept_uniq, dtype=np.float32),
                            np.array(kept_ent, dtype=np.float32),
                        ]
                    )
                    probs = predict_batch(feats)
                    for did, txt, prob in zip(kept_ids, kept_texts, probs):
                        row = f"{did}\t{txt}\n"
                        if prob >= threshold:
                            routed_rows_t1.append(row)
                        else:
                            routed_rows_t2.append(row)

        # process remainder
        if batch_lines:
            pbar.update(len(batch_lines))
            doc_ids = []
            texts = []
            for ln in batch_lines:
                did, txt = ln.split("\t", 1)
                doc_ids.append(did)
                texts.append(txt)
            batch_lines = []

            doc_lens = []
            uniq_counts = []
            entropies = []
            valid_mask = []
            for did, txt in zip(doc_ids, texts):
                tokens = tokenize(txt)
                dl = len(tokens)
                if dl == 0:
                    uniq_counts.append(0)
                    entropies.append(0.0)
                    doc_lens.append(0)
                else:
                    freq: Dict[str, int] = {}
                    for t in tokens:
                        freq[t] = freq.get(t, 0) + 1
                    uniq_counts.append(len(freq))
                    probs = np.array(list(freq.values()), dtype=np.float32) / float(dl)
                    entropies.append(float(-np.sum(probs * np.log(probs + 1e-12))))
                    doc_lens.append(dl)
                valid_mask.append(did in id_to_idx)

            vecs = []
            kept_ids = []
            kept_texts = []
            kept_lens = []
            kept_uniq = []
            kept_ent = []
            for did, txt, dl, uq, ent, ok in zip(doc_ids, texts, doc_lens, uniq_counts, entropies, valid_mask):
                if not ok:
                    continue
                idx = id_to_idx[did]
                vecs.append(emb_table[idx])
                kept_ids.append(did)
                kept_texts.append(txt)
                kept_lens.append(dl)
                kept_uniq.append(uq)
                kept_ent.append(ent)
            if vecs:
                emb_batch = np.stack(vecs, axis=0)
                norms = np.linalg.norm(emb_batch, axis=1)
                emb_normed = emb_batch.copy()
                faiss.normalize_L2(emb_normed)
                scores, _ = q_index.search(emb_normed, args.topk)
                static = scores.mean(axis=1)
                sim_max = scores.max(axis=1)
                sim_std = scores.std(axis=1)
                sim_p90 = np.percentile(scores, 90, axis=1)

                feats = np.column_stack(
                    [
                        static,
                        np.log1p(static),
                        sim_max,
                        sim_std,
                        sim_p90,
                        norms,
                        np.log1p(norms),
                        np.array(kept_lens, dtype=np.float32),
                        np.log1p(np.array(kept_lens, dtype=np.float32)),
                        np.array(kept_uniq, dtype=np.float32),
                        np.array(kept_ent, dtype=np.float32),
                    ]
                )
                probs = predict_batch(feats)
                for did, txt, prob in zip(kept_ids, kept_texts, probs):
                    row = f"{did}\t{txt}\n"
                    if prob >= threshold:
                        routed_rows_t1.append(row)
                    else:
                        routed_rows_t2.append(row)

    # Append to delta TSVs
    append_tsv(Path(args.delta_t1), routed_rows_t1)
    append_tsv(Path(args.delta_t2), routed_rows_t2)

    # Check thresholds
    t1_count = count_lines(Path(args.delta_t1))
    t2_count = count_lines(Path(args.delta_t2))
    print(f"[Ingest HNSW] Delta sizes: T1={t1_count}, T2={t2_count}")

    def rebuild_base(base_tsv: Path, delta_tsv: Path, index_dir: Path):
        # Merge base + delta TSVs
        base_lines = []
        if base_tsv.exists():
            with base_tsv.open("r", encoding="utf-8") as f:
                base_lines = f.readlines()
        with delta_tsv.open("r", encoding="utf-8") as f:
            delta_lines = f.readlines()
        combined = base_lines + delta_lines
        base_tsv.write_text("".join(combined))
        delta_tsv.unlink(missing_ok=True)
        # Build index from combined IDs
        ids_list = [ln.split("\t", 1)[0] for ln in combined if ln.strip()]
        vecs, num_ids = gather_embeddings(ids_list, emb_table, id_to_idx)
        if len(num_ids) > 0:
            rebuild_hnsw(index_dir, num_ids.tolist(), vecs, args.m, args.ef_construction)

    def rebuild_delta(delta_tsv: Path, index_dir: Path):
        if not delta_tsv.exists():
            return
        with delta_tsv.open("r", encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
        ids_list = [ln.split("\t", 1)[0] for ln in lines]
        vecs, num_ids = gather_embeddings(ids_list, emb_table, id_to_idx)
        if len(num_ids) == 0:
            return
        rebuild_hnsw(index_dir, num_ids.tolist(), vecs, args.m, args.ef_construction)

    # Roll or rebuild deltas
    if t1_count > DELTA_T1_THRESHOLD:
        print("[Ingest HNSW] T1 delta exceeds threshold; rolling into base.")
        rebuild_base(Path(args.base_t1), Path(args.delta_t1), Path(args.index_t1))
    else:
        print("[Ingest HNSW] Rebuilding T1 delta index...")
        rebuild_delta(Path(args.delta_t1), Path(args.index_t1_delta))

    if t2_count > DELTA_T2_THRESHOLD:
        print("[Ingest HNSW] T2 delta exceeds threshold; rolling into base.")
        rebuild_base(Path(args.base_t2), Path(args.delta_t2), Path(args.index_t2))
    else:
        print("[Ingest HNSW] Rebuilding T2 delta index...")
        rebuild_delta(Path(args.delta_t2), Path(args.index_t2_delta))


if __name__ == "__main__":
    main()
