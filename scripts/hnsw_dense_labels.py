"""
Compute dense static scores for HNSW tiering using top-K query similarities,
assign Tier-1/Tier-2 labels, and write tier-specific embedding files.

Steps:
- Load doc embeddings and query embeddings (HDF5 with id / embedding datasets)
- L2 normalize (to mirror HNSW retrieval flow)
- Build a Faiss inner-product index over query embeddings
- For each doc embedding, search topK queries; aggregate scores (avg of topK)
- Rank/label docs by score (Tier-1 = top tier_ratio), Tier-2 = rest
- Save labels, static scores, tier1/tier2 id lists, and tier-specific embedding files
"""

import argparse
import json
from pathlib import Path

import faiss
import h5py
import numpy as np
from tqdm import tqdm

from utils.config import (
    HNSW_TRAIN_EMB_PATH,
    HNSW_QUERY_EMB_PATH,
    HNSW_STATIC_SCORES_PATH,
    HNSW_LABELS_PATH,
    HNSW_TIER_RATIO,
    HNSW_TOPK_QUERIES,
    HNSW_TIERING_DIR,
    HNSW_T1_EMB_PATH,
    HNSW_T2_EMB_PATH,
)


def load_embeddings(path: Path):
    with h5py.File(path, "r") as f:
        ids = np.array(f["id"]).astype(str)
        emb = np.array(f["embedding"]).astype(np.float32)
    return ids, emb


def save_embeddings(path: Path, ids: np.ndarray, emb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("id", data=ids.astype("S"), compression="gzip")
        f.create_dataset("embedding", data=emb.astype(np.float32), compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute dense scores for HNSW tiering and write tier splits.")
    parser.add_argument("--doc-emb", default=HNSW_TRAIN_EMB_PATH)
    parser.add_argument("--query-emb", default=HNSW_QUERY_EMB_PATH)
    parser.add_argument("--topk", type=int, default=HNSW_TOPK_QUERIES)
    parser.add_argument("--tier-ratio", type=float, default=HNSW_TIER_RATIO)
    parser.add_argument("--scores-out", default=HNSW_STATIC_SCORES_PATH)
    parser.add_argument("--labels-out", default=HNSW_LABELS_PATH)
    parser.add_argument("--tier-dir", default=HNSW_TIERING_DIR)
    parser.add_argument("--t1-emb-out", default=HNSW_T1_EMB_PATH)
    parser.add_argument("--t2-emb-out", default=HNSW_T2_EMB_PATH)
    parser.add_argument("--tier1-ids-out", default=None, help="Optional path for Tier-1 IDs list.")
    parser.add_argument("--tier2-ids-out", default=None, help="Optional path for Tier-2 IDs list.")
    parser.add_argument("--batch-size", type=int, default=50000, help="Batch size for scoring docs.")
    args = parser.parse_args()

    # Load embeddings
    print("[HNSW tiering] Loading embeddings...")
    doc_ids, doc_emb = load_embeddings(Path(args.doc_emb))
    query_ids, query_emb = load_embeddings(Path(args.query_emb))

    # Normalize to align with HNSW flow
    print("[HNSW tiering] Normalizing embeddings...")
    faiss.normalize_L2(doc_emb)
    faiss.normalize_L2(query_emb)

    # Build query index
    print("[HNSW tiering] Building query index...")
    dim = query_emb.shape[1]
    q_index = faiss.IndexFlatIP(dim)
    q_index.add(query_emb)

    # Search topK for all docs (batched for progress visibility)
    print("[HNSW tiering] Scoring docs against query index...")
    num_docs = doc_emb.shape[0]
    agg_scores = np.zeros(num_docs, dtype=np.float32)
    sim_max = np.zeros(num_docs, dtype=np.float32)
    sim_std = np.zeros(num_docs, dtype=np.float32)
    sim_p90 = np.zeros(num_docs, dtype=np.float32)
    for start in tqdm(range(0, num_docs, args.batch_size), desc="Scoring", unit="batch"):
        end = min(num_docs, start + args.batch_size)
        scores, _ = q_index.search(doc_emb[start:end], args.topk)
        agg_scores[start:end] = scores.mean(axis=1)  # avg topK
        sim_max[start:end] = scores.max(axis=1)
        sim_std[start:end] = scores.std(axis=1)
        sim_p90[start:end] = np.percentile(scores, 90, axis=1)

    # Save raw scores + aggregates
    print("[HNSW tiering] Saving raw scores + aggregates...")
    Path(args.scores_out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.scores_out, score=agg_scores, sim_max=sim_max, sim_std=sim_std, sim_p90=sim_p90)

    # Label by tier ratio
    
    order = np.argsort(-agg_scores)
    cutoff = max(1, int(len(doc_ids) * args.tier_ratio))
    tier1_indices = order[:cutoff]
    tier2_indices = order[cutoff:]

    labels = {doc_ids[i]: 1 for i in tier1_indices}
    labels.update({doc_ids[i]: 0 for i in tier2_indices})

    Path(args.labels_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.labels_out, "w", encoding="utf-8") as f:
        json.dump(labels, f)

    # Write id lists
    print("[HNSW tiering] Writing T1/T2 id lists...")
    tier_dir = Path(args.tier_dir)
    tier_dir.mkdir(parents=True, exist_ok=True)
    t1_ids_path = Path(args.tier1_ids_out) if args.tier1_ids_out else tier_dir / "tier1_ids_hnsw.txt"
    t2_ids_path = Path(args.tier2_ids_out) if args.tier2_ids_out else tier_dir / "tier2_ids_hnsw.txt"
    with open(t1_ids_path, "w", encoding="utf-8") as f:
        for i in tier1_indices:
            f.write(f"{doc_ids[i]}\n")
    with open(t2_ids_path, "w", encoding="utf-8") as f:
        for i in tier2_indices:
            f.write(f"{doc_ids[i]}\n")

    # Write tier-specific embeddings
    print("[HNSW tiering] Writing T1/T2 embeddings...")
    save_embeddings(Path(args.t1_emb_out), doc_ids[tier1_indices], doc_emb[tier1_indices])
    save_embeddings(Path(args.t2_emb_out), doc_ids[tier2_indices], doc_emb[tier2_indices])

    print(f"[HNSW tiering] Labeled {len(doc_ids)} docs; Tier-1 cutoff {cutoff} docs.")


if __name__ == "__main__":
    main()
