# Query-Tier Inference – Strategies and Labeling Ideas

This note captures strategies for training a classifier to decide, per query, whether to search only Tier-1 or both Tier-1 & Tier-2 (e.g., `T1` vs `T1&T2`).

## Labeling Strategies

### 1) Qrels + Doc Tier Membership (simple, deterministic)
- For each query with qrels, check where its relevant docs reside given the current tier labels.
- If **all** relevant doc_ids are in Tier-1 ⇒ label `T1`.
- If **any** relevant doc_id is in Tier-2 (or its delta) ⇒ label `T1&T2`.
- Pros: Simple, no retrieval run needed. Cons: Depends entirely on current tier assignments; doesn’t account for retrieval cutoffs.

### 2) Performance-based (recall-oriented)
- Run retrieval on Tier-1 only (BM25_T1 and/or HNSW_T1), compute recall@K against qrels.
- Run retrieval on Tier-1+Tier-2, compute recall@K.
- If Tier-1 meets a target (e.g., finds at least one relevant doc, or achieves recall@K ≥ threshold), label `T1`. If adding Tier-2 improves recall or is needed to meet the target, label `T1&T2`.
- Pros: Reflects actual retrieval behavior. Cons: Requires retrieval runs to generate labels.

### 3) Hybrid rule
- Base label on qrels+tiers (strategy 1) as a floor: any relevant in Tier-2 ⇒ `T1&T2`.
- Optionally override to `T1&T2` if a Tier-1-only retrieval run misses all qrels (or fails recall target) but Tier-1+2 would succeed.
- Pros: Uses static knowledge (tiers) plus retrieval evidence.

## Candidate Query Features
- Token-based:
  - Query length, log1p(length)
  - Mean/max IDF of query terms (from BM25 lexicon)
  - Entropy of term distribution
  - Unique term count
- Embedding-based:
  - Query embedding (dot model) summary stats vs Tier-1/Tier-2 centroids (optional)
  - Raw query embedding as input to a small model (if using a neural classifier)
- Retrieval hints (if available):
  - BM25_T1 max score / sum score estimates (requires a quick probe or precomputed stats)
  - HNSW_T1 similarity proxies (if a fast head index is available)

## Model/Formulation
- Binary classifier: `T1` (0) vs `T1&T2` (1).
- Training data: queries with labels from strategies above.
- Evaluation: precision/recall on `T1&T2` (avoid missing queries that need Tier-2), overall accuracy/AUC.

## Notes
- Keep labels versioned: tied to the specific tier assignments and qrels snapshot.
- Feature computation should match the tokenizer/lexicon used by the tiered BM25 (and the embedding model if used).
- Start simple (strategy 1) to get a baseline; strategy 2/3 can improve realism once retrieval runs are available.*** End Patch
