# Document Static Score (Query-Log–Weighted BM25)

## Formula: query-log–weighted BM25 sum
- We want a single, global “prior” score per doc:  
  `StaticScore(d) = Σ_{t in d} QTF(t) * w(t, d)`
- `QTF(t)`: how many times term `t` appears across the query log (after the same preprocessing as the index). High if users search this term often.
- `w(t, d)`: BM25 term contribution for term `t` in document `d`, computed with the doc’s term frequency and corpus stats:  
  `w(t, d) = idf(t) * ((tf_td * (k1 + 1)) / (tf_td + k1 * (1 - b + b * len_d / avgdl)))`
  - Inputs:
    - `tf_td`: term frequency of `t` in doc `d`
    - `len_d`: length of doc `d` (tokens)
    - `avgdl`: average document length
    - `idf(t)`: inverse document frequency from the index
    - `k1`, `b`: BM25 params
- Intuition: BM25 gives within-doc relevance for `t`; QTF scales it by how popular `t` is among queries. Summing over all terms in `d` yields a “how appealing is this doc to our observed queries?” prior.

## Computation pass (single inverted-index sweep)
1) Initialize `doc_static_score[d] = 0` for every doc ID.  
2) Iterate over each term `t` in the lexicon:  
   - If `QTF(t) == 0`, skip (the term never appears in queries, so it can’t improve appeal).  
   - Set `q_weight = QTF(t)`.  
3) For that term’s postings list `(doc_id, tf_td)`:  
   - Compute `w_t_d = BM25_term_weight(tf_td, len_doc[doc_id], avgdl, idf(t), k1, b)` using the formula above.  
   - Accumulate: `doc_static_score[doc_id] += q_weight * w_t_d`.  
4) After all terms are processed, each `doc_static_score[d]` is the sum of BM25 contributions for all terms present in the doc, weighted by how frequently those terms appear in the query log.

## Why it works
- Terms common in user queries (high QTF) dominate the sum, so docs rich in those terms get higher scores.
- Docs with rare-in-query terms contribute little, even if the terms are frequent in the doc.
- Using the inverted index makes it efficient: we only touch docs that actually contain each term, and we reuse stored `tf`, `len_d`, `idf`, `avgdl`, `k1`, `b`.

## Normalization for labeling
- Once `doc_static_score` is computed for all docs, we min–max normalize to `[0,1]`:  
  `static_norm[d] = (score[d] - min_score) / (max_score - min_score)`
- Sort docs by `static_norm` descending, take the top target ratio (e.g., 40%) as Tier-1 labels, remainder Tier-2.

## Inference for new docs
- For a new doc, we recompute the same BM25 term weights using the base index stats (`idf`, `avgdl`, `k1`, `b`) and the fixed `QTF`. That static score feature feeds the trained tiering model alongside other doc-side features.

## How QTF drives the static score (summary)
- QTF definition: We precompute a global query term frequency map over the 200k-query log using the same preprocessing as the index (tokenization, lowercasing, stopword removal, stemming/lemmatization if enabled). For each token `t`, `qtf[t]` counts how many queries contain `t`.
- Static score formula: For each document `d`, we compute a query-log–weighted BM25 sum:  
  `StaticScore(d) = Σ_{t ∈ d} QTF(t) * w(t, d)`  
  where `w(t, d)` is the BM25 term weight using the doc’s `tf`, length, `avgdl`, `idf`, and `k1/b` from the index.
- Computation pass: Iterate terms in the inverted index. If `qtf[t] > 0`, scale every posting `(d, tf_td)` by `qtf[t] * BM25_term_weight(tf_td, len_d, avgdl, idf(t), k1, b)` and accumulate into `doc_static_score[d]`. Terms not seen in the query log (`qtf=0`) contribute nothing, so the static score emphasizes documents that overlap popular query terms.
- Normalization and tier labels: After scoring all docs, min–max normalize scores to `[0,1]`, rank by normalized score, and assign Tier-1 to the top target ratio (e.g., 40%) and Tier-2 to the rest. These labels feed the feature/label datasets and the XGBoost tiering model.
- Inference for new docs: During ingest, we reuse the same `qtf` and BM25 stats (`idf`, `avgdl`, `k1`, `b`) from the base index to compute a static score for each new doc. That static score is used as a feature (and log-transformed variant) alongside length/IDF/entropy features. The trained model then predicts Tier-1 vs Tier-2, and we route the doc into the appropriate delta.
- Why QTF helps: Weighting by QTF biases the static score toward terms that appear frequently in real queries, making the score a prior relevance signal aligned with observed query demand rather than raw document content alone.
