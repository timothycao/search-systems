# CS-GY 6913 Assignment 3: Search Systems

This repository implements a modular retrieval and reranking framework for the MS MARCO passage ranking task.  
The system supports:

- **Sparse retrieval:** BM25  
- **Dense retrieval:** HNSW  
- **Fusion reranking:** RRF, LSF  
- **Neural reranking:** BERT based Bi-Encoder  

The pipeline supports evaluation on **dev**, **eval1**, and **eval2** subsets.

---

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Organize dataset files

Your directory structure should match:

```plaintext
data/
├── collection/
│   ├── collection.tsv
│   ├── msmarco_passages_subset.tsv
│   └── msmarco_passages_embeddings_subset.h5
├── queries/
│   ├── queries.dev.tsv
│   ├── queries.eval.tsv
│   └── msmarco_queries_dev_eval_embeddings.h5
└── qrels/
    ├── qrels.dev.tsv
    ├── qrels.eval.one.tsv
    └── qrels.eval.two.tsv
```

---

# Usage

The repository provides four top-level scripts:

- `scripts.build` — Build BM25 or HNSW indexes  
- `scripts.run` — Run retrieval or reranking  
- `scripts.evaluate` — Compute aggregate performance  
- `scripts.bucket_evaluation` — Compute aggregate and bucketed performance 

Below is the generalized usage for each script.

---

# Build

Used for constructing **BM25** or **HNSW** indexes.

```bash
python -m scripts.build \
    --system <bm25 | hnsw> \
    [--track <time | memory>]
```

---

# Run

The `run` script supports **three types** of systems with different arguments.

Results are saved in: `results/<system>/`.

---

## 1. Retrieval Systems (BM25, HNSW)

Requires:

- Evaluation dataset split (dev / eval1 / eval2)

```bash
python -m scripts.run \
    --system <bm25 | hnsw> \
    --qrels <dev | eval1 | eval2> \
    --save <output_run_file> \
    [--track <time | memory>]
```

---

## 2. Fusion Reranking (RRF, LSF)

Requires:

- Two run files (BM25 and HNSW)

```bash
python -m scripts.run \
    --system <rrf | lsf> \
    --targets <bm25_run> <hnsw_run> \
    --save <output_run_file> \
    [--track <time | memory>]
```

---

## 3. Neural Cascading Rerank (Bi-Encoder)

Requires:

- One run file (BM25 / HNSW / RRF / LSF)  
- Evaluation dataset split (dev / eval1 / eval2)

```bash
python -m scripts.run \
    --system biencoder \
    --qrels <dev | eval1 | eval2> \
    --targets <run_file> \
    --save <output_run_file> \
    [--track <time | memory>]
```

---

# Evaluate

Aggregated performance evaluation using the appropriate metrics for each dataset:

- dev → **MRR@10, Recall@100, MAP**  
- eval1/eval2 → **MRR@10, Recall@100, NDCG@10, NDCG@100**

```bash
python -m scripts.evaluate \
    --system <bm25 | hnsw | rrf | lsf | biencoder> \
    --qrels <dev | eval1 | eval2> \
    --run <run_file>
```

---

# Bucketed Evaluation

Computes:

- Aggregated performance evaluation
- Performance evaluation for short, medium, long query buckets  

Results are saved in: `results/<system>/`.

```bash
python -m scripts.bucket_evaluation \
    --system <bm25 | hnsw | rrf | lsf | biencoder> \
    --qrels <dev | eval1 | eval2> \
    --run <run_file> \
    --save <results_file>
```

---

# Workflow Example

A full demonstration of the complete retrieval and reranking pipeline, covering  
BM25 and HNSW index construction, sparse and dense retrieval, fusion-based reranking  
(LSF and RRF), bi-encoder cascading reranking, and comprehensive evaluation, is  
provided in **Section 3: System Implementation and Execution Framework** within the **Report.pdf** file which may be found at the root of the repository.

**Section 3** walks through the entire workflow in detail, including:

- Building BM25 and HNSW indexes  
- Running retrieval/ranking for dev, eval1, and eval2  
- Applying LSF/RRF fusion reranking  
- Applying BERT bi-encoder cascading reranking  
- Evaluating aggregated/bucketed performance
 
