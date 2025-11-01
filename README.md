# CS-GY 6913 Assignment 3: Search Systems

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Organize dataset files

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

## Usage

### Build

```bash
python -m scripts.build \
    --system <bm25 | hnsw | rerank> \
    [--track <time | memory>]
```

### Run

```bash
python -m scripts.run \
    --system <bm25 | hnsw | rerank> \
    --qrels <dev | eval1 | eval2> \
    --save <filename> \
    [--track <time | memory>]
```

### Evaluate

```bash
python -m scripts.evaluate \
    --system <bm25 | hnsw | rerank> \
    --qrels <dev | eval1 | eval2> \
    --run <filename>
```
