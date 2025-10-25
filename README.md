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

## Usage

### Build

```bash
python -m scripts.build --system <bm25 | hnsw | rerank>
```

### Run

```bash
python -m scripts.run --system <bm25 | hnsw | rerank>
```
