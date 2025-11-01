# Top level directories
DATA_DIR: str = "data"              # input datasets
ARTIFACTS_DIR: str = "artifacts"    # build outputs
RUNS_DIR: str = "runs"              # evaluation outputs

# Collection (documents/passages)
COLLECTION_DIR: str = f"{DATA_DIR}/collection"
DATASET_PATH: str = f"{COLLECTION_DIR}/collection.tsv"
SUBSET_PATH: str = f"{COLLECTION_DIR}/msmarco_passages_subset.tsv"
SUBSET_EMBEDDINGS_PATH: str = f"{COLLECTION_DIR}/msmarco_passages_embeddings_subset.h5"

# Queries
QUERIES_DIR: str = f"{DATA_DIR}/queries"
QUERIES_DEV_PATH: str = f"{QUERIES_DIR}/queries.dev.tsv"
QUERIES_EVAL_PATH: str = f"{QUERIES_DIR}/queries.eval.tsv"
QUERIES_EMBEDDINGS_PATH: str = f"{QUERIES_DIR}/msmarco_queries_dev_eval_embeddings.h5"

# Qrels (relevance labels)
QRELS_DIR: str = f"{DATA_DIR}/qrels"
QRELS_DEV_PATH: str = f"{QRELS_DIR}/qrels.dev.tsv"
QRELS_EVAL1_PATH: str = f"{QRELS_DIR}/qrels.eval.one.tsv"
QRELS_EVAL2_PATH: str = f"{QRELS_DIR}/qrels.eval.two.tsv"