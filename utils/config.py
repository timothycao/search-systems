# Top level directories
DATA_DIR: str = "data"              # input datasets
ARTIFACTS_DIR: str = "artifacts"    # build outputs
RUNS_DIR: str = "runs"              # evaluation outputs

# Collection (documents/passages)
COLLECTION_DIR: str = f"{DATA_DIR}/collection"
DATASET_PATH: str = f"{COLLECTION_DIR}/collection.tsv"
SUBSET_PATH = None # None builds index over full collection
#SUBSET_PATH: str = f"{COLLECTION_DIR}/msmarco_passages_subset.tsv"
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

# Tiering config
TIERING_DIR: str = f"{ARTIFACTS_DIR}/tiering"
TIERING_QTF_PATH: str = f"{TIERING_DIR}/qtf.json"
TIERING_LABELS_PATH: str = f"{TIERING_DIR}/labels.json"
TIERING_MODEL_PATH: str = f"{TIERING_DIR}/model.json"
TIERING_THRESHOLD_PATH: str = f"{TIERING_DIR}/threshold.json"
TIERING_FEATURE_NAMES_PATH: str = f"{TIERING_DIR}/train.pkl"
TIER1_IDS_PATH: str = f"{TIERING_DIR}/tier1_ids.txt"
TIER2_IDS_PATH: str = f"{TIERING_DIR}/tier2_ids.txt"
DELTA_DIR: str = TIERING_DIR
DELTA_T1_THRESHOLD: int = 1000
DELTA_T2_THRESHOLD: int = 100000
