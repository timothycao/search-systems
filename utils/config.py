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

# HNSW tiering config
HNSW_MODEL_NAME: str = "sentence-transformers/msmarco-bert-base-dot-v5"
HNSW_EMBED_DIR: str = f"{ARTIFACTS_DIR}/hnsw_embeddings"
HNSW_DOC_EMB_PATH: str = f"{HNSW_EMBED_DIR}/doc_embeddings.h5"
HNSW_QUERY_EMB_PATH: str = f"{HNSW_EMBED_DIR}/query_embeddings.h5"
HNSW_TIERING_DIR: str = f"{ARTIFACTS_DIR}/tiering_dense"
HNSW_LABELS_PATH: str = f"{HNSW_TIERING_DIR}/labels.json"
HNSW_STATIC_SCORES_PATH: str = f"{HNSW_TIERING_DIR}/static_scores.npy"
HNSW_TIER_RATIO: float = 0.4
HNSW_TOPK_QUERIES: int = 25
HNSW_T1_EMB_PATH: str = f"{HNSW_EMBED_DIR}/doc_embeddings_t1.h5"
HNSW_T2_EMB_PATH: str = f"{HNSW_EMBED_DIR}/doc_embeddings_t2.h5"
HNSW_T1_DIR: str = f"{ARTIFACTS_DIR}/hnsw_T1"
HNSW_T2_DIR: str = f"{ARTIFACTS_DIR}/hnsw_T2"
