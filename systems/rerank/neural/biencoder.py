"""
BERT based Bi-Encoder Reranking System.
"""

import os
from typing import List, Tuple
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, util

from systems.rerank.base import RerankSystem
from utils.io import load_run, load_queries, load_passages_from_subset
from utils.config import (
    DATASET_PATH,
    SUBSET_PATH,
    QUERIES_EVAL_PATH,
)

# Set environment variables to limit thread usage for reproducibility
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


class BiEncoderSystem(RerankSystem):
    """Semantic reranking using a BERT-based bi-encoder model."""

    # cosine model: "sentence-transformers/msmarco-distilbert-base-v4"
    def __init__(self, model_name: str = "sentence-transformers/msmarco-bert-base-dot-v5"):
        super().__init__("BiEncoder")
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def build(self) -> None:
        """No setup required for BERT based bi-encoder rerank."""
        pass

    def rerank(
        self,
        runs: List[str],
        top_k: int = 100,
        queries_path: str = QUERIES_EVAL_PATH,
    ) -> List[Tuple[str, List[Tuple[str, float]]]]:
        """
        Re-rank top-k candidate documents from BM25 based on semantic similarity.

        Args:
            runs: List of run file paths (only the first one is used, e.g., BM25 results).
            top_k: Number of top candidates to rerank per query.
            queries_path: Path to the evaluation queries file.

        Returns:
            List of (query_id, ranked_results),
            where ranked_results = [(doc_id, similarity_score), ...].
        """
        path = runs[0]
        print(f"[{self.name}] Loading run: {path}")
        path_results = load_run(path)

        # Load queries and passages
        print(f"[{self.name}] Loading queries and subset passages...")
        queries = load_queries(queries_path)
        passages = load_passages_from_subset(DATASET_PATH, SUBSET_PATH)

        results: List[Tuple[str, List[Tuple[str, float]]]] = []

        print(f"[{self.name}] Performing semantic reranking...")

        for qid, doc_scores in tqdm(path_results.items(), desc=f"[{self.name}] Reranking", unit="query"):
            query_text = queries.get(qid)
            if not query_text:
                continue

            # Select top-k BM25 candidates
            candidate_ids = list(doc_scores.keys())[:top_k]
            candidate_texts = [passages.get(pid, "") for pid in candidate_ids]

            # Filter missing or empty passages
            valid_pairs = [(pid, text) for pid, text in zip(candidate_ids, candidate_texts) if text.strip()]
            if not valid_pairs:
                continue

            # Encode query and candidates
            query_emb = self.model.encode(query_text, convert_to_tensor=True, device=self.device)
            doc_embs = self.model.encode(
                [text for _, text in valid_pairs],
                batch_size=4,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False,
            )

            # Compute cosine similarities
            #query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=0)
            #doc_embs = torch.nn.functional.normalize(doc_embs, p=2, dim=1)
            #sims = util.cos_sim(query_emb, doc_embs).squeeze(0).cpu().tolist()

            # Compute dot-product similarities
            sims = (query_emb @ doc_embs.T).squeeze(0).cpu().tolist()

            ranked_docs = sorted(zip([pid for pid, _ in valid_pairs], sims), key=lambda x: x[1], reverse=True)
            results.append((qid, ranked_docs))

        print(f"[{self.name}] Semantic reranking complete for {len(results)} queries.")
        return results
