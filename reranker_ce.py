# reranker_ce.py
#
# Thin wrapper around a sentence-transformers CrossEncoder for reranking.
# Supports both:
#   - local checkpoint directories
#   - HuggingFace model names like "BAAI/bge-reranker-base"

from typing import List
from sentence_transformers import CrossEncoder


class RerankerCE:
    def __init__(self, ckpt: str):
        """
        ckpt can be:
          - a local directory path with config.json, model.safetensors, etc.
          - a HuggingFace model name like 'BAAI/bge-reranker-base'
        """
        self.model = CrossEncoder(ckpt, num_labels=1, max_length=256)
        print("-----------------------------Reranker loaded:", ckpt)

    def score(self, query: str, docs: List[str]) -> List[float]:
        """
        Score a list of documents for a single query.
        Returns a list of floats (higher = more relevant).
        """
        pairs = [(query, d) for d in docs]
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]

