# rag_pipeline.py

import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sentence_transformers import CrossEncoder

# --- config from env ---
BM25_INDEX_PATH = Path(os.getenv("BM25_INDEX_PATH", "data/bm25.idx"))
#RERANKER_PATH = Path(os.getenv("RERANKER_PATH", "reranker/checkpoints/modelA_miniLM_e2_bm25neg"))
RERANKER_PATH = Path(os.getenv("RERANKER_PATH", "/home/nerp/Documents/Github/apocalypse-mommy-llm-supervised-trainer/runs/bge_reranker_e2_bm25neg"))
BM25_TOPK = int(os.getenv("BM25_TOPK", "50"))
RERANK_TOPN = int(os.getenv("RERANK_TOPN", "3"))

# Globals we fill on first use
_bm25 = None
_docs: List[Dict] = []
_reranker: CrossEncoder | None = None


def _ensure_loaded():
    """
    Lazily load BM25 index + docs + reranker.
    Raises if files are missing.
    """
    global _bm25, _docs, _reranker

    if _bm25 is None or not _docs:
        if not BM25_INDEX_PATH.exists():
            raise FileNotFoundError(f"BM25 index not found at {BM25_INDEX_PATH}")
        with BM25_INDEX_PATH.open("rb") as f:
            bundle = pickle.load(f)
        _docs = bundle["docs"]          # list of dicts with at least "text"
        _bm25 = bundle["bm25"]          # BM25Okapi (rank_bm25)
        # tokenized docs may also exist in bundle["tokenized"], but BM25 already has them

    if _reranker is None:
        if not RERANKER_PATH.exists():
            raise FileNotFoundError(f"Reranker checkpoint not found at {RERANKER_PATH}")
        _reranker = CrossEncoder(str(RERANKER_PATH), num_labels=1, max_length=256)
        print(f"CrossEncoder loaded Checkpoint at: {RERANKER_PATH}")


def _simple_tokenize(text: str) -> List[str]:
    return text.lower().split()


def build_rag_prompt(query: str) -> Tuple[str, Dict]:
    """
    Given user query -> returns:
        new_query_with_context, debug_info
    """
    _ensure_loaded()

    # --- 1) first-stage retrieval with BM25 ---
    query_tokens = _simple_tokenize(query)
    scores = _bm25.get_scores(query_tokens)  # shape: [len(docs)]
    scores = np.array(scores)
    topk = min(BM25_TOPK, len(_docs))
    top_idx = scores.argsort()[::-1][:topk]

    candidates = [(_docs[i], float(scores[i])) for i in top_idx]

    # --- 2) rerank with CrossEncoder ---
    texts = [doc["text"] for doc, _ in candidates]
    ce_inputs = [(query, t) for t in texts]
    ce_scores = _reranker.predict(ce_inputs)  # shape: [topk]

    ce_scores = np.array(ce_scores).reshape(-1)
    rerank_idx = ce_scores.argsort()[::-1][:RERANK_TOPN]
    top_docs = [candidates[i][0] for i in rerank_idx]

    # --- 3) build context string ---
    parts = []
    for d in top_docs:
        title = d.get("title")
        if title:
            parts.append(f"### {title}\n{d['text']}")
        else:
            parts.append(d["text"])
    context = "\n\n---\n\n".join(parts)

    # --- 4) stitch into a new prompt for the LLM ---
    prompt = (
        "Use only the following context to answer the user's question. "
        "If the answer is not contained here, say you don't know.\n\n"
        f"{context}\n\n"
        f"User question: {query}"
    )

    debug = {
        "num_docs": len(_docs),
        "bm25_topk": topk,
        "rerank_topn": RERANK_TOPN,
        "titles": [d.get("title") for d in top_docs],
        "context_snippet": context[:200],
    }
    return prompt, debug

