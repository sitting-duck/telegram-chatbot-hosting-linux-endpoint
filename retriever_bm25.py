# retriever_bm25.py
import json, pickle, textwrap
from pathlib import Path
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------
# Survival-Data specific index (PDF-based bm25_survival_home.idx)
# ---------------------------------------------------------

# Absolute path to the Survival-Data repo on your Linux box
SURVIVAL_DATA_DIR = Path(
    "/home/nerp/Documents/Github/apocalypse-mommy-llm-supervised-trainer/external/Survival-Data"
)

# The BM25 index built from the HOME/ PDFs (build_survival_pdf_index.py)
SURVIVAL_BM25_INDEX_PATH = SURVIVAL_DATA_DIR / "bm25_survival_home.idx"


def load_survival_index():
    """
    Convenience loader for the Survival-Data BM25 index.

    Usage:
        from retriever_bm25 import load_survival_index, retrieve
        bm25, docs = load_survival_index()
        results = retrieve("knife for hiking in the bush", bm25, docs, top_k=10)
    """
    bm25, docs = load_index(str(SURVIVAL_BM25_INDEX_PATH))
    return bm25, docs


def debug_retrieve(query: str, bm25=None, docs: List[Dict] | None = None, top_k: int = 5):
    """
    Debug helper: print top-k docs for a query with scores and snippets.

    If bm25/docs are not provided, it will load the Survival-Data index by default.
    """
    if bm25 is None or docs is None:
        bm25, docs = load_survival_index()

    scores = bm25.get_scores(_tokenize(query))
    idx_scores: List[Tuple[int, float]] = sorted(
        enumerate(scores), key=lambda x: x[1], reverse=True
    )[:top_k]

    print(f"\n=== Debug for query: {query!r} ===")
    for rank, (i, s) in enumerate(idx_scores, start=1):
        doc = docs[i]
        title = doc.get("title", "")
        source = doc.get("source_path", doc.get("id", ""))
        snippet = textwrap.shorten(doc.get("text", ""), width=320, placeholder=" ...")
        print(f"\n#{rank}  score={float(s):.3f}")
        print(f"Title: {title}")
        print(f"Source: {source}")
        print(snippet)
        print("-" * 80)


# ---------------------------------------------------------
# Generic JSONL-based BM25 builder / loader
# ---------------------------------------------------------
# Expected corpus: JSONL with {"id": "...", "text": "...", "title": "..."} (title optional)

def _tokenize(s: str) -> List[str]:
    return s.lower().split()  # simple + fast; swap in better tokenization if you like


def build_index(corpus_path: str, index_path: str):
    docs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("text") or obj.get("body") or ""
            if not text.strip():
                continue
            docs.append(
                {
                    "id": obj.get("id"),
                    "title": obj.get("title", ""),
                    "text": text,
                }
            )
    tokenized = [_tokenize(d["text"]) for d in docs]
    bm25 = BM25Okapi(tokenized)
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump({"docs": docs, "bm25": bm25}, f)
    return len(docs)


def load_index(index_path: str):
    with open(index_path, "rb") as f:
        blob = pickle.load(f)
    return blob["bm25"], blob["docs"]


def retrieve(query: str, bm25, docs: List[Dict], top_k: int = 50) -> List[Dict]:
    scores = bm25.get_scores(_tokenize(query))
    idx_scores: List[Tuple[int, float]] = sorted(
        enumerate(scores), key=lambda x: x[1], reverse=True
    )[:top_k]
    return [{**docs[i], "score": float(s)} for i, s in idx_scores]

