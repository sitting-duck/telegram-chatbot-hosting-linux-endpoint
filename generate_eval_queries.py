# generate_eval_queries.py
#
# Generate an evaluation dataset (queries + relevant chunk IDs)
# for the Survival-Data HOME index.
#
# - Loads BM25 index via retriever_bm25.load_survival_index()
# - Filters to reasonable survival/manual categories
# - Drops chunks that look like citation/reference blocks
# - Creates a query for each sampled chunk:
#     * Prefer: paraphrase using Ollama (nice human question)
#     * Fallback: heuristic template built from a cleaned sentence
# - Rejects bad questions (too citation-y, too numeric, not survival-ish)
# - APPENDS to data/survival_eval_queries.jsonl, avoiding duplicate chunk IDs
#
# Per run, it adds up to NUM_NEW_QUERIES new entries.
#
# Each line:
#   {"query": "...", "relevant_ids": ["HOME/...pdf#123"]}

import os
import re
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

import httpx

from retriever_bm25 import load_survival_index  # must exist in your repo


# -----------------------------
# SETTINGS
# -----------------------------

# How many *new* eval queries to add per run
NUM_NEW_QUERIES = 50

# Where to write the JSONL (this file will be appended to)
OUTPUT_PATH = Path("data/survival_eval_queries.jsonl")

# Categories from your index we consider "instructional" and useful
ALLOWED_CATEGORIES = {
    "Basics",
    "Water",
    "Medicine",
    "Edible-and-Medicinal-Plants",
    "How-to-and-Bushcraft",
    "Food-Preservation",
    "Food-Production-&-Recipes",
    "Hunting-&-Trapping",
    "Knots",
    "Quick-Guides-&-Checklists",
    "Survival-Manuals",
    "Food-Production-&-Recipes",
}

# Titles that look like narrative/journals/memoirs – avoid for eval
BLOCKED_TITLE_KEYWORDS = [
    "journal",
    "journals",
    "out-of-captivity",
    "diary",
    "memoir",
    "novel",
]

# Heuristic query templates by category
QUESTION_TEMPLATES = {
    "Basics": [
        "What is the basic procedure for {}?",
        "How do I safely perform {}?",
        "What do I need to know about {}?",
        "Beginner guide: {} — what should I do?",
    ],
    "Water": [
        "How should I handle {}?",
        "What are the correct steps for {}?",
        "What should I know about {} in an emergency?",
    ],
    "Medicine": [
        "How do I treat {}?",
        "What is the field method for {}?",
        "In an emergency, how should I manage {}?",
    ],
    "Edible-and-Medicinal-Plants": [
        "How can I use {}?",
        "Is {} safe to eat or use?",
        "What survival uses does {} have?",
    ],
    "How-to-and-Bushcraft": [
        "Bushcraft question: how do I {}?",
        "Step-by-step guide for {}?",
        "What is the safest way to {}?",
    ],
    "Food-Preservation": [
        "How do I preserve food using {}?",
        "What should I know about {} preservation?",
        "Explain the process of {} in food storage.",
    ],
    "Food-Production-&-Recipes": [
        "How can I use {} in food production or recipes?",
        "What is a good way to prepare {}?",
        "How can I grow or produce {}?",
    ],
    "Hunting-&-Trapping": [
        "How can I {} while hunting or trapping?",
        "What is the correct method for {} in trapping?",
        "What should I know about {} when hunting?",
    ],
    "Knots": [
        "What is the {} knot used for?",
        "How do I tie a {} knot?",
        "When should I use the {} knot?",
    ],
    "Quick-Guides-&-Checklists": [
        "Give me a quick guide for {}.",
        "What checklist should I follow for {}?",
        "What are the key steps for {}?",
    ],
    "Survival-Manuals": [
        "Explain {} in basic survival terms.",
        "What is the recommended approach to {}?",
        "What do survival manuals recommend about {}?",
    ],
}

# Trail stopwords to avoid queries ending in "of the", "and", etc.
STOPWORDS_TRAIL = {"and", "or", "the", "to", "of", "in", "for", "with"}

# Survival-ish keywords to require in the final question
SURV_KEYWORDS = [
    "treat",
    "treatment",
    "store",
    "storage",
    "build",
    "build a",
    "cook",
    "prepare",
    "preserve",
    "preservation",
    "tie",
    "knot",
    "shelter",
    "trap",
    "trapping",
    "hunting",
    "hunt",
    "water",
    "wound",
    "injury",
    "injuries",
    "emergency",
    "first aid",
    "poisonous",
    "edible",
    "safe to eat",
    "forage",
    "foraging",
    "navigation",
    "navigate",
    "map",
    "compass",
    "radio",
    "signal",
    "signaling",
    "fire",
    "hypothermia",
    "cold weather",
    "heat",
    "dehydration",
    "bandage",
    "bleeding",
]

# Ollama / Qwen configuration (reuse your existing env if possible)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "").strip()  # e.g., "qwen2.5"


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def is_blocked_title(title: str) -> bool:
    t = (title or "").lower()
    return any(key in t for key in BLOCKED_TITLE_KEYWORDS)


def looks_like_citation_block(text: str) -> bool:
    """Heuristically detect reference / citation-heavy chunks and skip them."""
    t = text.strip()
    if not t:
        return True

    # Many digits or multiple years = likely reference section
    year_matches = re.findall(r"\b(19|20)\d{2}\b", t)
    digit_ratio = sum(ch.isdigit() for ch in t) / max(len(t), 1)

    if len(year_matches) >= 2:
        return True
    if digit_ratio > 0.25:
        return True

    # Obvious journal-ish markers
    if re.search(r"\b(Ann\s+Oncol|Anticancer\s+Research|Clin\s+Pharm|Oncology|Eur\s+J)\b", t):
        return True

    # Very reference-like: many semicolons/parentheses/commas in short span
    head = t[:260]
    if sum(ch in ";()" for ch in head) >= 6:
        return True

    return False


def filter_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only docs from allowed categories and non-narrative, non-reference chunks."""
    eligible = []
    for d in docs:
        cat = d.get("category", "")
        if cat and cat not in ALLOWED_CATEGORIES:
            continue
        title = d.get("title") or ""
        if is_blocked_title(title):
            continue
        text = (d.get("text") or "").strip()
        if len(text) < 60:  # ignore tiny chunks
            continue
        if looks_like_citation_block(text):
            continue
        eligible.append(d)
    return eligible


def pick_sentence(text: str) -> str:
    """Rough sentence selection: get a reasonably clean, mid-length sentence."""
    # Normalize whitespace
    text = " ".join(text.split())
    # Rough sentence split
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for s in sentences:
        s = s.strip()
        if len(s) < 40 or len(s) > 220:
            continue
        # Avoid diary-style first-person narrative
        if s.lower().startswith(("i ", "we ", "he ", "she ", "they ")):
            continue
        # Skip obviously reference-like sentences
        if looks_like_citation_block(s):
            continue
        # If it passed all filters, take it
        return s
    # Fallback: first 200 chars of the whole thing
    return text[:200]


def shorten_phrase(sentence: str, max_tokens: int = 10) -> str:
    """Turn a sentence into a short phrase for template filling."""
    tokens = sentence.split()
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    # Strip trailing punctuation and junk stopwords
    while tokens:
        last = tokens[-1].rstrip(",.:;")
        if last.lower() in STOPWORDS_TRAIL:
            tokens.pop()
        else:
            break
    return " ".join(tokens)


def make_heuristic_query(chunk: Dict[str, Any]) -> str:
    """Heuristic query based purely on chunk text + category templates."""
    text = (chunk.get("text") or "").strip()
    category = chunk.get("category", "")
    sentence = pick_sentence(text)
    phrase = shorten_phrase(sentence, max_tokens=12)

    templates = QUESTION_TEMPLATES.get(category, QUESTION_TEMPLATES["Basics"])
    template = random.choice(templates)
    return template.format(phrase)


def call_ollama_for_query(chunk: Dict[str, Any], timeout: float = 25.0) -> Optional[str]:
    """
    Ask Ollama to produce a single natural user question that this chunk would answer.
    Returns None if anything goes wrong or Ollama is not configured.
    """
    if not OLLAMA_MODEL:
        return None

    text = (chunk.get("text") or "").strip()
    if not text:
        return None

    # Keep the prompt from getting too huge
    max_chars = 900
    excerpt = text[:max_chars]

    system_prompt = (
        "You are helping to create evaluation queries for a survival question-answering system. "
        "Given an excerpt from a survival-related document (manual, first aid, plants, water, "
        "bushcraft, etc.), you will write ONE realistic question that a user might ask that "
        "could be answered by this excerpt.\n"
        "- Focus on practical survival, wilderness medicine, water, food, navigation, shelters, "
        "  plants/herbs, or basic safety.\n"
        "- Ignore citation-like details such as journal names, years, volume/issue numbers, "
        "  and author lists.\n"
        "- Do NOT mention that you see an excerpt or a document.\n"
        "- Output only the question, nothing else."
    )

    user_prompt = f"EXCERPT:\n\"\"\"\n{excerpt}\n\"\"\"\n\nWrite one realistic survival or first-aid question in plain English."

    try:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        url = f"{OLLAMA_URL.rstrip('/')}/api/chat"
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = (data.get("message", {}) or {}).get("content", "")
        q = (content or "").strip()

        if not q:
            return None

        # If the model replies with multiple lines, grab first line
        q = q.split("\n")[0].strip()

        # Remove leading numbering if present ("1. ", "- ", etc.)
        q = re.sub(r"^(\d+[\).\s]+|-+\s+)", "", q).strip()

        # Ensure it looks like a question
        if not q.endswith("?"):
            q = q.rstrip(".") + "?"

        return q

    except Exception:
        return None


def make_query_with_ollama_fallback(chunk: Dict[str, Any]) -> str:
    """
    Try to create a natural query via Ollama; if that fails, use the heuristic.
    """
    q_ollama = call_ollama_for_query(chunk)
    if q_ollama:
        return q_ollama
    return make_heuristic_query(chunk)


def is_good_query(q: str) -> bool:
    """
    Final sanity filter for generated queries.
    We reject:
      - too short / too long
      - no '?'
      - too many digits
      - obvious journal/citation patterns
      - no survival-ish keyword
    """
    q = (q or "").strip()
    if len(q) < 25 or len(q) > 200:
        return False
    if "?" not in q:
        return False

    letters = [c for c in q if c.isalpha()]
    if len(letters) < 10:
        return False

    digit_ratio = sum(c.isdigit() for c in q) / max(len(q), 1)
    if digit_ratio > 0.20:
        return False

    # Obvious citation-like patterns in the query itself
    if re.search(r"\b(Ann\s+Oncol|Anticancer\s+Research|Clin\s+Pharm|Oncology|Eur\s+J|J\s*[A-Z][a-z]+)\b", q):
        return False
    if re.search(r"\b(19|20)\d{2}\b", q) and any(w in q for w in ["Research", "Oncol", "J ", "Ann "]):
        return False

    # Must contain at least one survival-ish keyword
    lower_q = q.lower()
    if not any(kw in lower_q for kw in SURV_KEYWORDS):
        return False

    return True


def load_existing_ids_and_queries(path: Path) -> tuple[Set[str], Set[str]]:
    """Load existing relevant_ids and queries from an existing JSONL file."""
    existing_ids: Set[str] = set()
    existing_queries: Set[str] = set()
    if not path.exists():
        return existing_ids, existing_queries

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            q = (obj.get("query") or "").strip()
            if q:
                existing_queries.add(q)
            rel_ids = obj.get("relevant_ids") or []
            for rid in rel_ids:
                existing_ids.add(str(rid))
    return existing_ids, existing_queries


# -----------------------------
# MAIN
# -----------------------------

def main():
    random.seed(42)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading existing eval file (if any)...")
    existing_ids, existing_queries = load_existing_ids_and_queries(OUTPUT_PATH)
    print(f"Existing entries: {len(existing_queries)} queries, {len(existing_ids)} unique chunk IDs")

    print("Loading Survival-Data BM25 index...")
    bm25, docs = load_survival_index()
    print(f"Loaded {len(docs)} chunks total.")

    print("Filtering to survival/manual-style, non-reference chunks...")
    eligible_docs = filter_docs(docs)
    print(f"Eligible chunks after filtering: {len(eligible_docs)}")

    if not eligible_docs:
        print("No eligible chunks found after filtering; aborting.")
        return

    # Shuffle and then walk until we collect NUM_NEW_QUERIES that pass is_good_query
    random.shuffle(eligible_docs)

    added_records = []
    added_count = 0

    # Open in append mode
    with OUTPUT_PATH.open("a", encoding="utf-8") as f:
        for chunk in eligible_docs:
            if added_count >= NUM_NEW_QUERIES:
                break

            chunk_id = str(chunk.get("id"))
            if not chunk_id:
                continue

            # Skip if this chunk ID has already been used in prior runs
            if chunk_id in existing_ids:
                continue

            q = make_query_with_ollama_fallback(chunk)
            if not is_good_query(q):
                continue

            # Skip if this exact query string already exists
            if q in existing_queries:
                continue

            rel_ids = [chunk_id]
            rec = {"query": q, "relevant_ids": rel_ids}
            added_records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            added_count += 1
            existing_ids.add(chunk_id)
            existing_queries.add(q)

            print(f"[+{added_count}/{NUM_NEW_QUERIES}] {q}  ->  {rel_ids[0]}")

    print(f"\nAppended {len(added_records)} new evaluation queries to {OUTPUT_PATH}")
    if added_count < NUM_NEW_QUERIES:
        print("Note: ran out of eligible, unused chunks before reaching NUM_NEW_QUERIES; "
              "you can relax filters or increase categories if you want more per run.")


if __name__ == "__main__":
    main()

