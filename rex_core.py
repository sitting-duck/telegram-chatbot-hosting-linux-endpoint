# rex_core.py — shared config + RAG + Ollama streaming

import os
import json
import logging
from pathlib import Path

import httpx

from retriever_bm25 import load_survival_index, retrieve as bm25_retrieve

logging.basicConfig(level=logging.INFO)

# --- Env/config (moved from main.py) ---
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN env var.")

WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
OLLAMA_URL     = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL", "qwen2.5")

NUM_PREDICT = int(os.environ.get("NUM_PREDICT", "180"))
NUM_CTX     = int(os.environ.get("NUM_CTX", "2048"))
KEEP_ALIVE  = os.environ.get("KEEP_ALIVE", "30m")

MAX_TOTAL_CHARS = int(os.environ.get("MAX_TOTAL_CHARS", "3500"))
SUBSCRIBERS_FILE = Path(os.environ.get("SUBSCRIBERS_FILE", "subscribers.json"))

# RAG knobs
MIN_BM25_SCORE   = float(os.environ.get("MIN_BM25_SCORE", "2.0"))
MAX_CONTEXT_DOCS = int(os.environ.get("MAX_CONTEXT_DOCS", "5"))

# --- Load Survival-Data BM25 index at import time ---
try:
    BM25, DOCS = load_survival_index()
    logging.info("Rex: loaded %s survival chunks from Survival-Data index", len(DOCS))
except Exception:
    logging.exception("Rex: failed to load Survival-Data BM25 index")
    BM25, DOCS = None, []

# rex_core.py — shared config + RAG + Ollama streaming

import os
import logging
from pathlib import Path

import httpx

from retriever_bm25 import load_survival_index, retrieve as bm25_retrieve

logging.basicConfig(level=logging.INFO)

# --- Env/config (moved from main.py) ---
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN env var.")

WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
OLLAMA_URL     = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL", "qwen2.5")

NUM_PREDICT = int(os.environ.get("NUM_PREDICT", "180"))
NUM_CTX     = int(os.environ.get("NUM_CTX", "2048"))
KEEP_ALIVE  = os.environ.get("KEEP_ALIVE", "30m")

MAX_TOTAL_CHARS = int(os.environ.get("MAX_TOTAL_CHARS", "3500"))
SUBSCRIBERS_FILE = Path(os.environ.get("SUBSCRIBERS_FILE", "subscribers.json"))

# RAG knobs
MIN_BM25_SCORE   = float(os.environ.get("MIN_BM25_SCORE", "2.0"))
MAX_CONTEXT_DOCS = int(os.environ.get("MAX_CONTEXT_DOCS", "5"))

# --- Load Survival-Data BM25 index at import time ---
try:
    BM25, DOCS = load_survival_index()
    logging.info("Rex: loaded %s survival chunks from Survival-Data index", len(DOCS))
except Exception:
    logging.exception("Rex: failed to load Survival-Data BM25 index")
    BM25, DOCS = None, []

async def stream_ollama(prompt: str, sys_prompt: str | None = None):
    """
    Stream tokens from the local Ollama instance using the OpenAI-compatible
    /v1/chat/completions endpoint with stream=True.

    Ollama prefixes each JSON chunk line with 'data: ' and ends with 'data: [DONE]'.
    This function strips the prefix, parses the JSON, and yields delta.content.
    """
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "options": {
            "num_predict": NUM_PREDICT,
            "num_ctx": NUM_CTX,
            "keep_alive": KEEP_ALIVE,
        },
    }

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_URL}/v1/chat/completions",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                if not raw_line:
                    continue

                line = raw_line.strip()
                # Handle "data: [DONE]"
                if line.startswith("data:"):
                    line = line[len("data:") :].strip()
                if not line:
                    continue
                if line == "[DONE]":
                    break

                try:
                    data = json.loads(line)
                except Exception as e:
                    logging.warning("Bad line from Ollama: %r (%s)", raw_line, e)
                    continue

                choices = data.get("choices") or []
                if not choices:
                    continue

                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    yield delta

                # Optional: stop if finish_reason appears
                if choices[0].get("finish_reason") is not None:
                    break
