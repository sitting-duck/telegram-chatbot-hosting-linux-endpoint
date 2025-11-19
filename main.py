# main.py — Telegram + RAG + reranker + analytics + web Rex hook

import os, sys, json, logging
from time import monotonic
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, Request, HTTPException
from telegram import Update
from telegram.constants import ChatAction
from telegram.error import BadRequest
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

from affiliate_catalog import find_matches, preset_for_scenario
from analytics_logger import (
    log_interaction,
    log_affiliate_impressions,
    log_system,
    categorize,
)
from retriever_bm25 import load_survival_index, retrieve as bm25_retrieve
from web_rex import register_web_routes

# --- Wire up training repo so we can import reranker_ce ---
TRAIN_REPO = os.environ.get("TRAIN_REPO")
if TRAIN_REPO and TRAIN_REPO not in sys.path:
    sys.path.append(TRAIN_REPO)

try:
    from reranker_ce import RerankerCE
except Exception:
    RerankerCE = None  # type: ignore

logging.basicConfig(level=logging.INFO)

# --- Env/config ---
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
BM25_TOPK        = int(os.environ.get("BM25_TOPK", "50"))
RERANK_TOPN      = int(os.environ.get("RERANK_TOPN", "3"))

# Reranker checkpoint (from your training repo)
RERANKER_PATH = os.environ.get("RERANKER_PATH")
RERANKER_MODEL = None

if RERANKER_PATH and RerankerCE is not None:
    try:
        logging.info(f"Loading reranker from {RERANKER_PATH} ...")
        RERANKER_MODEL = RerankerCE(RERANKER_PATH)
        logging.info("Reranker loaded OK.")
    except Exception:
        logging.exception("Failed to load reranker; continuing without it.")
        RERANKER_MODEL = None
else:
    if not RERANKER_PATH:
        logging.info("No RERANKER_PATH set; running without reranker.")
    if RerankerCE is None:
        logging.info("reranker_ce import failed; running without reranker.")

# --- Load Survival-Data BM25 index at startup ---
try:
    BM25, DOCS = load_survival_index()
    logging.info(f"Rex: loaded {len(DOCS)} survival chunks from Survival-Data index")
except Exception:
    logging.exception("Rex: failed to load Survival-Data BM25 index")
    BM25, DOCS = None, []

# --- Helper: cross-encoder rerank wrapper ---
def rerank_candidates(query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """
    Take BM25 candidates (each with 'text', 'score', etc.), use RERANKER_MODEL
    to get better ordering, and return top_k docs.
    """
    global RERANKER_MODEL
    if RERANKER_MODEL is None:
        return candidates[:top_k]

    # Prepare pairs (query, passage)
    passages = [c.get("text") or "" for c in candidates]
    pairs = [(query, p) for p in passages]

    try:
        # Try RERANKER_MODEL.predict first, else fall back to underlying .model.predict
        if hasattr(RERANKER_MODEL, "predict"):
            scores = RERANKER_MODEL.predict(pairs)  # type: ignore[arg-type]
        elif hasattr(RERANKER_MODEL, "model") and hasattr(RERANKER_MODEL.model, "predict"):
            scores = RERANKER_MODEL.model.predict(pairs)  # type: ignore[union-attr]
        else:
            logging.warning("RERANKER_MODEL has no predict or model.predict; skipping rerank.")
            return candidates[:top_k]
    except Exception:
        logging.exception("Error while reranking candidates; falling back to BM25 order.")
        return candidates[:top_k]

    # Attach scores and sort
    scored: List[Dict[str, Any]] = []
    for cand, s in zip(candidates, scores):
        c = dict(cand)
        try:
            c["rerank_score"] = float(s)
        except Exception:
            c["rerank_score"] = 0.0
        scored.append(c)

    scored.sort(key=lambda d: d.get("rerank_score", 0.0), reverse=True)
    return scored[:top_k]


# --- Build PTB app ---
tg_app = Application.builder().token(BOT_TOKEN).build()

# --- Streaming call to Ollama ---
async def stream_ollama(prompt: str, sys_prompt: str | None = None):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": (
            ([{"role": "system", "content": sys_prompt}] if sys_prompt else [])
            + [{"role": "user", "content": prompt}]
        ),
        "stream": True,
        "options": {
            "num_predict": NUM_PREDICT,
            "num_ctx": NUM_CTX,
            "temperature": float(os.getenv("TEMP", "0.3")),
            "top_p": float(os.getenv("TOP_P", "0.9")),
            "repeat_penalty": float(os.getenv("REPEAT_PENALTY", "1.15")),
        },
        "keep_alive": KEEP_ALIVE,
    }
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                chunk = (data.get("message", {}) or {}).get("content", "")
                if chunk:
                    yield chunk

# --- Telegram helpers ---
MAX_LEN = 4096

async def edit_throttled(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    message_id: int,
    new_full_text: str,
    last_edit_time: float,
    last_text: str,
    min_interval: float = 0.25,
):
    display = new_full_text if len(new_full_text) <= MAX_LEN else new_full_text[-MAX_LEN:]
    if display == last_text:
        return last_edit_time, last_text

    now = monotonic()
    if (now - last_edit_time) < min_interval:
        return last_edit_time, last_text

    try:
        await context.bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=display or "…")
    except BadRequest as e:
        if "not modified" not in str(e).lower():
            raise
    return now, display

async def send_final(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    message_id: int,
    full_text: str,
    last_text: str,
):
    first = (full_text[:MAX_LEN] or "…")
    if first != last_text:
        try:
            await context.bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=first)
        except BadRequest as e:
            if "not modified" not in str(e).lower():
                raise
    i = MAX_LEN
    while i < len(full_text):
        await context.bot.send_message(chat_id=chat_id, text=full_text[i:i+MAX_LEN])
        i += MAX_LEN

# --- Affiliate helpers ---
def _fmt_aff_line(item) -> str:
    return f"• {item.title}\n  {item.url}"

async def maybe_suggest_affiliates(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_msg = (update.message.text or "").lower()
    matches = find_matches(user_msg, max_items=2)
    presets = preset_for_scenario(user_msg)
    seen, suggestions = set(), []
    for it in (matches + presets):
        if it.url not in seen:
            suggestions.append(it); seen.add(it.url)
    if suggestions:
        blurb = "*Helpful gear for this topic:*\n" + "\n".join(_fmt_aff_line(it) for it in suggestions)
        await update.message.reply_text(blurb, disable_web_page_preview=False)

        try:
            user_id = update.effective_user.id if update.effective_user else 0
        except Exception:
            user_id = 0
        items_for_log = [{"title": it.title, "url": it.url} for it in suggestions]
        log_affiliate_impressions(
            user_id=user_id,
            message=update.message.text or "",
            category=None,
            items=items_for_log,
        )

# --- Subscriber storage ---
def _ensure_subs_parent():
    SUBSCRIBERS_FILE.parent.mkdir(parents=True, exist_ok=True)

def _load_subscribers() -> set[int]:
    try:
        raw = SUBSCRIBERS_FILE.read_text(encoding="utf-8")
        data = json.loads(raw)
        return set(int(x) for x in data)
    except Exception:
        return set()

def _save_subscribers(subs: set[int]) -> None:
    _ensure_subs_parent()
    SUBSCRIBERS_FILE.write_text(json.dumps(sorted(list(subs))), encoding="utf-8")

# --- Commands ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Webhook online ✅  Streaming model replies. Send me a message.")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/start – status\n"
        "/help – this help\n"
        "/buy <keywords> – quick affiliate links\n"
        "/subscribe – get daily videos sent here\n"
        "/unsubscribe – stop daily videos\n"
        "/debug_rag <query> – show top RAG chunks (dev only)\n"
        "/norag <prompt> – bypass RAG and query the model directly"
    )

async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    subs = _load_subscribers()
    cid = update.effective_chat.id
    if cid in subs:
        return await update.message.reply_text("You’re already subscribed to the daily video ✅")
    subs.add(cid)
    _save_subscribers(subs)
    await update.message.reply_text("Subscribed! I’ll send you the daily 30-sec survival video 📹")

async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    subs = _load_subscribers()
    cid = update.effective_chat.id
    if cid not in subs:
        return await update.message.reply_text("You’re not currently subscribed.")
    subs.remove(cid)
    _save_subscribers(subs)
    await update.message.reply_text("Unsubscribed. No more daily videos.")

async def buy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await update.message.reply_text("Usage: /buy <keywords>  e.g., /buy radio")
    q = " ".join(context.args)
    items = find_matches(q, max_items=3)
    if not items:
        return await update.message.reply_text("No matching items yet—try different keywords.")
    text = "*Suggested items:*\n" + "\n".join(_fmt_aff_line(it) for it in items)
    await update.message.reply_text(text, disable_web_page_preview=False)

# --- /debug_rag ---
async def debug_rag_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if BM25 is None or not DOCS:
        return await update.message.reply_text("RAG index not loaded. (BM25/DOCS is empty)")

    if not context.args:
        return await update.message.reply_text(
            "Usage: /debug_rag <query>\n\n"
            "Example: /debug_rag knife for hiking in the bush"
        )

    q = " ".join(context.args)
    results = bm25_retrieve(q, BM25, DOCS, top_k=5)

    if not results:
        return await update.message.reply_text(f"No results for query:\n\n{q}")

    lines = [f"*RAG debug for:* `{q}`"]
    for i, hit in enumerate(results, start=1):
        title = hit.get("title") or "(no title)"
        source = hit.get("source_path") or hit.get("id") or "?"
        score = hit.get("score", 0.0)
        text = hit.get("text", "")
        snippet = text[:350].replace("\n", " ")
        if len(text) > 350:
            snippet += " …"
        lines.append(
            f"\n*#{i}*  score={score:.3f}\n"
            f"*Title:* {title}\n"
            f"*Source:* `{source}`\n"
            f"*Snippet:* {snippet}"
        )

    reply = "\n".join(lines)
    if len(reply) > MAX_LEN:
        reply = reply[:MAX_LEN - 20] + "\n\n…(truncated)"

    await update.message.reply_text(reply, parse_mode="Markdown")

# --- /norag ---
async def norag_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await update.message.reply_text(
            "Usage: /norag <prompt>\n\n"
            "Example: /norag Explain the urban ember protocol"
        )

    user_text = " ".join(context.args)
    chat_id = update.effective_chat.id

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    msg = await update.message.reply_text("…")
    buffer = ""
    last_edit = 0.0
    last_text = "…"

    concise_sys_prompt = (
        "You are a helpful, concise assistant replying for a Telegram bot. "
        "Keep answers under ~180 words (≈900 characters) unless the user asks for details. "
        "Prefer short bullets for steps; avoid long stories."
    )

    start_ts = monotonic()

    try:
        async for chunk in stream_ollama(user_text, sys_prompt=concise_sys_prompt):
            buffer += chunk

            if len(buffer) >= MAX_TOTAL_CHARS:
                buffer = buffer[:MAX_TOTAL_CHARS] + "\n\n…(truncated for length)"
                break

            last_edit, last_text = await edit_throttled(
                context, chat_id, msg.message_id, buffer, last_edit, last_text
            )

        await send_final(context, chat_id, msg.message_id, buffer or "No response.", last_text)

        elapsed_ms = int((monotonic() - start_ts) * 1000)
        reply_len = len(buffer or "")
        try:
            user_id = update.effective_user.id if update.effective_user else 0
        except Exception:
            user_id = 0
        log_interaction(
            user_id=user_id,
            message=user_text or "",
            reply_len=reply_len,
            response_time_ms=elapsed_ms,
            category=categorize(user_text or ""),
            error=False,
            meta={"model": OLLAMA_MODEL, "mode": "norag"},
        )

    except Exception as e:
        logging.exception("/norag streaming error")
        try:
            await context.bot.edit_message_text(chat_id=chat_id, message_id=msg.message_id, text=f"Model error: {e}")
        except BadRequest:
            await context.bot.send_message(chat_id=chat_id, text=f"Model error: {e}")

        elapsed_ms = int((monotonic() - start_ts) * 1000)
        reply_len = len(buffer or "")
        try:
            user_id = update.effective_user.id if update.effective_user else 0
        except Exception:
            user_id = 0
        log_interaction(
            user_id=user_id,
            message=user_text or "",
            reply_len=reply_len,
            response_time_ms=elapsed_ms,
            category=categorize(user_text or ""),
            error=True,
            meta={"model": OLLAMA_MODEL, "mode": "norag", "exception": str(e)},
        )

# --- Text handler (Rex + RAG + optional reranker) ---
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    chat_id = update.effective_chat.id

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    msg = await update.message.reply_text("…")
    buffer = ""
    last_edit = 0.0
    last_text = "…"

    start_ts = monotonic()

    try:
        if BM25 is not None and DOCS:
            # 1) BM25 retrieve
            candidates = bm25_retrieve(user_text, BM25, DOCS, top_k=BM25_TOPK)
            best_score = candidates[0]["score"] if candidates else 0.0

            context_docs: List[Dict[str, Any]] = []
            if candidates and best_score >= MIN_BM25_SCORE:
                # 2) optional rerank
                if RERANKER_MODEL is not None:
                    ranked = rerank_candidates(user_text, candidates, top_k=max(MAX_CONTEXT_DOCS, RERANK_TOPN))
                else:
                    ranked = candidates
                context_docs = ranked[:MAX_CONTEXT_DOCS]

            context_chunks: List[str] = []
            for hit in context_docs:
                title = hit.get("title") or ""
                prefix = f"[{title}] " if title else ""
                context_chunks.append(prefix + (hit.get("text") or ""))
            context_block = "\n\n".join(context_chunks)

            survival_sys_prompt = (
                "You are Rex, an experienced survival and preparedness instructor talking to a beginner. "
                "Be concise, practical, and safety-focused. Use short paragraphs or bullets for steps. "
                "Prefer concrete, real-world advice about bushcraft, homesteading, emergency preparedness, "
                "medical and safety basics, and gear usage. If something is highly uncertain or depends on "
                "local regulations, say so explicitly."
            )

            if context_block:
                model_user_prompt = (
                    f"User question:\n{user_text}\n\n"
                    "Here are relevant excerpts from a survival knowledge base. "
                    "Use them when they clearly help answer the question:\n\n"
                    f"{context_block}\n\n"
                    "Now answer the question above as Rex, using the excerpts plus your own survival expertise. "
                    "Do NOT mention 'documents' or 'context'; just answer normally."
                )
            else:
                model_user_prompt = (
                    f"The user asked:\n{user_text}\n\n"
                    "Answer as Rex the survival instructor using your general survival knowledge. "
                    "If something is uncertain or risky, explain the tradeoffs and safety precautions."
                )

            async for chunk in stream_ollama(model_user_prompt, sys_prompt=survival_sys_prompt):
                buffer += chunk

                if len(buffer) >= MAX_TOTAL_CHARS:
                    buffer = buffer[:MAX_TOTAL_CHARS] + "\n\n…(truncated for length)"
                    break

                last_edit, last_text = await edit_throttled(
                    context, chat_id, msg.message_id, buffer, last_edit, last_text
                )

            mode = "rag_rerank" if RERANKER_MODEL is not None else "rag"

        else:
            concise_sys_prompt = (
                "You are a helpful, concise assistant replying for a Telegram bot. "
                "Keep answers under ~180 words (≈900 characters) unless the user asks for details. "
                "Prefer short bullets for steps; avoid long stories."
            )

            async for chunk in stream_ollama(user_text, sys_prompt=concise_sys_prompt):
                buffer += chunk

                if len(buffer) >= MAX_TOTAL_CHARS:
                    buffer = buffer[:MAX_TOTAL_CHARS] + "\n\n…(truncated for length)"
                    break

                last_edit, last_text = await edit_throttled(
                    context, chat_id, msg.message_id, buffer, last_edit, last_text
                )

            mode = "fallback"

        await send_final(context, chat_id, msg.message_id, buffer or "No response.", last_text)

        await maybe_suggest_affiliates(update, context)

        elapsed_ms = int((monotonic() - start_ts) * 1000)
        reply_len = len(buffer or "")
        try:
            user_id = update.effective_user.id if update.effective_user else 0
        except Exception:
            user_id = 0
        log_interaction(
            user_id=user_id,
            message=user_text or "",
            reply_len=reply_len,
            response_time_ms=elapsed_ms,
            category=categorize(user_text or ""),
            error=False,
            meta={"model": OLLAMA_MODEL, "mode": mode},
        )

    except Exception as e:
        logging.exception("Streaming error")
        try:
            await context.bot.edit_message_text(chat_id=chat_id, message_id=msg.message_id, text=f"Model error: {e}")
        except BadRequest:
            await context.bot.send_message(chat_id=chat_id, text=f"Model error: {e}")

        elapsed_ms = int((monotonic() - start_ts) * 1000)
        reply_len = len(buffer or "")
        try:
            user_id = update.effective_user.id if update.effective_user else 0
        except Exception:
            user_id = 0
        log_interaction(
            user_id=user_id,
            message=user_text or "",
            reply_len=reply_len,
            response_time_ms=elapsed_ms,
            category=categorize(user_text or ""),
            error=True,
            meta={"model": OLLAMA_MODEL, "mode": "rag", "exception": str(e)},
        )

# Register Telegram handlers
tg_app.add_handler(CommandHandler("start", start))
tg_app.add_handler(CommandHandler("help", help_cmd))
tg_app.add_handler(CommandHandler("subscribe", subscribe))
tg_app.add_handler(CommandHandler("unsubscribe", unsubscribe))
tg_app.add_handler(CommandHandler("buy", buy_cmd))
tg_app.add_handler(CommandHandler("debug_rag", debug_rag_cmd))
tg_app.add_handler(CommandHandler("norag", norag_cmd))
tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

# --- FastAPI app + lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        log_system(level="info", msg="bot_start", meta={"model": OLLAMA_MODEL})
    except Exception:
        pass
    await tg_app.initialize()
    yield
    await tg_app.shutdown()

app = FastAPI(lifespan=lifespan)

# Attach web UI + /api/rex
register_web_routes(
    app,
    ctx={
        "stream_ollama": stream_ollama,
        "BM25": BM25,
        "DOCS": DOCS,
        "bm25_retrieve": bm25_retrieve,
        "MAX_TOTAL_CHARS": MAX_TOTAL_CHARS,
        "OLLAMA_MODEL": OLLAMA_MODEL,
        "MIN_BM25_SCORE": MIN_BM25_SCORE,
        "MAX_CONTEXT_DOCS": MAX_CONTEXT_DOCS,
        "categorize": categorize,
        "log_interaction": log_interaction,
        "BM25_TOPK": BM25_TOPK,
        "RERANK_TOPN": RERANK_TOPN,
        "rerank_candidates": rerank_candidates if RERANKER_MODEL is not None else None,
    },
)

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.post(f"/telegram/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    if WEBHOOK_SECRET and request.headers.get("X-Telegram-Bot-Api-Secret-Token") != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="bad secret")
    update = Update.de_json(await request.json(), tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}

