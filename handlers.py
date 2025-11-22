# handlers.py
import logging
from typing import Set, Tuple

from telegram import Update
from telegram.constants import ChatAction
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from time import monotonic

# --- imports from your other modules / config ---
# Adjust these imports to match your actual structure:
from rex_core import BM25, DOCS, bm25_retrieve, stream_ollama
from analytics_logger import log_interaction, categorize, log_affiliate_impressions
from affiliate_catalog import preset_for_scenario, find_matches
from retriever_bm25 import retrieve as bm25_retrieve

from rex_core import (
    BM25,
    DOCS,
    MIN_BM25_SCORE,
    MAX_CONTEXT_DOCS,
    MAX_TOTAL_CHARS,
    OLLAMA_MODEL,
    stream_ollama,
)

from chat_utils import (
    edit_throttled,
    send_final,
)

# Track which messages we've already handled
PROCESSED_MESSAGES: Set[Tuple[int, int]] = set()


# --- Text handler (Rex + RAG) ---
# --- Text handler (Rex + RAG) ---
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Ignore edited messages – Telegram Desktop often sends these
    if update.edited_message is not None:
        logging.info("Ignoring edited_message in on_text: %s", update.update_id)
        return

    message = update.effective_message
    if not message or not message.text:
        return

    chat_id = message.chat_id
    message_id = message.message_id
    user_text = message.text.strip()

    logging.info(
        "on_text received chat_id=%s message_id=%s text=%r",
        chat_id,
        message_id,
        user_text,
    )

    # Hard dedupe: only handle each (chat_id, message_id) once
    key = (chat_id, message_id)
    if key in PROCESSED_MESSAGES:
        logging.info("Already processed message %s/%s, skipping", chat_id, message_id)
        return
    PROCESSED_MESSAGES.add(key)

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    msg = await message.reply_text("…")
    buffer = ""
    last_edit = 0.0
    last_text = "…"

    start_ts = monotonic()

    try:
        if BM25 is not None and DOCS:
            results = bm25_retrieve(user_text, BM25, DOCS, top_k=8)
            best_score = results[0]["score"] if results else 0.0

            context_chunks = []
            if results and best_score >= MIN_BM25_SCORE:
                for hit in results[:MAX_CONTEXT_DOCS]:
                    title = hit.get("title") or ""
                    prefix = f"[{title}] " if title else ""
                    context_chunks.append(prefix + hit["text"])
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

            mode = "rag"

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

