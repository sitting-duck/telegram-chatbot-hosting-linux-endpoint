# main.py - web server + routing logic for Telegram chatbot
import json
import logging

from fastapi import FastAPI, Request, HTTPException

from telegram import Update
from telegram.ext import Application, MessageHandler, filters
from telegram.request import HTTPXRequest

from handlers import on_text
from rex_core import BOT_TOKEN, WEBHOOK_SECRET, stream_ollama, BM25, DOCS
from retriever_bm25 import retrieve as bm25_retrieve
from analytics_logger import categorize, log_interaction
from web_rex import register_web_routes  # web UI / API routes

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# --- Telegram app with slightly higher timeouts ---
request = HTTPXRequest(
    connect_timeout=15.0,
    read_timeout=30.0,
    write_timeout=30.0,
    pool_timeout=30.0,
)

tg_app = Application.builder().token(BOT_TOKEN).request(request).build()

# register handlers
tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
# if you have /start, /help, etc., import and add them here:
# from handlers import on_start
# tg_app.add_handler(CommandHandler("start", on_start))


# --- Register web routes (HTML/API endpoints for ngrok site) ---
# web_rex.register_web_routes expects a context dict with several keys
ctx = {
    "stream_ollama": stream_ollama,
    "bm25_retrieve": bm25_retrieve,
    "BM25": BM25,
    "DOCS": DOCS,
    "categorize": categorize,
    "log_interaction": log_interaction,
}
register_web_routes(app, ctx)


@app.on_event("startup")
async def startup_event():
    logging.info("Starting up, initializing Telegram app")
    await tg_app.initialize()
    logging.info("Telegram app initialized")


@app.on_event("shutdown")
async def shutdown_event():
    logging.info("Shutting down Telegram app")
    await tg_app.shutdown()


@app.post(f"/telegram/{BOT_TOKEN}")
async def telegram_webhook(request: Request):
    # Optional secret check
    if WEBHOOK_SECRET and request.headers.get("X-Telegram-Bot-Api-Secret-Token") != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="bad secret")

    data = await request.json()
    logging.info("Telegram webhook payload: %s", json.dumps(data, ensure_ascii=False))

    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}


# --- Affiliate helpers (if you still want this here) ---
def _fmt_aff_line(item) -> str:
    return f"• {item.title}\n  {item.url}"

