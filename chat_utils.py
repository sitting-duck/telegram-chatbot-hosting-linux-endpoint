import logging
from time import monotonic
from telegram.error import BadRequest
from telegram.ext import ContextTypes

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
