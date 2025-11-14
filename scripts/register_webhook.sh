#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root & .env
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"

# Read config (allow env overrides or .env values)
get_env () { grep -E "^$1=" "$ENV_FILE" 2>/dev/null | cut -d= -f2- | tr -d '"'; }

mkdir -p "$REPO_ROOT"
touch "$ENV_FILE"

TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-$(get_env TELEGRAM_BOT_TOKEN)}"
PUBLIC_URL="${PUBLIC_URL:-$(get_env PUBLIC_URL)}"
BOT_PATH="${BOT_PATH:-/telegram/webhook}"

if [[ -z "${TELEGRAM_BOT_TOKEN:-}" ]]; then
  echo "TELEGRAM_BOT_TOKEN is empty. Put it in $ENV_FILE and re-run." >&2
  exit 1
fi
if [[ -z "${PUBLIC_URL:-}" ]]; then
  echo "PUBLIC_URL is empty. Start ngrok (or use your domain), set PUBLIC_URL in $ENV_FILE, and re-run." >&2
  exit 1
fi

CUR_SECRET="$(get_env WEBHOOK_SECRET || true)"
if [[ -z "${CUR_SECRET:-}" ]]; then
  echo "WEBHOOK_SECRET is empty — generating new secret..."
  NEW_SECRET="$(openssl rand -base64 48 | tr -d '\n' | tr '/+' '_-')"
  # Portable edit using Python
  python3 - "$ENV_FILE" "$NEW_SECRET" <<'PY'
import os, re, sys
env_path = sys.argv[1]
new_secret = sys.argv[2]
try:
    with open(env_path, "r", encoding="utf-8") as f:
        data = f.read()
except FileNotFoundError:
    data = ""
if re.search(r"^WEBHOOK_SECRET=", data, flags=re.M):
    data = re.sub(r"^WEBHOOK_SECRET=.*", f'WEBHOOK_SECRET="{new_secret}"', data, flags=re.M)
else:
    if data and not data.endswith("\n"):
        data += "\n"
    data += f'WEBHOOK_SECRET="{new_secret}"\n'
with open(env_path, "w", encoding="utf-8") as f:
    f.write(data)
print(f"Updated {env_path}")
PY
  WEBHOOK_SECRET="$NEW_SECRET"
else
  WEBHOOK_SECRET="$CUR_SECRET"
fi

WEBHOOK_URL="${PUBLIC_URL%/}${BOT_PATH}?secret=${WEBHOOK_SECRET}"
echo "Registering webhook to: $WEBHOOK_URL"

curl -sS "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/setWebhook" \
  -d "url=${WEBHOOK_URL}" \
  -d "allowed_updates[]=message" \
  -d "allowed_updates[]=callback_query" | jq .

echo "Done."

