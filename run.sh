#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export BRIDGE_PORT="${BRIDGE_PORT:-12345}"
export BRIDGE_HOST="${BRIDGE_HOST:-127.0.0.1}"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "[bridge] $PY not found. Install Python 3.10+ first." >&2
    exit 1
fi

if ! "$PY" -c "import fastapi, uvicorn, httpx" >/dev/null 2>&1; then
    echo "[bridge] Installing dependencies..."
    "$PY" -m pip install --user -r requirements.txt \
        || "$PY" -m pip install --user --break-system-packages -r requirements.txt
fi

exec "$PY" main.py
