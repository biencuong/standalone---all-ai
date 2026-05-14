# -*- coding: utf-8 -*-
"""Entry point for the Multi-Provider OAuth Bridge.

Run:
    python main.py            # foreground
    pythonw main.py           # hidden (Windows)

Env (most useful):
    BRIDGE_HOST=127.0.0.1            bind host
    BRIDGE_PORT=12345                bind port
    BRIDGE_API_KEY=                  optional bearer auth for the bridge itself
    BRIDGE_POOL_STRATEGY=least_load  least_load | round_robin | random
    BRIDGE_MAX_FAILOVER=3            max accounts to try per request
    BRIDGE_LOCALE=vi                 vi | en (error messages)
    BRIDGE_CROSS_PROVIDER_FALLBACK=  comma-separated providers to try after primary
                                     (e.g. "claude,gemini" when chatgpt pool exhausted)

See README.md for the full env reference per provider.
"""
from __future__ import annotations

import atexit
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import uvicorn

from core import PID_FILE, SETTINGS, logger
from bridge import app


def _write_pid() -> None:
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()), encoding="ascii")
    atexit.register(_clear_pid)


def _clear_pid() -> None:
    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
    except OSError:
        pass


def main() -> None:
    _write_pid()
    logger.info(
        "Starting Multi-Provider OAuth Bridge on http://%s:%s",
        SETTINGS.host, SETTINGS.port,
    )
    logger.info("UI:                 http://%s:%s/", SETTINGS.host, SETTINGS.port)
    logger.info("Chat completions:   http://%s:%s/v1/chat/completions", SETTINGS.host, SETTINGS.port)
    logger.info("Models:             http://%s:%s/v1/models", SETTINGS.host, SETTINGS.port)
    logger.info("OAuth token:        http://%s:%s/v1/oauth/token", SETTINGS.host, SETTINGS.port)
    uvicorn.run(app, host=SETTINGS.host, port=SETTINGS.port, log_level="info")


if __name__ == "__main__":
    main()
