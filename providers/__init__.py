# -*- coding: utf-8 -*-
"""Provider implementations.

Each provider exposes:
- A `register(pool, app)` function that wires the provider into the
  AccountPool + FastAPI app (adds the token-store factory, OAuth routes).
- `Executor`-style functions taking `(chat_request, account) -> ...` for
  the unified bridge to call.
- A `MODELS` list for `/v1/models` aggregation.

Providers MUST translate upstream-specific errors into the core error
hierarchy (`QuotaExhausted`, `AuthRevoked`, `UpstreamServerError`,
`UpstreamClientError`) so the unified failover loop can react.
"""
from . import codex, google, anthropic  # noqa: F401

__all__ = ["codex", "google", "anthropic"]
