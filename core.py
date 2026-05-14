# -*- coding: utf-8 -*-
"""Core infrastructure: config, shared HTTP client, SSE, errors, OAuth helpers.

Shared across all providers (Codex / Google / Anthropic). Keep this module
import-light so providers can pull only what they need.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional
from urllib.parse import parse_qs, urlparse

import httpx

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ACCOUNTS_DIR = DATA_DIR / "accounts"
LOG_FILE = DATA_DIR / "bridge.log"
POOL_STATE_FILE = DATA_DIR / "pool_state.json"
ROUTE_GROUPS_FILE = DATA_DIR / "route_groups.json"
PENDING_AUTH_FILE = DATA_DIR / "pending_auth.json"
PID_FILE = DATA_DIR / "bridge.pid"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ACCOUNTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config (env-based; keep it simple, no pydantic dep)
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on", "y", "t"}


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_list(name: str, default: Optional[list[str]] = None) -> list[str]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return list(default or [])
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    host: str = _env_str("BRIDGE_HOST", _env_str("OPENAI_BRIDGE_HOST", "127.0.0.1"))
    port: int = _env_int("BRIDGE_PORT", _env_int("OPENAI_BRIDGE_PORT", 12345))
    api_key: str = _env_str("BRIDGE_API_KEY", "")  # optional auth for bridge itself
    locale: str = _env_str("BRIDGE_LOCALE", "vi")  # vi | en
    max_failover_attempts: int = _env_int("BRIDGE_MAX_FAILOVER", 3)
    pool_strategy: str = _env_str("BRIDGE_POOL_STRATEGY", "least_load")
    sse_keepalive_seconds: int = _env_int("BRIDGE_SSE_KEEPALIVE", 15)
    upstream_max_attempts: int = _env_int("BRIDGE_UPSTREAM_RETRIES", 3)
    log_max_bytes: int = _env_int("BRIDGE_LOG_MAX_BYTES", 10 * 1024 * 1024)
    log_backup_count: int = _env_int("BRIDGE_LOG_BACKUP_COUNT", 5)
    enable_cors: bool = _env_bool("BRIDGE_ENABLE_CORS", True)
    cross_provider_fallback: list[str] = field(
        default_factory=lambda: _env_list("BRIDGE_CROSS_PROVIDER_FALLBACK", [])
    )

    # Codex specifics (kept for backward compat with existing env var names)
    codex_default_model: str = _env_str("OPENAI_CODEX_DEFAULT_MODEL", "gpt-5.5")
    codex_default_instructions: str = _env_str(
        "OPENAI_CODEX_DEFAULT_INSTRUCTIONS",
        "You are ChatGPT, a helpful assistant.",
    )
    codex_inject_default_instructions: bool = _env_bool(
        "OPENAI_CODEX_INJECT_DEFAULT_INSTRUCTIONS", True
    )
    codex_reasoning_effort: str = _env_str("OPENAI_CODEX_REASONING_EFFORT", "medium")
    codex_verbosity: str = _env_str("OPENAI_CODEX_VERBOSITY", "medium")
    codex_reasoning_summary: str = _env_str("OPENAI_CODEX_REASONING_SUMMARY", "")
    codex_passthrough_sampling: bool = _env_bool("OPENAI_CODEX_PASSTHROUGH_SAMPLING", False)
    codex_default_store: bool = _env_bool("OPENAI_CODEX_DEFAULT_STORE", False)
    codex_default_include: list[str] = field(
        default_factory=lambda: _env_list("OPENAI_CODEX_DEFAULT_INCLUDE", [])
    )
    codex_audio_model: str = _env_str("OPENAI_CODEX_AUDIO_MODEL", "gpt-5.4")
    codex_video_frames: int = _env_int("OPENAI_CODEX_VIDEO_FRAMES", 6)
    codex_video_fps: str = _env_str("OPENAI_CODEX_VIDEO_FPS", "1/3")
    codex_video_max_width: int = _env_int("OPENAI_CODEX_VIDEO_MAX_WIDTH", 1280)
    codex_originator: str = _env_str("OPENAI_CODEX_ORIGINATOR", "codex_cli_rs")
    codex_beta: str = _env_str("OPENAI_CODEX_BETA", "")
    codex_user_agent: str = _env_str("OPENAI_CODEX_USER_AGENT", "")
    codex_rate_limit_warn_remaining: int = _env_int(
        "OPENAI_CODEX_RATE_LIMIT_WARN_REMAINING", 5
    )
    codex_rate_limit_warn_percent: float = _env_float(
        "OPENAI_CODEX_RATE_LIMIT_WARN_PERCENT", 15.0
    )

    # Google specifics
    google_default_model: str = _env_str("GOOGLE_GEMINI_DEFAULT_MODEL", "auto-gemini-3")
    google_code_assist_project: str = _env_str(
        "GOOGLE_CODE_ASSIST_PROJECT", _env_str("GOOGLE_CLOUD_PROJECT", "")
    )
    google_ignore_server_project: bool = _env_bool(
        "GOOGLE_CODE_ASSIST_IGNORE_SERVER_PROJECT", False
    )
    google_skip_load_code_assist: bool = _env_bool(
        "GOOGLE_CODE_ASSIST_SKIP_LOAD", False
    )
    google_use_user_project_header: bool = _env_bool(
        "GOOGLE_CODE_ASSIST_USER_PROJECT_HEADER", False
    )
    google_oauth_prompt: str = _env_str("GOOGLE_OAUTH_PROMPT", "consent")
    google_user_agent: str = _env_str("GOOGLE_GEMINI_USER_AGENT", "google-gemini-cli")

    # Anthropic specifics
    anthropic_default_model: str = _env_str("ANTHROPIC_DEFAULT_MODEL", "claude-sonnet-4-6")
    anthropic_api_key: str = _env_str("ANTHROPIC_API_KEY", "")
    anthropic_beta: str = _env_str("ANTHROPIC_BETA", "")
    anthropic_oauth_beta: str = _env_str("ANTHROPIC_OAUTH_BETA", "oauth-2025-04-20")
    anthropic_allow_legacy_oauth: bool = _env_bool("ANTHROPIC_ALLOW_LEGACY_OAUTH", False)

    # DeepSeek specifics
    deepseek_default_model: str = _env_str("DEEPSEEK_DEFAULT_MODEL", "deepseek-v4-flash")
    deepseek_api_key: str = _env_str("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = _env_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com")


SETTINGS = Settings()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging() -> logging.Logger:
    """Configure root logging once with rotating file + stdout."""
    root = logging.getLogger()
    if any(getattr(h, "_bridge_setup", False) for h in root.handlers):
        return logging.getLogger("bridge")

    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=SETTINGS.log_max_bytes,
        backupCount=SETTINGS.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    file_handler._bridge_setup = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    stream_handler._bridge_setup = True  # type: ignore[attr-defined]
    root.addHandler(stream_handler)

    return logging.getLogger("bridge")


logger = setup_logging()

# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class BridgeError(Exception):
    """Base error."""


class QuotaExhausted(BridgeError):
    """Upstream returned 429 / usage_limit_reached for an account."""

    def __init__(
        self,
        message: str,
        status_code: int = 429,
        resets_at: float = 0.0,
        reason: str = "",
        raw_body: str = "",
    ):
        super().__init__(message)
        self.status_code = status_code
        self.resets_at = resets_at
        self.reason = reason
        self.raw_body = raw_body


class AuthRevoked(BridgeError):
    """Upstream returned 401 / refresh token rejected."""

    def __init__(self, message: str, status_code: int = 401, raw_body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.raw_body = raw_body


class UpstreamServerError(BridgeError):
    """5xx or network-level failure. Failover may help (different account)."""

    def __init__(self, message: str, status_code: int = 500, raw_body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.raw_body = raw_body


class UpstreamClientError(BridgeError):
    """4xx that is not quota/auth. Surface to client."""

    def __init__(self, message: str, status_code: int = 400, raw_body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.raw_body = raw_body


class NoAccountAvailable(BridgeError):
    """Pool has no healthy account for the requested provider."""

    def __init__(self, provider: str, tried: Optional[list[str]] = None):
        super().__init__(
            f"No healthy account for provider '{provider}'. Tried: {tried or []}"
        )
        self.provider = provider
        self.tried = tried or []


# ---------------------------------------------------------------------------
# Shared httpx.AsyncClient
# ---------------------------------------------------------------------------


_http_client: Optional[httpx.AsyncClient] = None


async def get_http_client() -> httpx.AsyncClient:
    """Return process-wide shared AsyncClient. Lazy-initialized."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = _build_http_client()
    return _http_client


def _build_http_client() -> httpx.AsyncClient:
    timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=5.0)
    limits = httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=30.0,
    )
    # http2 requires h2 package; fall back if missing.
    try:
        import h2  # noqa: F401
        http2 = True
    except Exception:
        http2 = False
    return httpx.AsyncClient(
        http2=http2,
        timeout=timeout,
        limits=limits,
        follow_redirects=False,
        headers={"Accept": "application/json, text/event-stream"},
    )


async def close_http_client() -> None:
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        try:
            await _http_client.aclose()
        except Exception:
            pass
    _http_client = None


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def sse_data(payload: Dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def sse_done() -> bytes:
    return b"data: [DONE]\n\n"


def sse_comment(text: str = "keepalive") -> bytes:
    return f": {text}\n\n".encode("utf-8")


async def sse_with_keepalive(
    producer: Callable[[], "AsyncIterableBytes"],
    interval: float,
) -> "AsyncGeneratorBytes":
    """Wrap an SSE producer so idle gaps >= `interval` emit a comment line.

    Useful behind Nginx/Cloudflare which kill idle streams ~60s.
    """
    import asyncio as _asyncio

    queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()

    async def _pump() -> None:
        try:
            async for chunk in producer():
                await queue.put(chunk)
        except Exception as exc:
            logger.exception("SSE producer crashed: %s", exc)
        finally:
            await queue.put(None)

    task = _asyncio.create_task(_pump())
    try:
        while True:
            try:
                item = await _asyncio.wait_for(queue.get(), timeout=interval)
            except _asyncio.TimeoutError:
                yield sse_comment()
                continue
            if item is None:
                return
            yield item
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except Exception:
                pass


# Just for type hints — not strictly enforced
AsyncIterableBytes = Any
AsyncGeneratorBytes = Any


# ---------------------------------------------------------------------------
# OAuth PKCE helpers
# ---------------------------------------------------------------------------


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def generate_pkce_pair() -> tuple[str, str]:
    verifier = b64url(os.urandom(32))
    challenge = b64url(hashlib.sha256(verifier.encode("utf-8")).digest())
    return verifier, challenge


def generate_state() -> str:
    return b64url(os.urandom(16))


def decode_jwt_payload(token: str) -> Dict[str, Any]:
    if not token or "." not in token:
        return {}
    try:
        _, payload, _ = token.split(".", 2)
        payload += "=" * ((4 - len(payload) % 4) % 4)
        return json.loads(base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8"))
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# OAuth callback HTTP server (shared across all providers)
# ---------------------------------------------------------------------------

CALLBACK_PORT_START = 1455
CALLBACK_PORT_END = 1554
PENDING_AUTH_TTL = 600


def find_free_port() -> int:
    for port in range(CALLBACK_PORT_START, CALLBACK_PORT_END + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        f"No free callback port in range {CALLBACK_PORT_START}-{CALLBACK_PORT_END}"
    )


@dataclass
class PendingAuth:
    state: str
    slot_id: str
    provider: str
    verifier: str
    redirect_uri: str
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "provider": self.provider,
            "verifier": self.verifier,
            "redirect_uri": self.redirect_uri,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, state: str, d: Dict[str, Any]) -> "PendingAuth":
        return cls(
            state=state,
            slot_id=d.get("slot_id", ""),
            provider=d.get("provider", ""),
            verifier=d.get("verifier", ""),
            redirect_uri=d.get("redirect_uri", ""),
            created_at=float(d.get("created_at") or time.time()),
        )


class CallbackBroker:
    """Thread-safe registry of pending OAuth flows + the local callback server.

    Each provider has its own callback path (e.g. /auth/callback for Codex,
    /oauth2callback for Google, /claude/callback for Anthropic). The single
    HTTP server routes by path and delegates handling to the registered
    `process_callback` function for the matching provider.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._server: Optional[ThreadingHTTPServer] = None
        self._port: Optional[int] = None
        self._pending: Dict[str, PendingAuth] = {}
        self._handlers: Dict[str, Callable[[str, Dict[str, Any], PendingAuth], Awaitable[bool]]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._load_pending()

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # -- pending state ------------------------------------------------------

    def add_pending(self, pa: PendingAuth) -> None:
        with self._lock:
            self._pending[pa.state] = pa
            self._save_pending()

    def pop_pending(self, state: str) -> Optional[PendingAuth]:
        with self._lock:
            pa = self._pending.pop(state, None)
            if pa is not None:
                self._save_pending()
            return pa

    def _save_pending(self) -> None:
        try:
            data = {s: pa.to_dict() for s, pa in self._pending.items()}
            PENDING_AUTH_FILE.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            logger.exception("Failed to save pending OAuth state")

    def _load_pending(self) -> None:
        if not PENDING_AUTH_FILE.exists():
            return
        try:
            raw = json.loads(PENDING_AUTH_FILE.read_text(encoding="utf-8"))
            cutoff = time.time() - PENDING_AUTH_TTL
            if isinstance(raw, dict):
                for state, d in raw.items():
                    if isinstance(d, dict):
                        pa = PendingAuth.from_dict(state, d)
                        if pa.created_at >= cutoff:
                            self._pending[state] = pa
        except Exception:
            logger.exception("Failed to load pending OAuth state")

    # -- handler registration ----------------------------------------------

    def register_handler(
        self,
        path: str,
        handler: Callable[[str, Dict[str, Any], PendingAuth], Awaitable[bool]],
    ) -> None:
        """Register an async coroutine that handles an OAuth callback for `path`.

        Handler receives (code, query_params, pending_auth) and returns True on
        success. It runs on the main asyncio loop via run_coroutine_threadsafe.
        """
        with self._lock:
            self._handlers[path] = handler

    # -- HTTP server --------------------------------------------------------

    def ensure_server(self) -> int:
        with self._lock:
            if self._server is not None and self._port is not None:
                return self._port
            port = find_free_port()
            broker = self

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):  # noqa: N802
                    parsed = urlparse(self.path)
                    handler = broker._handlers.get(parsed.path)
                    if handler is None:
                        self.send_response(404)
                        self.end_headers()
                        self.wfile.write(b"Not Found")
                        return

                    query = {k: v[0] if v else "" for k, v in parse_qs(parsed.query).items()}
                    state = query.get("state", "")
                    code = query.get("code", "")
                    err = query.get("error", "")
                    err_desc = query.get("error_description", "")

                    pa = broker.pop_pending(state) if state else None
                    success = False
                    detail = ""
                    if err:
                        detail = f"{err} {err_desc}".strip()
                    elif not code or not state or pa is None:
                        detail = "Missing or invalid OAuth callback parameters"
                    else:
                        if broker._loop is not None:
                            future = asyncio.run_coroutine_threadsafe(
                                handler(code, query, pa), broker._loop
                            )
                            try:
                                success = bool(future.result(timeout=30))
                            except Exception as exc:
                                detail = f"Callback handler failed: {exc}"
                                logger.exception("Callback handler raised")
                        else:
                            detail = "Server loop not ready"

                    html = _render_callback_html(success, detail, pa.slot_id if pa else "")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(html.encode("utf-8"))

                def log_message(self, format, *args):  # noqa: A002
                    logger.info("oauth-callback " + format, *args)

            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self._server = server
            self._port = port
            logger.info("OAuth callback server on http://127.0.0.1:%s", port)
            return port

    def shutdown(self) -> None:
        with self._lock:
            if self._server is not None:
                try:
                    self._server.shutdown()
                    self._server.server_close()
                except Exception:
                    pass
                self._server = None
                self._port = None


def _render_callback_html(success: bool, detail: str, slot_id: str) -> str:
    title = "Signed in" if success else "Authentication failed"
    body = (
        f"<h1>{title}</h1>"
        f"<p>Slot: <code>{slot_id or '-'}</code></p>"
        f"<p>{detail or 'You may close this page.'}</p>"
    )
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title></head>"
        f"<body style='font-family:system-ui;padding:40px;background:#0a1f52;color:#fff'>{body}</body></html>"
    )


CALLBACK_BROKER = CallbackBroker()


# ---------------------------------------------------------------------------
# Misc utilities
# ---------------------------------------------------------------------------


def nested_get(data: Any, *path: str) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def safe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return None
