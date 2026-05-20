# -*- coding: utf-8 -*-
"""Google Gemini provider (via Code Assist / gemini-cli OAuth).

OAuth flow with accounts.google.com, upstream at cloudcode-pa.googleapis.com
/v1internal:{generateContent|streamGenerateContent}. Maps OpenAI Chat
Completions <-> Code Assist GenerateContentRequest.
"""
from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import platform
import re
import sys
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import httpx

from core import (
    AuthRevoked,
    CALLBACK_BROKER,
    PendingAuth,
    QuotaExhausted,
    SETTINGS,
    UpstreamClientError,
    UpstreamServerError,
    decode_jwt_payload,
    generate_state,
    get_http_client,
    logger,
    safe_int,
    sse_comment,
    sse_data,
    sse_done,
)

PROVIDER = "google"

CLIENT_ID_ENV_KEYS = (
    "GOOGLE_OAUTH_CLIENT_ID",
    "GEMINI_CLI_OAUTH_CLIENT_ID",
    "GOOGLE_GEMINI_OAUTH_CLIENT_ID",
)
CLIENT_SECRET_ENV_KEYS = (
    "GOOGLE_OAUTH_CLIENT_SECRET",
    "GEMINI_CLI_OAUTH_CLIENT_SECRET",
    "GOOGLE_GEMINI_OAUTH_CLIENT_SECRET",
)
AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
USERINFO_ENDPOINT = "https://www.googleapis.com/oauth2/v2/userinfo"
CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com"
CODE_ASSIST_API_VERSION = "v1internal"
CALLBACK_PATH = "/oauth2callback"
SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]
REFRESH_SAFETY_WINDOW = 60
_capacity_wait_env = safe_int(os.getenv("GOOGLE_GEMINI_CAPACITY_WAIT_MAX_SECONDS"))
CAPACITY_AUTO_WAIT_MAX_SECONDS = max(0, _capacity_wait_env if _capacity_wait_env is not None else 75)


_GEMINI_CLI_OAUTH_CLIENT_CACHE: Optional[Tuple[str, str]] = None


def _first_env(keys: Tuple[str, ...]) -> str:
    for key in keys:
        value = (os.getenv(key) or "").strip()
        if value:
            return value
    return ""


def _gemini_cli_bundle_dirs() -> List[Path]:
    dirs: List[Path] = []
    package_dir = (os.getenv("GEMINI_CLI_PACKAGE_DIR") or "").strip()
    if package_dir:
        root = Path(package_dir)
        dirs.extend([root / "bundle", root])

    appdata = (os.getenv("APPDATA") or "").strip()
    if appdata:
        dirs.append(Path(appdata) / "npm" / "node_modules" / "@google" / "gemini-cli" / "bundle")

    home = Path.home()
    dirs.extend([
        home / "AppData" / "Roaming" / "npm" / "node_modules" / "@google" / "gemini-cli" / "bundle",
        home / ".npm-global" / "lib" / "node_modules" / "@google" / "gemini-cli" / "bundle",
        Path("/usr/local/lib/node_modules/@google/gemini-cli/bundle"),
        Path("/opt/homebrew/lib/node_modules/@google/gemini-cli/bundle"),
    ])

    seen: set[str] = set()
    out: List[Path] = []
    for d in dirs:
        key = str(d)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _extract_gemini_cli_oauth_client(path: Path) -> Optional[Tuple[str, str]]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    if "OAUTH_CLIENT_ID" not in text or "OAUTH_CLIENT_SECRET" not in text:
        return None
    client_id = re.search(r'OAUTH_CLIENT_ID\s*=\s*["\']([^"\']+)["\']', text)
    client_secret = re.search(r'OAUTH_CLIENT_SECRET\s*=\s*["\']([^"\']+)["\']', text)
    if not client_id or not client_secret:
        return None
    return client_id.group(1), client_secret.group(1)


def _gemini_cli_oauth_client() -> Optional[Tuple[str, str]]:
    global _GEMINI_CLI_OAUTH_CLIENT_CACHE
    if _GEMINI_CLI_OAUTH_CLIENT_CACHE:
        return _GEMINI_CLI_OAUTH_CLIENT_CACHE

    for bundle_dir in _gemini_cli_bundle_dirs():
        if not bundle_dir.exists():
            continue
        for path in bundle_dir.glob("chunk-*.js"):
            found = _extract_gemini_cli_oauth_client(path)
            if found:
                _GEMINI_CLI_OAUTH_CLIENT_CACHE = found
                logger.info("Using OAuth client from installed Gemini CLI package")
                return found
    return None


def _oauth_client() -> Tuple[str, str]:
    client_id = _first_env(CLIENT_ID_ENV_KEYS)
    client_secret = _first_env(CLIENT_SECRET_ENV_KEYS)
    if not client_id or not client_secret:
        found = _gemini_cli_oauth_client()
        if found:
            client_id = client_id or found[0]
            client_secret = client_secret or found[1]
    if not client_id or not client_secret:
        raise UpstreamClientError(
            "Google OAuth client is not configured. Set GOOGLE_OAUTH_CLIENT_ID "
            "and GOOGLE_OAUTH_CLIENT_SECRET, or install Gemini CLI on this machine.",
            status_code=500,
        )
    return client_id, client_secret

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

DEFAULT_GEMINI_MODEL = "gemini-3auto"

GEMINI_MODEL_CATALOG: List[Dict[str, Any]] = [
    {"id": "gemini-3auto", "name": "Gemini 3 Auto",
     "description": "Current working Gemini 3 route that avoids unavailable 3.5 models."},
    {"id": "auto-gemini-3.5", "name": "Auto Gemini 3.5",
     "description": "Gemini 3.5 router default."},
    {"id": "auto-gemini-3", "name": "Auto Gemini 3",
     "description": "Gemini CLI router default."},
    {"id": "auto-gemini-2.5", "name": "Auto Gemini 2.5",
     "description": "Gemini CLI 2.5 router default."},
    {"id": "gemini-3.5-flash", "name": "Gemini 3.5 Flash",
     "description": "Fast Gemini 3.5 model."},
    {"id": "gemini-3.5-flash-lite", "name": "Gemini 3.5 Flash Lite",
     "description": "Low latency Gemini 3.5 Flash Lite model."},
    {"id": "gemini-3-pro-preview", "name": "Gemini 3 Pro Preview",
     "description": "Current Gemini Pro preview model."},
    {"id": "gemini-3.1-pro-preview", "name": "Gemini 3.1 Pro Preview",
     "description": "Gemini Pro preview variant."},
    {"id": "gemini-3.1-flash-preview", "name": "Gemini 3.1 Flash Preview",
     "description": "Gemini Flash preview variant."},
    {"id": "gemini-3-flash-preview", "name": "Gemini 3 Flash Preview",
     "description": "Current Gemini Flash preview model."},
    {"id": "gemini-3.1-flash-lite-preview", "name": "Gemini 3.1 Flash Lite Preview",
     "description": "Gemini Flash Lite preview variant."},
    {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro",
     "description": "High capability Gemini 2.5."},
    {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash",
     "description": "Fast Gemini 2.5."},
    {"id": "gemini-2.5-flash-lite", "name": "Gemini 2.5 Flash Lite",
     "description": "Low latency Gemini 2.5."},
]

MODEL_ALIASES = {
    "gemini": DEFAULT_GEMINI_MODEL,
    "gemini-auto": DEFAULT_GEMINI_MODEL,
    "gemini3auto": "gemini-3auto",
    "gemini-3-auto": "gemini-3auto",
    "gemini-auto-3": "gemini-3auto",
    "auto-gemini-3-current": "gemini-3auto",
    "gemini-auto-3.5": "auto-gemini-3.5",
    "gemini-auto-2.5": "auto-gemini-2.5",
    "gemini-pro": "gemini-3-pro-preview",
    "gemini-3.5": "gemini-3.5-flash",
    "gemini-3.5-flash-preview": "gemini-3.5-flash",
    "gemini-3.5-flash-lite-preview": "gemini-3.5-flash-lite",
    "gemini-3.5-flashlite": "gemini-3.5-flash-lite",
    "gemini-3-pro": "gemini-3-pro-preview",
    "gemini-3.1-pro": "gemini-3.1-pro-preview",
    "gemini-3.1-flash": "gemini-3.1-flash-preview",
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemini-3.1-flash-lite": "gemini-3.1-flash-lite-preview",
    "gemini-flash": "gemini-3-flash-preview",
    "gemini-flash-lite": "gemini-2.5-flash-lite",
    "gemini-2.5-pro-preview": "gemini-2.5-pro",
}

MODEL_FALLBACK_CHAINS: Dict[str, List[str]] = {
    # Mirrors Gemini CLI 0.42.x: Gemini 3 auto tries Pro, then Flash as last resort.
    "gemini-3auto": [
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
    ],
    "auto-gemini-3.5": [
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3-flash-preview",
    ],
    "auto-gemini-3": [
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
    ],
    "auto-gemini-2.5": ["gemini-2.5-pro", "gemini-2.5-flash"],
    "gemini-3.1-pro-preview": [
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
    ],
    "gemini-3-pro-preview": [
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
    ],
    "gemini-3.5-flash": [
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3-flash-preview",
    ],
    "gemini-3.5-flash-lite": [
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-3-flash-preview",
    ],
    "gemini-3-flash-preview": [
        "gemini-3-flash-preview",
    ],
    "gemini-3.1-flash-preview": [
        "gemini-3.1-flash-preview",
        "gemini-3-flash-preview",
    ],
    "gemini-3.1-flash-lite-preview": [
        "gemini-3.1-flash-lite-preview",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
    ],
    "gemini-2.5-pro": ["gemini-2.5-pro", "gemini-2.5-flash"],
    "gemini-2.5-flash": [
        "gemini-2.5-flash",
    ],
    "gemini-2.5-flash-lite": [
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ],
}

MODEL_COOLDOWNS: Dict[str, float] = {}
NOT_FOUND_COOLDOWN_SECONDS = 30 * 60


def normalize_model(model: Optional[str], default: Optional[str] = None) -> str:
    raw = (model or "").strip()
    if not raw:
        raw = default or SETTINGS.google_default_model or DEFAULT_GEMINI_MODEL
    if raw.lower().startswith("models/"):
        raw = raw.split("/", 1)[1]
    return MODEL_ALIASES.get(raw.lower(), raw)


def _account_cooldown_key(account: Any, model: str) -> str:
    slot_id = str(getattr(account, "slot_id", "") or "global")
    return f"{slot_id}:{model}"


def _model_chain(model: str) -> List[str]:
    chain = MODEL_FALLBACK_CHAINS.get(model, [model])
    seen: set[str] = set()
    out: List[str] = []
    for item in chain:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _is_model_cooled_down(model: str, account: Any = None, now: Optional[float] = None) -> bool:
    until = MODEL_COOLDOWNS.get(_account_cooldown_key(account, model), 0)
    return bool(until and until > (now or time.time()))


def _model_candidates(model: str, account: Any = None) -> List[str]:
    now = time.time()
    return [item for item in _model_chain(model) if not _is_model_cooled_down(item, account, now)]


def _record_model_unavailable(model: str, exc: "_GoogleModelUnavailable", account: Any = None) -> None:
    reason = (exc.reason or "").upper()
    cooldown = 0.0
    if reason == "NOT_FOUND" or exc.status_code == 404:
        cooldown = NOT_FOUND_COOLDOWN_SECONDS
    elif exc.retry_delay:
        cooldown = max(1.0, float(exc.retry_delay))
    if cooldown <= 0:
        return
    until = time.time() + cooldown
    key = _account_cooldown_key(account, model)
    MODEL_COOLDOWNS[key] = max(MODEL_COOLDOWNS.get(key, 0), until)
    logger.info(
        "Google model %s cooled down for %ds on %s after %s",
        model,
        int(cooldown),
        str(getattr(account, "slot_id", "") or "global"),
        reason or exc.status_code,
    )


def _model_cooldown_wait_seconds(requested_model: str, account: Any = None) -> float:
    now = time.time()
    waits: List[int] = []
    for item in _model_chain(requested_model):
        until = MODEL_COOLDOWNS.get(_account_cooldown_key(account, item), 0)
        if until <= now:
            continue
        wait = int(max(1.0, (until - now) + 0.999))
        waits.append(wait)
    return float(min(waits)) if waits else 0.0


def _can_auto_wait_capacity(delay: float, waited: float) -> bool:
    if delay <= 0 or CAPACITY_AUTO_WAIT_MAX_SECONDS <= 0:
        return False
    return waited + delay <= CAPACITY_AUTO_WAIT_MAX_SECONDS


async def _sleep_capacity_cooldown(requested_model: str, account: Any, delay: float) -> None:
    logger.info(
        "Google model route %s is cooling down on %s; waiting %ds before retry",
        requested_model,
        str(getattr(account, "slot_id", "") or "global"),
        int(delay),
    )
    await asyncio.sleep(delay)


async def _stream_capacity_cooldown(requested_model: str, account: Any, delay: float) -> AsyncGenerator[bytes, None]:
    logger.info(
        "Google stream route %s is cooling down on %s; waiting %ds before retry",
        requested_model,
        str(getattr(account, "slot_id", "") or "global"),
        int(delay),
    )
    deadline = time.time() + delay
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return
        yield sse_comment(f"gemini capacity wait {int(remaining + 0.999)}s")
        await asyncio.sleep(min(max(1, SETTINGS.sse_keepalive_seconds), remaining))


def _model_cooldown_error(requested_model: str, account: Any = None) -> UpstreamClientError:
    now = time.time()
    parts: List[str] = []
    waits: List[int] = []
    for item in _model_chain(requested_model):
        until = MODEL_COOLDOWNS.get(_account_cooldown_key(account, item), 0)
        if until <= now:
            continue
        wait = int(max(1.0, (until - now) + 0.999))
        waits.append(wait)
        parts.append(f"{item}:cooldown {wait}s")
    wait_text = f" Wait {min(waits)}s before retrying." if waits else ""
    detail = f" Cooldowns: {', '.join(parts)}." if parts else ""
    return UpstreamClientError(
        "Google model capacity/rate limit; all fallback models are cooling down. "
        f"Requested {requested_model}.{wait_text}{detail}",
        status_code=429,
    )


def build_models_list() -> List[Dict[str, Any]]:
    created = int(time.time())
    return [
        {"id": m["id"], "object": "model", "created": created, "owned_by": "google",
         "name": m.get("name", m["id"]), "description": m.get("description", ""),
         "provider": PROVIDER}
        for m in GEMINI_MODEL_CATALOG
    ]


# ---------------------------------------------------------------------------
# Token store
# ---------------------------------------------------------------------------


class GoogleTokenStore:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / "oauth.json"
        self._lock = asyncio.Lock()
        # Extra metadata file: code_assist project / tier
        self.ca_path = self.folder / "code_assist.json"

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Bad google oauth.json: %s", exc)
            return {}

    def save(self, data: Dict[str, Any]) -> None:
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def clear(self) -> None:
        for p in (self.path, self.ca_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    def status(self) -> Dict[str, Any]:
        self.import_gemini_cli_credentials()
        data = self.load()
        now = time.time()
        token = data.get("access_token") or ""
        refresh = data.get("refresh_token") or ""
        expires_at = float(data.get("expires_at") or 0)
        active = bool(token and expires_at > now + REFRESH_SAFETY_WINDOW)
        if active:
            state, notice = "active", "Google session active."
        elif refresh:
            state, notice = "refreshable", "Will refresh on demand."
        elif token:
            state, notice = "expired", "Google session expired."
        else:
            state, notice = "missing", "Not signed in with Google."
        return {
            "logged_in": active,
            "email": data.get("email") or "",
            "account_id": data.get("account_id") or "",
            "expires_at": expires_at,
            "expires_in": max(0, int(expires_at - now)) if expires_at else 0,
            "has_refresh_token": bool(refresh),
            "has_saved_session": bool(token or refresh),
            "session_state": state,
            "session_notice": notice,
        }

    async def get_access_token(self) -> str:
        async with self._lock:
            self.import_gemini_cli_credentials()
            data = self.load()
            token = data.get("access_token")
            expires_at = float(data.get("expires_at") or 0)
            if token and expires_at > time.time() + REFRESH_SAFETY_WINDOW:
                return token
            refresh = data.get("refresh_token")
            if not refresh:
                raise AuthRevoked("No Google refresh token")
            new = await self._refresh(refresh)
            data.update(new)
            self.save(data)
            return data["access_token"]

    async def refresh_if_needed(
        self,
        min_valid_seconds: int = REFRESH_SAFETY_WINDOW,
        *,
        force: bool = False,
    ) -> bool:
        async with self._lock:
            self.import_gemini_cli_credentials()
            data = self.load()
            token = data.get("access_token")
            expires_at = float(data.get("expires_at") or 0)
            refresh = data.get("refresh_token")
            now = time.time()
            if not refresh:
                if force or not token:
                    raise AuthRevoked("No Google refresh token")
                return False
            if not force and token and expires_at > now + max(0, int(min_valid_seconds)):
                return False
            new = await self._refresh(refresh)
            for key, value in new.items():
                if value not in ("", 0, 0.0, None) or key in {"access_token", "expires_at"}:
                    data[key] = value
            self.save(data)
            return True

    async def _refresh(self, refresh_token: str) -> Dict[str, Any]:
        client = await get_http_client()
        client_id, client_secret = _oauth_client()
        resp = await client.post(
            TOKEN_ENDPOINT,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30.0,
        )
        if resp.status_code in (400, 401):
            raise AuthRevoked(f"Refresh failed: {resp.text}", status_code=resp.status_code)
        if resp.status_code >= 400:
            raise UpstreamServerError(f"Refresh failed: {resp.text}", status_code=resp.status_code)
        payload = resp.json()
        return {
            "access_token": payload["access_token"],
            "refresh_token": payload.get("refresh_token") or refresh_token,
            "id_token": payload.get("id_token") or "",
            "expires_at": time.time() + int(payload.get("expires_in") or 3600),
        }

    # -- Code Assist project tracking --------------------------------------

    def load_code_assist(self) -> Dict[str, Any]:
        if not self.ca_path.exists():
            return {}
        try:
            return json.loads(self.ca_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save_code_assist(self, data: Dict[str, Any]) -> None:
        self.ca_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def import_gemini_cli_credentials(self, *, force: bool = False) -> bool:
        if not SETTINGS.google_import_gemini_cli_credentials:
            return False
        data = _load_gemini_cli_oauth()
        if not data:
            return False
        imported = _saved_payload_from_gemini_cli(data)
        if not imported.get("access_token") and not imported.get("refresh_token"):
            return False

        current = self.load()
        current_expiry = float(current.get("expires_at") or 0)
        imported_expiry = float(imported.get("expires_at") or 0)
        if not force and current_expiry and current_expiry >= imported_expiry:
            if current.get("refresh_token") or not imported.get("refresh_token"):
                return False

        if not imported.get("refresh_token") and current.get("refresh_token"):
            imported["refresh_token"] = current["refresh_token"]
        for key in ("email", "account_id"):
            if not imported.get(key) and current.get(key):
                imported[key] = current[key]
        merged = dict(current)
        merged.update({k: v for k, v in imported.items() if v not in ("", 0, 0.0)})
        self.save(merged)
        logger.info("Imported Gemini CLI OAuth credentials from %s", _gemini_cli_oauth_path())
        return True


# ---------------------------------------------------------------------------
# OAuth
# ---------------------------------------------------------------------------


def build_auth_url(state: str, port: int) -> str:
    client_id, _ = _oauth_client()
    redirect_uri = f"http://127.0.0.1:{port}{CALLBACK_PATH}"
    params = {
        "redirect_uri": redirect_uri,
        "access_type": "offline",
        "scope": " ".join(SCOPES),
        "state": state,
        "response_type": "code",
        "client_id": client_id,
        "include_granted_scopes": "true",
    }
    if SETTINGS.google_oauth_prompt:
        params["prompt"] = SETTINGS.google_oauth_prompt
    return f"{AUTH_ENDPOINT}?{urlencode(params)}"


async def exchange_code(code: str, redirect_uri: str) -> Dict[str, Any]:
    client = await get_http_client()
    client_id, client_secret = _oauth_client()
    resp = await client.post(
        TOKEN_ENDPOINT,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "client_secret": client_secret,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


async def fetch_userinfo(access_token: str) -> Dict[str, Any]:
    client = await get_http_client()
    resp = await client.get(
        USERINFO_ENDPOINT, headers={"Authorization": f"Bearer {access_token}"}, timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}


def build_saved_payload(token_payload: Dict[str, Any], userinfo: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    expires_in = int(token_payload.get("expires_in") or 3600)
    id_token = token_payload.get("id_token") or ""
    jwt = decode_jwt_payload(id_token)
    userinfo = userinfo or {}
    return {
        "access_token": token_payload.get("access_token") or "",
        "refresh_token": token_payload.get("refresh_token") or "",
        "id_token": id_token,
        "expires_at": time.time() + expires_in,
        "email": jwt.get("email") or userinfo.get("email") or "",
        "account_id": jwt.get("sub") or userinfo.get("id") or "",
    }


def _gemini_cli_oauth_path() -> Path:
    configured = SETTINGS.google_gemini_cli_oauth_creds or _first_env((
        "GEMINI_CLI_OAUTH_CREDS",
        "GOOGLE_GEMINI_CLI_OAUTH_CREDS",
    ))
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".gemini" / "oauth_creds.json"


def _expiry_seconds(data: Dict[str, Any]) -> float:
    raw = data.get("expires_at")
    if raw is None:
        raw = data.get("expiry_date")
    try:
        value = float(raw or 0)
    except (TypeError, ValueError):
        return 0.0
    if value > 10_000_000_000:
        value = value / 1000.0
    return value


def _load_gemini_cli_oauth() -> Dict[str, Any]:
    path = _gemini_cli_oauth_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Bad Gemini CLI oauth creds at %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _saved_payload_from_gemini_cli(data: Dict[str, Any]) -> Dict[str, Any]:
    id_token = str(data.get("id_token") or "")
    jwt = decode_jwt_payload(id_token)
    return {
        "access_token": str(data.get("access_token") or ""),
        "refresh_token": str(data.get("refresh_token") or ""),
        "id_token": id_token,
        "expires_at": _expiry_seconds(data),
        "email": jwt.get("email") or "",
        "account_id": jwt.get("sub") or "",
    }


async def start_login(slot_id: str) -> str:
    from accounts import POOL
    if POOL.get(slot_id) is None:
        raise ValueError(f"Unknown slot: {slot_id}")
    port = CALLBACK_BROKER.ensure_server()
    state = generate_state()
    redirect_uri = f"http://127.0.0.1:{port}{CALLBACK_PATH}"
    CALLBACK_BROKER.add_pending(PendingAuth(
        state=state, slot_id=slot_id, provider=PROVIDER,
        verifier="", redirect_uri=redirect_uri,
    ))
    return build_auth_url(state, port)


async def handle_callback(code: str, query: Dict[str, Any], pa: PendingAuth) -> bool:
    from accounts import POOL
    acc = POOL.get(pa.slot_id)
    if acc is None:
        return False
    try:
        payload = await exchange_code(code, pa.redirect_uri)
        userinfo = {}
        if payload.get("access_token"):
            try:
                userinfo = await fetch_userinfo(payload["access_token"])
            except Exception:
                logger.exception("Failed to fetch Google userinfo")
        saved = build_saved_payload(payload, userinfo)
        old = acc.token_store.load()
        if not saved.get("refresh_token") and old.get("refresh_token"):
            saved["refresh_token"] = old["refresh_token"]
        acc.token_store.save(saved)
        # Force code assist re-check
        acc.token_store.save_code_assist({})
        await POOL.mark_valid(acc)
        logger.info("Google slot %s signed in (%s)", pa.slot_id, saved.get("email"))
        return True
    except Exception:
        logger.exception("Google token exchange failed")
        return False


# ---------------------------------------------------------------------------
# Code Assist preflight + headers
# ---------------------------------------------------------------------------


def _platform_id() -> str:
    machine = platform.machine().lower()
    is_arm = "arm" in machine or "aarch" in machine
    if sys.platform.startswith("win"):
        return "WINDOWS_AMD64"
    if sys.platform == "darwin":
        return "DARWIN_ARM64" if is_arm else "DARWIN_AMD64"
    return "LINUX_ARM64" if is_arm else "LINUX_AMD64"


def _client_metadata(project: str = "") -> Dict[str, Any]:
    meta = {
        "ideType": "GEMINI_CLI",
        "platform": _platform_id(),
        "pluginType": "GEMINI",
        "ideName": "gemini-cli",
    }
    if project:
        meta["duetProject"] = project
    return meta


def _code_assist_url(method: str) -> str:
    return f"{CODE_ASSIST_ENDPOINT}/{CODE_ASSIST_API_VERSION}:{method}"


def _active_project(account) -> str:
    if SETTINGS.google_code_assist_project:
        return SETTINGS.google_code_assist_project
    if SETTINGS.google_ignore_server_project:
        return ""
    ca = account.token_store.load_code_assist()
    return str(ca.get("project") or "").strip()


def _google_headers(token: str, project: str = "") -> Dict[str, str]:
    h = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": SETTINGS.google_user_agent or "google-gemini-cli",
    }
    if SETTINGS.google_use_user_project_header and project:
        h["x-goog-user-project"] = project
    return h


def _header_value(headers: Any, *names: str) -> str:
    for name in names:
        try:
            value = headers.get(name)
        except Exception:
            value = None
        if value not in (None, ""):
            return str(value)
    return ""


def _generic_rate_limit_dimension(headers: Any, dim: str) -> Optional[Dict[str, Any]]:
    if dim == "overall":
        limit = safe_int(_header_value(headers, "x-ratelimit-limit", "x-rate-limit-limit", "ratelimit-limit"))
        remaining = safe_int(_header_value(headers, "x-ratelimit-remaining", "x-rate-limit-remaining", "ratelimit-remaining"))
        reset = _header_value(headers, "x-ratelimit-reset", "x-rate-limit-reset", "ratelimit-reset")
    else:
        limit = safe_int(_header_value(headers, f"x-ratelimit-limit-{dim}", f"x-rate-limit-limit-{dim}", f"ratelimit-limit-{dim}"))
        remaining = safe_int(_header_value(headers, f"x-ratelimit-remaining-{dim}", f"x-rate-limit-remaining-{dim}", f"ratelimit-remaining-{dim}"))
        reset = _header_value(headers, f"x-ratelimit-reset-{dim}", f"x-rate-limit-reset-{dim}", f"ratelimit-reset-{dim}")
    if limit is None and remaining is None and not reset:
        return None
    out: Dict[str, Any] = {}
    if limit is not None:
        out["limit"] = limit
    if remaining is not None:
        out["remaining"] = remaining
    if reset:
        out["reset"] = reset
    if limit and remaining is not None:
        out["percent_remaining"] = round((remaining / limit) * 100, 1)
    return out


def _extract_rate_limit(headers: Any, model: str, raw_body: str = "") -> Dict[str, Any]:
    limits: Dict[str, Dict[str, Any]] = {}
    for dim in ("requests", "tokens", "input-tokens", "output-tokens", "overall"):
        parsed = _generic_rate_limit_dimension(headers, dim)
        if parsed:
            limits[dim] = parsed

    retry_after = safe_int(_header_value(headers, "retry-after")) or 0
    error_info: Dict[str, Any] = {}
    if raw_body:
        error = _google_error(raw_body)
        if error:
            info = _detail(error, "ErrorInfo")
            retry_delay = _retry_delay_seconds(error)
            quota_failure = _detail(error, "QuotaFailure")
            if info.get("reason") or error.get("status"):
                error_info["reason"] = str(info.get("reason") or error.get("status"))
            if retry_delay:
                retry_after = max(retry_after, int(retry_delay))
            violations = quota_failure.get("violations") if isinstance(quota_failure, dict) else None
            if isinstance(violations, list):
                error_info["quota_violations"] = [
                    {
                        "quota_id": str(v.get("quotaId") or ""),
                        "description": str(v.get("description") or ""),
                    }
                    for v in violations if isinstance(v, dict)
                ]

    raw_headers = {
        k: v for k, v in {
            "retry-after": _header_value(headers, "retry-after"),
            "x-goog-quota-project": _header_value(headers, "x-goog-quota-project"),
            "x-goog-user-project": _header_value(headers, "x-goog-user-project"),
        }.items() if v
    }
    if not limits and not retry_after and not error_info and not raw_headers:
        return {}
    out: Dict[str, Any] = {
        "provider": PROVIDER,
        "model": model,
        "updated_at": int(time.time()),
    }
    if limits:
        out["limits"] = limits
    if retry_after:
        out["retry_after_seconds"] = retry_after
    if error_info:
        out["error"] = error_info
    if raw_headers:
        out["headers"] = raw_headers
    return out


async def _ensure_code_assist_ready(account) -> None:
    if SETTINGS.google_skip_load_code_assist:
        return
    ca = account.token_store.load_code_assist()
    checked_at = float(ca.get("checked_at") or 0)
    if checked_at and time.time() - checked_at < 600:
        return

    token = await account.token_store.get_access_token()
    project_hint = SETTINGS.google_code_assist_project or _active_project(account)
    payload: Dict[str, Any] = {
        "metadata": _client_metadata(project_hint),
        "mode": "FULL_ELIGIBILITY_CHECK",
    }
    if project_hint:
        payload["cloudaicompanionProject"] = project_hint
    result = await _code_assist_call(account, "loadCodeAssist", payload, token)
    server_project = str(result.get("cloudaicompanionProject") or "").strip()
    if SETTINGS.google_code_assist_project:
        active_project = SETTINGS.google_code_assist_project
    elif SETTINGS.google_ignore_server_project:
        active_project = ""
    else:
        active_project = server_project or project_hint
    tier_obj = result.get("paidTier") or result.get("currentTier") or {}
    tier_name = ""
    if isinstance(tier_obj, dict):
        tier_name = str(tier_obj.get("name") or tier_obj.get("id") or "")
    account.token_store.save_code_assist({
        "project": active_project, "tier": tier_name, "checked_at": time.time(),
    })
    account.meta.tier = tier_name
    account.save_meta()


async def _code_assist_call(account, method: str, payload: Dict[str, Any], token: str) -> Dict[str, Any]:
    from accounts import POOL
    client = await get_http_client()
    resp = await client.post(
        _code_assist_url(method),
        headers=_google_headers(token, _active_project(account)),
        json=payload,
        timeout=60.0,
    )
    model = str(payload.get("model") or method)
    raw = resp.text if resp.status_code >= 400 else ""
    rl = _extract_rate_limit(resp.headers, model, raw)
    if rl:
        await POOL.record_rate_limit(account, rl)
    if resp.status_code >= 400:
        raise _classify_error(resp.status_code, resp.text, method)
    data = resp.json()
    return data if isinstance(data, dict) else {}


class _GoogleModelUnavailable(UpstreamServerError):
    """Transient Google model capacity/rate-limit issue; do not mark account exhausted."""

    def __init__(
        self,
        message: str,
        status_code: int = 429,
        raw_body: str = "",
        reason: str = "",
        retry_delay: float = 0.0,
        model: str = "",
    ):
        super().__init__(message, status_code=status_code, raw_body=raw_body)
        self.reason = reason
        self.retry_delay = retry_delay
        self.model = model


def _google_error(raw_body: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw_body)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    err = parsed.get("error")
    return err if isinstance(err, dict) else {}


def _detail(error: Dict[str, Any], suffix: str) -> Dict[str, Any]:
    details = error.get("details") or []
    if not isinstance(details, list):
        return {}
    for item in details:
        if isinstance(item, dict) and str(item.get("@type") or "").endswith(suffix):
            return item
    return {}


def _duration_seconds(value: Any) -> float:
    if value is None:
        return 0.0
    raw = str(value).strip()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(ms|s)?", raw)
    if not match:
        return 0.0
    amount = float(match.group(1))
    return amount / 1000.0 if match.group(2) == "ms" else amount


def _retry_delay_seconds(error: Dict[str, Any]) -> float:
    retry_info = _detail(error, "RetryInfo")
    delay = _duration_seconds(retry_info.get("retryDelay"))
    if delay:
        return delay
    message = str(error.get("message") or "")
    patterns = (
        r"(?:retry|reset)\s+(?:in|after)\s+([0-9]+(?:\.[0-9]+)?\s*(?:ms|s)?)",
        r"after\s+([0-9]+(?:\.[0-9]+)?\s*(?:ms|s)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            delay = _duration_seconds(match.group(1).replace(" ", ""))
            if delay:
                return delay
    return 0.0


def _has_daily_quota_failure(error: Dict[str, Any]) -> bool:
    quota = _detail(error, "QuotaFailure")
    violations = quota.get("violations") if quota else None
    if not isinstance(violations, list):
        return False
    for item in violations:
        if not isinstance(item, dict):
            continue
        quota_id = str(item.get("quotaId") or "")
        if "PerDay" in quota_id or "Daily" in quota_id:
            return True
    return False


def _classify_error(status: int, raw_body: str, method: str) -> Exception:
    if status == 401:
        return AuthRevoked(f"{method}: {raw_body[:300]}", status_code=status, raw_body=raw_body)
    if status == 429:
        error = _google_error(raw_body)
        error_info = _detail(error, "ErrorInfo")
        reason = str(error_info.get("reason") or error.get("status") or "RESOURCE_EXHAUSTED")
        message = str(error.get("message") or raw_body[:300])
        retry_delay = _retry_delay_seconds(error)
        if reason in {"RATE_LIMIT_EXCEEDED", "MODEL_CAPACITY_EXHAUSTED"}:
            return _GoogleModelUnavailable(
                f"{method}: {message}",
                status_code=status,
                raw_body=raw_body,
                reason=reason,
                retry_delay=retry_delay,
            )
        if (
            reason in {"QUOTA_EXHAUSTED", "INSUFFICIENT_G1_CREDITS_BALANCE"}
            or _has_daily_quota_failure(error)
        ):
            resets_at = time.time() + (retry_delay or 300)
            return QuotaExhausted(
                f"{method}: {message}",
                status_code=status,
                resets_at=resets_at,
                reason=reason.lower(),
                raw_body=raw_body,
            )
        # Unknown 429s from Code Assist are usually transient model capacity.
        if "capacity" in message.lower() or retry_delay:
            return _GoogleModelUnavailable(
                f"{method}: {message}",
                status_code=status,
                raw_body=raw_body,
                reason=reason,
                retry_delay=retry_delay,
            )
        resets_at = time.time() + 300
        return QuotaExhausted(
            f"{method}: {message}",
            status_code=status, resets_at=resets_at,
            reason=reason.lower(), raw_body=raw_body,
        )
    if status in (400, 404):
        error = _google_error(raw_body)
        reason = str(error.get("status") or "MODEL_UNAVAILABLE")
        message = str(error.get("message") or raw_body[:300])
        lower = message.lower()
        if status == 404 and method in {"generateContent", "streamGenerateContent"}:
            return _GoogleModelUnavailable(
                f"{method}: {message}",
                status_code=status,
                raw_body=raw_body,
                reason=reason,
            )
        if (
            "model" in lower
            and any(term in lower for term in (
                "not found",
                "not available",
                "unavailable",
                "unsupported",
                "invalid",
                "unknown",
            ))
        ):
            return _GoogleModelUnavailable(
                f"{method}: {message}",
                status_code=status,
                raw_body=raw_body,
                reason=reason,
            )
    if status == 403:
        return UpstreamClientError(
            f"{method} forbidden: {raw_body[:300]}", status_code=status, raw_body=raw_body,
        )
    if 500 <= status < 600:
        return UpstreamServerError(f"{method}: {raw_body[:300]}", status_code=status, raw_body=raw_body)
    return UpstreamClientError(f"{method}: {raw_body[:300]}", status_code=status, raw_body=raw_body)


# ---------------------------------------------------------------------------
# Request mapping
# ---------------------------------------------------------------------------


def _text_part(text: Any) -> Dict[str, str]:
    return {"text": str(text or "")}


def _image_part_from_url(url: str) -> Dict[str, Any]:
    if url.startswith("data:") and "," in url:
        header, data = url.split(",", 1)
        mime = header[5:].split(";", 1)[0] or "application/octet-stream"
        return {"inlineData": {"mimeType": mime, "data": data}}
    mime = mimetypes.guess_type(url)[0] or "application/octet-stream"
    return {"fileData": {"mimeType": mime, "fileUri": url}}


def _parts_from_openai_content(content: Any) -> List[Dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [_text_part(content)] if content else []
    if not isinstance(content, list):
        return [_text_part(json.dumps(content, ensure_ascii=False))]
    parts: List[Dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            parts.append(_text_part(item))
            continue
        if not isinstance(item, dict):
            parts.append(_text_part(json.dumps(item, ensure_ascii=False)))
            continue
        item_type = item.get("type")
        if item_type in ("text", "input_text"):
            parts.append(_text_part(item.get("text") or ""))
        elif item_type in ("image_url", "input_image"):
            image = item.get("image_url") or item.get("image") or {}
            url = image.get("url") if isinstance(image, dict) else image
            if url:
                parts.append(_image_part_from_url(str(url)))
        elif item.get("text"):
            parts.append(_text_part(item.get("text")))
    return [p for p in parts if p]


def _parse_tool_args(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except Exception:
        return {"arguments": str(raw)}


def _schema_ref_target(root: Dict[str, Any], ref: str) -> Optional[Any]:
    if not ref.startswith("#/"):
        return None
    current: Any = root
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _google_schema_type(value: Any) -> Any:
    if isinstance(value, list):
        non_null = [v for v in value if str(v).lower() != "null"]
        return _google_schema_type(non_null[0]) if non_null else None
    if not isinstance(value, str):
        return value
    lower = value.lower()
    if lower == "integer":
        return "number"
    if lower in {"object", "array", "string", "number", "boolean", "null"}:
        return lower
    return value


def _normalize_google_schema(schema: Any, root: Optional[Dict[str, Any]] = None, seen: Optional[set[str]] = None) -> Dict[str, Any]:
    """Convert JSON Schema into the smaller schema subset accepted by Code Assist."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    root = root or schema
    seen = seen or set()

    ref = schema.get("$ref")
    if isinstance(ref, str) and ref not in seen:
        target = _schema_ref_target(root, ref)
        if isinstance(target, dict):
            merged = deepcopy(target)
            for key, value in schema.items():
                if key != "$ref":
                    merged[key] = value
            seen.add(ref)
            return _normalize_google_schema(merged, root, seen)

    for choice_key in ("anyOf", "oneOf", "allOf"):
        choices = schema.get(choice_key)
        if isinstance(choices, list) and choices:
            nullable = any(isinstance(c, dict) and str(c.get("type")).lower() == "null" for c in choices)
            if choice_key == "allOf":
                merged: Dict[str, Any] = {}
                for choice in choices:
                    if isinstance(choice, dict):
                        merged.update(_normalize_google_schema(choice, root, seen))
                for key, value in schema.items():
                    if key not in {"allOf", "$defs", "definitions"}:
                        merged.setdefault(key, value)
                if nullable:
                    merged["nullable"] = True
                return _normalize_google_schema(merged, root, seen)
            first = next((c for c in choices if isinstance(c, dict) and str(c.get("type")).lower() != "null"), None)
            if isinstance(first, dict):
                base = dict(first)
                for key, value in schema.items():
                    if key not in {choice_key, "$defs", "definitions"}:
                        base.setdefault(key, value)
                if nullable:
                    base["nullable"] = True
                return _normalize_google_schema(base, root, seen)

    out: Dict[str, Any] = {}
    nullable = False
    raw_type = schema.get("type")
    if isinstance(raw_type, list) and any(str(v).lower() == "null" for v in raw_type):
        nullable = True
    typ = _google_schema_type(raw_type)
    if typ and typ != "null":
        out["type"] = typ
    if schema.get("description"):
        out["description"] = str(schema["description"])
    if schema.get("format"):
        out["format"] = str(schema["format"])
    if "enum" in schema and isinstance(schema["enum"], list):
        enum = [v for v in schema["enum"] if v is not None]
        if enum:
            out["enum"] = enum
        if len(enum) != len(schema["enum"]):
            nullable = True
    if "const" in schema:
        out["enum"] = [schema["const"]]
    if schema.get("nullable") is True or nullable:
        out["nullable"] = True

    props = schema.get("properties")
    if isinstance(props, dict):
        out["type"] = out.get("type") or "object"
        cleaned_props: Dict[str, Any] = {}
        for name, prop_schema in props.items():
            cleaned_props[str(name)] = _normalize_google_schema(prop_schema, root, set(seen))
        out["properties"] = cleaned_props
        required = schema.get("required")
        if isinstance(required, list):
            req = [str(x) for x in required if str(x) in cleaned_props]
            if req:
                out["required"] = req

    items = schema.get("items")
    if isinstance(items, dict):
        out["type"] = out.get("type") or "array"
        out["items"] = _normalize_google_schema(items, root, set(seen))
    elif isinstance(items, list) and items:
        out["type"] = out.get("type") or "array"
        first_item = next((x for x in items if isinstance(x, dict)), None)
        if first_item:
            out["items"] = _normalize_google_schema(first_item, root, set(seen))

    for numeric_key in ("minimum", "maximum", "minItems", "maxItems", "minLength", "maxLength"):
        if numeric_key in schema:
            out[numeric_key] = schema[numeric_key]

    if not out:
        out = {"type": "object", "properties": {}}
    return out


def _messages_to_gemini(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    contents: List[Dict[str, Any]] = []
    system_parts: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "user").lower()
        if role in {"system", "developer"}:
            system_parts.extend(_parts_from_openai_content(m.get("content")))
            continue
        gemini_role = "model" if role == "assistant" else "user"
        parts = _parts_from_openai_content(m.get("content"))
        if role == "tool":
            tool_name = m.get("name") or m.get("tool_call_id") or "tool"
            text = "".join(p.get("text", "") for p in parts if "text" in p)
            parts = [_text_part(f"Tool result from {tool_name}:\n{text}")]
        for tc in m.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            name = fn.get("name")
            if name:
                parts.append({"functionCall": {"name": name, "args": _parse_tool_args(fn.get("arguments"))}})
        if parts:
            contents.append({"role": gemini_role, "parts": parts})
    system_instr = {"role": "system", "parts": system_parts} if system_parts else None
    if not contents:
        contents.append({"role": "user", "parts": [_text_part("")]})
    return contents, system_instr


def _convert_tools(tools: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(tools, list):
        return None
    decl: List[Dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        fn = t.get("function") or {}
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        d: Dict[str, Any] = {"name": fn["name"]}
        if fn.get("description"):
            d["description"] = fn["description"]
        if fn.get("parameters"):
            d["parameters"] = _normalize_google_schema(fn["parameters"])
        decl.append(d)
    return [{"functionDeclarations": decl}] if decl else None


def _gen_config(req: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    mapping = {
        "temperature": "temperature", "top_p": "topP", "top_k": "topK",
        "presence_penalty": "presencePenalty", "frequency_penalty": "frequencyPenalty",
        "seed": "seed",
    }
    for k, target in mapping.items():
        if req.get(k) is not None:
            cfg[target] = req[k]
    mx = req.get("max_output_tokens") or req.get("max_completion_tokens") or req.get("max_tokens")
    if mx is not None:
        try:
            cfg["maxOutputTokens"] = int(mx)
        except (TypeError, ValueError):
            pass
    stop = req.get("stop")
    if isinstance(stop, str):
        cfg["stopSequences"] = [stop]
    elif isinstance(stop, list):
        cfg["stopSequences"] = [str(s) for s in stop]
    rf = req.get("response_format")
    if isinstance(rf, dict):
        if rf.get("type") in {"json_object", "json_schema"}:
            cfg["responseMimeType"] = "application/json"
        schema = rf.get("json_schema")
        if isinstance(schema, dict):
            cfg["responseJsonSchema"] = _normalize_google_schema(schema.get("schema") or schema)
    eb = req.get("extra_body")
    if isinstance(eb, dict):
        ge = eb.get("generationConfig") or eb.get("generation_config")
        if isinstance(ge, dict):
            cfg.update(ge)
    return cfg


def build_request(
    req: Dict[str, Any],
    account,
    model_override: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    raw_model = req.get("model")
    requested_model = normalize_model(str(raw_model or ""), DEFAULT_GEMINI_MODEL)
    model = model_override or requested_model
    msgs = req.get("messages")
    if not isinstance(msgs, list):
        msgs = [{"role": "user", "content": str(req.get("prompt") or "")}]
    contents, system_instr = _messages_to_gemini(msgs)
    body: Dict[str, Any] = {"contents": contents}
    if system_instr:
        body["systemInstruction"] = system_instr
    tools = _convert_tools(req.get("tools"))
    if tools:
        body["tools"] = tools
    cfg = _gen_config(req)
    if cfg:
        body["generationConfig"] = cfg
    session_id = str(req.get("session_id") or req.get("conversation_id") or "").strip()
    if session_id:
        body["session_id"] = session_id
    payload: Dict[str, Any] = {
        "model": model,
        "user_prompt_id": str(req.get("user_prompt_id") or uuid.uuid4()),
        "request": body,
    }
    project = _active_project(account)
    if project:
        payload["project"] = project
    return payload, model


def _requested_model(req: Dict[str, Any]) -> str:
    return normalize_model(str(req.get("model") or ""), DEFAULT_GEMINI_MODEL)


def _model_fallback_error(requested_model: str, failures: List[_GoogleModelUnavailable]) -> UpstreamClientError:
    if not failures:
        return UpstreamClientError(f"No Google model available for {requested_model}", status_code=429)
    parts = []
    for err in failures:
        retry = f", retry={int(err.retry_delay)}s" if err.retry_delay else ""
        parts.append(f"{err.model or '?'}:{err.reason or 'RESOURCE_EXHAUSTED'}{retry}")
    last = failures[-1]
    return UpstreamClientError(
        "Google model capacity/rate limit; account quota is not exhausted. "
        f"Requested {requested_model}. Tried: {', '.join(parts)}. Last error: {last}",
        status_code=429,
        raw_body=last.raw_body,
    )


def _candidate(resp: Dict[str, Any]) -> Dict[str, Any]:
    r = resp.get("response") or {}
    if not isinstance(r, dict):
        return {}
    cands = r.get("candidates") or []
    if isinstance(cands, list) and cands and isinstance(cands[0], dict):
        return cands[0]
    return {}


def _extract_text_and_tools(resp: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], str]:
    cand = _candidate(resp)
    content = cand.get("content") or {}
    parts = content.get("parts") if isinstance(content, dict) else []
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    if isinstance(parts, list):
        for p in parts:
            if not isinstance(p, dict):
                continue
            if p.get("text"):
                text_parts.append(str(p["text"]))
            fc = p.get("functionCall") or p.get("function_call")
            if isinstance(fc, dict) and fc.get("name"):
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": fc["name"],
                        "arguments": json.dumps(fc.get("args") or {}, ensure_ascii=False),
                    },
                })
    finish = str(cand.get("finishReason") or cand.get("finish_reason") or "STOP")
    return "".join(text_parts), tool_calls, finish


def _map_finish_reason(r: str, has_tools: bool) -> str:
    n = (r or "").upper()
    if has_tools:
        return "tool_calls"
    if n in {"MAX_TOKENS", "TOKEN_LIMIT"}:
        return "length"
    if n in {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}:
        return "content_filter"
    return "stop"


def _usage(resp: Dict[str, Any]) -> Dict[str, Any]:
    r = resp.get("response") or {}
    usage = r.get("usageMetadata") if isinstance(r, dict) else {}
    if not isinstance(usage, dict):
        usage = {}
    prompt = int(usage.get("promptTokenCount") or 0)
    completion = int(usage.get("candidatesTokenCount") or 0)
    total = int(usage.get("totalTokenCount") or (prompt + completion))
    out: Dict[str, Any] = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}
    thoughts = int(usage.get("thoughtsTokenCount") or 0)
    if thoughts:
        out["completion_tokens_details"] = {"reasoning_tokens": thoughts}
    cached = int(usage.get("cachedContentTokenCount") or 0)
    if cached:
        out["prompt_tokens_details"] = {"cached_tokens": cached}
    return out


async def _emit_completion_as_stream(
    result: Dict[str, Any],
    response_id: str,
    created: int,
    include_usage: bool,
) -> AsyncGenerator[bytes, None]:
    choice = (result.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    model = str(result.get("model") or "")
    content = message.get("content")
    if content:
        yield sse_data({
            "id": response_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
        })
    for idx, tc in enumerate(message.get("tool_calls") or []):
        yield sse_data({
            "id": response_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0,
                         "delta": {"tool_calls": [{**tc, "index": idx}]},
                         "finish_reason": None}],
        })
    yield sse_data({
        "id": response_id, "object": "chat.completion.chunk",
        "created": created, "model": model,
        "choices": [{"index": 0, "delta": {},
                     "finish_reason": choice.get("finish_reason") or "stop"}],
    })
    usage = result.get("usage")
    if include_usage and usage:
        yield sse_data({
            "id": response_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [], "usage": usage,
        })
    yield sse_done()


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


async def complete_completion(req: Dict[str, Any], account) -> Dict[str, Any]:
    await _ensure_code_assist_ready(account)
    token = await account.token_store.get_access_token()
    requested_model = _requested_model(req)
    created = int(time.time())
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    model_failures: List[_GoogleModelUnavailable] = []
    ca_response: Optional[Dict[str, Any]] = None
    model = requested_model
    waited_capacity = 0.0
    while ca_response is None:
        candidates = _model_candidates(requested_model, account)
        if not candidates:
            delay = _model_cooldown_wait_seconds(requested_model, account)
            if not _can_auto_wait_capacity(delay, waited_capacity):
                raise _model_cooldown_error(requested_model, account)
            await _sleep_capacity_cooldown(requested_model, account, delay)
            waited_capacity += delay
            continue
        cycle_failures: List[_GoogleModelUnavailable] = []
        for candidate in candidates:
            payload, model = build_request(req, account, model_override=candidate)
            try:
                ca_response = await _code_assist_call(account, "generateContent", payload, token)
                break
            except _GoogleModelUnavailable as exc:
                exc.model = candidate
                _record_model_unavailable(candidate, exc, account)
                cycle_failures.append(exc)
                model_failures.append(exc)
                logger.warning(
                    "Google model %s unavailable for slot %s (%s%s); trying fallback",
                    candidate,
                    getattr(account, "slot_id", "?"),
                    exc.reason or "RESOURCE_EXHAUSTED",
                    f", retry={int(exc.retry_delay)}s" if exc.retry_delay else "",
                )
        if ca_response is not None:
            break
        delay = _model_cooldown_wait_seconds(requested_model, account)
        if cycle_failures and _can_auto_wait_capacity(delay, waited_capacity):
            await _sleep_capacity_cooldown(requested_model, account, delay)
            waited_capacity += delay
            continue
        break
    if ca_response is None:
        raise _model_fallback_error(requested_model, model_failures)
    text, tool_calls, finish = _extract_text_and_tools(ca_response)
    message: Dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    result = {
        "id": response_id, "object": "chat.completion",
        "created": created, "model": model,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": _map_finish_reason(finish, bool(tool_calls))}],
        "usage": _usage(ca_response),
    }
    if account.health.rate_limit:
        result["_bridge_rate_limit"] = account.health.rate_limit
    return result


async def stream_completion(req: Dict[str, Any], account) -> AsyncGenerator[bytes, None]:
    from accounts import POOL
    await _ensure_code_assist_ready(account)
    token = await account.token_store.get_access_token()
    requested_model = _requested_model(req)
    created = int(time.time())
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
    last_usage: Dict[str, Any] = {}
    emitted_tools = False
    model_failures: List[_GoogleModelUnavailable] = []

    client = await get_http_client()
    opened_stream = False
    model = requested_model
    waited_capacity = 0.0
    while not opened_stream:
        candidates = _model_candidates(requested_model, account)
        if not candidates:
            delay = _model_cooldown_wait_seconds(requested_model, account)
            if not _can_auto_wait_capacity(delay, waited_capacity):
                raise _model_cooldown_error(requested_model, account)
            async for chunk in _stream_capacity_cooldown(requested_model, account, delay):
                yield chunk
            waited_capacity += delay
            continue
        cycle_failures: List[_GoogleModelUnavailable] = []
        for candidate in candidates:
            payload, model = build_request(req, account, model_override=candidate)
            async with client.stream(
                "POST",
                _code_assist_url("streamGenerateContent"),
                params={"alt": "sse"},
                headers=_google_headers(token, _active_project(account)),
                json=payload,
            ) as resp:
                rl = _extract_rate_limit(resp.headers, candidate)
                if rl:
                    await POOL.record_rate_limit(account, rl)
                if resp.status_code >= 400:
                    detail = (await resp.aread()).decode("utf-8", "ignore")
                    rl = _extract_rate_limit(resp.headers, candidate, detail)
                    if rl:
                        await POOL.record_rate_limit(account, rl)
                    err = _classify_error(resp.status_code, detail, "streamGenerateContent")
                    if isinstance(err, _GoogleModelUnavailable):
                        err.model = candidate
                        _record_model_unavailable(candidate, err, account)
                        cycle_failures.append(err)
                        model_failures.append(err)
                        logger.warning(
                            "Google stream model %s unavailable for slot %s (%s%s); trying fallback",
                            candidate,
                            getattr(account, "slot_id", "?"),
                            err.reason or "RESOURCE_EXHAUSTED",
                            f", retry={int(err.retry_delay)}s" if err.retry_delay else "",
                        )
                        continue
                    raise err

                opened_stream = True
                buffered: List[str] = []
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        buffered.append(line[6:].strip())
                        continue
                    if line != "" or not buffered:
                        continue
                    raw = "\n".join(buffered)
                    buffered = []
                    if raw == "[DONE]":
                        break
                    try:
                        ca_resp = json.loads(raw)
                    except Exception:
                        continue
                    if not isinstance(ca_resp, dict):
                        continue
                    text, tool_calls, finish = _extract_text_and_tools(ca_resp)
                    last_usage = _usage(ca_resp) or last_usage
                    if text:
                        yield sse_data({
                            "id": response_id, "object": "chat.completion.chunk",
                            "created": created, "model": model,
                            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                        })
                    for idx, tc in enumerate(tool_calls):
                        emitted_tools = True
                        yield sse_data({
                            "id": response_id, "object": "chat.completion.chunk",
                            "created": created, "model": model,
                            "choices": [{"index": 0,
                                         "delta": {"tool_calls": [{**tc, "index": idx}]},
                                         "finish_reason": None}],
                        })
                    if finish and finish.upper() not in {"", "STOP"}:
                        break
            if opened_stream:
                break
        if opened_stream:
            break
        delay = _model_cooldown_wait_seconds(requested_model, account)
        if cycle_failures and _can_auto_wait_capacity(delay, waited_capacity):
            async for chunk in _stream_capacity_cooldown(requested_model, account, delay):
                yield chunk
            waited_capacity += delay
            continue
        break

    if not opened_stream:
        raise _model_fallback_error(requested_model, model_failures)

    yield sse_data({
        "id": response_id, "object": "chat.completion.chunk",
        "created": created, "model": model,
        "choices": [{"index": 0, "delta": {},
                     "finish_reason": "tool_calls" if emitted_tools else "stop"}],
    })
    if include_usage and last_usage:
        usage_chunk = {
            "id": response_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [], "usage": last_usage,
        }
        if account.health.rate_limit:
            usage_chunk["_bridge_rate_limit"] = account.health.rate_limit
        yield sse_data(usage_chunk)
    yield sse_done()


def register(pool) -> None:
    pool.register_provider(PROVIDER, GoogleTokenStore)
    CALLBACK_BROKER.register_handler(CALLBACK_PATH, handle_callback)
