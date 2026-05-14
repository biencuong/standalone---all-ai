# -*- coding: utf-8 -*-
"""Anthropic Claude provider.

The supported authentication path is the official Anthropic Console API key
(`ANTHROPIC_API_KEY` or a per-slot key saved through the local UI). Legacy
Claude Code OAuth import is intentionally disabled unless
`ANTHROPIC_ALLOW_LEGACY_OAUTH=1` is set, because Claude Code subscription
OAuth is not a stable third-party API credential.

Maps OpenAI Chat Completions <-> Anthropic Messages API. Supports:
- system message → top-level `system`
- user/assistant content with text + image_url + tool_result blocks
- tools (function calling) with proper JSON schema
- tool_calls streaming via content_block events
- thinking (extended thinking) → `delta.reasoning_content`
- usage tokens (input/output, cached, reasoning)
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
from urllib.parse import urlencode

from core import (
    AuthRevoked,
    CALLBACK_BROKER,
    PendingAuth,
    QuotaExhausted,
    SETTINGS,
    UpstreamClientError,
    UpstreamServerError,
    generate_pkce_pair,
    generate_state,
    get_http_client,
    logger,
    nested_get,
    safe_int,
    sse_data,
    sse_done,
)

PROVIDER = "anthropic"

# Legacy OAuth values are kept only for explicit opt-in migration/testing.
CLIENT_ID = os.getenv("ANTHROPIC_CLIENT_ID") or "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
AUTH_ENDPOINT = os.getenv("ANTHROPIC_AUTH_URL") or "https://platform.claude.com/oauth/authorize"
TOKEN_ENDPOINT = os.getenv("ANTHROPIC_TOKEN_URL") or "https://platform.claude.com/v1/oauth/token"
MESSAGES_ENDPOINT = "https://api.anthropic.com/v1/messages"
CALLBACK_PATH = "/claude/callback"

# OAuth scope used by Claude Code
SCOPES = "user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"
CLI_CREDENTIALS_FILE = Path.home() / ".claude" / ".credentials.json"

# Anthropic API headers
ANTHROPIC_VERSION = "2023-06-01"
OAUTH_BETA = "oauth-2025-04-20"

REFRESH_SAFETY_WINDOW = 60


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-sonnet-4-6"

CLAUDE_MODEL_CATALOG: List[Dict[str, Any]] = [
    {"id": "claude-opus-4-7", "name": "Claude Opus 4.7",
     "description": "Frontier intelligence for complex reasoning."},
    {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6",
     "description": "Balanced speed and capability."},
    {"id": "claude-haiku-4-5", "name": "Claude Haiku 4.5",
     "description": "Fast and cost-efficient."},
    {"id": "claude-opus-4-1", "name": "Claude Opus 4.1",
     "description": "Previous-gen frontier."},
    {"id": "claude-sonnet-3-7", "name": "Claude Sonnet 3.7",
     "description": "Stable Sonnet 3.x."},
]

MODEL_ALIASES: Dict[str, str] = {
    "claude": DEFAULT_MODEL,
    "claude-opus": "claude-opus-4-7",
    "claude-opus-latest": "claude-opus-4-7",
    "claude-sonnet": "claude-sonnet-4-6",
    "claude-sonnet-latest": "claude-sonnet-4-6",
    "claude-haiku": "claude-haiku-4-5",
    "claude-haiku-latest": "claude-haiku-4-5",
    "claude-4-7": "claude-opus-4-7",
    "claude-4-6": "claude-sonnet-4-6",
    "claude-4-5": "claude-haiku-4-5",
}


def normalize_model(model: Optional[str], default: Optional[str] = None) -> str:
    raw = (model or "").strip().lower()
    if not raw:
        return default or SETTINGS.anthropic_default_model or DEFAULT_MODEL
    return MODEL_ALIASES.get(raw, raw)


def build_models_list() -> List[Dict[str, Any]]:
    created = int(time.time())
    return [
        {"id": m["id"], "object": "model", "created": created, "owned_by": "anthropic",
         "name": m["name"], "description": m["description"], "provider": PROVIDER}
        for m in CLAUDE_MODEL_CATALOG
    ]


# ---------------------------------------------------------------------------
# Token store
# ---------------------------------------------------------------------------


class AnthropicTokenStore:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / "oauth.json"
        self._lock = asyncio.Lock()

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Bad anthropic oauth.json: %s", exc)
            return {}

    def save(self, data: Dict[str, Any]) -> None:
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def status(self) -> Dict[str, Any]:
        data = self.load()
        now = time.time()
        api_key = data.get("api_key") or ""
        if api_key:
            return {
                "logged_in": True,
                "email": data.get("email") or "",
                "account_id": data.get("account_id") or "",
                "expires_at": 0,
                "expires_in": 0,
                "has_refresh_token": False,
                "has_saved_session": True,
                "session_state": "active",
                "session_notice": "Anthropic API key configured.",
                "auth_type": "api_key",
            }

        token = data.get("access_token") or ""
        refresh = data.get("refresh_token") or ""
        expires_at = float(data.get("expires_at") or 0)
        legacy_allowed = SETTINGS.anthropic_allow_legacy_oauth
        active = bool(legacy_allowed and token and expires_at > now + REFRESH_SAFETY_WINDOW)
        if token or refresh:
            if legacy_allowed:
                if active:
                    state, notice = "active", "Legacy Claude OAuth session active."
                elif refresh:
                    state, notice = "refreshable", "Legacy OAuth will refresh on demand."
                else:
                    state, notice = "expired", "Legacy Claude OAuth session expired."
            else:
                state = "legacy_oauth_disabled"
                notice = "Claude Code OAuth is disabled; configure an Anthropic API key."
        else:
            state, notice = "missing", "Not signed in with Claude."
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
            "auth_type": "legacy_oauth" if token or refresh else "",
        }

    def save_api_key(self, api_key: str) -> None:
        key = (api_key or "").strip()
        if not key:
            raise ValueError("Empty Anthropic API key")
        self.save({
            "auth_type": "api_key",
            "api_key": key,
            "source": "anthropic-console",
            "created_at": time.time(),
        })

    def credential_type(self) -> str:
        data = self.load()
        if data.get("api_key"):
            return "api_key"
        if data.get("access_token") or data.get("refresh_token"):
            return "legacy_oauth"
        return ""

    async def get_credential(self) -> tuple[str, str]:
        data = self.load()
        api_key = data.get("api_key") or ""
        if api_key:
            return "api_key", api_key
        if not SETTINGS.anthropic_allow_legacy_oauth:
            raise AuthRevoked("No Anthropic API key configured")
        return "legacy_oauth", await self.get_access_token()

    async def get_access_token(self) -> str:
        async with self._lock:
            data = self.load()
            api_key = data.get("api_key") or ""
            if api_key:
                return api_key
            if not SETTINGS.anthropic_allow_legacy_oauth:
                raise AuthRevoked("No Anthropic API key configured")
            token = data.get("access_token")
            expires_at = float(data.get("expires_at") or 0)
            if token and expires_at > time.time() + REFRESH_SAFETY_WINDOW:
                return token
            refresh = data.get("refresh_token")
            if not refresh:
                raise AuthRevoked("No Anthropic refresh token")
            new = await self._refresh(refresh)
            data.update(new)
            self.save(data)
            return data["access_token"]

    async def _refresh(self, refresh_token: str) -> Dict[str, Any]:
        client = await get_http_client()
        request_body = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CLIENT_ID,
        }
        resp = await client.post(
            TOKEN_ENDPOINT,
            json=request_body,
            headers={"Content-Type": "application/json"},
            timeout=30.0,
        )
        if resp.status_code == 400 and "invalid request" in resp.text.lower():
            resp = await client.post(
                TOKEN_ENDPOINT,
                data=request_body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30.0,
            )
        if resp.status_code in (400, 401):
            raise AuthRevoked(f"Refresh failed: {resp.text}", status_code=resp.status_code)
        if resp.status_code >= 400:
            raise UpstreamServerError(f"Refresh failed: {resp.text}", status_code=resp.status_code)
        payload = resp.json()
        return {
            "access_token": payload.get("access_token") or "",
            "refresh_token": payload.get("refresh_token") or refresh_token,
            "expires_at": time.time() + int(payload.get("expires_in") or 3600),
        }


def _load_cli_oauth() -> Dict[str, Any]:
    if not CLI_CREDENTIALS_FILE.exists():
        return {}
    try:
        raw = json.loads(CLI_CREDENTIALS_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Cannot read Claude CLI credentials: %s", exc)
        return {}
    oauth = raw.get("claudeAiOauth")
    if not isinstance(oauth, dict):
        return {}
    access_token = oauth.get("accessToken") or oauth.get("access_token") or ""
    refresh_token = oauth.get("refreshToken") or oauth.get("refresh_token") or ""
    if not access_token and not refresh_token:
        return {}
    try:
        expires_at = float(oauth.get("expiresAt") or oauth.get("expires_at") or 0)
    except (TypeError, ValueError):
        expires_at = 0.0
    if expires_at > 10_000_000_000:
        expires_at = expires_at / 1000.0
    scopes = oauth.get("scopes") or []
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": expires_at or (time.time() + 3600),
        "email": oauth.get("email") or "",
        "account_id": oauth.get("accountId") or oauth.get("account_id") or "",
        "scope": " ".join(scopes) if isinstance(scopes, list) else str(scopes or ""),
        "subscription_type": oauth.get("subscriptionType") or "",
        "rate_limit_tier": oauth.get("rateLimitTier") or "",
        "source": "claude-code-cli",
    }


async def import_cli_credentials(pool, slot_id: str = "") -> bool:
    saved = _load_cli_oauth()
    if not saved:
        return False
    target = pool.get(slot_id) if slot_id else None
    if target is None:
        existing = pool.by_provider(PROVIDER)
        target = existing[0] if existing else None
    if target is None:
        target = await pool.create_slot(PROVIDER, alias="Claude Code", slot_id=slot_id or "anthropic-1")
    target.token_store.save(saved)
    tier = saved.get("subscription_type") or saved.get("rate_limit_tier") or ""
    if tier and target.meta.tier != tier:
        target.meta.tier = tier
        target.save_meta()
    await pool.mark_valid(target)
    logger.info("Imported Claude Code OAuth credentials into slot %s", target.slot_id)
    return True


async def import_env_api_key(pool, slot_id: str = "") -> bool:
    api_key = SETTINGS.anthropic_api_key.strip()
    if not api_key:
        return False
    target = pool.get(slot_id) if slot_id else None
    if target is None:
        existing = pool.by_provider(PROVIDER)
        target = existing[0] if existing else None
    if target is None:
        target = await pool.create_slot(PROVIDER, alias="Claude API", slot_id=slot_id or "anthropic-1")
    target.token_store.save_api_key(api_key)
    if target.meta.tier != "api_key":
        target.meta.tier = "api_key"
        target.save_meta()
    await pool.mark_valid(target)
    logger.info("Imported Anthropic API key into slot %s", target.slot_id)
    return True


async def save_api_key_for_slot(pool, slot_id: str, api_key: str) -> bool:
    target = pool.get(slot_id)
    if target is None or target.provider != PROVIDER:
        return False
    target.token_store.save_api_key(api_key)
    if target.meta.tier != "api_key":
        target.meta.tier = "api_key"
        target.save_meta()
    await pool.mark_valid(target)
    return True


# ---------------------------------------------------------------------------
# OAuth
# ---------------------------------------------------------------------------


def build_auth_url(state: str, verifier: str, port: int) -> str:
    import base64 as _b64
    import hashlib as _h
    challenge = _b64.urlsafe_b64encode(
        _h.sha256(verifier.encode("utf-8")).digest()
    ).rstrip(b"=").decode("ascii")
    redirect_uri = f"http://localhost:{port}{CALLBACK_PATH}"
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPES,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    return f"{AUTH_ENDPOINT}?{urlencode(params)}"


async def exchange_code(code: str, verifier: str, redirect_uri: str) -> Dict[str, Any]:
    client = await get_http_client()
    request_body = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": CLIENT_ID,
        "code_verifier": verifier,
    }
    resp = await client.post(
        TOKEN_ENDPOINT,
        json=request_body,
        headers={"Content-Type": "application/json"},
        timeout=30.0,
    )
    if resp.status_code == 400 and "invalid request" in resp.text.lower():
        resp = await client.post(
            TOKEN_ENDPOINT,
            data=request_body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30.0,
        )
    resp.raise_for_status()
    return resp.json()


def build_saved_payload(token_payload: Dict[str, Any]) -> Dict[str, Any]:
    expires_in = int(token_payload.get("expires_in") or 3600)
    return {
        "access_token": token_payload.get("access_token") or "",
        "refresh_token": token_payload.get("refresh_token") or "",
        "expires_at": time.time() + expires_in,
        "email": token_payload.get("account", {}).get("email_address", "")
        if isinstance(token_payload.get("account"), dict) else "",
        "account_id": token_payload.get("account", {}).get("uuid", "")
        if isinstance(token_payload.get("account"), dict) else "",
        "scope": token_payload.get("scope", ""),
    }


async def start_login(slot_id: str) -> str:
    from accounts import POOL
    if POOL.get(slot_id) is None:
        raise ValueError(f"Unknown slot: {slot_id}")
    if await import_env_api_key(POOL, slot_id):
        return f"http://{SETTINGS.host}:{SETTINGS.port}/"
    if not SETTINGS.anthropic_allow_legacy_oauth:
        raise ValueError("Claude now requires an Anthropic API key. Set ANTHROPIC_API_KEY or save a key in the UI.")
    if await import_cli_credentials(POOL, slot_id):
        return f"http://{SETTINGS.host}:{SETTINGS.port}/"
    port = CALLBACK_BROKER.ensure_server()
    verifier, _ = generate_pkce_pair()
    state = generate_state()
    redirect_uri = f"http://localhost:{port}{CALLBACK_PATH}"
    CALLBACK_BROKER.add_pending(PendingAuth(
        state=state, slot_id=slot_id, provider=PROVIDER,
        verifier=verifier, redirect_uri=redirect_uri,
    ))
    return build_auth_url(state, verifier, port)


async def handle_callback(code: str, query: Dict[str, Any], pa: PendingAuth) -> bool:
    from accounts import POOL
    acc = POOL.get(pa.slot_id)
    if acc is None:
        return False
    try:
        payload = await exchange_code(code, pa.verifier, pa.redirect_uri)
        saved = build_saved_payload(payload)
        acc.token_store.save(saved)
        await POOL.mark_valid(acc)
        logger.info("Claude slot %s signed in (%s)", pa.slot_id, saved.get("email"))
        return True
    except Exception:
        logger.exception("Claude token exchange failed")
        return False


# ---------------------------------------------------------------------------
# Request mapping: OpenAI Chat <-> Anthropic Messages
# ---------------------------------------------------------------------------


def _text_block(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text}


def _convert_image_url(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    iu = item.get("image_url")
    url = None
    detail = "auto"
    if isinstance(iu, dict):
        url = iu.get("url")
        detail = iu.get("detail", "auto")
    elif isinstance(iu, str):
        url = iu
    if not url:
        return None
    if url.startswith("data:") and "," in url:
        header, data = url.split(",", 1)
        mime = header[5:].split(";", 1)[0] or "image/png"
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": mime, "data": data},
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def _convert_user_content(content: Any) -> List[Dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [_text_block(content)] if content else []
    if not isinstance(content, list):
        return [_text_block(str(content))]
    out: List[Dict[str, Any]] = []
    for it in content:
        if isinstance(it, str):
            if it:
                out.append(_text_block(it))
            continue
        if not isinstance(it, dict):
            continue
        t = it.get("type")
        if t in ("text", "input_text"):
            txt = it.get("text") or ""
            if txt:
                out.append(_text_block(txt))
        elif t in ("image_url", "input_image", "image"):
            blk = _convert_image_url(it)
            if blk:
                out.append(blk)
        elif t in ("document", "input_file", "file", "pdf"):
            # Anthropic documents support PDF (base64 application/pdf)
            file_obj = it.get("file") if isinstance(it.get("file"), dict) else it
            file_data = file_obj.get("file_data")
            file_url = file_obj.get("file_url") or file_obj.get("url")
            if isinstance(file_data, str):
                payload = file_data
                mime = "application/pdf"
                if payload.startswith("data:"):
                    header, data = payload.split(",", 1)
                    mime = header[5:].split(";", 1)[0] or mime
                    payload = data
                out.append({
                    "type": "document",
                    "source": {"type": "base64", "media_type": mime, "data": payload},
                })
            elif isinstance(file_url, str):
                out.append({"type": "document", "source": {"type": "url", "url": file_url}})
    return out


def _convert_assistant_content(content: Any) -> List[Dict[str, Any]]:
    # Anthropic assistant blocks accept text + tool_use
    if isinstance(content, str):
        return [_text_block(content)] if content else []
    if not isinstance(content, list):
        return [_text_block(str(content))]
    out: List[Dict[str, Any]] = []
    for it in content:
        if isinstance(it, dict) and it.get("type") in ("text", "output_text"):
            txt = it.get("text") or ""
            if txt:
                out.append(_text_block(txt))
    return out


def _messages_to_anthropic(messages: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], str]:
    system_chunks: List[str] = []
    anth_messages: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "user").lower()
        if role in ("system", "developer"):
            txt = m.get("content")
            if isinstance(txt, str):
                if txt:
                    system_chunks.append(txt)
            elif isinstance(txt, list):
                for it in txt:
                    if isinstance(it, dict) and it.get("type") in ("text", "input_text") and it.get("text"):
                        system_chunks.append(it["text"])
            continue

        if role == "tool":
            tool_id = m.get("tool_call_id") or m.get("tool_use_id") or ""
            content = m.get("content")
            tool_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            anth_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": tool_text,
                }],
            })
            continue

        if role == "assistant":
            blocks = _convert_assistant_content(m.get("content"))
            for tc in m.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args_obj = json.loads(args)
                    except Exception:
                        args_obj = {"_raw": args}
                else:
                    args_obj = args or {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": fn.get("name") or "",
                    "input": args_obj,
                })
            if blocks:
                anth_messages.append({"role": "assistant", "content": blocks})
            continue

        # user (and unknown roles treated as user)
        blocks = _convert_user_content(m.get("content"))
        if blocks:
            anth_messages.append({"role": "user", "content": blocks})

    return anth_messages, "\n\n".join(system_chunks)


def _convert_tools(tools: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(tools, list):
        return None
    out: List[Dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        fn = t.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        schema = fn.get("parameters") or {"type": "object", "properties": {}}
        block: Dict[str, Any] = {
            "name": name,
            "description": fn.get("description") or "",
            "input_schema": schema,
        }
        out.append(block)
    return out or None


def _convert_tool_choice(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, str):
        if value == "auto":
            return {"type": "auto"}
        if value == "none":
            return None
        if value == "required":
            return {"type": "any"}
        return None
    if isinstance(value, dict) and value.get("type") == "function":
        fn = value.get("function") or {}
        name = fn.get("name")
        if name:
            return {"type": "tool", "name": name}
    return None


def build_messages_body(req: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
    raw_model = req.get("model")
    model = normalize_model(str(raw_model or ""))
    messages = req.get("messages") or []
    anth_msgs, system = _messages_to_anthropic(messages)

    body: Dict[str, Any] = {
        "model": model,
        "messages": anth_msgs or [{"role": "user", "content": [_text_block("")]}],
    }
    if system:
        body["system"] = system

    # max_tokens is REQUIRED for Anthropic API
    max_tokens = (
        req.get("max_output_tokens")
        or req.get("max_completion_tokens")
        or req.get("max_tokens")
        or 8192
    )
    try:
        body["max_tokens"] = int(max_tokens)
    except (TypeError, ValueError):
        body["max_tokens"] = 8192

    for k in ("temperature", "top_p", "top_k", "stop_sequences"):
        if req.get(k) is not None:
            body[k] = req[k]
    stop = req.get("stop")
    if isinstance(stop, str):
        body["stop_sequences"] = [stop]
    elif isinstance(stop, list):
        body["stop_sequences"] = [str(s) for s in stop]

    tools = _convert_tools(req.get("tools"))
    if tools:
        body["tools"] = tools
    tc = _convert_tool_choice(req.get("tool_choice"))
    if tc:
        body["tool_choice"] = tc

    # Extended thinking — reasoning effort -> thinking budget
    re_effort = req.get("reasoning_effort") or nested_get(req, "reasoning", "effort") \
        or nested_get(req, "extra_body", "reasoning_effort")
    if re_effort:
        budget = {"low": 1024, "medium": 4096, "high": 16384, "xhigh": 32768}.get(
            str(re_effort).lower(), 4096
        )
        body["thinking"] = {"type": "enabled", "budget_tokens": budget}

    # Metadata
    metadata = req.get("metadata") or nested_get(req, "extra_body", "metadata")
    if isinstance(metadata, dict) and metadata.get("user_id"):
        body["metadata"] = {"user_id": str(metadata["user_id"])}

    return body, model


def _anthropic_headers(token: str, auth_type: str = "api_key") -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "anthropic-version": ANTHROPIC_VERSION,
    }
    if auth_type == "api_key":
        headers["x-api-key"] = token
        if SETTINGS.anthropic_beta:
            headers["anthropic-beta"] = SETTINGS.anthropic_beta
    else:
        headers["Authorization"] = f"Bearer {token}"
        headers["anthropic-beta"] = SETTINGS.anthropic_oauth_beta or OAUTH_BETA
    return headers


def _header_value(headers: Any, *names: str) -> str:
    for name in names:
        try:
            value = headers.get(name)
        except Exception:
            value = None
        if value not in (None, ""):
            return str(value)
    return ""


def _retry_after_seconds(headers: Any) -> int:
    raw = _header_value(headers, "retry-after")
    if not raw:
        return 0
    parsed = safe_int(raw)
    return parsed or 0


def _rate_limit_dimension(headers: Any, dim: str) -> Optional[Dict[str, Any]]:
    prefix = f"anthropic-ratelimit-{dim}"
    limit = safe_int(_header_value(headers, f"{prefix}-limit"))
    remaining = safe_int(_header_value(headers, f"{prefix}-remaining"))
    reset = _header_value(headers, f"{prefix}-reset")
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


def _extract_rate_limit(headers: Any, model: str) -> Dict[str, Any]:
    limits: Dict[str, Dict[str, Any]] = {}
    for dim in ("requests", "tokens", "input-tokens", "output-tokens"):
        parsed = _rate_limit_dimension(headers, dim)
        if parsed:
            limits[dim] = parsed
    retry_after = _retry_after_seconds(headers)
    request_id = _header_value(headers, "request-id", "x-request-id")
    if not limits and not retry_after and not request_id:
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
    if request_id:
        out["request_id"] = request_id
    return out


def _classify_error(status: int, raw_body: str, model: str, headers: Any = None) -> Exception:
    try:
        payload = json.loads(raw_body) if raw_body else {}
    except Exception:
        payload = {}
    inner = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(inner, dict):
        inner = payload if isinstance(payload, dict) else {}
    err_msg = inner.get("message") or raw_body or ""

    if status == 401:
        return AuthRevoked(err_msg, status_code=status, raw_body=raw_body)
    if status == 429:
        resets_at = time.time() + (_retry_after_seconds(headers) or 60)
        return QuotaExhausted(err_msg, status_code=status, resets_at=resets_at,
                              reason="rate_limited", raw_body=raw_body)
    if status == 403 and "credit" in err_msg.lower():
        return QuotaExhausted(err_msg, status_code=status,
                              resets_at=time.time() + 3600,
                              reason="credit_exhausted", raw_body=raw_body)
    if 500 <= status < 600:
        return UpstreamServerError(err_msg, status_code=status, raw_body=raw_body)
    return UpstreamClientError(err_msg, status_code=status, raw_body=raw_body)


def _usage_to_openai(usage: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(usage, dict):
        return {}
    inp = int(usage.get("input_tokens") or 0)
    out = int(usage.get("output_tokens") or 0)
    cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    total = inp + out + cache_creation + cache_read
    result: Dict[str, Any] = {
        "prompt_tokens": inp + cache_creation + cache_read,
        "completion_tokens": out,
        "total_tokens": total,
    }
    if cache_read:
        result["prompt_tokens_details"] = {"cached_tokens": cache_read}
    return result


def _map_stop_reason(reason: Optional[str], has_tool: bool) -> str:
    r = (reason or "").lower()
    if has_tool or r == "tool_use":
        return "tool_calls"
    if r == "max_tokens":
        return "length"
    if r in ("stop_sequence", "end_turn"):
        return "stop"
    return "stop"


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


async def complete_completion(req: Dict[str, Any], account) -> Dict[str, Any]:
    from accounts import POOL
    auth_type, token = await account.token_store.get_credential()
    body, model = build_messages_body(req)
    created = int(time.time())
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    client = await get_http_client()
    resp = await client.post(
        MESSAGES_ENDPOINT,
        headers=_anthropic_headers(token, auth_type),
        json=body,
        timeout=120.0,
    )
    rl = _extract_rate_limit(resp.headers, model)
    if rl:
        await POOL.record_rate_limit(account, rl)
    if resp.status_code >= 400:
        raise _classify_error(resp.status_code, resp.text, model, resp.headers)
    data = resp.json()

    text_parts: List[str] = []
    reasoning_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for block in data.get("content") or []:
        if not isinstance(block, dict):
            continue
        bt = block.get("type")
        if bt == "text":
            text_parts.append(block.get("text") or "")
        elif bt == "thinking":
            reasoning_parts.append(block.get("thinking") or "")
        elif bt == "tool_use":
            tool_calls.append({
                "id": block.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": block.get("name") or "",
                    "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                },
            })

    message: Dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts).strip()
    if tool_calls:
        message["tool_calls"] = tool_calls
    finish = _map_stop_reason(data.get("stop_reason"), bool(tool_calls))
    result = {
        "id": response_id, "object": "chat.completion",
        "created": created, "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": _usage_to_openai(data.get("usage") or {}),
    }
    if rl:
        result["_bridge_rate_limit"] = rl
    return result


async def stream_completion(req: Dict[str, Any], account) -> AsyncGenerator[bytes, None]:
    from accounts import POOL
    auth_type, token = await account.token_store.get_credential()
    body, model = build_messages_body(req)
    body["stream"] = True
    created = int(time.time())
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    include_usage = bool((req.get("stream_options") or {}).get("include_usage"))

    tool_index_map: Dict[int, int] = {}  # anthropic block_index -> OpenAI tool_calls index
    next_tool_idx = 0
    emitted_tool = False
    last_usage: Dict[str, Any] = {}
    last_stop_reason: Optional[str] = None

    client = await get_http_client()
    async with client.stream(
        "POST", MESSAGES_ENDPOINT,
        headers=_anthropic_headers(token, auth_type),
        json=body,
        timeout=None,
    ) as resp:
        rl = _extract_rate_limit(resp.headers, model)
        if rl:
            await POOL.record_rate_limit(account, rl)
        if resp.status_code >= 400:
            raw = await resp.aread()
            raise _classify_error(resp.status_code, raw.decode("utf-8", "ignore"), model, resp.headers)

        async for line in resp.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            raw = line[6:]
            if raw == "[DONE]":
                break
            try:
                evt = json.loads(raw)
            except Exception:
                continue
            etype = evt.get("type")

            if etype == "message_start":
                msg = evt.get("message") or {}
                u = msg.get("usage") or {}
                if u:
                    last_usage = _usage_to_openai(u)

            elif etype == "content_block_start":
                block = evt.get("content_block") or {}
                bt = block.get("type")
                idx = evt.get("index", 0)
                if bt == "tool_use":
                    tool_idx = next_tool_idx
                    next_tool_idx += 1
                    tool_index_map[idx] = tool_idx
                    emitted_tool = True
                    yield sse_data({
                        "id": response_id, "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"tool_calls": [{
                                "index": tool_idx,
                                "id": block.get("id"),
                                "type": "function",
                                "function": {"name": block.get("name") or "", "arguments": ""},
                            }]},
                            "finish_reason": None,
                        }],
                    })

            elif etype == "content_block_delta":
                delta = evt.get("delta") or {}
                dt = delta.get("type")
                idx = evt.get("index", 0)
                if dt == "text_delta":
                    txt = delta.get("text") or ""
                    if txt:
                        yield sse_data({
                            "id": response_id, "object": "chat.completion.chunk",
                            "created": created, "model": model,
                            "choices": [{"index": 0, "delta": {"content": txt}, "finish_reason": None}],
                        })
                elif dt == "thinking_delta":
                    txt = delta.get("thinking") or ""
                    if txt:
                        yield sse_data({
                            "id": response_id, "object": "chat.completion.chunk",
                            "created": created, "model": model,
                            "choices": [{"index": 0, "delta": {"reasoning_content": txt}, "finish_reason": None}],
                        })
                elif dt == "input_json_delta":
                    partial = delta.get("partial_json") or ""
                    tool_idx = tool_index_map.get(idx)
                    if partial and tool_idx is not None:
                        yield sse_data({
                            "id": response_id, "object": "chat.completion.chunk",
                            "created": created, "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"tool_calls": [{
                                    "index": tool_idx,
                                    "function": {"arguments": partial},
                                }]},
                                "finish_reason": None,
                            }],
                        })

            elif etype == "message_delta":
                d = evt.get("delta") or {}
                if d.get("stop_reason"):
                    last_stop_reason = d["stop_reason"]
                u = evt.get("usage") or {}
                if u:
                    # message_delta carries cumulative output token count
                    merged = dict(last_usage) if last_usage else {}
                    out_tokens = int(u.get("output_tokens") or 0)
                    if out_tokens:
                        merged["completion_tokens"] = out_tokens
                        in_tok = merged.get("prompt_tokens", 0)
                        merged["total_tokens"] = in_tok + out_tokens
                    last_usage = merged

            elif etype == "message_stop":
                break

    finish = _map_stop_reason(last_stop_reason, emitted_tool)
    yield sse_data({
        "id": response_id, "object": "chat.completion.chunk",
        "created": created, "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
    })
    if include_usage and last_usage:
        usage_chunk = {
            "id": response_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [], "usage": last_usage,
        }
        if rl:
            usage_chunk["_bridge_rate_limit"] = rl
        yield sse_data(usage_chunk)
    yield sse_done()


def register(pool) -> None:
    pool.register_provider(PROVIDER, AnthropicTokenStore)
    CALLBACK_BROKER.register_handler(CALLBACK_PATH, handle_callback)
