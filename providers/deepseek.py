# -*- coding: utf-8 -*-
"""DeepSeek provider.

DeepSeek exposes an OpenAI-compatible Chat Completions API. This provider
keeps the bridge contract simple: store one API key per account slot, forward
OpenAI-compatible request bodies, and let the bridge handle account/model
rotation and failover.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from core import (
    AuthRevoked,
    QuotaExhausted,
    SETTINGS,
    UpstreamClientError,
    UpstreamServerError,
    get_http_client,
    logger,
    safe_int,
)

PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-v4-flash"

MODEL_CATALOG: List[Dict[str, Any]] = [
    {"id": "deepseek-v4-flash", "name": "DeepSeek V4 Flash",
     "description": "Fast, cost-efficient DeepSeek V4 model."},
    {"id": "deepseek-v4-pro", "name": "DeepSeek V4 Pro",
     "description": "DeepSeek V4 model for harder reasoning and coding tasks."},
    {"id": "deepseek-chat", "name": "DeepSeek Chat (legacy)",
     "description": "Legacy alias for non-thinking DeepSeek V4 Flash; deprecated after 2026-07-24."},
    {"id": "deepseek-reasoner", "name": "DeepSeek Reasoner (legacy)",
     "description": "Legacy alias for thinking DeepSeek V4 Flash; deprecated after 2026-07-24."},
]

MODEL_ALIASES: Dict[str, str] = {
    "deepseek": DEFAULT_MODEL,
    "deepseek-flash": "deepseek-v4-flash",
    "deepseek-pro": "deepseek-v4-pro",
    "deepseek-v4": "deepseek-v4-flash",
}


def normalize_model(model: Optional[str], default: Optional[str] = None) -> str:
    raw = (model or "").strip().lower()
    if not raw:
        return default or SETTINGS.deepseek_default_model or DEFAULT_MODEL
    return MODEL_ALIASES.get(raw, raw)


def build_models_list() -> List[Dict[str, Any]]:
    created = int(time.time())
    return [
        {
            "id": m["id"],
            "object": "model",
            "created": created,
            "owned_by": "deepseek",
            "name": m["name"],
            "description": m["description"],
            "provider": PROVIDER,
        }
        for m in MODEL_CATALOG
    ]


class DeepSeekTokenStore:
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
            logger.warning("Bad deepseek oauth.json: %s", exc)
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
        api_key = data.get("api_key") or ""
        return {
            "logged_in": bool(api_key),
            "email": data.get("email") or "",
            "account_id": data.get("account_id") or "",
            "expires_at": 0,
            "expires_in": 0,
            "has_refresh_token": False,
            "has_saved_session": bool(api_key),
            "session_state": "active" if api_key else "missing",
            "session_notice": "DeepSeek API key configured." if api_key else "Add a DeepSeek API key.",
            "auth_type": "api_key" if api_key else "",
        }

    async def get_access_token(self) -> str:
        async with self._lock:
            api_key = self.load().get("api_key") or ""
            if not api_key:
                raise RuntimeError("DeepSeek API key is missing.")
            return api_key

    async def save_api_key(self, api_key: str) -> None:
        async with self._lock:
            data = self.load()
            data.update({
                "api_key": api_key.strip(),
                "created_at": data.get("created_at") or time.time(),
                "updated_at": time.time(),
            })
            self.save(data)


def register(pool) -> None:
    pool.register_provider(PROVIDER, lambda folder: DeepSeekTokenStore(folder))


async def import_env_api_key(pool) -> None:
    api_key = SETTINGS.deepseek_api_key
    if not api_key:
        return
    for acc in pool.by_provider(PROVIDER):
        if isinstance(acc.token_store, DeepSeekTokenStore) and acc.token_store.load().get("api_key"):
            return
    acc = await pool.create_slot(provider=PROVIDER, alias="DeepSeek API")
    await acc.token_store.save_api_key(api_key)
    logger.info("Imported DEEPSEEK_API_KEY into slot %s", acc.slot_id)


async def start_login(slot_id: str) -> str:
    raise ValueError("DeepSeek uses an API key. Click API key in the UI and paste DEEPSEEK_API_KEY.")


async def save_api_key_for_slot(pool, slot_id: str, api_key: str) -> bool:
    acc = pool.get(slot_id)
    if acc is None or acc.provider != PROVIDER:
        return False
    await acc.token_store.save_api_key(api_key)
    await pool.mark_valid(acc)
    return True


def _chat_endpoint() -> str:
    return f"{SETTINGS.deepseek_base_url.rstrip('/')}/chat/completions"


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }


def _extract_rate_limit(headers: Any, model: str) -> Dict[str, Any]:
    limits: Dict[str, Dict[str, Any]] = {}
    for name in ("requests", "tokens"):
        limit = safe_int(headers.get(f"x-ratelimit-limit-{name}"))
        remaining = safe_int(headers.get(f"x-ratelimit-remaining-{name}"))
        reset = headers.get(f"x-ratelimit-reset-{name}") or ""
        if limit is not None or remaining is not None or reset:
            item: Dict[str, Any] = {}
            if limit is not None:
                item["limit"] = limit
            if remaining is not None:
                item["remaining"] = remaining
            if limit and remaining is not None:
                item["percent_remaining"] = round((remaining / limit) * 100, 2)
            if reset:
                item["reset"] = reset
            limits[name] = item

    info: Dict[str, Any] = {
        "provider": PROVIDER,
        "model": model,
        "updated_at": int(time.time()),
    }
    retry_after = safe_int(headers.get("retry-after"))
    if retry_after is not None:
        info["retry_after_seconds"] = retry_after
    request_id = headers.get("x-request-id") or headers.get("cf-ray") or ""
    if request_id:
        info["request_id"] = request_id
    if limits:
        info["limits"] = limits
    return info


def _classify_error(status: int, raw_body: str, model: str, headers: Any) -> Exception:
    try:
        payload = json.loads(raw_body) if raw_body else {}
    except Exception:
        payload = {}
    inner = payload.get("error") if isinstance(payload, dict) else {}
    message = ""
    if isinstance(inner, dict):
        message = str(inner.get("message") or inner.get("code") or "")
    message = message or raw_body[:500] or f"DeepSeek HTTP {status}"
    retry_after = safe_int(headers.get("retry-after")) or 0
    if status in {401, 403}:
        return AuthRevoked(message, status_code=status, raw_body=raw_body)
    if status in {402, 429}:
        resets_at = time.time() + (retry_after if retry_after > 0 else 3600)
        return QuotaExhausted(
            message,
            status_code=status,
            resets_at=resets_at,
            reason="quota_or_rate_limit",
            raw_body=raw_body,
        )
    if status >= 500:
        return UpstreamServerError(message, status_code=status, raw_body=raw_body)
    return UpstreamClientError(message, status_code=status, raw_body=raw_body)


async def complete_completion(req: Dict[str, Any], account) -> Dict[str, Any]:
    from accounts import POOL

    model = normalize_model(str(req.get("model") or ""), SETTINGS.deepseek_default_model)
    body = dict(req)
    body["model"] = model
    body["stream"] = False

    api_key = await account.token_store.get_access_token()
    client = await get_http_client()
    resp = await client.post(_chat_endpoint(), headers=_headers(api_key), json=body)
    rate = _extract_rate_limit(resp.headers, model)
    if rate:
        await POOL.record_rate_limit(account, rate)
    text = resp.text
    if resp.status_code >= 400:
        raise _classify_error(resp.status_code, text, model, resp.headers)
    try:
        data = resp.json()
    except Exception:
        raise UpstreamServerError("DeepSeek returned a non-JSON response", raw_body=text)
    data.setdefault("model", model)
    data["_bridge_rate_limit"] = rate
    return data


async def stream_completion(req: Dict[str, Any], account) -> AsyncGenerator[bytes, None]:
    from accounts import POOL

    model = normalize_model(str(req.get("model") or ""), SETTINGS.deepseek_default_model)
    body = dict(req)
    body["model"] = model
    body["stream"] = True

    api_key = await account.token_store.get_access_token()
    client = await get_http_client()
    async with client.stream("POST", _chat_endpoint(), headers=_headers(api_key), json=body) as resp:
        rate = _extract_rate_limit(resp.headers, model)
        if rate:
            await POOL.record_rate_limit(account, rate)
        if resp.status_code >= 400:
            raw = (await resp.aread()).decode("utf-8", errors="replace")
            raise _classify_error(resp.status_code, raw, model, resp.headers)
        async for chunk in resp.aiter_raw():
            if chunk:
                yield chunk
