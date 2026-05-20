# -*- coding: utf-8 -*-
"""Unified multi-provider OAuth bridge.

Routes:
- GET  /                              UI (account grid, per-slot login)
- GET  /api/accounts                  list all account slots + health
- POST /api/accounts                  create a new slot
- DELETE /api/accounts/{slot_id}      remove a slot
- POST /api/accounts/{slot_id}/login  begin OAuth flow for that slot
- POST /api/accounts/{slot_id}/refresh   force refresh
- POST /api/accounts/{slot_id}/logout    clear tokens
- PATCH /api/accounts/{slot_id}       update alias/enabled
- GET  /health                        liveness
- GET  /v1/models                     aggregated model list
- POST /v1/chat/completions           with failover across pool
- POST /v1/messages                   Anthropic-compatible shim for Claude Code
- POST /v1/messages/count_tokens      approximate Anthropic token count
- POST /v1/audio/transcriptions       Whisper-compatible (Codex backend)
- GET/POST /v1/oauth/token            return live bearer (per slot or first)
- GET  /.well-known/openai-bridge     discovery
"""
from __future__ import annotations

import asyncio
import copy
import json
import random
import re
import time
import uuid
from contextlib import asynccontextmanager, suppress
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from fastapi import Body, FastAPI, File, Form, HTTPException, Header, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from accounts import POOL, migrate_legacy_data, resolve_provider
from auth_keys import API_KEYS, ManagedKeyAuth
from core import (
    AuthRevoked,
    CALLBACK_BROKER,
    NoAccountAvailable,
    QuotaExhausted,
    ROUTE_GROUPS_FILE,
    SETTINGS,
    UpstreamClientError,
    UpstreamServerError,
    close_http_client,
    get_http_client,
    logger,
    sse_data,
    sse_done,
    sse_with_keepalive,
)
from providers import anthropic as p_anthropic
from providers import codex as p_codex
from providers import deepseek as p_deepseek
from providers import google as p_google

PROVIDER_EXECUTORS = {
    "chatgpt": {
        "complete": p_codex.complete_completion,
        "stream": p_codex.stream_completion,
        "start_login": p_codex.start_login,
        "models": p_codex.build_models_list,
        "label": "ChatGPT (Codex)",
    },
    "google": {
        "complete": p_google.complete_completion,
        "stream": p_google.stream_completion,
        "start_login": p_google.start_login,
        "models": p_google.build_models_list,
        "label": "Google Gemini",
    },
    "anthropic": {
        "complete": p_anthropic.complete_completion,
        "stream": p_anthropic.stream_completion,
        "start_login": p_anthropic.start_login,
        "save_api_key": p_anthropic.save_api_key_for_slot,
        "models": p_anthropic.build_models_list,
        "label": "Anthropic Claude",
    },
    "deepseek": {
        "complete": p_deepseek.complete_completion,
        "stream": p_deepseek.stream_completion,
        "start_login": p_deepseek.start_login,
        "save_api_key": p_deepseek.save_api_key_for_slot,
        "models": p_deepseek.build_models_list,
        "label": "DeepSeek",
    },
}

GROUP_MODES = {"priority", "round_robin", "random"}
ROUTE_GROUPS: Dict[str, Dict[str, Any]] = {}
_ROUTE_GROUP_RR: Dict[str, int] = {}
CLAUDE_CODE_MODEL_ROUTES = {
    "claude-opus-4-7": "gpt-5.5-xhigh",
    "claude-opus": "gpt-5.5-xhigh",
    "claude-opus-latest": "gpt-5.5-xhigh",
    "claude-sonnet-4-6": "gpt-5.5-high",
    "claude-sonnet": "gpt-5.5-high",
    "claude-sonnet-latest": "gpt-5.5-high",
    "claude-haiku-4-5": "gpt-5.5-low",
    "claude-haiku-4-5-20251001": "gpt-5.5-low",
    "claude-haiku": "gpt-5.5-low",
    "claude-haiku-latest": "gpt-5.5-low",
}
CLAUDE_CODE_LOCAL_ROUTE_GROUPS = ("claude-local", "claude-3-5-sonnet-latest")


def _provider_default_model(provider: str) -> str:
    if provider == "google":
        return SETTINGS.google_default_model
    if provider == "anthropic":
        return SETTINGS.anthropic_default_model
    if provider == "deepseek":
        return SETTINGS.deepseek_default_model
    return SETTINGS.codex_default_model


def _provider_models() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for prov, execs in PROVIDER_EXECUTORS.items():
        for item in execs["models"]():
            model = dict(item)
            model["provider"] = model.get("provider") or prov
            out.append(model)
    return out


def _known_route_model_ids() -> set[str]:
    ids = {p.lower() for p in PROVIDER_EXECUTORS}
    for model in _provider_models():
        model_id = str(model.get("id") or "").strip().lower()
        if model_id:
            ids.add(model_id)
    return ids


def _normalize_group_name(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9_.-]+", "-", raw).strip("-")


def _normalize_group_mode(value: Any) -> str:
    mode = str(value or "priority").strip().lower()
    return mode if mode in GROUP_MODES else "priority"


def _normalize_route_item(raw: Any) -> Dict[str, str]:
    if isinstance(raw, str):
        value = raw.strip()
        if value.startswith("@"):
            group = _normalize_group_name(value[1:])
            if not group:
                raise ValueError("Empty group item")
            return {"kind": "group", "group": group}
        provider = value.lower()
        if provider in PROVIDER_EXECUTORS:
            return {"kind": "provider", "provider": provider}
        model = value
        return {
            "kind": "model",
            "provider": resolve_provider(model),
            "model": model,
        }

    if not isinstance(raw, dict):
        raise ValueError("Route item must be an object or string")

    kind = str(raw.get("kind") or raw.get("type") or "").strip().lower()
    provider = str(raw.get("provider") or "").strip().lower()

    if kind == "group":
        group = _normalize_group_name(raw.get("group") or raw.get("name"))
        if not group:
            raise ValueError("Empty group item")
        return {"kind": "group", "group": group}

    if kind == "provider" or (provider and not raw.get("model")):
        provider = provider or str(raw.get("id") or "").strip().lower()
        if provider not in PROVIDER_EXECUTORS:
            raise ValueError(f"Unknown provider in route item: {provider}")
        return {"kind": "provider", "provider": provider}

    model = str(raw.get("model") or raw.get("id") or "").strip()
    if not model:
        raise ValueError("Route model item is missing model")
    provider = provider or resolve_provider(model)
    if provider not in PROVIDER_EXECUTORS:
        raise ValueError(f"Unknown provider in model route item: {provider}")
    if model.lower() in PROVIDER_EXECUTORS:
        return {"kind": "provider", "provider": model.lower()}
    return {"kind": "model", "provider": provider, "model": model}


def _normalize_group_config(
    name_value: Any,
    body: Dict[str, Any],
    *,
    strict_group_refs: bool = True,
) -> Dict[str, Any]:
    name = _normalize_group_name(name_value or body.get("name"))
    if not name:
        raise ValueError("Missing group name")
    if name in PROVIDER_EXECUTORS:
        raise ValueError("Group name cannot be a provider id")
    if name in _known_route_model_ids():
        raise ValueError("Group name cannot be the same as a model id")

    raw_items = body.get("items")
    if raw_items is None:
        raw_items = body.get("providers")
    if raw_items is None:
        raw_items = body.get("models")
    if not isinstance(raw_items, list):
        raise ValueError("Group needs an items list")

    items: List[Dict[str, str]] = []
    for raw_item in raw_items:
        item = _normalize_route_item(raw_item)
        if item.get("kind") == "group":
            ref = item.get("group") or ""
            if ref == name:
                raise ValueError("Group cannot contain itself")
            if strict_group_refs and ref not in ROUTE_GROUPS:
                raise ValueError(f"Unknown nested group: {ref}")
        items.append(item)
    if not items:
        raise ValueError("Group needs at least one item")

    return {"name": name, "mode": _normalize_group_mode(body.get("mode")), "items": items}


def _load_route_groups() -> None:
    ROUTE_GROUPS.clear()
    if not ROUTE_GROUPS_FILE.exists():
        return
    try:
        raw = json.loads(ROUTE_GROUPS_FILE.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Cannot parse route_groups.json")
        return
    raw_groups = raw.get("groups") if isinstance(raw, dict) else raw
    if isinstance(raw_groups, dict):
        iterator = raw_groups.items()
    elif isinstance(raw_groups, list):
        iterator = ((item.get("name"), item) for item in raw_groups if isinstance(item, dict))
    else:
        return
    for name, cfg in iterator:
        try:
            group = _normalize_group_config(name, cfg, strict_group_refs=False)
        except Exception as exc:
            logger.warning("Skipping route group %r: %s", name, exc)
            continue
        ROUTE_GROUPS[group["name"]] = group


def _save_route_groups() -> None:
    payload = {
        "version": 1,
        "groups": {
            name: {"mode": cfg["mode"], "items": cfg["items"]}
            for name, cfg in ROUTE_GROUPS.items()
        },
    }
    ROUTE_GROUPS_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _route_groups_payload() -> List[Dict[str, Any]]:
    return [
        {"name": name, "mode": cfg["mode"], "items": list(cfg["items"])}
        for name, cfg in ROUTE_GROUPS.items()
    ]


def _ordered_group_items(group_name: str, seen: Optional[set[str]] = None) -> List[Dict[str, str]]:
    seen = set(seen or set())
    if group_name in seen:
        return []
    cfg = ROUTE_GROUPS.get(group_name)
    if not cfg:
        return []
    seen.add(group_name)
    items = list(cfg.get("items") or [])
    mode = cfg.get("mode") or "priority"
    if len(items) > 1 and mode == "random":
        items = random.sample(items, len(items))
    elif len(items) > 1 and mode == "round_robin":
        idx = _ROUTE_GROUP_RR.get(group_name, 0) % len(items)
        _ROUTE_GROUP_RR[group_name] = idx + 1
        items = items[idx:] + items[:idx]

    out: List[Dict[str, str]] = []
    for item in items:
        if item.get("kind") == "group":
            out.extend(_ordered_group_items(item.get("group") or "", seen))
        else:
            out.append(item)
    return out


def _attempt_from_item(item: Dict[str, str]) -> Optional[Dict[str, str]]:
    kind = item.get("kind")
    provider = item.get("provider") or ""
    if provider not in PROVIDER_EXECUTORS:
        return None
    if kind == "provider":
        model = _provider_default_model(provider)
        return {
            "provider": provider,
            "model": model,
            "label": provider,
            "kind": "provider",
        }
    model = item.get("model") or _provider_default_model(provider)
    return {
        "provider": provider,
        "model": model,
        "label": f"{provider}/{model}",
        "kind": "model",
    }


def _group_route_plan(name: str, *, kind: str = "group", display_name: Optional[str] = None) -> Dict[str, Any]:
    attempts = [
        attempt
        for item in _ordered_group_items(name)
        for attempt in [_attempt_from_item(item)]
        if attempt is not None
    ]
    providers = []
    for attempt in attempts:
        provider = attempt["provider"]
        if provider not in providers:
            providers.append(provider)
    return {
        "name": display_name or name,
        "kind": kind,
        "providers": providers,
        "attempts": attempts,
    }


def _claude_code_model_route(key: str) -> Optional[Dict[str, Any]]:
    mapped_model = CLAUDE_CODE_MODEL_ROUTES.get(key)
    if not mapped_model:
        return None
    return {
        "name": key,
        "kind": "claude_code_alias",
        "providers": ["chatgpt"],
        "attempts": [{
            "provider": "chatgpt",
            "model": mapped_model,
            "label": f"claude-alias/{key}->{mapped_model}",
            "kind": "model",
        }],
    }


def _claude_code_local_alias(key: str) -> str:
    if key not in CLAUDE_CODE_MODEL_ROUTES:
        return ""
    for group in CLAUDE_CODE_LOCAL_ROUTE_GROUPS:
        if group in ROUTE_GROUPS:
            return group
    return ""


def _resolve_route_plan(req: Dict[str, Any]) -> Dict[str, Any]:
    model = str(req.get("model") or "").strip()
    key = _normalize_group_name(model)

    if key in PROVIDER_EXECUTORS:
        attempt = _attempt_from_item({"kind": "provider", "provider": key})
        return {
            "name": key,
            "kind": "provider",
            "providers": [key],
            "attempts": [attempt] if attempt else [],
        }

    if key in ROUTE_GROUPS:
        return _group_route_plan(key)

    claude_code_route = _claude_code_model_route(key)
    if claude_code_route:
        return claude_code_route

    local_alias = _claude_code_local_alias(key)
    if local_alias:
        return _group_route_plan(local_alias, kind="claude_code_alias", display_name=key)

    provider = resolve_provider(model)
    attempts: List[Dict[str, str]] = [
        {
            "provider": provider,
            "model": model or _provider_default_model(provider),
            "label": f"{provider}/{model or _provider_default_model(provider)}",
            "kind": "model",
        }
    ]
    for entry in SETTINGS.cross_provider_fallback:
        mapped = resolve_provider(entry, default=provider)
        if mapped in PROVIDER_EXECUTORS and mapped not in [a["provider"] for a in attempts]:
            attempts.append({
                "provider": mapped,
                "model": _provider_default_model(mapped),
                "label": mapped,
                "kind": "provider",
            })
    return {
        "name": provider,
        "kind": "model",
        "providers": [a["provider"] for a in attempts],
        "attempts": attempts,
    }


def _request_for_attempt(req: Dict[str, Any], attempt: Dict[str, str]) -> Dict[str, Any]:
    routed = dict(req)
    routed["model"] = attempt["model"]
    return routed


def _looks_like_model_error(err: UpstreamClientError) -> bool:
    text = f"{err} {getattr(err, 'raw_body', '')}".lower()
    markers = (
        "model", "not found", "not supported", "unsupported",
        "invalid model", "unknown model", "does not exist",
    )
    return any(marker in text for marker in markers)


def _should_failover_client_error(plan: Dict[str, Any], err: UpstreamClientError) -> bool:
    if plan.get("kind") == "group":
        return True
    return len(plan.get("attempts") or []) > 1 and _looks_like_model_error(err)


def _refresh_safety_seconds(provider: str) -> int:
    if provider == "google":
        return SETTINGS.google_refresh_safety_seconds
    if provider == "chatgpt":
        return SETTINGS.codex_refresh_safety_seconds
    return 0


async def _refresh_saved_tokens_once() -> None:
    for acc in POOL.all_accounts():
        if not acc.enabled:
            continue
        refresh_if_needed = getattr(acc.token_store, "refresh_if_needed", None)
        if not callable(refresh_if_needed):
            continue
        safety_seconds = _refresh_safety_seconds(acc.provider)
        if safety_seconds <= 0:
            continue
        try:
            refreshed = await refresh_if_needed(min_valid_seconds=safety_seconds)
            if refreshed:
                await POOL.mark_valid(acc)
                logger.info("Refreshed OAuth token for %s (%s)", acc.slot_id, acc.provider)
        except AuthRevoked as exc:
            await POOL.mark_invalid(acc)
            logger.warning("OAuth refresh revoked for %s: %s", acc.slot_id, exc)
        except Exception as exc:
            logger.warning("OAuth auto-refresh failed for %s (%s): %s", acc.slot_id, acc.provider, exc)


async def _token_refresh_loop() -> None:
    interval = max(60, int(SETTINGS.oauth_refresh_interval_seconds or 300))
    while True:
        await _refresh_saved_tokens_once()
        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    migrate_legacy_data()
    p_codex.register(POOL)
    p_google.register(POOL)
    p_anthropic.register(POOL)
    p_deepseek.register(POOL)
    POOL.load_from_disk()
    _load_route_groups()
    try:
        await p_anthropic.import_env_api_key(POOL)
    except Exception:
        logger.exception("Failed to import Anthropic API key")
    try:
        await p_deepseek.import_env_api_key(POOL)
    except Exception:
        logger.exception("Failed to import DeepSeek API key")
    CALLBACK_BROKER.set_loop(asyncio.get_running_loop())
    await get_http_client()
    refresh_task: Optional[asyncio.Task] = None
    if SETTINGS.oauth_auto_refresh and SETTINGS.oauth_refresh_interval_seconds > 0:
        refresh_task = asyncio.create_task(_token_refresh_loop(), name="oauth-refresh-loop")
    logger.info(
        "Bridge ready: %d accounts across providers %s",
        len(POOL.all_accounts()),
        ", ".join(POOL.known_providers()),
    )
    try:
        yield
    finally:
        # Shutdown
        if refresh_task:
            refresh_task.cancel()
            with suppress(asyncio.CancelledError):
                await refresh_task
        CALLBACK_BROKER.shutdown()
        await close_http_client()
        logger.info("Bridge stopped")


app = FastAPI(title="Multi-Provider OAuth Bridge", lifespan=lifespan)

if SETTINGS.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ---------------------------------------------------------------------------
# Optional API-key auth on bridge itself
# ---------------------------------------------------------------------------


def _extract_auth_token(authorization: Optional[str], x_api_key: Optional[str] = None) -> str:
    tokens = _extract_auth_tokens(authorization, x_api_key)
    return tokens[0] if tokens else ""


def _extract_auth_tokens(authorization: Optional[str], x_api_key: Optional[str] = None) -> List[str]:
    tokens: List[str] = []
    if authorization:
        token = authorization.removeprefix("Bearer ").strip()
        if token:
            tokens.append(token)
    if x_api_key:
        token = x_api_key.strip()
        if token:
            tokens.append(token)
    out: List[str] = []
    seen = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _check_bridge_auth(authorization: Optional[str], x_api_key: Optional[str] = None) -> None:
    if not SETTINGS.api_key:
        return
    tokens = _extract_auth_tokens(authorization, x_api_key)
    if not tokens:
        raise HTTPException(401, "Missing bridge API key")
    if SETTINGS.api_key not in tokens:
        raise HTTPException(401, "Invalid bridge API key")


def _request_client_ip(request: Request) -> str:
    if SETTINGS.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for") or ""
        if forwarded:
            return forwarded.split(",", 1)[0].strip()
        real_ip = request.headers.get("x-real-ip") or ""
        if real_ip:
            return real_ip.strip()
    return request.client.host if request.client else ""


def _all_models_payload() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    created = int(time.time())
    for prov, execs in PROVIDER_EXECUTORS.items():
        if POOL.by_provider(prov):
            for model in execs["models"]():
                item = dict(model)
                item["provider"] = item.get("provider") or prov
                out.append(item)
            out.append({
                "id": prov,
                "object": "model",
                "created": created,
                "owned_by": "bridge",
                "name": f"{execs['label']} rotation",
                "description": "Provider alias: route to this provider and rotate enabled accounts.",
                "provider": prov,
                "route_kind": "provider",
            })
    for group in _route_groups_payload():
        out.append({
            "id": group["name"],
            "object": "model",
            "created": created,
            "owned_by": "bridge",
            "name": f"Route group: {group['name']}",
            "description": "Custom model/provider rotation group.",
            "provider": "bridge",
            "route_kind": "group",
            "mode": group["mode"],
            "items": group["items"],
        })
    return out


def _usage_total_tokens(usage: Any) -> int:
    if not isinstance(usage, dict):
        return 0
    try:
        total = int(usage.get("total_tokens") or 0)
    except (TypeError, ValueError):
        total = 0
    if total > 0:
        return total
    try:
        prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, prompt + completion)


def _allowed_model_tokens(requested_model: str) -> set[str]:
    key = _normalize_group_name(requested_model)
    tokens = {requested_model.strip().lower(), key}
    for item in _provider_models():
        if str(item.get("id") or "").strip().lower() == key:
            canonical = str(item.get("canonical_model") or "").strip().lower()
            if canonical:
                tokens.add(canonical)
            provider = str(item.get("provider") or "").strip().lower()
            if provider:
                tokens.add(provider)
            break
    try:
        plan = _resolve_route_plan({"model": requested_model})
    except Exception:
        return {t for t in tokens if t}
    if plan.get("name"):
        tokens.add(str(plan["name"]).lower())
    for provider in plan.get("providers") or []:
        tokens.add(str(provider).lower())
    for attempt in plan.get("attempts") or []:
        if attempt.get("model"):
            tokens.add(str(attempt["model"]).lower())
        if attempt.get("provider"):
            tokens.add(str(attempt["provider"]).lower())
    return {t for t in tokens if t}


def _managed_key_allows_model(auth: ManagedKeyAuth, requested_model: str) -> bool:
    allowed = {str(x).strip().lower() for x in auth.allowed_models if str(x).strip()}
    if "*" in allowed:
        return True
    key = _normalize_group_name(requested_model)
    if key in allowed or requested_model.strip().lower() in allowed:
        return True
    try:
        plan = _resolve_route_plan({"model": requested_model})
    except Exception:
        return bool(allowed.intersection(_allowed_model_tokens(requested_model)))
    if str(plan.get("name") or "").lower() in allowed:
        return True
    providers = {str(p).lower() for p in plan.get("providers") or [] if p}
    if providers and providers.issubset(allowed):
        return True
    if key in ROUTE_GROUPS:
        return False
    model_tokens = _allowed_model_tokens(requested_model) - providers
    return bool(allowed.intersection(model_tokens))


def _authorize_v1(
    request: Request,
    authorization: Optional[str] = None,
    x_api_key: Optional[str] = None,
    *,
    model: Optional[str] = None,
) -> Optional[ManagedKeyAuth]:
    tokens = _extract_auth_tokens(authorization, x_api_key)
    if SETTINGS.api_key and SETTINGS.api_key in tokens:
        return None
    if not tokens:
        if SETTINGS.api_key:
            raise HTTPException(401, "Missing API key")
        return None
    record = None
    for token in tokens:
        record = API_KEYS.authenticate(token)
        if record:
            break
    if not record:
        raise HTTPException(401, "Invalid API key")
    client_ip = _request_client_ip(request)
    ok, reason = API_KEYS.check_usable(record, client_ip)
    if not ok:
        code = 429 if reason == "exhausted" else 403
        if reason == "expired":
            code = 401
        raise HTTPException(code, f"API key denied: {reason}")
    auth = ManagedKeyAuth(
        key_id=str(record.get("id") or ""),
        name=str(record.get("name") or ""),
        prefix=str(record.get("prefix") or ""),
        allowed_models=[str(x) for x in record.get("allowed_models") or []],
        token_limit=int(record.get("token_limit") or 0),
        token_used=int(record.get("token_used") or 0),
        network_scope=str(record.get("network_scope") or "local"),
        client_ip=client_ip,
    )
    if model and not _managed_key_allows_model(auth, model):
        API_KEYS.record_usage(auth.key_id, 0, client_ip, error=f"model_denied:{model}")
        raise HTTPException(403, f"API key is not allowed to use model {model}")
    return auth


def _filter_models_for_key(models: List[Dict[str, Any]], auth: Optional[ManagedKeyAuth]) -> List[Dict[str, Any]]:
    if auth is None:
        return models
    return [
        item for item in models
        if _managed_key_allows_model(auth, str(item.get("id") or ""))
    ]


def _record_managed_usage(auth: Optional[ManagedKeyAuth], tokens: int, error: str = "") -> None:
    if auth is None:
        return
    API_KEYS.record_usage(auth.key_id, tokens, auth.client_ip, error=error)


def _chat_body_with_internal_usage(body: Dict[str, Any], auth: Optional[ManagedKeyAuth]) -> Tuple[Dict[str, Any], bool]:
    if auth is None:
        return body, bool((body.get("stream_options") or {}).get("include_usage"))
    client_wants_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    if not auth.token_limit and client_wants_usage:
        return body, True
    patched = copy.deepcopy(body)
    opts = patched.get("stream_options")
    if not isinstance(opts, dict):
        opts = {}
    opts["include_usage"] = True
    patched["stream_options"] = opts
    return patched, client_wants_usage


async def _meter_openai_stream(
    stream: AsyncGenerator[bytes, None],
    auth: Optional[ManagedKeyAuth],
    *,
    expose_usage: bool,
) -> AsyncGenerator[bytes, None]:
    total_tokens = 0
    async for chunk in stream:
        hide_chunk = False
        for payload in _iter_openai_sse_payloads(chunk):
            if isinstance(payload, dict) and isinstance(payload.get("usage"), dict):
                total_tokens = max(total_tokens, _usage_total_tokens(payload.get("usage")))
                if not expose_usage and payload.get("choices") == []:
                    hide_chunk = True
        if not hide_chunk:
            yield chunk
    _record_managed_usage(auth, total_tokens, "" if total_tokens else "missing_usage")


# ---------------------------------------------------------------------------
# Failover loop
# ---------------------------------------------------------------------------


async def _execute_failover_stream(req: Dict[str, Any]) -> AsyncGenerator[bytes, None]:
    plan = _resolve_route_plan(req)
    tried: List[str] = []
    last_error: Optional[Exception] = None

    for route_attempt in plan["attempts"]:
        prov = route_attempt["provider"]
        execs = PROVIDER_EXECUTORS.get(prov)
        if not execs:
            continue
        max_attempts = SETTINGS.max_failover_attempts
        attempts = 0
        while attempts < max_attempts:
            account = await POOL.acquire(prov)
            if account is None:
                tried.append(f"{route_attempt['label']}:no-account")
                break
            attempts += 1
            success = False
            routed_req = _request_for_attempt(req, route_attempt)
            try:
                async for chunk in execs["stream"](routed_req, account):
                    success = True
                    yield chunk
                await POOL.release(account, success=True)
                return
            except QuotaExhausted as e:
                await POOL.mark_exhausted(account, e.resets_at, e.reason)
                tried.append(f"{account.slot_id}/{route_attempt['model']}:429")
                last_error = e
            except AuthRevoked as e:
                await POOL.mark_invalid(account)
                tried.append(f"{account.slot_id}/{route_attempt['model']}:401")
                last_error = e
            except UpstreamServerError as e:
                await POOL.release(account, success=False)
                tried.append(f"{account.slot_id}/{route_attempt['model']}:{e.status_code}")
                last_error = e
            except UpstreamClientError as e:
                await POOL.release(account, success=False)
                tried.append(f"{account.slot_id}/{route_attempt['model']}:{e.status_code}")
                last_error = e
                if _should_failover_client_error(plan, e):
                    break
                yield sse_data(_format_error_chunk(req, e))
                yield sse_done()
                return
            except Exception as e:
                logger.exception("Unexpected error in stream for slot %s", account.slot_id)
                await POOL.release(account, success=False)
                last_error = e
                break
            if success:
                # We already yielded chunks; can't retry mid-stream
                return

    yield sse_data(_format_no_account_chunk(req, plan, tried, last_error))
    yield sse_done()


async def _execute_failover_complete(req: Dict[str, Any]) -> Dict[str, Any]:
    plan = _resolve_route_plan(req)
    tried: List[str] = []
    last_error: Optional[Exception] = None

    for route_attempt in plan["attempts"]:
        prov = route_attempt["provider"]
        execs = PROVIDER_EXECUTORS.get(prov)
        if not execs:
            continue
        max_attempts = SETTINGS.max_failover_attempts
        attempts = 0
        while attempts < max_attempts:
            account = await POOL.acquire(prov)
            if account is None:
                tried.append(f"{route_attempt['label']}:no-account")
                break
            attempts += 1
            routed_req = _request_for_attempt(req, route_attempt)
            try:
                result = await execs["complete"](routed_req, account)
                await POOL.release(account, success=True)
                return result
            except QuotaExhausted as e:
                await POOL.mark_exhausted(account, e.resets_at, e.reason)
                tried.append(f"{account.slot_id}/{route_attempt['model']}:429")
                last_error = e
            except AuthRevoked as e:
                await POOL.mark_invalid(account)
                tried.append(f"{account.slot_id}/{route_attempt['model']}:401")
                last_error = e
            except UpstreamServerError as e:
                await POOL.release(account, success=False)
                tried.append(f"{account.slot_id}/{route_attempt['model']}:{e.status_code}")
                last_error = e
            except UpstreamClientError as e:
                await POOL.release(account, success=False)
                tried.append(f"{account.slot_id}/{route_attempt['model']}:{e.status_code}")
                last_error = e
                if _should_failover_client_error(plan, e):
                    break
                return _format_error_completion(req, e)
            except Exception as e:
                logger.exception("Unexpected error for slot %s", account.slot_id)
                await POOL.release(account, success=False)
                last_error = e
                break

    return _format_no_account_completion(req, plan, tried, last_error)


def _format_error_chunk(req: Dict[str, Any], err: Exception) -> Dict[str, Any]:
    msg = f"Upstream error ({type(err).__name__}): {err}"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": req.get("model") or "",
        "choices": [{"index": 0, "delta": {"content": msg}, "finish_reason": "stop"}],
    }


def _format_error_completion(req: Dict[str, Any], err: Exception) -> Dict[str, Any]:
    msg = f"Upstream error ({type(err).__name__}): {err}"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.get("model") or "",
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": msg},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _format_no_account_chunk(
    req: Dict[str, Any], plan: Dict[str, Any], tried: List[str], last: Optional[Exception]
) -> Dict[str, Any]:
    msg = _no_account_message(plan, tried, last)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": req.get("model") or "",
        "choices": [{"index": 0, "delta": {"content": msg}, "finish_reason": "stop"}],
    }


def _format_no_account_completion(
    req: Dict[str, Any], plan: Dict[str, Any], tried: List[str], last: Optional[Exception]
) -> Dict[str, Any]:
    msg = _no_account_message(plan, tried, last)
    providers = plan.get("providers") or []
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.get("model") or "",
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": msg},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "_bridge_error": {
            "route": plan.get("name") or "",
            "providers": providers,
            "tried": tried,
            "last_error": str(last) if last else "",
        },
    }


def _no_account_message(plan: Dict[str, Any], tried: List[str], last: Optional[Exception]) -> str:
    route_name = plan.get("name") or "route"
    providers = plan.get("providers") or []
    all_acc = [
        acc for provider in providers
        for acc in POOL.by_provider(provider)
    ]
    healthy = [
        acc for provider in providers
        for acc in POOL.healthy(provider)
    ]
    if not all_acc:
        if SETTINGS.locale == "en":
            return (f"⛔ No account configured for route `{route_name}`. "
                    f"Add one at http://{SETTINGS.host}:{SETTINGS.port}/")
        return (f"⛔ Chưa có account nào cho route `{route_name}`. "
                f"Thêm tại http://{SETTINGS.host}:{SETTINGS.port}/")
    extra = ""
    if tried:
        extra = f"\n\nĐã thử: {', '.join(tried)}"
    if last:
        extra += f"\nLỗi cuối: {last}"
    if SETTINGS.locale == "en":
        return (
            f"⛔ All accounts for route `{route_name}` are unavailable ({len(all_acc)} total, "
            f"{len(healthy)} healthy).{extra}"
        )
    return (
        f"⛔ Tất cả account cho route `{route_name}` không khả dụng "
        f"({len(all_acc)} tổng, {len(healthy)} healthy).{extra}\n\n"
        f"💡 Vào http://{SETTINGS.host}:{SETTINGS.port}/ để thêm hoặc đăng nhập lại."
    )


# ---------------------------------------------------------------------------
# Anthropic Messages compatibility shim
# ---------------------------------------------------------------------------


def _anthropic_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                typ = item.get("type")
                if typ in ("text", "input_text", "output_text"):
                    parts.append(str(item.get("text") or ""))
                elif typ == "tool_result":
                    parts.append(_anthropic_text(item.get("content")))
        return "\n".join(p for p in parts if p)
    return str(content)


def _anthropic_source_url(source: Any) -> str:
    if not isinstance(source, dict):
        return ""
    if source.get("type") == "url":
        return str(source.get("url") or "")
    if source.get("type") == "base64":
        media_type = str(source.get("media_type") or "application/octet-stream")
        data = str(source.get("data") or "")
        return f"data:{media_type};base64,{data}" if data else ""
    return str(source.get("url") or source.get("uri") or source.get("path") or "")


def _anthropic_content_to_openai(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")

    out: List[Dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            if item:
                out.append({"type": "text", "text": item})
            continue
        if not isinstance(item, dict):
            continue
        typ = item.get("type")
        if typ in ("text", "input_text", "output_text"):
            text = item.get("text") or ""
            if text:
                out.append({"type": "text", "text": text})
        elif typ == "image":
            url = _anthropic_source_url(item.get("source"))
            if url:
                out.append({"type": "image_url", "image_url": {"url": url}})
        elif typ in ("document", "pdf"):
            url = _anthropic_source_url(item.get("source"))
            if not url:
                continue
            file_obj: Dict[str, Any] = {
                "filename": item.get("name") or item.get("filename") or "document"
            }
            if url.startswith("data:"):
                file_obj["file_data"] = url
            else:
                file_obj["file_url"] = url
            out.append({"type": "file", "file": file_obj})
    if not out:
        return ""
    if len(out) == 1 and out[0].get("type") == "text":
        return out[0].get("text") or ""
    return out


def _split_anthropic_user_content(content: Any) -> Tuple[Any, List[Dict[str, Any]]]:
    if not isinstance(content, list):
        return content, []
    regular: List[Any] = []
    tool_results: List[Dict[str, Any]] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "tool_result":
            tool_results.append(item)
        else:
            regular.append(item)
    return regular, tool_results


def _anthropic_system_to_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return _anthropic_text(system)
    return ""


def _anthropic_tool_args(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value or {}, ensure_ascii=False)
    except Exception:
        return "{}"


def _anthropic_messages_to_openai(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    system_text = _anthropic_system_to_text(body.get("system"))
    if system_text:
        messages.append({"role": "system", "content": system_text})

    for raw_msg in body.get("messages") or []:
        if not isinstance(raw_msg, dict):
            continue
        role = str(raw_msg.get("role") or "user").lower()
        content = raw_msg.get("content")
        if role == "assistant":
            tool_calls: List[Dict[str, Any]] = []
            text_blocks: List[str] = []
            blocks = content if isinstance(content, list) else [
                {"type": "text", "text": _anthropic_text(content)}
            ]
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                typ = block.get("type")
                if typ == "tool_use":
                    tool_calls.append({
                        "id": block.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": block.get("name") or "",
                            "arguments": _anthropic_tool_args(block.get("input")),
                        },
                    })
                elif typ in ("text", "input_text", "output_text"):
                    text = block.get("text") or ""
                    if text:
                        text_blocks.append(text)
            msg: Dict[str, Any] = {"role": "assistant", "content": "\n".join(text_blocks)}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
            continue

        regular_content, tool_results = _split_anthropic_user_content(content)
        converted = _anthropic_content_to_openai(regular_content)
        if converted:
            messages.append({"role": "user", "content": converted})
        for result in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": result.get("tool_use_id") or result.get("id") or "",
                "content": _anthropic_text(result.get("content")),
            })

    if not messages:
        messages.append({"role": "user", "content": ""})
    return messages


def _anthropic_tools_to_openai(tools: Any) -> List[Dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    out: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not name:
            continue
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": tool.get("description") or "",
                "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
            },
        })
    return out


def _anthropic_tool_choice_to_openai(choice: Any) -> Any:
    if not isinstance(choice, dict):
        return choice
    typ = choice.get("type")
    if typ == "auto":
        return "auto"
    if typ == "any":
        return "required"
    if typ == "none":
        return "none"
    if typ == "tool" and choice.get("name"):
        return {"type": "function", "function": {"name": choice["name"]}}
    return None


def _anthropic_request_to_openai_chat(body: Dict[str, Any]) -> Dict[str, Any]:
    original_model = body.get("model") or SETTINGS.anthropic_default_model
    req: Dict[str, Any] = {
        "model": original_model,
        "_anthropic_model": original_model,
        "messages": _anthropic_messages_to_openai(body),
        "stream": bool(body.get("stream")),
    }
    if body.get("max_tokens") is not None:
        req["max_tokens"] = body["max_tokens"]
    for key in ("temperature", "top_p", "top_k"):
        if body.get(key) is not None:
            req[key] = body[key]
    if body.get("stop_sequences") is not None:
        req["stop"] = body["stop_sequences"]
    tools = _anthropic_tools_to_openai(body.get("tools"))
    if tools:
        req["tools"] = tools
    tool_choice = _anthropic_tool_choice_to_openai(body.get("tool_choice"))
    if tool_choice is not None:
        req["tool_choice"] = tool_choice
    thinking = body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        budget = int(thinking.get("budget_tokens") or 4096)
        req["reasoning_effort"] = (
            "xhigh" if budget >= 32768 else "high" if budget >= 16384 else "medium"
        )
    return req


def _parse_tool_input(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except Exception:
        return {"_raw": str(raw)}


def _openai_usage_to_anthropic(usage: Any) -> Dict[str, int]:
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
    }


def _openai_finish_to_anthropic(reason: Any) -> str:
    value = str(reason or "").lower()
    if value in ("length", "max_tokens"):
        return "max_tokens"
    if value in ("tool_calls", "function_call"):
        return "tool_use"
    if value == "stop_sequence":
        return "stop_sequence"
    return "end_turn"


def _openai_completion_to_anthropic(req: Dict[str, Any], completion: Dict[str, Any]) -> Dict[str, Any]:
    choices = completion.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    content: List[Dict[str, Any]] = []
    reasoning = message.get("reasoning_content")
    if reasoning:
        content.append({"type": "thinking", "thinking": str(reasoning)})
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": str(text)})
    for tc in message.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        content.append({
            "type": "tool_use",
            "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
            "name": fn.get("name") or "",
            "input": _parse_tool_input(fn.get("arguments")),
        })
    if not content:
        content.append({"type": "text", "text": ""})
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": req.get("_anthropic_model") or completion.get("model") or req.get("model") or "",
        "content": content,
        "stop_reason": _openai_finish_to_anthropic(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": _openai_usage_to_anthropic(completion.get("usage")),
    }


def _anthropic_sse(event: str, payload: Dict[str, Any]) -> bytes:
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n".encode("utf-8")


def _iter_openai_sse_payloads(chunk: bytes) -> List[Any]:
    text = chunk.decode("utf-8", "ignore")
    payloads: List[Any] = []
    for event in text.split("\n\n"):
        data_lines = []
        for line in event.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if not data_lines:
            continue
        raw = "\n".join(data_lines)
        if raw == "[DONE]":
            payloads.append("[DONE]")
            continue
        try:
            payloads.append(json.loads(raw))
        except Exception:
            continue
    return payloads


def _tool_state(tool_states: Dict[int, Dict[str, Any]], openai_index: int) -> Dict[str, Any]:
    if openai_index not in tool_states:
        tool_states[openai_index] = {
            "block_index": -1,
            "id": f"toolu_{uuid.uuid4().hex[:24]}",
            "name": "",
            "arguments": "",
            "started": False,
            "stopped": False,
        }
    return tool_states[openai_index]


async def _execute_anthropic_stream(
    req: Dict[str, Any],
    usage_sink: Optional[Dict[str, int]] = None,
) -> AsyncGenerator[bytes, None]:
    message_id = f"msg_{uuid.uuid4().hex}"
    model = req.get("model") or ""
    yield _anthropic_sse("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    next_block_index = 0
    text_block_index = -1
    text_open = False
    text_closed = False
    tool_states: Dict[int, Dict[str, Any]] = {}
    output_tokens = 0
    stop_reason = "end_turn"

    async for chunk in _execute_failover_stream(req):
        for payload in _iter_openai_sse_payloads(chunk):
            if payload == "[DONE]" or not isinstance(payload, dict):
                continue
            for choice in payload.get("choices") or []:
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                if text:
                    if text_block_index < 0:
                        text_block_index = next_block_index
                        next_block_index += 1
                        text_open = True
                        yield _anthropic_sse("content_block_start", {
                            "type": "content_block_start",
                            "index": text_block_index,
                            "content_block": {"type": "text", "text": ""},
                        })
                    elif not text_open and not text_closed:
                        text_open = True
                    yield _anthropic_sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": text_block_index,
                        "delta": {"type": "text_delta", "text": text},
                    })

                for tool_delta in delta.get("tool_calls") or []:
                    if not isinstance(tool_delta, dict):
                        continue
                    if text_open and not text_closed:
                        text_closed = True
                        text_open = False
                        yield _anthropic_sse("content_block_stop", {
                            "type": "content_block_stop",
                            "index": text_block_index,
                        })
                    openai_index = int(tool_delta.get("index") or 0)
                    state = _tool_state(tool_states, openai_index)
                    if tool_delta.get("id"):
                        state["id"] = tool_delta["id"]
                    fn = tool_delta.get("function") or {}
                    if fn.get("name"):
                        state["name"] = fn["name"]
                    if state["block_index"] < 0:
                        state["block_index"] = next_block_index
                        next_block_index += 1
                    if not state["started"] and state["name"]:
                        state["started"] = True
                        yield _anthropic_sse("content_block_start", {
                            "type": "content_block_start",
                            "index": state["block_index"],
                            "content_block": {
                                "type": "tool_use",
                                "id": state["id"],
                                "name": state["name"],
                                "input": {},
                            },
                        })
                    args = fn.get("arguments") or ""
                    if args:
                        state["arguments"] += args
                        if state["started"]:
                            yield _anthropic_sse("content_block_delta", {
                                "type": "content_block_delta",
                                "index": state["block_index"],
                                "delta": {"type": "input_json_delta", "partial_json": args},
                            })

                finish = choice.get("finish_reason")
                if finish:
                    stop_reason = _openai_finish_to_anthropic(finish)
            usage = payload.get("usage")
            if isinstance(usage, dict):
                output_tokens = int(usage.get("completion_tokens") or output_tokens)
                if usage_sink is not None:
                    usage_sink["total_tokens"] = max(
                        int(usage_sink.get("total_tokens") or 0),
                        _usage_total_tokens(usage),
                    )

    if text_open and not text_closed:
        text_closed = True
        yield _anthropic_sse("content_block_stop", {
            "type": "content_block_stop",
            "index": text_block_index,
        })

    for openai_index in sorted(tool_states):
        state = tool_states[openai_index]
        if not state["started"]:
            state["started"] = True
            name = state["name"] or "tool"
            yield _anthropic_sse("content_block_start", {
                "type": "content_block_start",
                "index": state["block_index"],
                "content_block": {
                    "type": "tool_use",
                    "id": state["id"],
                    "name": name,
                    "input": {},
                },
            })
            if state["arguments"]:
                yield _anthropic_sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": state["block_index"],
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": state["arguments"],
                    },
                })
        if not state["stopped"]:
            state["stopped"] = True
            yield _anthropic_sse("content_block_stop", {
                "type": "content_block_stop",
                "index": state["block_index"],
            })

    if text_block_index < 0 and not tool_states:
        yield _anthropic_sse("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        })
        yield _anthropic_sse("content_block_stop", {
            "type": "content_block_stop",
            "index": 0,
        })

    yield _anthropic_sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })
    yield _anthropic_sse("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# Routes - /v1
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {
        "ok": True,
        "providers": POOL.known_providers(),
        "accounts": len(POOL.all_accounts()),
        "uptime": int(time.time()),
    }


@app.get("/v1/models")
async def list_models(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    auth = _authorize_v1(request, authorization, x_api_key)
    out = _filter_models_for_key(_all_models_payload(), auth)
    return {"object": "list", "data": out}


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    body: Dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    auth = _authorize_v1(request, authorization, x_api_key, model=str(body.get("model") or SETTINGS.codex_default_model))
    stream = bool(body.get("stream"))
    if stream:
        metered_body, expose_usage = _chat_body_with_internal_usage(body, auth)
        async def producer():
            stream_gen = _execute_failover_stream(metered_body)
            async for chunk in _meter_openai_stream(stream_gen, auth, expose_usage=expose_usage):
                yield chunk
        gen = sse_with_keepalive(producer, SETTINGS.sse_keepalive_seconds)
        return StreamingResponse(
            gen, media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    result = await _execute_failover_complete(body)
    _record_managed_usage(auth, _usage_total_tokens(result.get("usage")), "")
    return JSONResponse(result)


@app.post("/v1/messages")
async def anthropic_messages(
    request: Request,
    body: Dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    auth = _authorize_v1(request, authorization, x_api_key, model=str(body.get("model") or SETTINGS.anthropic_default_model))
    chat_req = _anthropic_request_to_openai_chat(body)
    if chat_req.get("stream"):
        metered_req, _ = _chat_body_with_internal_usage(chat_req, auth)
        usage_sink: Dict[str, int] = {}
        async def producer():
            async for chunk in _execute_anthropic_stream(metered_req, usage_sink):
                yield chunk
            _record_managed_usage(
                auth,
                int(usage_sink.get("total_tokens") or 0),
                "" if usage_sink.get("total_tokens") else "missing_usage",
            )

        gen = sse_with_keepalive(producer, SETTINGS.sse_keepalive_seconds)
        return StreamingResponse(
            gen,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    chat_req["stream"] = False
    completion = await _execute_failover_complete(chat_req)
    _record_managed_usage(auth, _usage_total_tokens(completion.get("usage")), "")
    return JSONResponse(_openai_completion_to_anthropic(chat_req, completion))


@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(
    request: Request,
    body: Dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    auth = _authorize_v1(request, authorization, x_api_key, model=str(body.get("model") or SETTINGS.anthropic_default_model))
    text = _anthropic_system_to_text(body.get("system"))
    text += "\n" + json.dumps(body.get("messages") or [], ensure_ascii=False)
    text += "\n" + json.dumps(body.get("tools") or [], ensure_ascii=False)
    # Claude Code uses this endpoint for budgeting. This local shim gives a
    # conservative approximation because the real tokenizer is provider-side.
    _record_managed_usage(auth, 0, "")
    return {"input_tokens": max(1, len(text) // 3)}


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    response_format: str = Form("json"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    stream: Optional[bool] = Form(False),
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    auth = _authorize_v1(request, authorization, x_api_key, model=model)
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty audio file")
    # Audio currently routes to Codex (Codex Responses has audio input)
    account = await POOL.acquire("chatgpt")
    if account is None:
        raise HTTPException(503, "No healthy ChatGPT account for audio transcription")
    try:
        transcript = await p_codex.transcribe_audio(
            data, filename=file.filename or "audio.mp3",
            account=account, hint_prompt=prompt or "",
            language=language, model=model,
        )
        _record_managed_usage(auth, 0, "")
        await POOL.release(account, success=True)
    except QuotaExhausted as e:
        await POOL.mark_exhausted(account, e.resets_at, e.reason)
        raise HTTPException(429, f"Quota exhausted: {e}")
    except Exception:
        await POOL.release(account, success=False)
        raise
    if stream:
        async def gen():
            yield sse_data({"text": transcript, "type": "transcript.text.done"})
            yield sse_done()
        return StreamingResponse(gen(), media_type="text/event-stream")
    if response_format == "text":
        return HTMLResponse(content=transcript, media_type="text/plain; charset=utf-8")
    return {"text": transcript}


@app.get("/v1/oauth/token")
@app.post("/v1/oauth/token")
async def get_oauth_token(
    slot_id: Optional[str] = None,
    provider: Optional[str] = None,
    authorization: Optional[str] = Header(None),
):
    _check_bridge_auth(authorization)
    if slot_id:
        acc = POOL.get(slot_id)
        if acc is None:
            raise HTTPException(404, f"Unknown slot: {slot_id}")
    else:
        prov = provider or "chatgpt"
        candidates = POOL.healthy(prov) or POOL.by_provider(prov)
        if not candidates:
            raise HTTPException(401, f"No account for provider {prov}")
        acc = candidates[0]
    try:
        token = await acc.token_store.get_access_token()
    except Exception as exc:
        raise HTTPException(401, f"Cannot get token: {exc}")
    s = acc.token_store.status()
    token_type = "Bearer"
    if acc.provider == "anthropic" and s.get("auth_type") == "api_key":
        token_type = "x-api-key"
    return {
        "access_token": token, "token_type": token_type,
        "expires_in": s.get("expires_in", 0),
        "account_id": s.get("account_id", ""),
        "email": s.get("email", ""),
        "slot_id": acc.slot_id,
        "provider": acc.provider,
    }


@app.get("/.well-known/openai-bridge")
async def well_known():
    return {
        "name": "multi-provider-oauth-bridge",
        "version": "2.0.0",
        "host": SETTINGS.host,
        "port": SETTINGS.port,
        "providers": POOL.known_providers(),
        "accounts": len(POOL.all_accounts()),
        "endpoints": {
            "ui": "/",
            "models": "/v1/models",
            "chat_completions": "/v1/chat/completions",
            "audio_transcriptions": "/v1/audio/transcriptions",
            "oauth_token": "/v1/oauth/token",
            "accounts": "/api/accounts",
            "api_keys": "/api/keys",
            "health": "/health",
        },
    }


# ---------------------------------------------------------------------------
# Routes — /api/accounts (slot manager)
# ---------------------------------------------------------------------------


@app.get("/api/accounts")
async def api_list_accounts(authorization: Optional[str] = Header(None)):
    _check_bridge_auth(authorization)
    accounts = [await a.status() for a in POOL.all_accounts()]
    return {
        "providers": [
            {
                "id": p,
                "label": PROVIDER_EXECUTORS[p]["label"],
                "api_key": bool(PROVIDER_EXECUTORS[p].get("save_api_key")),
            }
            for p in POOL.known_providers()
        ],
        "accounts": accounts,
        "pool_strategy": SETTINGS.pool_strategy,
        "max_failover": SETTINGS.max_failover_attempts,
        "cross_provider_fallback": SETTINGS.cross_provider_fallback,
        "models": _provider_models(),
        "route_groups": _route_groups_payload(),
    }


@app.get("/api/groups")
async def api_list_groups(authorization: Optional[str] = Header(None)):
    _check_bridge_auth(authorization)
    return {
        "providers": [
            {
                "id": p,
                "label": PROVIDER_EXECUTORS[p]["label"],
                "api_key": bool(PROVIDER_EXECUTORS[p].get("save_api_key")),
            }
            for p in POOL.known_providers()
        ],
        "models": _provider_models(),
        "groups": _route_groups_payload(),
        "modes": sorted(GROUP_MODES),
    }


@app.get("/api/keys")
async def api_list_keys(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    _check_bridge_auth(authorization, x_api_key)
    return {"keys": API_KEYS.list_public()}


@app.post("/api/keys")
async def api_create_key(
    body: Dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    _check_bridge_auth(authorization, x_api_key)
    record, secret = API_KEYS.create(body)
    return {"key": record, "secret": secret}


@app.patch("/api/keys/{key_id}")
async def api_patch_key(
    key_id: str,
    body: Dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    _check_bridge_auth(authorization, x_api_key)
    record = API_KEYS.patch(key_id, body)
    if record is None:
        raise HTTPException(404, f"Unknown API key: {key_id}")
    return {"key": record}


@app.post("/api/keys/{key_id}/rotate")
async def api_rotate_key(
    key_id: str,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    _check_bridge_auth(authorization, x_api_key)
    rotated = API_KEYS.rotate(key_id)
    if rotated is None:
        raise HTTPException(404, f"Unknown API key: {key_id}")
    record, secret = rotated
    return {"key": record, "secret": secret}


@app.post("/api/keys/{key_id}/reset-usage")
async def api_reset_key_usage(
    key_id: str,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    _check_bridge_auth(authorization, x_api_key)
    record = API_KEYS.patch(key_id, {"token_used": 0, "status": "active"})
    if record is None:
        raise HTTPException(404, f"Unknown API key: {key_id}")
    return {"key": record}


@app.delete("/api/keys/{key_id}")
async def api_delete_key(
    key_id: str,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    _check_bridge_auth(authorization, x_api_key)
    if not API_KEYS.delete(key_id):
        raise HTTPException(404, f"Unknown API key: {key_id}")
    return {"ok": True}


@app.post("/api/groups")
async def api_save_group(
    body: Dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(None),
):
    _check_bridge_auth(authorization)
    try:
        group = _normalize_group_config(body.get("name"), body)
    except ValueError as e:
        raise HTTPException(400, str(e))
    ROUTE_GROUPS[group["name"]] = group
    _save_route_groups()
    return {"ok": True, "group": group}


@app.delete("/api/groups/{group_name}")
async def api_delete_group(
    group_name: str,
    authorization: Optional[str] = Header(None),
):
    _check_bridge_auth(authorization)
    name = _normalize_group_name(group_name)
    if name not in ROUTE_GROUPS:
        raise HTTPException(404, f"Unknown route group: {group_name}")
    ROUTE_GROUPS.pop(name, None)
    _ROUTE_GROUP_RR.pop(name, None)
    _save_route_groups()
    return {"ok": True}


@app.post("/api/accounts")
async def api_create_account(
    body: Dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(None),
):
    _check_bridge_auth(authorization)
    provider = str(body.get("provider") or "").strip().lower()
    alias = str(body.get("alias") or "").strip()
    slot_id = str(body.get("slot_id") or "").strip()
    if provider not in PROVIDER_EXECUTORS:
        raise HTTPException(400, f"Unknown provider: {provider}")
    try:
        acc = await POOL.create_slot(provider=provider, alias=alias, slot_id=slot_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return await acc.status()


@app.delete("/api/accounts/{slot_id}")
async def api_delete_account(
    slot_id: str, authorization: Optional[str] = Header(None),
):
    _check_bridge_auth(authorization)
    ok = await POOL.delete_slot(slot_id)
    if not ok:
        raise HTTPException(404, f"Unknown slot: {slot_id}")
    return {"ok": True}


@app.patch("/api/accounts/{slot_id}")
async def api_update_account(
    slot_id: str,
    body: Dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(None),
):
    _check_bridge_auth(authorization)
    if POOL.get(slot_id) is None:
        raise HTTPException(404, f"Unknown slot: {slot_id}")
    if "alias" in body:
        await POOL.update_alias(slot_id, str(body["alias"]).strip())
    if "enabled" in body:
        await POOL.set_enabled(slot_id, bool(body["enabled"]))
    if "rotation_enabled" in body:
        await POOL.set_rotation_enabled(slot_id, bool(body["rotation_enabled"]))
    if "tier" in body:
        await POOL.update_tier(slot_id, str(body["tier"]).strip())
    acc = POOL.get(slot_id)
    return await acc.status()


@app.post("/api/accounts/{slot_id}/login")
async def api_start_login(slot_id: str, authorization: Optional[str] = Header(None)):
    _check_bridge_auth(authorization)
    acc = POOL.get(slot_id)
    if acc is None:
        raise HTTPException(404, f"Unknown slot: {slot_id}")
    starter = PROVIDER_EXECUTORS[acc.provider]["start_login"]
    try:
        auth_url = await starter(slot_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"auth_url": auth_url, "slot_id": slot_id, "provider": acc.provider}


@app.post("/api/accounts/{slot_id}/api-key")
async def api_save_api_key(
    slot_id: str,
    body: Dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(None),
):
    _check_bridge_auth(authorization)
    acc = POOL.get(slot_id)
    if acc is None:
        raise HTTPException(404, f"Unknown slot: {slot_id}")
    saver = PROVIDER_EXECUTORS.get(acc.provider, {}).get("save_api_key")
    if saver is None:
        raise HTTPException(400, f"API key setup is not supported for {acc.provider} slots")
    api_key = str(body.get("api_key") or "").strip()
    if not api_key:
        raise HTTPException(400, "Missing api_key")
    ok = await saver(POOL, slot_id, api_key)
    if not ok:
        raise HTTPException(400, "Could not save API key")
    return await acc.status()


@app.post("/api/accounts/{slot_id}/refresh")
async def api_refresh(slot_id: str, authorization: Optional[str] = Header(None)):
    _check_bridge_auth(authorization)
    acc = POOL.get(slot_id)
    if acc is None:
        raise HTTPException(404, f"Unknown slot: {slot_id}")
    try:
        refresh_if_needed = getattr(acc.token_store, "refresh_if_needed", None)
        if callable(refresh_if_needed):
            await refresh_if_needed(force=True)
        else:
            await acc.token_store.get_access_token()
        await POOL.mark_valid(acc)
        return await acc.status()
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=401)


@app.post("/api/accounts/{slot_id}/logout")
async def api_logout(slot_id: str, authorization: Optional[str] = Header(None)):
    _check_bridge_auth(authorization)
    acc = POOL.get(slot_id)
    if acc is None:
        raise HTTPException(404, f"Unknown slot: {slot_id}")
    try:
        acc.token_store.clear()
    except Exception:
        logger.exception("Failed to clear token for %s", slot_id)
    return await acc.status()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


INDEX_HTML = r"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Multi-Provider OAuth Bridge</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: #101214;
      color: #f5f7fb;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      line-height: 1.4;
    }
    .topbar {
      position: sticky;
      top: 0;
      z-index: 20;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding: 12px 20px;
      background: #15181c;
      border-bottom: 1px solid #2b3037;
      box-shadow: 0 8px 24px rgba(0,0,0,.22);
    }
    .brand h1 { margin: 0; font-size: 18px; letter-spacing: 0; }
    .status-line, .toolbar-actions, .inline-row, .actions, .filters {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .status-line { margin-top: 6px; color: #aeb7c4; font-size: 12px; }
    .toolbar-actions { justify-content: flex-end; }
    .base-url { color: #cbd3df; font-size: 12px; white-space: nowrap; }
    .wrap { max-width: 1260px; margin: 0 auto; padding: 16px 20px 28px; }
    .tabs {
      display: flex;
      gap: 6px;
      margin-bottom: 14px;
      overflow-x: auto;
      padding-bottom: 2px;
    }
    .tab-button {
      background: transparent;
      color: #cbd3df;
      border: 1px solid #303640;
      min-width: 112px;
    }
    .tab-button.active {
      background: #254f9c;
      border-color: #4d7fe2;
      color: #fff;
    }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    .panel {
      background: #181c21;
      border: 1px solid #2d333b;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 14px;
    }
    .panel-heading {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    h2 { margin: 0; font-size: 16px; letter-spacing: 0; }
    h3 { margin: 0 0 8px; font-size: 13px; color: #dbe2ec; letter-spacing: 0; }
    .muted { color: #aab3bf; font-size: 13px; }
    .meta { color: #8f9aaa; font-size: 12px; }
    .empty { color: #aab3bf; padding: 22px; text-align: center; border: 1px dashed #3a414b; border-radius: 8px; }
    code {
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 5px;
      padding: 2px 6px;
      color: #e6edf3;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }
    button, select, input {
      font: inherit;
      border-radius: 7px;
    }
    button {
      border: 1px solid #3d6ed0;
      background: #2f6fe4;
      color: #fff;
      cursor: pointer;
      padding: 7px 12px;
      font-size: 13px;
      min-height: 34px;
    }
    button:hover { filter: brightness(1.08); }
    button:disabled { opacity: .55; cursor: wait; }
    button.secondary { background: #2b313a; border-color: #4a525e; }
    button.ghost { background: transparent; border-color: #454c56; color: #d2d8e2; }
    button.danger { background: #8d2e2e; border-color: #b34b4b; }
    button.small { padding: 5px 9px; min-height: 28px; font-size: 12px; }
    select, input {
      background: #11151a;
      border: 1px solid #3a414b;
      color: #f5f7fb;
      padding: 7px 10px;
      min-height: 34px;
      font-size: 13px;
    }
    input::placeholder { color: #778292; }
    .field { display: flex; flex-direction: column; gap: 5px; min-width: 160px; }
    .field label, .field-inline label { color: #aab3bf; font-size: 12px; }
    .field-inline { display: flex; align-items: center; gap: 7px; }
    .wide-input { width: min(340px, 100%); }
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(120px, 1fr));
      gap: 10px;
    }
    .metric {
      border: 1px solid #2f3741;
      border-radius: 8px;
      padding: 12px;
      background: #14181d;
      min-height: 78px;
    }
    .metric-value { font-size: 28px; line-height: 1; font-weight: 700; }
    .metric-label { margin-top: 8px; color: #aab3bf; font-size: 12px; }
    .quick-actions { margin-top: 14px; }
    .alert-list { display: grid; gap: 8px; }
    .alert {
      border: 1px solid #3b424d;
      border-left-width: 4px;
      border-radius: 8px;
      padding: 10px 12px;
      background: #14181d;
    }
    .alert.warning { border-left-color: #d69b2d; }
    .alert.error { border-left-color: #d35d55; }
    .alert.info { border-left-color: #4e86e8; }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      border-radius: 999px;
      padding: 3px 9px;
      font-size: 12px;
      font-weight: 600;
      white-space: nowrap;
    }
    .pill.green { background: #1d6d42; color: #e9fff2; }
    .pill.red { background: #843232; color: #fff1f1; }
    .pill.amber { background: #8a6119; color: #fff3d2; }
    .pill.grey { background: #39414d; color: #d7dde7; }
    .pill.blue { background: #254f9c; color: #eef4ff; }
    .pill.outline { background: transparent; border: 1px solid #424a55; color: #cbd3df; }
    .account-tools, .add-form, .route-controls, .key-form {
      display: flex;
      align-items: flex-end;
      gap: 10px;
      flex-wrap: wrap;
    }
    .account-table {
      border: 1px solid #2d333b;
      border-radius: 8px;
      overflow: visible;
    }
    .account-row {
      display: grid;
      grid-template-columns: minmax(190px, 1.15fr) minmax(126px, .7fr) minmax(150px, .75fr) minmax(190px, 1fr) minmax(260px, 1.1fr);
      gap: 12px;
      align-items: center;
      padding: 12px;
      border-bottom: 1px solid #2d333b;
      background: #161a1f;
    }
    .account-row:last-child { border-bottom: 0; }
    .account-row.key-row {
      grid-template-columns: minmax(180px, 1fr) minmax(110px, .55fr) minmax(150px, .75fr) minmax(210px, 1fr) minmax(280px, 1.15fr);
    }
    .account-row.header-row {
      color: #aab3bf;
      background: #11151a;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .02em;
    }
    .slot-name { font-weight: 700; word-break: break-word; }
    .slot-id {
      color: #8f9aaa;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      word-break: break-all;
    }
    .quota-line { color: #cbd3df; font-size: 12px; line-height: 1.5; }
    .quota-line span { display: inline-block; margin-right: 8px; }
    .more-menu { position: relative; display: inline-block; }
    .more-menu summary {
      list-style: none;
      cursor: pointer;
      border: 1px solid #454c56;
      border-radius: 7px;
      padding: 5px 9px;
      min-height: 28px;
      color: #d2d8e2;
      background: #20252c;
      font-size: 12px;
    }
    .more-menu summary::-webkit-details-marker { display: none; }
    .more-menu[open] .menu {
      position: absolute;
      right: 0;
      top: 32px;
      display: grid;
      gap: 6px;
      min-width: 130px;
      padding: 8px;
      background: #0f1318;
      border: 1px solid #333b46;
      border-radius: 8px;
      box-shadow: 0 12px 28px rgba(0,0,0,.35);
      z-index: 10;
    }
    .route-workspace {
      display: grid;
      grid-template-columns: minmax(250px, .85fr) minmax(0, 1.6fr);
      gap: 14px;
      align-items: start;
    }
    .group-list, .route-palette, .route-list {
      border: 1px solid #2f3741;
      border-radius: 8px;
      background: #14181d;
    }
    .group-list { max-height: 620px; overflow: auto; }
    .group-card {
      display: grid;
      gap: 8px;
      width: 100%;
      padding: 12px;
      border: 0;
      border-bottom: 1px solid #2d333b;
      border-radius: 0;
      background: transparent;
      text-align: left;
      color: #f5f7fb;
    }
    .group-card:last-child { border-bottom: 0; }
    .group-card.active { background: #1d2634; }
    .group-card-title { display: flex; justify-content: space-between; gap: 8px; align-items: center; }
    .group-items { color: #bec7d4; font-size: 12px; line-height: 1.5; overflow-wrap: anywhere; }
    .editor-grid {
      display: grid;
      grid-template-columns: minmax(180px, 1fr) minmax(150px, .65fr);
      gap: 10px;
      margin-bottom: 12px;
    }
    .tag-picker {
      display: grid;
      grid-template-columns: 1fr 150px 150px;
      gap: 8px;
      margin: 12px 0 8px;
    }
    .route-palette {
      max-height: 300px;
      overflow: auto;
      padding: 8px;
    }
    .route-chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      max-width: 100%;
      margin: 4px;
      padding: 6px 10px;
      border: 1px solid #3b566f;
      border-radius: 999px;
      background: #182432;
      color: #fff;
      font-size: 12px;
      cursor: grab;
      vertical-align: top;
    }
    .route-chip.provider, .route-item.provider { border-color: #3e7b56; background: #173524; }
    .route-chip.group, .route-item.group { border-color: #8b6a35; background: #3a2b16; }
    .route-chip.model, .route-item.model { border-color: #3b609f; background: #182842; }
    .route-list {
      min-height: 150px;
      padding: 8px;
    }
    .route-item {
      display: grid;
      grid-template-columns: 34px minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
      padding: 8px;
      margin-bottom: 8px;
      border: 1px solid #3b609f;
      border-radius: 8px;
      cursor: grab;
    }
    .route-item:last-child { margin-bottom: 0; }
    .route-index {
      color: #cbd3df;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }
    .route-label { overflow-wrap: anywhere; }
    .settings-grid {
      display: grid;
      grid-template-columns: minmax(260px, .85fr) minmax(300px, 1.15fr);
      gap: 14px;
    }
    .setting-block {
      border: 1px solid #2f3741;
      border-radius: 8px;
      padding: 12px;
      background: #14181d;
    }
    .kv {
      display: grid;
      grid-template-columns: minmax(130px, .55fr) minmax(0, 1fr);
      gap: 8px;
      padding: 7px 0;
      border-bottom: 1px solid #272e37;
      font-size: 13px;
    }
    .kv:last-child { border-bottom: 0; }
    .usage-list { margin: 8px 0 0; padding-left: 18px; color: #cbd3df; }
    .usage-list li { margin: 5px 0; }
    .secret-box {
      display: none;
      margin-top: 12px;
      border: 1px solid #4f7d50;
      border-radius: 8px;
      padding: 10px 12px;
      background: #102016;
      color: #dff7e5;
      overflow-wrap: anywhere;
    }
    select[multiple] { min-height: 120px; min-width: 260px; }
    @media (max-width: 920px) {
      .topbar { align-items: flex-start; flex-direction: column; }
      .toolbar-actions { justify-content: flex-start; }
      .metrics-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .account-row { grid-template-columns: 1fr; gap: 8px; }
      .account-row.header-row { display: none; }
      .account-row > div::before {
        display: block;
        color: #8f9aaa;
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 3px;
        text-transform: uppercase;
      }
      .account-row > div:nth-child(1)::before { content: attr(data-label); }
      .account-row > div:nth-child(2)::before { content: attr(data-label); }
      .account-row > div:nth-child(3)::before { content: attr(data-label); }
      .account-row > div:nth-child(4)::before { content: attr(data-label); }
      .account-row > div:nth-child(5)::before { content: attr(data-label); }
      .route-workspace, .settings-grid { grid-template-columns: 1fr; }
      .tag-picker, .editor-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 560px) {
      .wrap { padding: 12px; }
      .topbar { padding: 10px 12px; }
      .metrics-grid { grid-template-columns: 1fr; }
      .tabs { gap: 5px; }
      .tab-button { min-width: auto; white-space: nowrap; }
      .panel { padding: 12px; }
      .base-url { white-space: normal; }
      button, select, input { width: 100%; }
      .actions button, .toolbar-actions button, .tabs button, .route-item button { width: auto; }
      .field, .wide-input { width: 100%; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand">
      <h1 id="appTitle">Multi-Provider OAuth Bridge</h1>
      <div class="status-line">
        <span id="serviceStatus" class="pill grey">Loading</span>
        <span id="providerSummary">-</span>
      </div>
    </div>
    <div class="toolbar-actions">
      <div class="base-url"><span id="baseUrlLabel">Base URL</span>: <code>http://__HOST__:__PORT__/v1</code></div>
      <label class="field-inline">
        <span id="languageLabel">Language</span>
        <select id="langSelect" onchange="setLang(this.value)">
          <option value="vi">Tiếng Việt</option>
          <option value="en">English</option>
        </select>
      </label>
      <button id="adminKeyButton" class="secondary" onclick="setAdminToken()">Admin key</button>
      <button id="refreshButton" class="secondary" onclick="fetchData()">Refresh</button>
    </div>
  </header>

  <main class="wrap">
    <nav class="tabs" aria-label="Bridge sections">
      <button class="tab-button active" id="tabOverviewButton" onclick="setActiveTab('overview')">Overview</button>
      <button class="tab-button" id="tabAccountsButton" onclick="setActiveTab('accounts')">Accounts</button>
      <button class="tab-button" id="tabGroupsButton" onclick="setActiveTab('groups')">Route groups</button>
      <button class="tab-button" id="tabKeysButton" onclick="setActiveTab('keys')">API keys</button>
      <button class="tab-button" id="tabSettingsButton" onclick="setActiveTab('settings')">Settings</button>
    </nav>

    <section id="tab-overview" class="tab-panel active">
      <div class="panel">
        <div class="panel-heading">
          <h2 id="overviewTitle">Overview</h2>
          <div class="actions quick-actions">
            <button id="quickAddAccountButton" onclick="showAddAccount()">Add account</button>
            <button id="quickCreateGroupButton" class="secondary" onclick="showRouteBuilder()">Create route group</button>
          </div>
        </div>
        <div id="overviewStats" class="metrics-grid"></div>
      </div>
      <div class="panel">
        <div class="panel-heading">
          <h2 id="alertsTitle">Attention</h2>
        </div>
        <div id="alertsList" class="alert-list"></div>
      </div>
    </section>

    <section id="tab-accounts" class="tab-panel">
      <div class="panel">
        <div class="panel-heading">
          <div>
            <h2><span id="accountsTitle">Accounts</span> <span class="meta">(<span id="accountCount">0</span>)</span></h2>
            <div class="muted" id="addHint">ChatGPT and Gemini use OAuth Login. Claude and DeepSeek use API keys.</div>
          </div>
        </div>
        <div class="add-form">
          <div class="field">
            <label id="providerFieldLabel" for="newProvider">Provider</label>
            <select id="newProvider"></select>
          </div>
          <div class="field">
            <label id="aliasFieldLabel" for="newAlias">Alias</label>
            <input id="newAlias" class="wide-input" placeholder="Alias (optional, e.g. 'Personal')">
          </div>
          <button id="createButton" onclick="createAccount()">Create slot</button>
        </div>
      </div>

      <div class="panel">
        <div class="account-tools">
          <div class="field">
            <label id="accountProviderFilterLabel" for="accountProviderFilter">Provider</label>
            <select id="accountProviderFilter" onchange="setAccountProviderFilter(this.value)"></select>
          </div>
          <div class="field">
            <label id="accountStatusFilterLabel" for="accountStatusFilter">Status</label>
            <select id="accountStatusFilter" onchange="setAccountStatusFilter(this.value)"></select>
          </div>
        </div>
        <div id="accountList" style="margin-top:12px"></div>
      </div>
    </section>

    <section id="tab-groups" class="tab-panel">
      <div class="route-workspace">
        <div class="panel">
          <div class="panel-heading">
            <h2 id="groupsTitle">Route groups</h2>
            <button id="newGroupButton" class="small secondary" onclick="clearGroupBuilder()">New</button>
          </div>
          <div id="groupList" class="group-list"></div>
        </div>

        <div class="panel">
          <div class="panel-heading">
            <h2 id="groupEditorTitle">Group editor</h2>
            <div class="actions">
              <button id="saveGroupButton" onclick="saveGroup()">Save group</button>
              <button id="clearGroupButton" class="secondary" onclick="clearGroupBuilder()">Clear</button>
              <button id="deleteGroupButton" class="danger" onclick="deleteActiveGroup()">Delete</button>
            </div>
          </div>
          <div class="editor-grid">
            <div class="field">
              <label id="groupNameLabel" for="groupName">Group name</label>
              <input id="groupName" placeholder="Group name, e.g. all">
            </div>
            <div class="field">
              <label id="groupModeLabel" for="groupMode">Mode</label>
              <select id="groupMode">
                <option value="priority" id="modePriority">Priority</option>
                <option value="round_robin" id="modeRoundRobin">Round-robin</option>
                <option value="random" id="modeRandom">Random</option>
              </select>
            </div>
          </div>

          <h3 id="availableTagsTitle">Available tags</h3>
          <div class="tag-picker">
            <input id="routeSearch" placeholder="Search provider, model, or group" oninput="setRouteSearch(this.value)">
            <select id="routeKindFilter" onchange="setRouteKindFilter(this.value)"></select>
            <select id="routeProviderFilter" onchange="setRouteProviderFilter(this.value)"></select>
          </div>
          <div class="meta" id="routePaletteCount"></div>
          <div id="routePalette" class="route-palette"></div>

          <h3 id="selectedRouteTitle" style="margin-top:14px">Selected route order</h3>
          <div id="selectedRouteItems" class="route-list" ondragover="allowRouteDrop(event)" ondrop="dropRouteItem(event)"></div>
        </div>
      </div>
    </section>

    <section id="tab-keys" class="tab-panel">
      <div class="panel">
        <div class="panel-heading">
          <div>
            <h2 id="keysTitle">API keys</h2>
            <div class="muted" id="keysHint">Create limited client keys for /v1 usage.</div>
          </div>
        </div>
        <div class="key-form">
          <div class="field">
            <label id="keyNameLabel" for="keyName">Name</label>
            <input id="keyName" class="wide-input" placeholder="Client name">
          </div>
          <div class="field">
            <label id="keyHoursLabel" for="keyHours">Hours</label>
            <input id="keyHours" type="number" min="0" placeholder="0 = no expiry">
          </div>
          <div class="field">
            <label id="keyTokenLimitLabel" for="keyTokenLimit">Token limit</label>
            <input id="keyTokenLimit" type="number" min="0" placeholder="0 = unlimited">
          </div>
          <div class="field">
            <label id="keyNetworkLabel" for="keyNetwork">Network</label>
            <select id="keyNetwork">
              <option value="local">Localhost</option>
              <option value="lan">LAN</option>
              <option value="internet">Internet</option>
            </select>
          </div>
          <div class="field">
            <label id="keyModelsLabel" for="keyModels">Allowed models</label>
            <select id="keyModels" multiple></select>
          </div>
          <button id="createKeyButton" onclick="createManagedKey()">Create key</button>
        </div>
        <div id="keySecretBox" class="secret-box"></div>
      </div>

      <div class="panel">
        <div id="keyList"></div>
      </div>
    </section>

    <section id="tab-settings" class="tab-panel">
      <div class="panel">
        <div class="panel-heading">
          <h2 id="settingsTitle">Settings</h2>
        </div>
        <div class="settings-grid">
          <div class="setting-block">
            <h3 id="runtimeTitle">Runtime</h3>
            <div class="kv"><div class="meta" id="settingsBaseUrlLabel">Base URL</div><div><code>http://__HOST__:__PORT__/v1</code></div></div>
            <div class="kv"><div class="meta" id="settingsStrategyLabel">Strategy</div><div id="settingsStrategy">-</div></div>
            <div class="kv"><div class="meta" id="settingsFallbackLabel">Cross-provider fallback</div><div id="settingsFallback">-</div></div>
            <div class="kv"><div class="meta" id="settingsFailoverLabel">Max failover</div><div id="settingsMaxFailover">-</div></div>
            <div class="kv"><div class="meta" id="settingsModelsLabel">Visible models</div><div id="settingsModelCount">-</div></div>
          </div>
          <div class="setting-block">
            <h3 id="usageTitle">Usage</h3>
            <div id="usageContent" class="muted"></div>
          </div>
        </div>
      </div>
    </section>
  </main>

  <script>
    const DEFAULT_LANG = '__LOCALE__';
    const BASE_URL = 'http://__HOST__:__PORT__/v1';
    const TEXT = {
      en: {
        title: 'Multi-Provider OAuth Bridge',
        overview: 'Overview',
        accounts: 'Accounts',
        groups: 'Route groups',
        settings: 'Settings',
        attention: 'Attention',
        serviceOnline: 'Service online',
        serviceError: 'Service error',
        loading: 'Loading',
        providerSummary: '{providers} providers, {accounts} accounts',
        language: 'Language',
        baseUrl: 'Base URL',
        refresh: 'Refresh',
        addAccount: 'Add account',
        createRouteGroup: 'Create route group',
        totalAccounts: 'total accounts',
        healthyAccounts: 'healthy',
        exhaustedAccounts: 'exhausted',
        invalidAccounts: 'invalid',
        routeGroups: 'route groups',
        noAlerts: 'No urgent alerts.',
        alertMore: '+{count} more alerts',
        needsSetupTitle: 'Account needs setup',
        needsSetupBody: '{slot} is enabled but not ready.',
        exhaustedTitle: 'Quota or rate limit',
        exhaustedBody: '{slot} retries in {time}.',
        lowQuotaTitle: 'Quota nearly used',
        lowQuotaBody: '{slot}: {detail}',
        badGroupTitle: 'Route group issue',
        badGroupBody: '{group}: {detail}',
        addHint: 'ChatGPT and Gemini use OAuth Login. Claude and DeepSeek use API keys.',
        provider: 'Provider',
        alias: 'Alias',
        aliasPlaceholder: "Alias (optional, e.g. 'Personal')",
        createSlot: 'Create slot',
        filterAllProviders: 'All providers',
        filterAllStatuses: 'All statuses',
        status: 'Status',
        slot: 'Slot',
        state: 'State',
        rotation: 'Rotation',
        quotaRate: 'Quota / rate',
        actions: 'Actions',
        empty: 'No accounts match the current filters.',
        disabled: 'disabled',
        invalid: 'invalid',
        exhausted: 'exhausted',
        healthy: 'healthy',
        refreshable: 'refreshable',
        expired: 'expired',
        notLoggedIn: 'not logged in',
        inFlight: 'in-flight',
        expiresIn: 'expires in',
        tier: 'tier',
        auth: 'auth',
        rotationOn: 'on',
        rotationOff: 'off',
        turnRotationOn: 'Rotate on',
        turnRotationOff: 'Rotate off',
        apiKey: 'API key',
        login: 'Login',
        disable: 'Disable',
        enable: 'Enable',
        logout: 'Logout',
        delete: 'Delete',
        more: 'More',
        rateWaiting: 'waiting for response headers',
        model: 'model',
        updated: 'updated',
        retry: 'retry',
        reset: 'reset',
        reason: 'reason',
        now: 'now',
        ago: 'ago',
        newGroup: 'New',
        groupEditor: 'Group editor',
        groupName: 'Group name',
        groupNamePlaceholder: 'Group name, e.g. all',
        mode: 'Mode',
        modePriority: 'Priority',
        modeRoundRobin: 'Round-robin',
        modeRandom: 'Random',
        saveGroup: 'Save group',
        clear: 'Clear',
        availableTags: 'Available tags',
        routeSearchPlaceholder: 'Search provider, model, or group',
        kindAll: 'All tags',
        kindProvider: 'Providers',
        kindModel: 'Models',
        kindGroup: 'Groups',
        selectedRoute: 'Selected route order',
        selectedEmpty: 'Drop provider, model, or group tags here.',
        noTags: 'No matching tags.',
        tagsShown: 'Showing {shown} of {total} tags',
        noGroups: 'No route groups yet.',
        groupProvider: 'provider',
        groupModel: 'model',
        groupGroup: 'group',
        up: 'Up',
        down: 'Down',
        remove: 'Remove',
        runtime: 'Runtime',
        strategy: 'Strategy',
        fallback: 'Cross-provider fallback',
        maxFailover: 'Max failover',
        visibleModels: 'Visible models',
        on: 'on',
        off: 'off',
        usage: 'Usage',
        usageHtml: 'Point OpenAI-compatible clients at <code>http://__HOST__:__PORT__/v1</code>.<ul class="usage-list"><li>Model examples: <code>gpt-5.5</code>, <code>gemini-2.5-pro</code>, <code>claude-sonnet-4-6</code>, <code>deepseek-v4-pro</code>.</li><li>Provider aliases: <code>chatgpt</code>, <code>google</code>, <code>anthropic</code>, <code>deepseek</code>.</li><li>Route group aliases use the group name, for example <code>all</code> or any saved group.</li></ul>',
        createFailed: 'Create failed: ',
        saveFailed: 'Save failed: ',
        saveGroupFailed: 'Save group failed: ',
        deleteGroupConfirm: 'Delete route group {name}?',
        groupNameRequired: 'Enter a group name.',
        groupItemsRequired: 'Add at least one provider, model, or group tag.',
        pasteApiKey: 'Paste API key for ',
        logoutConfirm: 'Logout {slot}? Token will be cleared.',
        deleteConfirm: 'Delete {slot} permanently?',
        missingProvider: 'missing provider {provider}',
        missingGroup: 'missing group {group}',
        missingModel: 'missing model {model}',
        selfReference: 'references itself',
        apiKeys: 'API keys',
        adminKey: 'Admin key',
        adminKeyPrompt: 'Enter admin BRIDGE_API_KEY for this browser.',
        keysHint: 'Create limited client keys for /v1 usage.',
        keyName: 'Name',
        keyNamePlaceholder: 'Client name',
        keyHours: 'Hours',
        keyTokenLimit: 'Token limit',
        keyNetwork: 'Network',
        keyModels: 'Allowed models',
        createKey: 'Create key',
        keyCreated: 'Copy this key now. It will not be shown again:',
        keyStatus: 'Status',
        keyUsage: 'Usage',
        keyPolicy: 'Policy',
        keyLastUsed: 'Last used',
        keyNever: 'never',
        keyNoExpiry: 'no expiry',
        keyUnlimited: 'unlimited',
        keyNetworkLocal: 'Localhost',
        keyNetworkLan: 'LAN',
        keyNetworkInternet: 'Internet',
        keyDisable: 'Disable',
        keyActivate: 'Activate',
        keyRotate: 'Rotate',
        keyResetUsage: 'Reset usage',
        keyEdit: 'Edit',
        keyModelsPrompt: 'Allowed models, comma-separated. Use * for all.',
        keyRotateConfirm: 'Rotate this key? Existing secret will stop working.',
        keyDeleteConfirm: 'Delete API key {name}?',
        noKeys: 'No managed API keys yet.'
      },
      vi: {
        title: 'Multi-Provider OAuth Bridge',
        overview: 'Tổng quan',
        accounts: 'Account',
        groups: 'Nhóm route',
        settings: 'Cài đặt',
        attention: 'Cần chú ý',
        serviceOnline: 'Service chạy',
        serviceError: 'Service lỗi',
        loading: 'Đang tải',
        providerSummary: '{providers} provider, {accounts} account',
        language: 'Ngôn ngữ',
        baseUrl: 'Base URL',
        refresh: 'Refresh',
        addAccount: 'Thêm account',
        createRouteGroup: 'Tạo nhóm route',
        totalAccounts: 'tổng account',
        healthyAccounts: 'sẵn sàng',
        exhaustedAccounts: 'hết quota',
        invalidAccounts: 'token hỏng',
        routeGroups: 'nhóm route',
        noAlerts: 'Không có cảnh báo quan trọng.',
        alertMore: '+{count} cảnh báo khác',
        needsSetupTitle: 'Account cần thiết lập',
        needsSetupBody: '{slot} đang bật nhưng chưa sẵn sàng.',
        exhaustedTitle: 'Quota hoặc rate limit',
        exhaustedBody: '{slot} thử lại sau {time}.',
        lowQuotaTitle: 'Quota gần hết',
        lowQuotaBody: '{slot}: {detail}',
        badGroupTitle: 'Nhóm route có lỗi',
        badGroupBody: '{group}: {detail}',
        addHint: 'ChatGPT và Gemini dùng OAuth Login. Claude và DeepSeek dùng API key.',
        provider: 'Provider',
        alias: 'Tên gợi nhớ',
        aliasPlaceholder: "Tên gợi nhớ (tuỳ chọn, ví dụ 'Cá nhân')",
        createSlot: 'Tạo slot',
        filterAllProviders: 'Tất cả provider',
        filterAllStatuses: 'Tất cả trạng thái',
        status: 'Trạng thái',
        slot: 'Slot',
        state: 'Trạng thái',
        rotation: 'Quay vòng',
        quotaRate: 'Quota / rate',
        actions: 'Thao tác',
        empty: 'Không có account khớp bộ lọc.',
        disabled: 'đã tắt',
        invalid: 'token hỏng',
        exhausted: 'hết quota',
        healthy: 'sẵn sàng',
        refreshable: 'có thể refresh',
        expired: 'hết hạn',
        notLoggedIn: 'chưa đăng nhập',
        inFlight: 'đang chạy',
        expiresIn: 'hết hạn sau',
        tier: 'gói',
        auth: 'xác thực',
        rotationOn: 'bật',
        rotationOff: 'tắt',
        turnRotationOn: 'Bật quay vòng',
        turnRotationOff: 'Tắt quay vòng',
        apiKey: 'API key',
        login: 'Đăng nhập',
        disable: 'Tắt',
        enable: 'Bật',
        logout: 'Đăng xuất',
        delete: 'Xoá',
        more: 'Thêm',
        rateWaiting: 'đang chờ response headers',
        model: 'model',
        updated: 'cập nhật',
        retry: 'thử lại',
        reset: 'reset',
        reason: 'lý do',
        now: 'vừa xong',
        ago: 'trước',
        newGroup: 'Tạo mới',
        groupEditor: 'Sửa nhóm',
        groupName: 'Tên nhóm',
        groupNamePlaceholder: 'Tên nhóm, ví dụ all',
        mode: 'Mode',
        modePriority: 'Ưu tiên',
        modeRoundRobin: 'Quay vòng',
        modeRandom: 'Ngẫu nhiên',
        saveGroup: 'Lưu nhóm',
        clear: 'Xoá chọn',
        availableTags: 'Tag có thể chọn',
        routeSearchPlaceholder: 'Tìm provider, model, hoặc group',
        kindAll: 'Tất cả tag',
        kindProvider: 'Provider',
        kindModel: 'Model',
        kindGroup: 'Group',
        selectedRoute: 'Thứ tự route đã chọn',
        selectedEmpty: 'Thả tag provider, model, hoặc group vào đây.',
        noTags: 'Không có tag khớp.',
        tagsShown: 'Đang hiển thị {shown}/{total} tag',
        noGroups: 'Chưa có nhóm route.',
        groupProvider: 'provider',
        groupModel: 'model',
        groupGroup: 'group',
        up: 'Lên',
        down: 'Xuống',
        remove: 'Xoá',
        runtime: 'Runtime',
        strategy: 'Chiến lược',
        fallback: 'Fallback khác provider',
        maxFailover: 'Số lần failover tối đa',
        visibleModels: 'Model đang expose',
        on: 'bật',
        off: 'tắt',
        usage: 'Cách dùng',
        usageHtml: 'Trỏ client OpenAI-compatible vào <code>http://__HOST__:__PORT__/v1</code>.<ul class="usage-list"><li>Ví dụ model: <code>gpt-5.5</code>, <code>gemini-2.5-pro</code>, <code>claude-sonnet-4-6</code>, <code>deepseek-v4-pro</code>.</li><li>Alias provider: <code>chatgpt</code>, <code>google</code>, <code>anthropic</code>, <code>deepseek</code>.</li><li>Alias nhóm route chính là tên nhóm đã lưu, ví dụ <code>all</code> hoặc nhóm bất kỳ.</li></ul>',
        createFailed: 'Tạo thất bại: ',
        saveFailed: 'Lưu thất bại: ',
        saveGroupFailed: 'Lưu nhóm thất bại: ',
        deleteGroupConfirm: 'Xoá nhóm route {name}?',
        groupNameRequired: 'Nhập tên nhóm.',
        groupItemsRequired: 'Thêm ít nhất một tag provider, model, hoặc group.',
        pasteApiKey: 'Dán API key cho ',
        logoutConfirm: 'Đăng xuất {slot}? Token sẽ bị xoá.',
        deleteConfirm: 'Xoá vĩnh viễn {slot}?',
        missingProvider: 'thiếu provider {provider}',
        missingGroup: 'thiếu group {group}',
        missingModel: 'thiếu model {model}',
        selfReference: 'trỏ tới chính nó'
      }
    };

    let currentLang = (localStorage.getItem('bridgeLang') || (DEFAULT_LANG || 'vi')).toLowerCase();
    if (!TEXT[currentLang]) currentLang = currentLang.startsWith('en') ? 'en' : 'vi';
    let activeTab = 'overview';
    let lastAccountsData = null;
    let lastGroupsData = null;
    let lastModelsData = null;
    let lastKeysData = null;
    let lastFetchOk = false;
    let adminToken = localStorage.getItem('bridgeAdminToken') || '';
    let apiKeyProviders = new Set();
    let selectedRouteItems = [];
    let activeGroupName = '';
    let renderedGroups = [];
    let paletteItems = [];
    let accountProviderFilter = 'all';
    let accountStatusFilter = 'all';
    let routeSearch = '';
    let routeKindFilter = 'all';
    let routeProviderFilter = 'all';

    function el(id) { return document.getElementById(id); }
    function setText(id, value) { const node = el(id); if (node) node.innerText = value; }
    function setHtml(id, value) { const node = el(id); if (node) node.innerHTML = value; }

    function t(key, vars = {}) {
      let value = (TEXT[currentLang] && TEXT[currentLang][key]) || TEXT.en[key] || key;
      Object.keys(vars).forEach(k => value = value.replaceAll('{'+k+'}', vars[k]));
      return value;
    }

    function setLang(lang) {
      currentLang = TEXT[lang] ? lang : 'vi';
      localStorage.setItem('bridgeLang', currentLang);
      applyLanguage();
    }

    function applyLanguage() {
      document.documentElement.lang = currentLang;
      document.title = t('title');
      el('langSelect').value = currentLang;
      setText('appTitle', t('title'));
      setText('languageLabel', t('language'));
      setText('baseUrlLabel', t('baseUrl'));
      setText('refreshButton', t('refresh'));
      setText('adminKeyButton', t('adminKey'));
      setText('tabOverviewButton', t('overview'));
      setText('tabAccountsButton', t('accounts'));
      setText('tabGroupsButton', t('groups'));
      setText('tabKeysButton', t('apiKeys'));
      setText('tabSettingsButton', t('settings'));
      setText('overviewTitle', t('overview'));
      setText('alertsTitle', t('attention'));
      setText('quickAddAccountButton', t('addAccount'));
      setText('quickCreateGroupButton', t('createRouteGroup'));
      setText('accountsTitle', t('accounts'));
      setText('addHint', t('addHint'));
      setText('providerFieldLabel', t('provider'));
      setText('aliasFieldLabel', t('alias'));
      el('newAlias').placeholder = t('aliasPlaceholder');
      setText('createButton', t('createSlot'));
      setText('accountProviderFilterLabel', t('provider'));
      setText('accountStatusFilterLabel', t('status'));
      setText('groupsTitle', t('groups'));
      setText('newGroupButton', t('newGroup'));
      setText('groupEditorTitle', t('groupEditor'));
      setText('saveGroupButton', t('saveGroup'));
      setText('clearGroupButton', t('clear'));
      setText('deleteGroupButton', t('delete'));
      setText('groupNameLabel', t('groupName'));
      el('groupName').placeholder = t('groupNamePlaceholder');
      setText('groupModeLabel', t('mode'));
      setText('modePriority', t('modePriority'));
      setText('modeRoundRobin', t('modeRoundRobin'));
      setText('modeRandom', t('modeRandom'));
      setText('availableTagsTitle', t('availableTags'));
      el('routeSearch').placeholder = t('routeSearchPlaceholder');
      setText('selectedRouteTitle', t('selectedRoute'));
      setText('keysTitle', t('apiKeys'));
      setText('keysHint', t('keysHint'));
      setText('keyNameLabel', t('keyName'));
      el('keyName').placeholder = t('keyNamePlaceholder');
      setText('keyHoursLabel', t('keyHours'));
      setText('keyTokenLimitLabel', t('keyTokenLimit'));
      setText('keyNetworkLabel', t('keyNetwork'));
      setText('keyModelsLabel', t('keyModels'));
      setText('createKeyButton', t('createKey'));
      setText('settingsTitle', t('settings'));
      setText('runtimeTitle', t('runtime'));
      setText('settingsBaseUrlLabel', t('baseUrl'));
      setText('settingsStrategyLabel', t('strategy'));
      setText('settingsFallbackLabel', t('fallback'));
      setText('settingsFailoverLabel', t('maxFailover'));
      setText('settingsModelsLabel', t('visibleModels'));
      setText('usageTitle', t('usage'));
      setHtml('usageContent', t('usageHtml'));
      renderAll();
    }

    function setActiveTab(tab) {
      activeTab = tab;
      ['overview', 'accounts', 'groups', 'keys', 'settings'].forEach(name => {
        el('tab-' + name).classList.toggle('active', name === tab);
        el('tab' + name.charAt(0).toUpperCase() + name.slice(1) + 'Button').classList.toggle('active', name === tab);
      });
    }

    function showAddAccount() {
      setActiveTab('accounts');
      setTimeout(() => el('newAlias').focus(), 0);
    }

    function showRouteBuilder() {
      setActiveTab('groups');
      clearGroupBuilder();
      setTimeout(() => el('groupName').focus(), 0);
    }

    function authHeaders(extra = {}) {
      const headers = Object.assign({}, extra);
      if (adminToken) headers.Authorization = 'Bearer ' + adminToken;
      return headers;
    }

    function setAdminToken() {
      const value = prompt(t('adminKeyPrompt'), adminToken);
      if (value === null) return;
      adminToken = value.trim();
      if (adminToken) localStorage.setItem('bridgeAdminToken', adminToken);
      else localStorage.removeItem('bridgeAdminToken');
      fetchData();
    }

    async function fetchJson(url) {
      const r = await fetch(url, {headers: authHeaders()});
      if (!r.ok) throw new Error(await r.text());
      return r.json();
    }

    async function fetchData() {
      const refresh = el('refreshButton');
      if (refresh) refresh.disabled = true;
      try {
        const accountsData = await fetchJson('/api/accounts');
        const [groupsData, modelsData, keysData] = await Promise.all([
          fetchJson('/api/groups').catch(() => null),
          fetchJson('/v1/models').catch(() => null),
          fetchJson('/api/keys').catch(() => null),
        ]);
        lastAccountsData = accountsData;
        lastGroupsData = groupsData;
        lastModelsData = modelsData;
        lastKeysData = keysData;
        lastFetchOk = true;
        renderAll();
      } catch (err) {
        lastFetchOk = false;
        renderToolbar();
        if (!lastAccountsData) {
          setHtml('overviewStats', `<div class="empty">${escapeHtml(String(err))}</div>`);
          setHtml('alertsList', `<div class="alert error">${escapeHtml(String(err))}</div>`);
        }
      } finally {
        if (refresh) refresh.disabled = false;
      }
    }

    async function fetchAccounts() {
      return fetchData();
    }

    function renderAll() {
      renderToolbar();
      if (!lastAccountsData) return;
      apiKeyProviders = new Set((providersCatalog() || []).filter(p => p.api_key).map(p => p.id));
      renderProviderControls();
      renderOverview();
      renderAccountSection();
      renderGroups();
      renderRouteBuilder();
      renderKeySection();
      renderSettings();
    }

    function providersCatalog() {
      return (lastGroupsData && lastGroupsData.providers) || (lastAccountsData && lastAccountsData.providers) || [];
    }

    function groupsCatalog() {
      return (lastGroupsData && lastGroupsData.groups) || (lastAccountsData && lastAccountsData.route_groups) || [];
    }

    function modelsCatalog() {
      return (lastGroupsData && lastGroupsData.models) || (lastAccountsData && lastAccountsData.models) || [];
    }

    function exposedModels() {
      return (lastModelsData && Array.isArray(lastModelsData.data)) ? lastModelsData.data : [];
    }

    function keyModelOptions() {
      const options = [['*', '* - all models']];
      providersCatalog().forEach(p => options.push([p.id, 'provider: ' + p.id]));
      groupsCatalog().forEach(g => options.push([g.name, 'group: ' + g.name]));
      exposedModels().forEach(m => options.push([m.id, (m.provider || 'model') + ': ' + m.id]));
      const seen = new Set();
      return options.filter(([value]) => {
        if (seen.has(value)) return false;
        seen.add(value);
        return true;
      });
    }

    function renderToolbar() {
      const status = el('serviceStatus');
      if (status) {
        status.className = 'pill ' + (lastFetchOk ? 'green' : (lastAccountsData ? 'amber' : 'grey'));
        status.innerText = lastFetchOk ? t('serviceOnline') : (lastAccountsData ? t('serviceError') : t('loading'));
      }
      const providers = providersCatalog().length;
      const accounts = lastAccountsData ? (lastAccountsData.accounts || []).length : 0;
      setText('providerSummary', t('providerSummary', {providers, accounts}));
    }

    function renderProviderControls() {
      const providers = providersCatalog();
      setSelectOptions(el('newProvider'), providers.map(p => [p.id, providerLabel(p)]), null, false);
      const providerOptions = [['all', t('filterAllProviders')]].concat(providers.map(p => [p.id, providerLabel(p)]));
      setSelectOptions(el('accountProviderFilter'), providerOptions, accountProviderFilter, false);
      setSelectOptions(el('routeProviderFilter'), providerOptions, routeProviderFilter, false);
      setSelectOptions(el('keyNetwork'), [
        ['local', t('keyNetworkLocal')],
        ['lan', t('keyNetworkLan')],
        ['internet', t('keyNetworkInternet')],
      ], el('keyNetwork').value || 'local', false);
      setSelectOptions(el('keyModels'), keyModelOptions(), null, false);
      const statusOptions = [
        ['all', t('filterAllStatuses')],
        ['healthy', t('healthy')],
        ['exhausted', t('exhausted')],
        ['invalid', t('invalid')],
        ['expired', t('expired')],
        ['refreshable', t('refreshable')],
        ['notLoggedIn', t('notLoggedIn')],
        ['disabled', t('disabled')],
      ];
      setSelectOptions(el('accountStatusFilter'), statusOptions, accountStatusFilter, false);
      const kindOptions = [
        ['all', t('kindAll')],
        ['provider', t('kindProvider')],
        ['model', t('kindModel')],
        ['group', t('kindGroup')],
      ];
      setSelectOptions(el('routeKindFilter'), kindOptions, routeKindFilter, false);
    }

    function setSelectOptions(select, options, keepValue, includeBlank) {
      if (!select) return;
      const current = keepValue === null ? select.value : keepValue;
      select.innerHTML = '';
      if (includeBlank) options = [['', '']].concat(options);
      options.forEach(([value, label]) => {
        const o = document.createElement('option');
        o.value = value;
        o.text = label;
        select.appendChild(o);
      });
      if (current && options.some(([value]) => value === current)) select.value = current;
    }

    function renderOverview() {
      const accounts = lastAccountsData.accounts || [];
      const stats = computeAccountStats(accounts);
      const metrics = [
        [stats.total, t('totalAccounts')],
        [stats.healthy, t('healthyAccounts')],
        [stats.exhausted, t('exhaustedAccounts')],
        [stats.invalid, t('invalidAccounts')],
        [groupsCatalog().length, t('routeGroups')],
      ];
      el('overviewStats').innerHTML = metrics.map(([value, label]) => `
        <div class="metric">
          <div class="metric-value">${escapeHtml(value)}</div>
          <div class="metric-label">${escapeHtml(label)}</div>
        </div>
      `).join('');

      const allAlerts = buildAlerts();
      const alerts = allAlerts.slice(0, 8);
      const root = el('alertsList');
      if (!allAlerts.length) {
        root.innerHTML = `<div class="empty">${t('noAlerts')}</div>`;
        return;
      }
      root.innerHTML = alerts.map(a => `
        <div class="alert ${a.level}">
          <div><b>${escapeHtml(a.title)}</b></div>
          <div class="muted">${escapeHtml(a.body)}</div>
        </div>
      `).join('') + (allAlerts.length > alerts.length
        ? `<div class="muted">${t('alertMore', {count: allAlerts.length - alerts.length})}</div>`
        : '');
    }

    function computeAccountStats(accounts) {
      const stats = {total: accounts.length, healthy: 0, exhausted: 0, invalid: 0};
      accounts.forEach(a => {
        const key = statusKey(a);
        if (key === 'healthy') stats.healthy += 1;
        if (key === 'exhausted') stats.exhausted += 1;
        if (key === 'invalid') stats.invalid += 1;
      });
      return stats;
    }

    function buildAlerts() {
      const alerts = [];
      const accounts = lastAccountsData.accounts || [];
      accounts.forEach(a => {
        const h = a.health || {};
        if (a.enabled && (h.invalid || !a.logged_in || a.session_state === 'expired' || a.session_state === 'missing')) {
          alerts.push({level: h.invalid ? 'error' : 'warning', title: t('needsSetupTitle'), body: t('needsSetupBody', {slot: accountName(a)})});
        }
        if (a.enabled && h.exhausted_in > 0) {
          alerts.push({level: 'warning', title: t('exhaustedTitle'), body: t('exhaustedBody', {slot: accountName(a), time: formatSecs(h.exhausted_in)})});
        }
        const low = lowQuotaDetail(h.rate_limit || {});
        if (a.enabled && low) {
          alerts.push({level: 'warning', title: t('lowQuotaTitle'), body: t('lowQuotaBody', {slot: accountName(a), detail: low})});
        }
      });
      validateGroups().forEach(issue => {
        alerts.push({level: 'error', title: t('badGroupTitle'), body: t('badGroupBody', issue)});
      });
      return alerts;
    }

    function validateGroups() {
      const providers = new Set(providersCatalog().map(p => p.id));
      const groups = groupsCatalog();
      const groupNames = new Set(groups.map(g => g.name));
      const modelKeys = new Set(modelsCatalog().map(m => `${m.provider}/${m.id}`));
      const issues = [];
      groups.forEach(group => {
        (group.items || []).forEach(item => {
          if (item.kind === 'provider' && !providers.has(item.provider)) {
            issues.push({group: group.name, detail: t('missingProvider', {provider: item.provider})});
          }
          if (item.kind === 'group') {
            if (item.group === group.name) issues.push({group: group.name, detail: t('selfReference')});
            else if (!groupNames.has(item.group)) issues.push({group: group.name, detail: t('missingGroup', {group: item.group})});
          }
          if (item.kind === 'model') {
            if (!providers.has(item.provider)) issues.push({group: group.name, detail: t('missingProvider', {provider: item.provider})});
            else if (!modelKeys.has(`${item.provider}/${item.model}`)) issues.push({group: group.name, detail: t('missingModel', {model: `${item.provider}/${item.model}`})});
          }
        });
      });
      return issues;
    }

    function renderAccountSection() {
      const data = lastAccountsData;
      const accounts = getFilteredAccounts(data.accounts || []);
      el('accountCount').innerText = String((data.accounts || []).length);
      const root = el('accountList');
      if (!accounts.length) {
        root.innerHTML = `<div class="empty">${t('empty')}</div>`;
        return;
      }
      const rows = [
        `<div class="account-row header-row"><div>${t('slot')}</div><div>${t('state')}</div><div>${t('rotation')}</div><div>${t('quotaRate')}</div><div>${t('actions')}</div></div>`
      ].concat(accounts.map(renderAccountRow));
      root.innerHTML = `<div class="account-table">${rows.join('')}</div>`;
    }

    function renderKeySection() {
      const root = el('keyList');
      if (!root) return;
      const keys = (lastKeysData && lastKeysData.keys) || [];
      if (!keys.length) {
        root.innerHTML = `<div class="empty">${t('noKeys')}</div>`;
        return;
      }
      const header = `<div class="account-row key-row header-row"><div>${t('keyName')}</div><div>${t('keyStatus')}</div><div>${t('keyUsage')}</div><div>${t('keyPolicy')}</div><div>${t('actions')}</div></div>`;
      root.innerHTML = `<div class="account-table">${header}${keys.map(renderKeyRow).join('')}</div>`;
    }

    function renderKeyRow(k) {
      const statusClass = {active: 'green', disabled: 'grey', expired: 'red', exhausted: 'amber'}[k.status] || 'grey';
      const limit = k.token_limit ? k.token_limit : t('keyUnlimited');
      const used = k.token_used || 0;
      const expires = k.expires_at ? `${t('expiresIn')} ${formatSecs(k.expires_in || 0)}` : t('keyNoExpiry');
      const models = (k.allowed_models || []).join(', ');
      const lastUsed = k.last_used_at ? `${formatAge(k.last_used_at)} ${k.last_used_ip ? '(' + escapeHtml(k.last_used_ip) + ')' : ''}` : t('keyNever');
      const actionStatus = k.status === 'disabled' ? 'active' : 'disabled';
      const actionLabel = k.status === 'disabled' ? t('keyActivate') : t('keyDisable');
      return `<div class="account-row key-row">
        <div data-label="${escapeHtml(t('keyName'))}">
          <div class="slot-name">${escapeHtml(k.name || k.id)}</div>
          <div class="slot-id">${escapeHtml(k.prefix || k.id)}</div>
          <div class="meta">${escapeHtml(t('keyLastUsed'))}: ${lastUsed}</div>
        </div>
        <div data-label="${escapeHtml(t('keyStatus'))}">
          <span class="pill ${statusClass}">${escapeHtml(k.status)}</span>
          <div class="meta">${escapeHtml(expires)}</div>
        </div>
        <div data-label="${escapeHtml(t('keyUsage'))}">
          <div>${escapeHtml(String(used))} / ${escapeHtml(String(limit))}</div>
          <div class="meta">${escapeHtml(k.last_error || '')}</div>
        </div>
        <div data-label="${escapeHtml(t('keyPolicy'))}">
          <div>${escapeHtml(k.network_scope || 'local')}</div>
          <div class="meta">${escapeHtml(models || '-')}</div>
        </div>
        <div data-label="${escapeHtml(t('actions'))}" class="actions">
          <button class="small secondary" onclick="patchManagedKey('${k.id}', {status: '${actionStatus}'})">${actionLabel}</button>
          <button class="small secondary" onclick="editManagedKey('${k.id}')">${t('keyEdit')}</button>
          <button class="small secondary" onclick="resetManagedKeyUsage('${k.id}')">${t('keyResetUsage')}</button>
          <button class="small secondary" onclick="rotateManagedKey('${k.id}')">${t('keyRotate')}</button>
          <button class="small danger" onclick="deleteManagedKey('${k.id}')">${t('delete')}</button>
        </div>
      </div>`;
    }

    function getFilteredAccounts(accounts) {
      return accounts.filter(a => {
        const providerOk = accountProviderFilter === 'all' || a.provider === accountProviderFilter;
        const statusOk = accountStatusFilter === 'all' || statusKey(a) === accountStatusFilter;
        return providerOk && statusOk;
      });
    }

    function setAccountProviderFilter(value) {
      accountProviderFilter = value || 'all';
      renderAccountSection();
    }

    function setAccountStatusFilter(value) {
      accountStatusFilter = value || 'all';
      renderAccountSection();
    }

    function renderAccountRow(a) {
      const h = a.health || {};
      const expires = a.expires_in ? `${t('expiresIn')} ${formatSecs(a.expires_in)}` : '';
      const inflight = h.in_flight ? `${t('inFlight')}: ${h.in_flight}` : '';
      const auth = a.auth_type ? `${t('auth')}: ${a.auth_type}` : '';
      const tier = a.tier ? `${t('tier')}: ${a.tier}` : '';
      const details = [a.email || '-', expires, inflight, auth, tier].filter(Boolean).map(escapeHtml).join(' · ');
      const setupButton = apiKeyProviders.has(a.provider)
        ? `<button class="small" onclick="setApiKey('${a.slot_id}')">${t('apiKey')}</button>`
        : `<button class="small" onclick="loginSlot('${a.slot_id}')">${t('login')}</button>`;
      const rotationLabel = a.rotation_enabled ? t('rotationOn') : t('rotationOff');
      const enabledButton = `<button class="small secondary" onclick="toggleEnabled('${a.slot_id}', ${!a.enabled})">${a.enabled ? t('disable') : t('enable')}</button>`;
      return `<div class="account-row">
        <div data-label="${escapeHtml(t('slot'))}">
          <div class="slot-name">${escapeHtml(a.alias || a.slot_id)}</div>
          <div class="slot-id">${escapeHtml(a.slot_id)}</div>
          <div class="meta">${escapeHtml(providerLabel({id: a.provider}))} · ${details}</div>
        </div>
        <div data-label="${escapeHtml(t('state'))}">
          ${renderStatusBadge(a)}
          <div class="meta">${escapeHtml(a.session_notice || '')}</div>
        </div>
        <div data-label="${escapeHtml(t('rotation'))}">
          <span class="pill ${a.rotation_enabled ? 'green' : 'grey'}">${rotationLabel}</span>
          <div class="meta">${a.enabled ? t('on') : t('disabled')}</div>
        </div>
        <div data-label="${escapeHtml(t('quotaRate'))}">${renderRateLimit(h.rate_limit || {})}</div>
        <div data-label="${escapeHtml(t('actions'))}" class="actions">
          ${setupButton}
          <button class="small secondary" onclick="refreshSlot('${a.slot_id}')">${t('refresh')}</button>
          <button class="small secondary" onclick="toggleRotation('${a.slot_id}', ${!a.rotation_enabled})">${a.rotation_enabled ? t('turnRotationOff') : t('turnRotationOn')}</button>
          ${enabledButton}
          <details class="more-menu">
            <summary>${t('more')}</summary>
            <div class="menu">
              <button class="small secondary" onclick="logoutSlot('${a.slot_id}')">${t('logout')}</button>
              <button class="small danger" onclick="deleteSlot('${a.slot_id}')">${t('delete')}</button>
            </div>
          </details>
        </div>
      </div>`;
    }

    function statusKey(a) {
      const h = a.health || {};
      if (!a.enabled) return 'disabled';
      if (h.invalid) return 'invalid';
      if (h.exhausted_in > 0) return 'exhausted';
      if (a.logged_in) return 'healthy';
      if (a.session_state === 'refreshable') return 'refreshable';
      if (a.session_state === 'expired') return 'expired';
      return 'notLoggedIn';
    }

    function renderStatusBadge(a) {
      const key = statusKey(a);
      const className = {
        healthy: 'green',
        exhausted: 'amber',
        invalid: 'red',
        expired: 'red',
        refreshable: 'amber',
        disabled: 'grey',
        notLoggedIn: 'red',
      }[key] || 'grey';
      const h = a.health || {};
      const label = key === 'exhausted' ? `${t('exhausted')} ${formatSecs(h.exhausted_in)}` : t(key);
      return `<span class="pill ${className}">${escapeHtml(label)}</span>`;
    }

    function renderGroups() {
      const groups = groupsCatalog();
      renderedGroups = groups;
      const root = el('groupList');
      if (!groups.length) {
        root.innerHTML = `<div class="empty">${t('noGroups')}</div>`;
        return;
      }
      root.innerHTML = groups.map((g, idx) => `
        <button class="group-card ${activeGroupName === g.name ? 'active' : ''}" onclick="selectGroupByIndex(${idx})">
          <span class="group-card-title">
            <b>${escapeHtml(g.name)}</b>
            <span class="pill outline">${escapeHtml(modeLabel(g.mode))}</span>
          </span>
          <span class="group-items">${formatRouteTrail(g.items || [])}</span>
        </button>
      `).join('');
    }

    function selectGroupByIndex(index) {
      const group = renderedGroups[index];
      if (!group) return;
      activeGroupName = group.name;
      el('groupName').value = group.name;
      el('groupMode').value = group.mode || 'priority';
      selectedRouteItems = (group.items || []).map(x => Object.assign({}, x));
      renderGroups();
      renderRouteBuilder();
    }

    function clearGroupBuilder() {
      activeGroupName = '';
      el('groupName').value = '';
      el('groupMode').value = 'priority';
      selectedRouteItems = [];
      renderGroups();
      renderRouteBuilder();
    }

    function renderRouteBuilder() {
      renderRoutePalette();
      renderSelectedRouteItems();
    }

    function setRouteSearch(value) {
      routeSearch = (value || '').toLowerCase().trim();
      renderRoutePalette();
    }

    function setRouteKindFilter(value) {
      routeKindFilter = value || 'all';
      renderRoutePalette();
    }

    function setRouteProviderFilter(value) {
      routeProviderFilter = value || 'all';
      renderRoutePalette();
    }

    function renderRoutePalette() {
      const allItems = buildPaletteItems();
      const filtered = allItems.filter(item => {
        if (routeKindFilter !== 'all' && item.kind !== routeKindFilter) return false;
        if (routeProviderFilter !== 'all' && item.provider !== routeProviderFilter) return false;
        if (!routeSearch) return true;
        return itemLabel(item).toLowerCase().includes(routeSearch);
      });
      paletteItems = filtered.slice(0, 160);
      setText('routePaletteCount', t('tagsShown', {shown: paletteItems.length, total: filtered.length}));
      const root = el('routePalette');
      if (!paletteItems.length) {
        root.innerHTML = `<div class="empty">${t('noTags')}</div>`;
        return;
      }
      root.innerHTML = paletteItems.map((item, idx) => `
        <span class="route-chip ${routeItemClass(item)}" draggable="true"
          ondragstart="paletteDragStart(event, ${idx})" onclick="addPaletteItem(${idx})">
          ${escapeHtml(itemLabel(item))}
        </span>
      `).join('');
    }

    function buildPaletteItems() {
      const items = [];
      providersCatalog().forEach(p => items.push({kind: 'provider', provider: p.id}));
      groupsCatalog()
        .filter(g => !activeGroupName || g.name !== activeGroupName)
        .forEach(g => items.push({kind: 'group', group: g.name}));
      modelsCatalog().forEach(m => items.push({kind: 'model', provider: m.provider, model: m.id}));
      return items;
    }

    function paletteDragStart(ev, index) {
      ev.dataTransfer.setData('application/json', JSON.stringify({source: 'palette', item: paletteItems[index]}));
    }

    function selectedDragStart(ev, index) {
      ev.dataTransfer.setData('application/json', JSON.stringify({source: 'selected', index}));
    }

    function allowRouteDrop(ev) {
      ev.preventDefault();
    }

    function dropRouteItem(ev, targetIndex) {
      ev.preventDefault();
      ev.stopPropagation();
      let payload = {};
      try { payload = JSON.parse(ev.dataTransfer.getData('application/json') || '{}'); }
      catch (e) { return; }
      if (payload.source === 'selected') {
        const from = parseInt(payload.index);
        if (Number.isNaN(from) || from < 0 || from >= selectedRouteItems.length) return;
        const item = selectedRouteItems.splice(from, 1)[0];
        let to = targetIndex === undefined ? selectedRouteItems.length : targetIndex;
        if (from < to) to -= 1;
        selectedRouteItems.splice(Math.max(0, Math.min(to, selectedRouteItems.length)), 0, item);
      } else if (payload.source === 'palette' && payload.item) {
        const to = targetIndex === undefined ? selectedRouteItems.length : targetIndex;
        selectedRouteItems.splice(Math.max(0, Math.min(to, selectedRouteItems.length)), 0, Object.assign({}, payload.item));
      }
      renderSelectedRouteItems();
    }

    function addPaletteItem(index) {
      const item = paletteItems[index];
      if (!item) return;
      selectedRouteItems.push(Object.assign({}, item));
      renderSelectedRouteItems();
    }

    function moveRouteItem(index, delta) {
      const target = index + delta;
      if (target < 0 || target >= selectedRouteItems.length) return;
      const item = selectedRouteItems.splice(index, 1)[0];
      selectedRouteItems.splice(target, 0, item);
      renderSelectedRouteItems();
    }

    function removeRouteItem(index) {
      selectedRouteItems.splice(index, 1);
      renderSelectedRouteItems();
    }

    function renderSelectedRouteItems() {
      const root = el('selectedRouteItems');
      if (!selectedRouteItems.length) {
        root.innerHTML = `<div class="empty">${t('selectedEmpty')}</div>`;
        return;
      }
      root.innerHTML = selectedRouteItems.map((item, idx) => `
        <div class="route-item ${routeItemClass(item)}" draggable="true"
          ondragstart="selectedDragStart(event, ${idx})"
          ondragover="allowRouteDrop(event)" ondrop="dropRouteItem(event, ${idx})">
          <span class="route-index">${idx + 1}</span>
          <span class="route-label">${escapeHtml(itemLabel(item))}</span>
          <span class="actions">
            <button class="small secondary" onclick="moveRouteItem(${idx}, -1)">${t('up')}</button>
            <button class="small secondary" onclick="moveRouteItem(${idx}, 1)">${t('down')}</button>
            <button class="small danger" onclick="removeRouteItem(${idx})">${t('remove')}</button>
          </span>
        </div>
      `).join('');
    }

    function renderSettings() {
      const data = lastAccountsData || {};
      setText('settingsStrategy', data.pool_strategy || '-');
      const fallback = data.cross_provider_fallback || [];
      const fallbackOn = Array.isArray(fallback) ? fallback.length > 0 : Boolean(fallback);
      const fallbackText = Array.isArray(fallback) && fallback.length ? fallback.join(', ') : '';
      setText('settingsFallback', fallbackOn ? `${t('on')}${fallbackText ? ': ' + fallbackText : ''}` : t('off'));
      setText('settingsMaxFailover', data.max_failover === undefined ? '-' : String(data.max_failover));
      const visible = exposedModels();
      const modelCount = lastModelsData && Array.isArray(lastModelsData.data) ? visible.length : modelsCatalog().length;
      setText('settingsModelCount', String(modelCount));
    }

    function providerLabel(p) {
      if (!p) return '';
      if (p.id === 'chatgpt') return 'ChatGPT (Codex)';
      if (p.id === 'google') return 'Google Gemini';
      if (p.id === 'anthropic') return 'Anthropic Claude';
      if (p.id === 'deepseek') return 'DeepSeek';
      return p.label || p.id;
    }

    function accountName(a) {
      return a.alias || a.slot_id;
    }

    function itemLabel(item) {
      if (!item) return '';
      if (item.kind === 'provider') return `${t('groupProvider')}: ${item.provider}`;
      if (item.kind === 'group') return `${t('groupGroup')}: ${item.group}`;
      return `${t('groupModel')}: ${item.provider}/${item.model}`;
    }

    function routeItemClass(item) {
      if (!item) return 'model';
      return item.kind === 'provider' ? 'provider' : (item.kind === 'group' ? 'group' : 'model');
    }

    function modeLabel(mode) {
      if (mode === 'round_robin') return t('modeRoundRobin');
      if (mode === 'random') return t('modeRandom');
      return t('modePriority');
    }

    function formatRouteTrail(items) {
      if (!items.length) return '-';
      return items.map(item => escapeHtml(itemLabel(item))).join(' -> ');
    }

    function escapeHtml(s) {
      return String(s === undefined || s === null ? '' : s).replace(/[<>&"']/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;', "'":'&#39;'}[c]));
    }

    function formatSecs(s) {
      s = parseInt(s);
      if (!Number.isFinite(s) || s < 0) s = 0;
      if (s < 60) return s + 's';
      if (s < 3600) return Math.floor(s / 60) + 'm';
      return Math.floor(s / 3600) + 'h ' + Math.floor((s % 3600) / 60) + 'm';
    }

    function formatAge(ts) {
      const s = Math.max(0, Math.floor(Date.now() / 1000 - parseInt(ts)));
      if (s < 5) return t('now');
      return formatSecs(s) + ' ' + t('ago');
    }

    function renderRateLimit(rate) {
      if (!rate || !Object.keys(rate).length) return `<div class="quota-line">${t('rateWaiting')}</div>`;
      const bits = [];
      if (rate.model) bits.push(`<span>${t('model')}: ${escapeHtml(rate.model)}</span>`);
      if (rate.updated_at) bits.push(`<span>${t('updated')}: ${formatAge(rate.updated_at)}</span>`);
      if (rate.retry_after_seconds) bits.push(`<span>${t('retry')}: ${formatSecs(rate.retry_after_seconds)}</span>`);
      const limits = rate.limits || {};
      Object.keys(limits).forEach(name => {
        const v = limits[name] || {};
        const rem = v.remaining !== undefined ? v.remaining : '?';
        const lim = v.limit !== undefined ? v.limit : '?';
        const pct = v.percent_remaining !== undefined ? ` (${v.percent_remaining}%)` : '';
        const reset = v.reset ? ` ${t('reset')} ${escapeHtml(v.reset)}` : '';
        bits.push(`<span>${escapeHtml(name)}: ${escapeHtml(rem)}/${escapeHtml(lim)}${pct}${reset}</span>`);
      });
      if (rate.error && rate.error.reason) bits.push(`<span>${t('reason')}: ${escapeHtml(rate.error.reason)}</span>`);
      return `<div class="quota-line">${bits.join('')}</div>`;
    }

    function lowQuotaDetail(rate) {
      if (!rate || !rate.limits) return '';
      const parts = [];
      Object.keys(rate.limits).forEach(name => {
        const v = rate.limits[name] || {};
        const pct = Number(v.percent_remaining);
        const rem = Number(v.remaining);
        if (Number.isFinite(pct) && pct <= 15) parts.push(`${name} ${pct}%`);
        else if (Number.isFinite(rem) && rem <= 0) parts.push(`${name} 0`);
      });
      return parts.slice(0, 2).join(', ');
    }

    function selectedKeyModels() {
      const selected = Array.from(el('keyModels').selectedOptions || []).map(o => o.value).filter(Boolean);
      if (!selected.length || selected.includes('*')) return ['*'];
      return selected;
    }

    function showManagedSecret(secret) {
      const box = el('keySecretBox');
      if (!box) return;
      box.style.display = 'block';
      box.innerHTML = `${escapeHtml(t('keyCreated'))}<br><code>${escapeHtml(secret)}</code>`;
    }

    async function createManagedKey() {
      const hours = parseInt(el('keyHours').value || '0');
      const tokenLimit = parseInt(el('keyTokenLimit').value || '0');
      const body = {
        name: el('keyName').value.trim() || 'Client key',
        duration_hours: Number.isFinite(hours) ? Math.max(0, hours) : 0,
        token_limit: Number.isFinite(tokenLimit) ? Math.max(0, tokenLimit) : 0,
        network_scope: el('keyNetwork').value || 'local',
        allowed_models: selectedKeyModels(),
      };
      const r = await fetch('/api/keys', {
        method: 'POST',
        headers: authHeaders({'Content-Type': 'application/json'}),
        body: JSON.stringify(body),
      });
      if (!r.ok) { alert(t('createFailed') + await r.text()); return; }
      const data = await r.json();
      if (data.secret) showManagedSecret(data.secret);
      el('keyName').value = '';
      await fetchData();
    }

    async function patchManagedKey(id, body) {
      const r = await fetch('/api/keys/' + encodeURIComponent(id), {
        method: 'PATCH',
        headers: authHeaders({'Content-Type': 'application/json'}),
        body: JSON.stringify(body),
      });
      if (!r.ok) { alert(t('saveFailed') + await r.text()); return; }
      await fetchData();
    }

    async function editManagedKey(id) {
      const key = ((lastKeysData && lastKeysData.keys) || []).find(k => k.id === id);
      if (!key) return;
      const name = prompt(t('keyName'), key.name || '');
      if (name === null) return;
      const limit = prompt(t('keyTokenLimit'), String(key.token_limit || 0));
      if (limit === null) return;
      const models = prompt(t('keyModelsPrompt'), (key.allowed_models || ['*']).join(', '));
      if (models === null) return;
      const scope = prompt(t('keyNetwork'), key.network_scope || 'local');
      if (scope === null) return;
      await patchManagedKey(id, {
        name,
        token_limit: parseInt(limit || '0') || 0,
        allowed_models: models.split(',').map(x => x.trim()).filter(Boolean),
        network_scope: scope,
      });
    }

    async function rotateManagedKey(id) {
      if (!confirm(t('keyRotateConfirm'))) return;
      const r = await fetch('/api/keys/' + encodeURIComponent(id) + '/rotate', {
        method: 'POST',
        headers: authHeaders(),
      });
      if (!r.ok) { alert(t('saveFailed') + await r.text()); return; }
      const data = await r.json();
      if (data.secret) showManagedSecret(data.secret);
      await fetchData();
    }

    async function resetManagedKeyUsage(id) {
      await fetch('/api/keys/' + encodeURIComponent(id) + '/reset-usage', {
        method: 'POST',
        headers: authHeaders(),
      });
      await fetchData();
    }

    async function deleteManagedKey(id) {
      const key = ((lastKeysData && lastKeysData.keys) || []).find(k => k.id === id);
      if (!confirm(t('keyDeleteConfirm', {name: key ? (key.name || key.id) : id}))) return;
      await fetch('/api/keys/' + encodeURIComponent(id), {
        method: 'DELETE',
        headers: authHeaders(),
      });
      await fetchData();
    }

    async function createAccount() {
      const provider = el('newProvider').value;
      const alias = el('newAlias').value;
      const r = await fetch('/api/accounts', {
        method: 'POST', headers: authHeaders({'Content-Type': 'application/json'}),
        body: JSON.stringify({provider, alias}),
      });
      if (!r.ok) { alert(t('createFailed') + await r.text()); return; }
      el('newAlias').value = '';
      await fetchData();
    }

    async function saveGroup() {
      const name = el('groupName').value.trim();
      const mode = el('groupMode').value;
      if (!name) { alert(t('groupNameRequired')); return; }
      if (!selectedRouteItems.length) { alert(t('groupItemsRequired')); return; }
      const r = await fetch('/api/groups', {
        method: 'POST', headers: authHeaders({'Content-Type': 'application/json'}),
        body: JSON.stringify({name, mode, items: selectedRouteItems}),
      });
      if (!r.ok) { alert(t('saveGroupFailed') + await r.text()); return; }
      activeGroupName = name;
      await fetchData();
    }

    async function deleteGroup(name) {
      if (!confirm(t('deleteGroupConfirm', {name}))) return;
      await fetch('/api/groups/' + encodeURIComponent(name), {method: 'DELETE', headers: authHeaders()});
      if (activeGroupName === name || el('groupName').value.trim() === name) clearGroupBuilder();
      await fetchData();
    }

    async function deleteActiveGroup() {
      const name = el('groupName').value.trim();
      if (name) await deleteGroup(name);
    }

    async function loginSlot(slot) {
      const popup = window.open('about:blank', '_blank');
      const r = await fetch('/api/accounts/' + slot + '/login', {method: 'POST', headers: authHeaders()});
      const data = await r.json();
      if (data.auth_url) {
        if (popup) popup.location = data.auth_url;
        else window.location = data.auth_url;
      } else if (popup) {
        popup.close();
      }
      setTimeout(fetchData, 1500);
    }

    async function setApiKey(slot) {
      const key = prompt(t('pasteApiKey') + slot);
      if (!key) return;
      const r = await fetch('/api/accounts/' + slot + '/api-key', {
        method: 'POST', headers: authHeaders({'Content-Type': 'application/json'}),
        body: JSON.stringify({api_key: key}),
      });
      if (!r.ok) { alert(t('saveFailed') + await r.text()); return; }
      await fetchData();
    }

    async function refreshSlot(slot) {
      await fetch('/api/accounts/' + slot + '/refresh', {method: 'POST', headers: authHeaders()});
      await fetchData();
    }

    async function logoutSlot(slot) {
      if (!confirm(t('logoutConfirm', {slot}))) return;
      await fetch('/api/accounts/' + slot + '/logout', {method: 'POST', headers: authHeaders()});
      await fetchData();
    }

    async function deleteSlot(slot) {
      if (!confirm(t('deleteConfirm', {slot}))) return;
      await fetch('/api/accounts/' + slot, {method: 'DELETE', headers: authHeaders()});
      await fetchData();
    }

    async function toggleEnabled(slot, enabled) {
      await fetch('/api/accounts/' + slot, {
        method: 'PATCH', headers: authHeaders({'Content-Type': 'application/json'}),
        body: JSON.stringify({enabled}),
      });
      await fetchData();
    }

    async function toggleRotation(slot, rotation_enabled) {
      await fetch('/api/accounts/' + slot, {
        method: 'PATCH', headers: authHeaders({'Content-Type': 'application/json'}),
        body: JSON.stringify({rotation_enabled}),
      });
      await fetchData();
    }

    applyLanguage();
    fetchData();
    setInterval(fetchData, 3000);
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    html = (
        INDEX_HTML
        .replace("__HOST__", SETTINGS.host)
        .replace("__PORT__", str(SETTINGS.port))
        .replace("__LOCALE__", SETTINGS.locale or "vi")
    )
    return html
