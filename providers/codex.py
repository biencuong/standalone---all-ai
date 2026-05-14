# -*- coding: utf-8 -*-
"""ChatGPT / Codex provider.

OAuth flow with auth.openai.com, upstream at chatgpt.com/backend-api/codex/responses,
mapping OpenAI Chat Completions <-> Codex Responses API.

All 16 fixes from the original bridge are preserved (PDF/document, reasoning
summary streaming, real usage, structured output, strict tool calls, built-in
tools passthrough, previous_response_id+store, include param, image detail,
audio model override, finish_reason mapping, headers, video config,
include_usage, system message concat).
"""
from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import shutil
import subprocess
import tempfile
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
from urllib.parse import urlencode

import httpx

from core import (
    AuthRevoked,
    PendingAuth,
    QuotaExhausted,
    SETTINGS,
    UpstreamClientError,
    UpstreamServerError,
    CALLBACK_BROKER,
    decode_jwt_payload,
    generate_pkce_pair,
    generate_state,
    get_http_client,
    logger,
    nested_get,
    safe_int,
    sse_data,
    sse_done,
)

PROVIDER = "chatgpt"

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTH_ENDPOINT = "https://auth.openai.com/oauth/authorize"
TOKEN_ENDPOINT = "https://auth.openai.com/oauth/token"
RESPONSES_ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"
CALLBACK_PATH = "/auth/callback"

REFRESH_SAFETY_WINDOW = 60

# ---------------------------------------------------------------------------
# Model registry (ported from codex_models.py)
# ---------------------------------------------------------------------------

DEFAULT_CODEX_MODEL_ID = "gpt-5.5"
SUPPORTED_REASONING_EFFORTS = {"low", "medium", "high", "xhigh"}
SUPPORTED_VERBOSITY_LEVELS = {"low", "medium", "high"}
SUPPORTED_REASONING_SUMMARIES = {"auto", "concise", "detailed"}
REASONING_EFFORT_ORDER = ["low", "medium", "high", "xhigh"]
REASONING_EFFORT_LABELS = {
    "low": "Low", "medium": "Medium", "high": "High", "xhigh": "Extra High",
}

CODEX_MODEL_CATALOG: List[Dict[str, Any]] = [
    {"id": "gpt-5.5", "name": "GPT-5.5",
     "description": "Frontier model for complex coding, research, and real-world work.",
     "supports_multimodal": True, "supports_image": True,
     "default_reasoning_effort": "medium", "default_verbosity": "medium"},
    {"id": "gpt-5.4", "name": "GPT-5.4",
     "description": "Strong model for everyday coding.",
     "supports_multimodal": True, "supports_image": True,
     "default_reasoning_effort": "medium", "default_verbosity": "medium"},
    {"id": "gpt-5.4-mini", "name": "GPT-5.4 Mini",
     "description": "Small, fast, and cost-efficient.",
     "supports_multimodal": True, "supports_image": True,
     "default_reasoning_effort": "medium", "default_verbosity": "medium"},
    {"id": "gpt-5.3-codex", "name": "GPT-5.3 Codex",
     "description": "Coding-optimized model.",
     "supports_multimodal": False, "supports_image": False,
     "default_reasoning_effort": "medium", "default_verbosity": "medium"},
    {"id": "gpt-5.2", "name": "GPT-5.2",
     "description": "Optimized for long-running agents.",
     "supports_multimodal": True, "supports_image": True,
     "default_reasoning_effort": "medium", "default_verbosity": "medium"},
]


def _build_variants(base_id: str) -> List[Dict[str, Any]]:
    base = next((it for it in CODEX_MODEL_CATALOG if it["id"] == base_id), None)
    if not base:
        return []
    variants: List[Dict[str, Any]] = []
    for effort in REASONING_EFFORT_ORDER:
        v = deepcopy(base)
        v["id"] = f"{base_id}-{effort}"
        v["name"] = f"{base['name']} ({REASONING_EFFORT_LABELS[effort]})"
        v["canonical_model"] = base_id
        v["default_reasoning_effort"] = effort
        v["variant_reasoning_effort"] = effort
        variants.append(v)
    return variants


CODEX_MODEL_VARIANTS = _build_variants("gpt-5.5")
ALL_CODEX_MODELS = CODEX_MODEL_CATALOG + CODEX_MODEL_VARIANTS
EXPOSED_MODELS = CODEX_MODEL_VARIANTS + [m for m in CODEX_MODEL_CATALOG if m["id"] != "gpt-5.5"]
CODEX_MODEL_IDS = {m["id"] for m in ALL_CODEX_MODELS}

MODEL_ALIASES: Dict[str, str] = {
    "5.5": "gpt-5.5", "gpt5.5": "gpt-5.5",
    "gpt-5": "gpt-5.5", "gpt-5-chat-latest": "gpt-5.5",
    "gpt-5.5-codex": "gpt-5.5", "gpt-5.5-thinking": "gpt-5.5",
    "gpt-5-mini": "gpt-5.4-mini",
    "gpt-5.1": "gpt-5.2", "gpt-5.1-codex": "gpt-5.3-codex",
    "codex-mini": "gpt-5.3-codex", "codex-mini-latest": "gpt-5.3-codex",
    "gpt-5-codex": "gpt-5.3-codex",
    "gpt-5.5-pro": "gpt-5.5", "gpt-5.4-pro": "gpt-5.4",
    "gpt-5.3-codex-spark": "gpt-5.3-codex", "gpt-5.2-codex": "gpt-5.2",
}
MODEL_ALIASES.update({v["id"]: v["canonical_model"] for v in CODEX_MODEL_VARIANTS})


def normalize_model(model: Optional[str], default: Optional[str] = None) -> str:
    value = (model or "").strip().lower()
    if not value:
        return default or SETTINGS.codex_default_model
    return MODEL_ALIASES.get(value, value)


def _model_meta(model: str) -> Optional[Dict[str, Any]]:
    v = (model or "").strip().lower()
    for m in ALL_CODEX_MODELS:
        if m["id"] == v:
            return m
    n = normalize_model(model)
    for m in ALL_CODEX_MODELS:
        if m["id"] == n:
            return m
    return None


def _model_default_effort(model: str) -> str:
    m = _model_meta(model)
    eff = (m or {}).get("default_reasoning_effort") or "medium"
    return eff if eff in SUPPORTED_REASONING_EFFORTS else "medium"


def _model_variant_effort(model: Optional[str]) -> Optional[str]:
    m = _model_meta(model or "")
    eff = (m or {}).get("variant_reasoning_effort")
    return eff if eff in SUPPORTED_REASONING_EFFORTS else None


def _model_default_verbosity(model: str) -> str:
    m = _model_meta(model)
    v = (m or {}).get("default_verbosity") or "medium"
    return v if v in SUPPORTED_VERBOSITY_LEVELS else "medium"


def build_models_list() -> List[Dict[str, Any]]:
    created = int(time.time())
    out = []
    for m in EXPOSED_MODELS:
        out.append({
            "id": m["id"], "object": "model", "created": created, "owned_by": "openai",
            "name": m["name"], "description": m["description"],
            "supports_multimodal": bool(m.get("supports_multimodal")),
            "supports_image": bool(m.get("supports_image")),
            "canonical_model": m.get("canonical_model", m["id"]),
            "default_reasoning_effort": m.get("default_reasoning_effort"),
            "supported_reasoning_efforts": sorted(SUPPORTED_REASONING_EFFORTS),
            "default_verbosity": m.get("default_verbosity"),
            "supported_verbosity": sorted(SUPPORTED_VERBOSITY_LEVELS),
            "provider": PROVIDER,
        })
    return out


# ---------------------------------------------------------------------------
# Token store (per-slot)
# ---------------------------------------------------------------------------


class CodexTokenStore:
    """OAuth state on disk + async refresh."""

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
            logger.warning("Bad codex oauth.json: %s", exc)
            return {}

    def save(self, data: Dict[str, Any]) -> None:
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def status(self) -> Dict[str, Any]:
        data = self.load()
        now = time.time()
        token = data.get("access_token") or ""
        refresh = data.get("refresh_token") or ""
        expires_at = float(data.get("expires_at") or 0)
        active = bool(token and expires_at > now + REFRESH_SAFETY_WINDOW)
        if active:
            state, notice = "active", "Session active."
        elif refresh:
            state, notice = "refreshable", "Will refresh on demand."
        elif token:
            state, notice = "expired", "Session expired, please sign in again."
        else:
            state, notice = "missing", "Not signed in."
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
            data = self.load()
            token = data.get("access_token")
            expires_at = float(data.get("expires_at") or 0)
            if token and expires_at > time.time() + REFRESH_SAFETY_WINDOW:
                return token
            refresh = data.get("refresh_token")
            if not refresh:
                raise AuthRevoked("No refresh token")
            new = await self._refresh(refresh)
            data.update(new)
            self.save(data)
            return data["access_token"]

    async def _refresh(self, refresh_token: str) -> Dict[str, Any]:
        client = await get_http_client()
        resp = await client.post(
            TOKEN_ENDPOINT,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CLIENT_ID,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30.0,
        )
        if resp.status_code == 400 or resp.status_code == 401:
            raise AuthRevoked(f"Refresh failed: {resp.text}", status_code=resp.status_code)
        if resp.status_code >= 400:
            raise UpstreamServerError(f"Refresh failed: {resp.text}", status_code=resp.status_code)
        payload = resp.json()
        return {
            "access_token": payload["access_token"],
            "refresh_token": payload.get("refresh_token") or refresh_token,
            "expires_at": time.time() + int(payload.get("expires_in") or 3600),
        }


# ---------------------------------------------------------------------------
# OAuth flow (authorize URL + token exchange)
# ---------------------------------------------------------------------------


def build_auth_url(state: str, verifier: str, port: int) -> str:
    import hashlib as _hashlib
    challenge = base64.urlsafe_b64encode(
        _hashlib.sha256(verifier.encode("utf-8")).digest()
    ).rstrip(b"=").decode("ascii")
    redirect_uri = f"http://localhost:{port}{CALLBACK_PATH}"
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid profile email offline_access",
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
    }
    return f"{AUTH_ENDPOINT}?{urlencode(params)}"


async def exchange_code(code: str, verifier: str, redirect_uri: str) -> Dict[str, Any]:
    client = await get_http_client()
    resp = await client.post(
        TOKEN_ENDPOINT,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": CLIENT_ID,
            "code_verifier": verifier,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


def _decode_account(id_token: str) -> tuple[str, str]:
    if not id_token:
        return "", ""
    data = decode_jwt_payload(id_token)
    email = data.get("email") or ""
    auth_obj = data.get("https://api.openai.com/auth") or {}
    account_id = (
        auth_obj.get("chatgpt_account_id")
        or auth_obj.get("chatgpt_user_id")
        or data.get("sub")
        or ""
    )
    return email, account_id


def build_saved_payload(token_payload: Dict[str, Any]) -> Dict[str, Any]:
    expires_in = int(token_payload.get("expires_in") or 3600)
    email, account_id = _decode_account(token_payload.get("id_token") or "")
    return {
        "access_token": token_payload.get("access_token") or "",
        "refresh_token": token_payload.get("refresh_token") or "",
        "id_token": token_payload.get("id_token") or "",
        "expires_at": time.time() + expires_in,
        "email": email,
        "account_id": account_id,
    }


async def start_login(slot_id: str) -> str:
    """Begin OAuth flow for a given slot. Returns the auth URL."""
    from accounts import POOL
    if POOL.get(slot_id) is None:
        raise ValueError(f"Unknown slot: {slot_id}")
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
        logger.warning("Codex callback for unknown slot %s", pa.slot_id)
        return False
    try:
        payload = await exchange_code(code, pa.verifier, pa.redirect_uri)
        saved = build_saved_payload(payload)
        acc.token_store.save(saved)
        await POOL.mark_valid(acc)
        logger.info("Codex slot %s signed in (%s)", pa.slot_id, saved.get("email"))
        return True
    except Exception:
        logger.exception("Codex token exchange failed for slot %s", pa.slot_id)
        return False


# ---------------------------------------------------------------------------
# Multimodal helpers (PDF/image/audio/video) — ported & cleaned
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
AUDIO_EXTS = {".mp3", ".wav"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
DOCUMENT_EXTS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
    ".txt", ".md", ".csv", ".tsv", ".rtf",
    ".html", ".htm", ".json", ".xml", ".yaml", ".yml", ".log",
}
DOCUMENT_MIME_HINTS = (
    "pdf", "wordprocessing", "spreadsheet", "presentation",
    "text/", "application/json", "application/xml", "application/yaml",
    "application/octet-stream",
)


def _is_local_path(value: str) -> bool:
    if not value:
        return False
    return not (value.startswith("http://") or value.startswith("https://") or value.startswith("data:"))


def _strip_data_url(value: Any) -> tuple[str, str]:
    if not isinstance(value, str) or not value.startswith("data:"):
        return "", ""
    try:
        header, _, payload = value.partition(",")
        if not payload:
            return "", ""
        return payload, header[5:].split(";", 1)[0].strip().lower()
    except Exception:
        return "", ""


def _path_to_data_url(p: str) -> Optional[str]:
    try:
        path = Path(p)
        if not path.exists():
            return None
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"
    except Exception:
        return None


def _path_to_input_file(p: str) -> Optional[Dict[str, Any]]:
    try:
        path = Path(p)
        if not path.exists():
            return None
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return {"type": "input_file", "file_data": data, "filename": path.name}
    except Exception:
        return None


def _path_to_input_audio(p: str) -> Optional[Dict[str, Any]]:
    try:
        path = Path(p)
        if not path.exists():
            return None
        fmt = path.suffix.lower().lstrip(".")
        if fmt not in {"mp3", "wav"}:
            return None
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return {"type": "input_audio", "input_audio": {"data": data, "format": fmt}}
    except Exception:
        return None


def _guess_audio_format(filename: str = "", media_type: str = "") -> str:
    mt = (media_type or "").lower()
    fn = (filename or "").lower()
    if mt.endswith("/mpeg") or mt.endswith("/mp3") or fn.endswith(".mp3"):
        return "mp3"
    if mt.endswith("/wav") or mt.endswith("/x-wav") or fn.endswith(".wav"):
        return "wav"
    return ""


def _b64_to_input_audio(data_b64: str, filename: str = "", media_type: str = "") -> Optional[Dict[str, Any]]:
    fmt = _guess_audio_format(filename, media_type)
    if not fmt or not data_b64:
        return None
    try:
        base64.b64decode(data_b64, validate=False)
    except Exception:
        return None
    return {"type": "input_audio", "input_audio": {"data": data_b64, "format": fmt}}


def _video_to_parts(p: str) -> List[Dict[str, Any]]:
    path = Path(p)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not path.exists():
        return []
    parts: List[Dict[str, Any]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="codex_video_") as td:
            tdp = Path(td)
            frame_pattern = str(tdp / "frame_%02d.jpg")
            subprocess.run(
                [ffmpeg, "-y", "-i", str(path),
                 "-vf", f"fps={SETTINGS.codex_video_fps},scale='min({SETTINGS.codex_video_max_width},iw)':-2",
                 "-frames:v", str(SETTINGS.codex_video_frames), frame_pattern],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )
            for f in sorted(tdp.glob("frame_*.jpg"))[: SETTINGS.codex_video_frames]:
                du = _path_to_data_url(str(f))
                if du:
                    parts.append({"type": "input_image", "image_url": du, "detail": "auto"})
            audio_path = tdp / "audio.wav"
            subprocess.run(
                [ffmpeg, "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
                 "-c:a", "pcm_s16le", str(audio_path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )
            if audio_path.exists() and audio_path.stat().st_size > 44:
                ap = _path_to_input_audio(str(audio_path))
                if ap:
                    parts.append(ap)
    except Exception:
        logger.exception("ffmpeg extract failed for %s", p)
    return parts


def _extract_media_url(item: Dict[str, Any]) -> Optional[str]:
    source = item.get("source") or {}
    if isinstance(source, dict):
        url = source.get("url") or source.get("path") or source.get("uri")
        if isinstance(url, str) and url:
            return url
        data_value = source.get("data")
        if isinstance(data_value, str) and _is_local_path(data_value):
            return data_value
    for key in ("image_url", "video_url"):
        v = item.get(key)
        if isinstance(v, dict) and v.get("url"):
            return v["url"]
        if isinstance(v, str) and v:
            return v
    if isinstance(item.get("url"), str) and item["url"]:
        return item["url"]
    return None


def _extract_image_detail(item: Dict[str, Any]) -> str:
    iu = item.get("image_url")
    if isinstance(iu, dict) and iu.get("detail"):
        return str(iu["detail"]).strip().lower()
    src = item.get("source")
    if isinstance(src, dict) and src.get("detail"):
        return str(src["detail"]).strip().lower()
    if item.get("detail"):
        return str(item["detail"]).strip().lower()
    return "auto"


def _normalize_image_detail(d: str) -> str:
    return d if d in {"low", "high", "auto"} else "auto"


def _normalize_file_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    item_type = item.get("type")
    if item_type not in ("file", "input_file"):
        return None
    if item_type == "file" and isinstance(item.get("file"), dict):
        fobj = item["file"]
    else:
        fobj = item
    file_id = fobj.get("file_id") or fobj.get("id")
    file_data = fobj.get("file_data")
    file_url = fobj.get("file_url") or fobj.get("url")
    filename = fobj.get("filename") or fobj.get("name") or ""
    if file_id:
        out: Dict[str, Any] = {"type": "input_file", "file_id": file_id}
        if filename:
            out["filename"] = filename
        return out
    if isinstance(file_data, str) and file_data:
        payload, mime = _strip_data_url(file_data)
        if payload:
            file_data = payload
            if not filename:
                guess = mimetypes.guess_extension(mime or "") or ".bin"
                filename = f"document{guess}"
        if not filename:
            filename = "document"
        return {"type": "input_file", "file_data": file_data, "filename": filename}
    if isinstance(file_url, str) and file_url:
        if file_url.startswith("data:"):
            payload, mime = _strip_data_url(file_url)
            if payload:
                if not filename:
                    guess = mimetypes.guess_extension(mime or "") or ".bin"
                    filename = f"document{guess}"
                return {"type": "input_file", "file_data": payload, "filename": filename}
            return None
        if _is_local_path(file_url):
            return _path_to_input_file(file_url)
        return {"type": "input_file", "file_url": file_url,
                "filename": filename or Path(file_url).name or "document"}
    return None


def _document_part(
    media_url: str, item: Dict[str, Any],
    data_payload: str = "", data_mime: str = "",
) -> Optional[Dict[str, Any]]:
    if not media_url:
        return None
    if data_payload:
        filename = item.get("filename") or item.get("name")
        if not filename:
            guess = mimetypes.guess_extension(data_mime or "") or ".bin"
            filename = f"document{guess}"
        return {"type": "input_file", "file_data": data_payload, "filename": filename}
    if media_url.startswith("data:"):
        return None
    if _is_local_path(media_url):
        return _path_to_input_file(media_url)
    return {"type": "input_file", "file_url": media_url,
            "filename": item.get("filename") or Path(media_url).name or "document"}


def _convert_content(role: str, content: Any) -> List[Dict[str, Any]]:
    text_type = "output_text" if role == "assistant" else "input_text"
    out: List[Dict[str, Any]] = []
    if isinstance(content, str):
        if content:
            out.append({"type": text_type, "text": content})
        return out
    if not isinstance(content, list):
        txt = str(content or "")
        if txt:
            out.append({"type": text_type, "text": txt})
        return out

    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in ("text", "input_text", "output_text"):
            txt = item.get("text", "")
            if txt:
                out.append({"type": text_type, "text": txt})
            continue
        if role == "assistant":
            if item_type == "refusal":
                out.append({"type": "refusal", "refusal": item.get("refusal", item.get("text", ""))})
            continue
        if item_type in ("file", "input_file"):
            fp = _normalize_file_item(item)
            if fp:
                out.append(fp)
                continue
        media_url = _extract_media_url(item)
        ext = Path(media_url or "").suffix.lower() if media_url else ""
        data_payload, data_mime = _strip_data_url(media_url or "")
        is_doc_mime = bool(data_mime) and any(h in data_mime for h in DOCUMENT_MIME_HINTS) and "image/" not in data_mime

        if item_type in ("image", "image_url") or ext in IMAGE_EXTS:
            if media_url:
                if ext in DOCUMENT_EXTS or is_doc_mime:
                    dp = _document_part(media_url, item, data_payload, data_mime)
                    if dp:
                        out.append(dp)
                    continue
                if _is_local_path(media_url):
                    media_url = _path_to_data_url(media_url) or media_url
                detail = _normalize_image_detail(_extract_image_detail(item))
                out.append({"type": "input_image", "image_url": media_url, "detail": detail})
            continue

        if item_type in ("document", "pdf") or ext in DOCUMENT_EXTS or is_doc_mime:
            if media_url:
                dp = _document_part(media_url, item, data_payload, data_mime)
                if dp:
                    out.append(dp)
            continue

        if item_type in ("audio", "audio_url", "input_audio") or ext in AUDIO_EXTS:
            if media_url:
                if _is_local_path(media_url):
                    ap = _path_to_input_audio(media_url)
                    if ap:
                        out.append(ap)
                    else:
                        fp = _path_to_input_file(media_url)
                        if fp:
                            out.append(fp)
                else:
                    out.append({"type": "input_file", "file_url": media_url,
                                "filename": Path(media_url).name or "audio"})
            else:
                source = item.get("source") or {}
                if isinstance(source, dict):
                    maybe = source.get("data")
                    media_type = source.get("media_type") or ""
                    if isinstance(maybe, str):
                        if _is_local_path(maybe):
                            ap = _path_to_input_audio(maybe)
                            if ap:
                                out.append(ap)
                        else:
                            ap = _b64_to_input_audio(maybe, media_type=media_type)
                            if ap:
                                out.append(ap)
            continue

        if item_type in ("video", "video_url") or ext in VIDEO_EXTS:
            if media_url:
                if _is_local_path(media_url):
                    parts = _video_to_parts(media_url)
                    if parts:
                        out.extend(parts)
                    else:
                        fp = _path_to_input_file(media_url)
                        if fp:
                            out.append(fp)
                else:
                    out.append({"type": "input_file", "file_url": media_url,
                                "filename": Path(media_url).name or "video"})
            continue

        if item_type == "file" and media_url:
            dp = _document_part(media_url, item, data_payload, data_mime)
            if dp:
                out.append(dp)

    return out


# ---------------------------------------------------------------------------
# Request body builder
# ---------------------------------------------------------------------------


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for it in content:
            if isinstance(it, dict) and it.get("type") in ("text", "input_text", "output_text"):
                parts.append(it.get("text", ""))
        return "\n".join(p for p in parts if p)
    return str(content or "")


def _messages_mention_json(messages: Any, instructions: Any) -> bool:
    hay = []
    if instructions:
        hay.append(str(instructions))
    for msg in messages or []:
        try:
            hay.append(_message_text(msg.get("content")))
        except Exception:
            pass
    return "json" in "\n".join(hay).lower()


def _response_format_to_text_format(rf: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(rf, dict):
        return None
    rt = (rf.get("type") or "").strip().lower()
    if rt == "json_object":
        return {"type": "json_object"}
    if rt == "text":
        return {"type": "text"}
    if rt == "json_schema":
        js = rf.get("json_schema") if isinstance(rf.get("json_schema"), dict) else rf
        out: Dict[str, Any] = {
            "type": "json_schema",
            "name": js.get("name") or rf.get("name") or "response",
            "schema": js.get("schema") or rf.get("schema") or {},
        }
        if "strict" in js:
            out["strict"] = bool(js["strict"])
        elif "strict" in rf:
            out["strict"] = bool(rf["strict"])
        if js.get("description"):
            out["description"] = js["description"]
        return out
    return None


def _request_reasoning_effort(req: Dict[str, Any]) -> Optional[str]:
    if not isinstance(req, dict):
        return None
    r = req.get("reasoning")
    if isinstance(r, dict) and r.get("effort"):
        return str(r["effort"])
    if req.get("reasoning_effort"):
        return str(req["reasoning_effort"])
    eb = req.get("extra_body")
    if isinstance(eb, dict):
        er = eb.get("reasoning")
        if isinstance(er, dict) and er.get("effort"):
            return str(er["effort"])
    return None


def _request_reasoning_summary(req: Dict[str, Any]) -> Optional[str]:
    if not isinstance(req, dict):
        return None
    for v in (
        nested_get(req, "reasoning", "summary"),
        req.get("reasoning_summary"),
        nested_get(req, "extra_body", "reasoning", "summary"),
    ):
        if v:
            return str(v)
    return None


def _request_verbosity(req: Dict[str, Any]) -> Optional[str]:
    if not isinstance(req, dict):
        return None
    for v in (
        nested_get(req, "text", "verbosity"),
        req.get("verbosity"),
        nested_get(req, "extra_body", "text", "verbosity"),
    ):
        if v:
            return str(v)
    return None


def _normalize_lower(value: Any, allowed: set, fallback: str = "") -> str:
    v = (str(value or "")).strip().lower()
    return v if v in allowed else (fallback if fallback in allowed else "")


def _reasoning_block(model: str, req: Dict[str, Any], raw_model: str) -> Optional[Dict[str, Any]]:
    n = normalize_model(model)
    if n not in CODEX_MODEL_IDS:
        return None
    model_default = _model_default_effort(n)
    env_default = (SETTINGS.codex_reasoning_effort or "").lower()
    fallback = env_default if env_default in SUPPORTED_REASONING_EFFORTS else model_default
    requested = _request_reasoning_effort(req) or _model_variant_effort(raw_model)
    block = {"effort": _normalize_lower(requested, SUPPORTED_REASONING_EFFORTS, fallback)}
    summary = _normalize_lower(
        _request_reasoning_summary(req) or SETTINGS.codex_reasoning_summary,
        SUPPORTED_REASONING_SUMMARIES,
    )
    if summary:
        block["summary"] = summary
    return block


def _text_block(model: str, req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    n = normalize_model(model)
    if n not in CODEX_MODEL_IDS:
        return None
    env_default = (SETTINGS.codex_verbosity or "").lower()
    fallback = env_default if env_default in SUPPORTED_VERBOSITY_LEVELS else _model_default_verbosity(n)
    verbosity = _normalize_lower(_request_verbosity(req), SUPPORTED_VERBOSITY_LEVELS, fallback)
    return {"verbosity": verbosity} if verbosity else None


SAMPLING_KEYS = ("temperature", "top_p", "top_k",
                 "frequency_penalty", "presence_penalty", "seed", "stop")


def _responses_tool_choice(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return None
    if value.get("type") == "function" and isinstance(value.get("function"), dict):
        name = value["function"].get("name")
        return {"type": "function", "name": name} if name else None
    return value


def build_responses_body(req: Dict[str, Any], stream: bool) -> Dict[str, Any]:
    raw_model = req.get("model") or SETTINGS.codex_default_model
    model = normalize_model(raw_model)
    messages = req.get("messages") or []
    tools = req.get("tools") or []
    instructions = ""
    input_items: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        if role in ("system", "developer"):
            text = _message_text(msg.get("content"))
            if text:
                instructions = (instructions + "\n\n" + text) if instructions else text
            continue
        if role in ("user", "assistant"):
            conv = _convert_content(role, msg.get("content"))
            if conv:
                input_items.append({"role": role, "content": conv})
        elif role == "tool":
            input_items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id") or msg.get("tool_use_id") or "",
                "output": _message_text(msg.get("content")),
            })
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            input_items.append({
                "type": "function_call",
                "call_id": tc.get("id") or str(uuid.uuid4()),
                "name": fn.get("name") or "",
                "arguments": fn.get("arguments") or "{}",
            })

    store_value = req.get("store")
    if store_value is None:
        store_value = nested_get(req, "extra_body", "store")
    store_bool = bool(store_value) if store_value is not None else SETTINGS.codex_default_store

    prev_id = req.get("previous_response_id") or nested_get(req, "extra_body", "previous_response_id")
    if prev_id:
        store_bool = True

    body: Dict[str, Any] = {
        "model": model, "input": input_items, "stream": stream, "store": store_bool,
    }
    if prev_id:
        body["previous_response_id"] = prev_id

    reasoning = _reasoning_block(model, req, raw_model)
    if reasoning:
        body["reasoning"] = reasoning

    text_block = _text_block(model, req) or {}
    fmt = _response_format_to_text_format(
        req.get("response_format") or nested_get(req, "extra_body", "response_format")
    )
    if fmt:
        text_block["format"] = fmt
        if fmt.get("type") == "json_object" and not _messages_mention_json(messages, instructions):
            injected = False
            for item in input_items:
                if isinstance(item, dict) and item.get("role") == "user":
                    content = item.get("content")
                    if isinstance(content, list):
                        content.insert(0, {"type": "input_text", "text": "Output must be valid json."})
                        injected = True
                        break
            if not injected:
                instructions = (instructions + "\n\nOutput must be valid json.").strip()
    if text_block:
        body["text"] = text_block

    if instructions:
        body["instructions"] = instructions
    elif SETTINGS.codex_inject_default_instructions and model in CODEX_MODEL_IDS:
        body["instructions"] = SETTINGS.codex_default_instructions

    include = req.get("include")
    if include is None:
        include = nested_get(req, "extra_body", "include")
    if isinstance(include, list) and include:
        body["include"] = list(include)
    elif SETTINGS.codex_default_include:
        body["include"] = list(SETTINGS.codex_default_include)

    if tools:
        out_tools: List[Dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            t_type = (tool.get("type") or "").strip().lower()
            if t_type == "function":
                fn = tool.get("function") or {}
                params = fn.get("parameters")
                if not isinstance(params, dict):
                    params = {"type": "object", "properties": {},
                              "additionalProperties": False, "required": []}
                tool_def: Dict[str, Any] = {
                    "type": "function",
                    "name": fn.get("name") or "",
                    "description": fn.get("description") or "",
                    "parameters": params,
                }
                strict = fn.get("strict") if fn.get("strict") is not None else tool.get("strict")
                if strict is not None:
                    tool_def["strict"] = bool(strict)
                out_tools.append(tool_def)
            elif t_type:
                out_tools.append(tool)
        if out_tools:
            body["tools"] = out_tools

    tool_choice = _responses_tool_choice(
        req.get("tool_choice") or nested_get(req, "extra_body", "tool_choice")
    )
    if tool_choice is not None:
        body["tool_choice"] = tool_choice

    parallel = req.get("parallel_tool_calls") or nested_get(req, "extra_body", "parallel_tool_calls")
    if parallel is not None:
        body["parallel_tool_calls"] = bool(parallel)

    metadata = req.get("metadata") or nested_get(req, "extra_body", "metadata")
    if isinstance(metadata, dict):
        body["metadata"] = metadata

    if SETTINGS.codex_passthrough_sampling:
        for k in SAMPLING_KEYS:
            v = req.get(k)
            if v is None:
                v = nested_get(req, "extra_body", k)
            if v is not None:
                body[k] = v

    return body


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _convert_usage(usage: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(usage, dict) or not usage:
        return {}
    prompt = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    completion = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    out: Dict[str, Any] = {
        "prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total,
    }
    od = usage.get("output_tokens_details")
    if isinstance(od, dict):
        rt = int(od.get("reasoning_tokens") or 0)
        if rt:
            out["completion_tokens_details"] = {"reasoning_tokens": rt}
    id_ = usage.get("input_tokens_details")
    if isinstance(id_, dict):
        ct = int(id_.get("cached_tokens") or 0)
        if ct:
            out["prompt_tokens_details"] = {"cached_tokens": ct}
    return out


def _map_finish_reason(response_obj: Dict[str, Any], emitted_tool: bool) -> str:
    if not isinstance(response_obj, dict):
        return "tool_calls" if emitted_tool else "stop"
    inc = response_obj.get("incomplete_details") or {}
    if isinstance(inc, dict):
        reason = (inc.get("reason") or "").lower()
        if reason == "max_output_tokens":
            return "length"
        if reason in ("content_filter", "moderation", "safety"):
            return "content_filter"
    status = (response_obj.get("status") or "").lower()
    if status == "incomplete":
        return "length"
    return "tool_calls" if emitted_tool else "stop"


def _rate_limit_dimension(headers: Any, dim: str) -> Optional[Dict[str, Any]]:
    def gv(*names):
        for n in names:
            try:
                v = headers.get(n)
            except Exception:
                v = None
            if v not in (None, ""):
                return str(v)
        return None

    if dim == "overall":
        limit = safe_int(gv("x-ratelimit-limit", "x-rate-limit-limit", "ratelimit-limit"))
        remaining = safe_int(gv("x-ratelimit-remaining", "x-rate-limit-remaining", "ratelimit-remaining"))
        reset = gv("x-ratelimit-reset", "x-rate-limit-reset", "ratelimit-reset")
    else:
        limit = safe_int(gv(f"x-ratelimit-limit-{dim}", f"x-rate-limit-limit-{dim}", f"ratelimit-limit-{dim}"))
        remaining = safe_int(gv(f"x-ratelimit-remaining-{dim}", f"x-rate-limit-remaining-{dim}", f"ratelimit-remaining-{dim}"))
        reset = gv(f"x-ratelimit-reset-{dim}", f"x-rate-limit-reset-{dim}", f"ratelimit-reset-{dim}")
    if limit is None and remaining is None and not reset:
        return None
    data: Dict[str, Any] = {}
    if limit is not None:
        data["limit"] = limit
    if remaining is not None:
        data["remaining"] = remaining
    if reset:
        data["reset"] = reset
    if limit and remaining is not None:
        data["percent_remaining"] = round((remaining / limit) * 100, 1)
    return data


def _extract_rate_limit(headers: Any, model: str) -> Dict[str, Any]:
    limits: Dict[str, Dict[str, Any]] = {}
    for dim in ("requests", "tokens", "input-tokens", "output-tokens", "overall"):
        d = _rate_limit_dimension(headers, dim)
        if d:
            limits[dim] = d
    retry_after = None
    try:
        retry_after = safe_int(headers.get("retry-after"))
    except Exception:
        retry_after = None
    request_id = ""
    try:
        request_id = str(headers.get("x-request-id") or headers.get("request-id") or "")
    except Exception:
        request_id = ""
    if not limits and not retry_after and not request_id:
        return {}
    out: Dict[str, Any] = {"provider": PROVIDER, "model": model, "updated_at": int(time.time())}
    if limits:
        out["limits"] = limits
    if retry_after:
        out["retry_after_seconds"] = retry_after
    if request_id:
        out["request_id"] = request_id
    return out


def _request_include_usage(req: Dict[str, Any]) -> bool:
    so = req.get("stream_options")
    if not isinstance(so, dict):
        so = nested_get(req, "extra_body", "stream_options")
    return bool(isinstance(so, dict) and so.get("include_usage"))


def _classify_upstream_error(status: int, raw_body: str, model: str) -> Exception:
    """Map upstream HTTP error to one of our error classes."""
    try:
        payload = json.loads(raw_body) if raw_body else {}
    except Exception:
        payload = {}
    inner = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(inner, dict):
        inner = payload if isinstance(payload, dict) else {}
    err_type = (inner.get("type") or "").lower()
    err_msg = inner.get("message") or raw_body or ""

    if status == 429 and err_type == "usage_limit_reached":
        resets_at = float(inner.get("resets_at") or 0)
        return QuotaExhausted(
            err_msg, status_code=status, resets_at=resets_at,
            reason=inner.get("plan_type") or "usage_limit_reached",
            raw_body=raw_body,
        )
    if status == 429:
        resets_at = float(inner.get("resets_at") or (time.time() + 60))
        return QuotaExhausted(err_msg, status_code=status, resets_at=resets_at,
                              reason="rate_limited", raw_body=raw_body)
    if status == 401:
        return AuthRevoked(err_msg, status_code=status, raw_body=raw_body)
    if 500 <= status < 600:
        return UpstreamServerError(err_msg, status_code=status, raw_body=raw_body)
    return UpstreamClientError(err_msg, status_code=status, raw_body=raw_body)


def _codex_headers(token: str, account_id: str) -> Dict[str, str]:
    h = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if SETTINGS.codex_originator:
        h["originator"] = SETTINGS.codex_originator
    if SETTINGS.codex_beta:
        h["OpenAI-Beta"] = SETTINGS.codex_beta
    if SETTINGS.codex_user_agent:
        h["User-Agent"] = SETTINGS.codex_user_agent
    if account_id:
        h["chatgpt-account-id"] = account_id
    return h


# ---------------------------------------------------------------------------
# Executor — stream + non-stream
# ---------------------------------------------------------------------------


async def stream_completion(
    chat_request: Dict[str, Any],
    account,
) -> AsyncGenerator[bytes, None]:
    from accounts import POOL
    token = await account.token_store.get_access_token()
    account_id = account.token_store.status().get("account_id") or ""
    body = build_responses_body(chat_request, stream=True)
    model = body["model"]
    created = int(time.time())
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    tool_index_map: Dict[str, int] = {}
    emitted_tool = False
    reasoning_emitted = False
    include_usage = _request_include_usage(chat_request)

    client = await get_http_client()
    async with client.stream(
        "POST", RESPONSES_ENDPOINT,
        headers=_codex_headers(token, account_id),
        json=body,
    ) as resp:
        rl = _extract_rate_limit(resp.headers, model)
        if rl:
            await POOL.record_rate_limit(account, rl)

        if resp.status_code >= 400:
            raw = await resp.aread()
            raise _classify_upstream_error(resp.status_code, raw.decode("utf-8", "ignore"), model)

        async for line in resp.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            raw = line[6:]
            if raw == "[DONE]":
                break
            try:
                payload = json.loads(raw)
            except Exception:
                continue

            evt = payload.get("type") or payload.get("event")
            if evt == "response.output_text.delta":
                delta = payload.get("delta") or ""
                yield sse_data({
                    "id": response_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": 0, "delta": {"content": delta} if delta else {}, "finish_reason": None}],
                })

            elif evt in (
                "response.reasoning_summary_text.delta",
                "response.reasoning_summary.delta",
                "response.reasoning.delta",
                "response.reasoning_text.delta",
            ):
                delta = payload.get("delta") or ""
                if delta:
                    reasoning_emitted = True
                    yield sse_data({
                        "id": response_id, "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {"reasoning_content": delta}, "finish_reason": None}],
                    })

            elif evt == "response.reasoning_summary_part.added" and reasoning_emitted:
                yield sse_data({
                    "id": response_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": 0, "delta": {"reasoning_content": "\n\n"}, "finish_reason": None}],
                })

            elif evt == "response.output_item.added":
                item = payload.get("item") or {}
                if item.get("type") == "function_call":
                    item_id = item.get("id") or str(uuid.uuid4())
                    call_id = item.get("call_id") or item_id
                    name = item.get("name") or ""
                    idx = tool_index_map.setdefault(item_id, len(tool_index_map))
                    emitted_tool = True
                    yield sse_data({
                        "id": response_id, "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"tool_calls": [{
                                "index": idx, "id": call_id, "type": "function",
                                "function": {"name": name, "arguments": ""},
                            }]},
                            "finish_reason": None,
                        }],
                    })

            elif evt == "response.function_call_arguments.delta":
                item_id = payload.get("item_id") or ""
                delta = payload.get("delta") or ""
                idx = tool_index_map.setdefault(item_id, len(tool_index_map))
                emitted_tool = True
                yield sse_data({
                    "id": response_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"tool_calls": [{
                            "index": idx, "function": {"arguments": delta},
                        }]},
                        "finish_reason": None,
                    }],
                })

            elif evt == "response.completed":
                completed = payload.get("response") or {}
                finish_reason = _map_finish_reason(completed, emitted_tool)
                usage_payload = _convert_usage(completed.get("usage") or {})
                yield sse_data({
                    "id": response_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                })
                if include_usage and usage_payload:
                    usage_chunk = {
                        "id": response_id, "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [], "usage": usage_payload,
                    }
                    if account.health.rate_limit:
                        usage_chunk["_bridge_rate_limit"] = account.health.rate_limit
                    yield sse_data(usage_chunk)
                yield sse_done()
                return

        # Stream ended without explicit completed event
        yield sse_data({
            "id": response_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0, "delta": {},
                         "finish_reason": "tool_calls" if emitted_tool else "stop"}],
        })
        yield sse_done()


async def complete_completion(chat_request: Dict[str, Any], account) -> Dict[str, Any]:
    from accounts import POOL
    token = await account.token_store.get_access_token()
    account_id = account.token_store.status().get("account_id") or ""
    body = build_responses_body(chat_request, stream=True)
    model = body["model"]
    created = int(time.time())
    response_id = f"chatcmpl-{uuid.uuid4().hex}"

    text_parts: List[str] = []
    reasoning_parts: List[str] = []
    tool_calls: Dict[str, Dict[str, Any]] = {}
    completed_response: Dict[str, Any] = {}

    client = await get_http_client()
    async with client.stream(
        "POST", RESPONSES_ENDPOINT,
        headers=_codex_headers(token, account_id),
        json=body,
    ) as resp:
        rl = _extract_rate_limit(resp.headers, model)
        if rl:
            await POOL.record_rate_limit(account, rl)
        if resp.status_code >= 400:
            raw = await resp.aread()
            raise _classify_upstream_error(resp.status_code, raw.decode("utf-8", "ignore"), model)
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            raw = line[6:]
            if raw == "[DONE]":
                break
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            evt = payload.get("type") or payload.get("event")
            if evt == "response.output_text.delta":
                d = payload.get("delta") or ""
                if d:
                    text_parts.append(d)
            elif evt in (
                "response.reasoning_summary_text.delta",
                "response.reasoning_summary.delta",
                "response.reasoning.delta",
                "response.reasoning_text.delta",
            ):
                d = payload.get("delta") or ""
                if d:
                    reasoning_parts.append(d)
            elif evt == "response.reasoning_summary_part.added" and reasoning_parts:
                reasoning_parts.append("\n\n")
            elif evt == "response.output_item.added":
                item = payload.get("item") or {}
                if item.get("type") == "function_call":
                    item_id = item.get("id") or str(uuid.uuid4())
                    tool_calls[item_id] = {
                        "id": item.get("call_id") or item_id,
                        "type": "function",
                        "function": {"name": item.get("name") or "", "arguments": ""},
                    }
            elif evt == "response.function_call_arguments.delta":
                item_id = payload.get("item_id") or ""
                d = payload.get("delta") or ""
                if item_id not in tool_calls:
                    tool_calls[item_id] = {"id": item_id, "type": "function",
                                           "function": {"name": "", "arguments": ""}}
                tool_calls[item_id]["function"]["arguments"] += d
            elif evt == "response.completed":
                completed_response = payload.get("response") or {}

    message: Dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts).strip()
    if tool_calls:
        message["tool_calls"] = list(tool_calls.values())
    finish_reason = _map_finish_reason(completed_response, bool(tool_calls))
    usage = _convert_usage(completed_response.get("usage") or {}) or {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
    }
    result = {
        "id": response_id, "object": "chat.completion",
        "created": created, "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }
    if account.health.rate_limit:
        result["_bridge_rate_limit"] = account.health.rate_limit
    return result


# ---------------------------------------------------------------------------
# Audio transcription via Codex
# ---------------------------------------------------------------------------


async def transcribe_audio(
    audio_bytes: bytes,
    filename: str,
    account,
    *,
    hint_prompt: str = "",
    language: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    from accounts import POOL
    token = await account.token_store.get_access_token()
    account_id = account.token_store.status().get("account_id") or ""
    ext = Path(filename or "").suffix.lower().lstrip(".")
    fmt = ext if ext in {"mp3", "wav"} else _guess_audio_format(filename, "") or "mp3"
    data_b64 = base64.b64encode(audio_bytes).decode("ascii")

    requested = (model or "").strip()
    if requested.lower() in {"whisper-1", "whisper", "auto", ""}:
        audio_model = SETTINGS.codex_audio_model
    else:
        audio_model = normalize_model(requested, SETTINGS.codex_audio_model)

    prompt_text = ("Transcribe the supplied audio faithfully. "
                   "Return only the transcript text, with no commentary or labels.")
    if language:
        prompt_text += f" The audio language is likely {language}."
    if hint_prompt:
        prompt_text += f" Additional hint: {hint_prompt}"

    body = {
        "model": audio_model,
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt_text},
                {"type": "input_audio", "input_audio": {"data": data_b64, "format": fmt}},
            ],
        }],
        "stream": True,
        "store": False,
        "reasoning": {"effort": SETTINGS.codex_reasoning_effort or "medium"},
    }

    text_parts: List[str] = []
    client = await get_http_client()
    async with client.stream(
        "POST", RESPONSES_ENDPOINT,
        headers=_codex_headers(token, account_id),
        json=body,
    ) as resp:
        rl = _extract_rate_limit(resp.headers, audio_model)
        if rl:
            await POOL.record_rate_limit(account, rl)
        if resp.status_code >= 400:
            raw = await resp.aread()
            raise _classify_upstream_error(resp.status_code, raw.decode("utf-8", "ignore"), audio_model)
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            raw = line[6:]
            if raw == "[DONE]":
                break
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            if (payload.get("type") or payload.get("event")) == "response.output_text.delta":
                d = payload.get("delta") or ""
                if d:
                    text_parts.append(d)
    return "".join(text_parts).strip()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(pool) -> None:
    pool.register_provider(PROVIDER, CodexTokenStore)
    from core import CALLBACK_BROKER
    CALLBACK_BROKER.register_handler(CALLBACK_PATH, handle_callback)
