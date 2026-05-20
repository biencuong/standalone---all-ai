# -*- coding: utf-8 -*-
"""Managed client API keys for the local bridge.

These keys protect the OpenAI/Anthropic-compatible /v1 surface. The
environment BRIDGE_API_KEY remains the master/admin key for /api routes.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core import DATA_DIR, logger


API_KEYS_FILE = DATA_DIR / "api_keys.json"
KEY_PREFIX = "ak_"
VALID_SCOPES = {"local", "lan", "internet"}
VALID_STATUSES = {"active", "disabled", "expired", "exhausted"}


@dataclass
class ManagedKeyAuth:
    key_id: str
    name: str
    prefix: str
    allowed_models: List[str]
    token_limit: int
    token_used: int
    network_scope: str
    client_ip: str


def _now() -> float:
    return time.time()


def _hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _new_secret() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32).replace("-", "").replace("_", "")


def _new_id() -> str:
    return "key_" + secrets.token_hex(8)


def _clean_models(value: Any) -> List[str]:
    if isinstance(value, str):
        raw = [x.strip() for x in value.split(",")]
    elif isinstance(value, list):
        raw = [str(x).strip() for x in value]
    else:
        raw = []
    out: List[str] = []
    seen = set()
    for item in raw:
        if not item:
            continue
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(item)
    return out


def _clean_scope(value: Any) -> str:
    scope = str(value or "local").strip().lower()
    return scope if scope in VALID_SCOPES else "local"


def _clean_status(value: Any) -> str:
    status = str(value or "active").strip().lower()
    return status if status in VALID_STATUSES else "active"


def _clean_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return default


def _expiry_from_payload(payload: Dict[str, Any], current: float = 0) -> float:
    if "expires_at" in payload:
        try:
            return max(0.0, float(payload.get("expires_at") or 0))
        except (TypeError, ValueError):
            return current
    if "expires_in_seconds" in payload:
        seconds = _clean_int(payload.get("expires_in_seconds"))
        return _now() + seconds if seconds > 0 else 0.0
    if "duration_hours" in payload:
        hours = _clean_int(payload.get("duration_hours"))
        return _now() + hours * 3600 if hours > 0 else 0.0
    return current


def effective_status(record: Dict[str, Any], now: Optional[float] = None) -> str:
    status = _clean_status(record.get("status"))
    if status != "active":
        return status
    current = now or _now()
    expires_at = float(record.get("expires_at") or 0)
    token_limit = _clean_int(record.get("token_limit"))
    token_used = _clean_int(record.get("token_used"))
    if expires_at and expires_at <= current:
        return "expired"
    if token_limit and token_used >= token_limit:
        return "exhausted"
    return "active"


def _public_record(record: Dict[str, Any], now: Optional[float] = None) -> Dict[str, Any]:
    current = now or _now()
    expires_at = float(record.get("expires_at") or 0)
    token_limit = _clean_int(record.get("token_limit"))
    token_used = _clean_int(record.get("token_used"))
    return {
        "id": record.get("id") or "",
        "name": record.get("name") or "",
        "prefix": record.get("prefix") or "",
        "status": effective_status(record, current),
        "stored_status": _clean_status(record.get("status")),
        "created_at": float(record.get("created_at") or 0),
        "expires_at": expires_at,
        "expires_in": max(0, int(expires_at - current)) if expires_at else 0,
        "token_limit": token_limit,
        "token_used": token_used,
        "token_remaining": max(0, token_limit - token_used) if token_limit else 0,
        "allowed_models": list(record.get("allowed_models") or []),
        "network_scope": _clean_scope(record.get("network_scope")),
        "last_used_at": float(record.get("last_used_at") or 0),
        "last_used_ip": record.get("last_used_ip") or "",
        "last_error": record.get("last_error") or "",
    }


def ip_matches_scope(client_ip: str, scope: str) -> bool:
    scope = _clean_scope(scope)
    if scope == "internet":
        return True
    try:
        ip = ipaddress.ip_address((client_ip or "").split("%", 1)[0])
    except ValueError:
        return False
    if scope == "local":
        return ip.is_loopback
    return ip.is_loopback or ip.is_private or ip.is_link_local


class ManagedApiKeyStore:
    def __init__(self, path: Path = API_KEYS_FILE):
        self.path = path
        self._lock = threading.RLock()

    def _load_unlocked(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "keys": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Cannot parse managed API keys at %s", self.path)
            return {"version": 1, "keys": []}
        if not isinstance(data, dict):
            return {"version": 1, "keys": []}
        keys = data.get("keys")
        if not isinstance(keys, list):
            data["keys"] = []
        return data

    def _save_unlocked(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_public(self) -> List[Dict[str, Any]]:
        with self._lock:
            data = self._load_unlocked()
            now = _now()
            return [_public_record(k, now) for k in data.get("keys", []) if isinstance(k, dict)]

    def create(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        secret = _new_secret()
        now = _now()
        record = {
            "id": _new_id(),
            "name": str(payload.get("name") or "Client key").strip()[:120],
            "prefix": secret[:12],
            "secret_hash": _hash_secret(secret),
            "status": "active",
            "created_at": now,
            "expires_at": _expiry_from_payload(payload),
            "token_limit": _clean_int(payload.get("token_limit")),
            "token_used": 0,
            "allowed_models": _clean_models(payload.get("allowed_models")) or ["*"],
            "network_scope": _clean_scope(payload.get("network_scope")),
            "last_used_at": 0,
            "last_used_ip": "",
            "last_error": "",
        }
        with self._lock:
            data = self._load_unlocked()
            data.setdefault("keys", []).append(record)
            self._save_unlocked(data)
        return _public_record(record), secret

    def get_public(self, key_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            for record in self._load_unlocked().get("keys", []):
                if isinstance(record, dict) and record.get("id") == key_id:
                    return _public_record(record)
        return None

    def authenticate(self, secret: str) -> Optional[Dict[str, Any]]:
        secret_hash = _hash_secret((secret or "").strip())
        with self._lock:
            for record in self._load_unlocked().get("keys", []):
                if not isinstance(record, dict):
                    continue
                if hmac.compare_digest(str(record.get("secret_hash") or ""), secret_hash):
                    return dict(record)
        return None

    def check_usable(self, record: Dict[str, Any], client_ip: str) -> Tuple[bool, str]:
        status = effective_status(record)
        if status != "active":
            return False, status
        if not ip_matches_scope(client_ip, str(record.get("network_scope") or "local")):
            return False, "network_denied"
        return True, ""

    def patch(self, key_id: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self._lock:
            data = self._load_unlocked()
            for record in data.get("keys", []):
                if not isinstance(record, dict) or record.get("id") != key_id:
                    continue
                if "name" in payload:
                    record["name"] = str(payload.get("name") or "").strip()[:120]
                if "status" in payload:
                    record["status"] = _clean_status(payload.get("status"))
                if "expires_at" in payload or "expires_in_seconds" in payload or "duration_hours" in payload:
                    record["expires_at"] = _expiry_from_payload(payload, float(record.get("expires_at") or 0))
                if "token_limit" in payload:
                    record["token_limit"] = _clean_int(payload.get("token_limit"))
                if "token_used" in payload:
                    record["token_used"] = _clean_int(payload.get("token_used"))
                if "allowed_models" in payload:
                    record["allowed_models"] = _clean_models(payload.get("allowed_models")) or ["*"]
                if "network_scope" in payload:
                    record["network_scope"] = _clean_scope(payload.get("network_scope"))
                record["last_error"] = ""
                self._save_unlocked(data)
                return _public_record(record)
        return None

    def rotate(self, key_id: str) -> Optional[Tuple[Dict[str, Any], str]]:
        secret = _new_secret()
        with self._lock:
            data = self._load_unlocked()
            for record in data.get("keys", []):
                if not isinstance(record, dict) or record.get("id") != key_id:
                    continue
                record["prefix"] = secret[:12]
                record["secret_hash"] = _hash_secret(secret)
                record["status"] = "active"
                record["last_error"] = ""
                self._save_unlocked(data)
                return _public_record(record), secret
        return None

    def delete(self, key_id: str) -> bool:
        with self._lock:
            data = self._load_unlocked()
            keys = [k for k in data.get("keys", []) if not isinstance(k, dict) or k.get("id") != key_id]
            if len(keys) == len(data.get("keys", [])):
                return False
            data["keys"] = keys
            self._save_unlocked(data)
            return True

    def record_usage(self, key_id: str, tokens: int = 0, client_ip: str = "", error: str = "") -> Optional[Dict[str, Any]]:
        with self._lock:
            data = self._load_unlocked()
            for record in data.get("keys", []):
                if not isinstance(record, dict) or record.get("id") != key_id:
                    continue
                used = _clean_int(record.get("token_used")) + max(0, int(tokens or 0))
                record["token_used"] = used
                record["last_used_at"] = _now()
                record["last_used_ip"] = client_ip or record.get("last_used_ip") or ""
                record["last_error"] = error or ""
                limit = _clean_int(record.get("token_limit"))
                if limit and used >= limit and record.get("status") == "active":
                    record["status"] = "exhausted"
                self._save_unlocked(data)
                return _public_record(record)
        return None


API_KEYS = ManagedApiKeyStore()
