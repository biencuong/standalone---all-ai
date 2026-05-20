# -*- coding: utf-8 -*-
"""Multi-account pool: storage, health tracking, selection, failover.

An "account" is a folder under `data/accounts/<slot_id>/` containing:
- `oauth.json` — provider-specific token state (managed by the provider's TokenStore)
- `meta.json`  — {alias, provider, enabled, created_at, tier?}

The pool keeps `AccountHealth` (in-memory + persisted to `pool_state.json`):
- `exhausted_until` — unix ts when account regains quota
- `invalid` — refresh token revoked, needs re-login
- `in_flight`, `consecutive_failures`, `last_used_at`, etc.

Selection strategies: least_load (default), round_robin, random.
Routing: model id prefix → provider name.
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core import (
    ACCOUNTS_DIR,
    POOL_STATE_FILE,
    SETTINGS,
    logger,
    NoAccountAvailable,
)

# ---------------------------------------------------------------------------
# Model -> provider routing
# ---------------------------------------------------------------------------

PROVIDER_PREFIXES: Dict[str, tuple[str, ...]] = {
    "chatgpt": (
        "gpt-", "gpt5", "gpt5.", "o1", "o1-", "o3", "o3-", "o4", "o4-",
        "codex", "chatgpt", "5.",
    ),
    "google": (
        "gemini", "gemini-", "models/gemini", "auto-gemini",
    ),
    "anthropic": (
        "claude", "claude-",
    ),
    "deepseek": (
        "deepseek", "deepseek-",
    ),
}


def resolve_provider(model: str, default: str = "chatgpt") -> str:
    m = (model or "").lower().lstrip("/").strip()
    if not m:
        return default
    if m in PROVIDER_PREFIXES:
        return m
    for provider, prefixes in PROVIDER_PREFIXES.items():
        if any(m.startswith(p) for p in prefixes):
            return provider
    return default


# ---------------------------------------------------------------------------
# Account model
# ---------------------------------------------------------------------------


@dataclass
class AccountHealth:
    exhausted_until: float = 0.0
    last_429_reason: str = ""
    last_429_at: float = 0.0
    consecutive_failures: int = 0
    in_flight: int = 0
    last_used_at: float = 0.0
    last_success_at: float = 0.0
    invalid: bool = False
    rate_limit: Dict[str, Any] = field(default_factory=dict)

    def is_healthy(self, now: Optional[float] = None) -> bool:
        if self.invalid:
            return False
        if self.exhausted_until > (now or time.time()):
            return False
        return True

    def active_rate_limit(self, now: Optional[float] = None) -> Dict[str, Any]:
        if not self.rate_limit:
            return {}
        current = now or time.time()
        try:
            updated_at = float(self.rate_limit.get("updated_at") or 0)
            retry_after = float(self.rate_limit.get("retry_after_seconds") or 0)
        except (TypeError, ValueError):
            return {}
        if updated_at and retry_after and updated_at + retry_after > current:
            return self.rate_limit
        return {}


@dataclass
class AccountMeta:
    alias: str
    provider: str
    enabled: bool = True
    rotation_enabled: bool = True
    created_at: float = field(default_factory=time.time)
    tier: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AccountMeta":
        return cls(
            alias=str(d.get("alias") or ""),
            provider=str(d.get("provider") or ""),
            enabled=bool(d.get("enabled", True)),
            rotation_enabled=bool(d.get("rotation_enabled", True)),
            created_at=float(d.get("created_at") or time.time()),
            tier=str(d.get("tier") or ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Account:
    """Represents one credential slot. Owns its folder and (lazy) TokenStore."""

    def __init__(
        self,
        slot_id: str,
        folder: Path,
        meta: AccountMeta,
        token_store: Any,
    ):
        self.slot_id = slot_id
        self.folder = folder
        self.meta = meta
        self.token_store = token_store
        self.health = AccountHealth()

    @property
    def provider(self) -> str:
        return self.meta.provider

    @property
    def enabled(self) -> bool:
        return self.meta.enabled

    @property
    def alias(self) -> str:
        return self.meta.alias or self.slot_id

    def save_meta(self) -> None:
        (self.folder / "meta.json").write_text(
            json.dumps(self.meta.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    async def status(self) -> Dict[str, Any]:
        try:
            ts_status = self.token_store.status()
        except Exception as exc:
            logger.exception("status() failed for %s: %s", self.slot_id, exc)
            ts_status = {}
        now = time.time()
        return {
            "slot_id": self.slot_id,
            "provider": self.provider,
            "alias": self.alias,
            "enabled": self.enabled,
            "rotation_enabled": self.meta.rotation_enabled,
            "tier": self.meta.tier,
            "logged_in": bool(ts_status.get("logged_in")),
            "email": ts_status.get("email") or "",
            "account_id": ts_status.get("account_id") or "",
            "session_state": ts_status.get("session_state") or "missing",
            "session_notice": ts_status.get("session_notice") or "",
            "auth_type": ts_status.get("auth_type") or "",
            "expires_in": int(ts_status.get("expires_in") or 0),
            "has_saved_session": bool(ts_status.get("has_saved_session")),
            "health": {
                "in_flight": self.health.in_flight,
                "consecutive_failures": self.health.consecutive_failures,
                "exhausted_until": self.health.exhausted_until,
                "exhausted_in": max(0, int(self.health.exhausted_until - now))
                if self.health.exhausted_until > now else 0,
                "invalid": self.health.invalid,
                "last_used_at": self.health.last_used_at,
                "last_success_at": self.health.last_success_at,
                "last_429_reason": self.health.last_429_reason,
                "rate_limit": self.health.active_rate_limit(now),
            },
        }


# ---------------------------------------------------------------------------
# Account pool
# ---------------------------------------------------------------------------


# Provider -> factory(folder: Path) -> token_store
TokenStoreFactory = Callable[[Path], Any]


class AccountPool:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._accounts: List[Account] = []
        self._factories: Dict[str, TokenStoreFactory] = {}
        self._round_robin_idx: Dict[str, int] = {}

    # -- registration -------------------------------------------------------

    def register_provider(self, provider: str, factory: TokenStoreFactory) -> None:
        self._factories[provider] = factory

    def known_providers(self) -> List[str]:
        return list(self._factories.keys())

    # -- load / persist -----------------------------------------------------

    def load_from_disk(self) -> None:
        ACCOUNTS_DIR.mkdir(parents=True, exist_ok=True)
        self._accounts.clear()
        for child in sorted(ACCOUNTS_DIR.iterdir()):
            if not child.is_dir():
                continue
            meta_file = child / "meta.json"
            if not meta_file.exists():
                logger.warning("Skipping %s: no meta.json", child)
                continue
            try:
                meta = AccountMeta.from_dict(
                    json.loads(meta_file.read_text(encoding="utf-8"))
                )
            except Exception as exc:
                logger.exception("Cannot parse %s: %s", meta_file, exc)
                continue
            factory = self._factories.get(meta.provider)
            if factory is None:
                logger.warning(
                    "No factory for provider %r (slot %s). Did you register the provider?",
                    meta.provider, child.name,
                )
                continue
            try:
                store = factory(child)
            except Exception as exc:
                logger.exception("Cannot init token store for slot %s: %s", child.name, exc)
                continue
            acc = Account(slot_id=child.name, folder=child, meta=meta, token_store=store)
            self._accounts.append(acc)
        self._load_health()
        logger.info(
            "Loaded %d account(s): %s",
            len(self._accounts),
            ", ".join(f"{a.slot_id}({a.provider})" for a in self._accounts),
        )

    def _load_health(self) -> None:
        if not POOL_STATE_FILE.exists():
            return
        try:
            raw = json.loads(POOL_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Cannot parse pool_state.json")
            return
        if not isinstance(raw, dict):
            return
        now = time.time()
        for acc in self._accounts:
            h = raw.get(acc.slot_id)
            if not isinstance(h, dict):
                continue
            # Restore only durable fields (not in_flight, etc.)
            acc.health.exhausted_until = float(h.get("exhausted_until") or 0)
            acc.health.invalid = bool(h.get("invalid", False))
            acc.health.last_429_reason = str(h.get("last_429_reason") or "")
            acc.health.last_429_at = float(h.get("last_429_at") or 0)
            acc.health.last_success_at = float(h.get("last_success_at") or 0)
            if acc.health.exhausted_until <= now:
                acc.health.exhausted_until = 0
                acc.health.last_429_reason = ""
                acc.health.last_429_at = 0
            rl = h.get("rate_limit")
            if isinstance(rl, dict):
                acc.health.rate_limit = rl
                if not acc.health.active_rate_limit(now):
                    acc.health.rate_limit = {}

    def _persist_health(self) -> None:
        try:
            payload = {
                acc.slot_id: {
                    "exhausted_until": acc.health.exhausted_until,
                    "invalid": acc.health.invalid,
                    "last_429_reason": acc.health.last_429_reason,
                    "last_429_at": acc.health.last_429_at,
                    "last_success_at": acc.health.last_success_at,
                    "rate_limit": acc.health.rate_limit,
                }
                for acc in self._accounts
            }
            POOL_STATE_FILE.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            logger.exception("Failed to persist pool state")

    # -- CRUD ---------------------------------------------------------------

    async def create_slot(
        self, provider: str, alias: str = "", slot_id: str = ""
    ) -> Account:
        if provider not in self._factories:
            raise ValueError(f"Unknown provider: {provider}")
        async with self._lock:
            existing_ids = {a.slot_id for a in self._accounts}
            if not slot_id:
                # Generate next available id: chatgpt-1, chatgpt-2, ...
                i = 1
                while True:
                    candidate = f"{provider}-{i}"
                    if candidate not in existing_ids:
                        slot_id = candidate
                        break
                    i += 1
            if slot_id in existing_ids:
                raise ValueError(f"Slot {slot_id} already exists")

            folder = ACCOUNTS_DIR / slot_id
            folder.mkdir(parents=True, exist_ok=True)
            meta = AccountMeta(alias=alias or slot_id, provider=provider)
            store = self._factories[provider](folder)
            acc = Account(slot_id=slot_id, folder=folder, meta=meta, token_store=store)
            acc.save_meta()
            self._accounts.append(acc)
            return acc

    async def delete_slot(self, slot_id: str) -> bool:
        async with self._lock:
            for i, acc in enumerate(self._accounts):
                if acc.slot_id == slot_id:
                    # Best-effort cleanup
                    try:
                        for f in acc.folder.iterdir():
                            try:
                                f.unlink()
                            except Exception:
                                pass
                        acc.folder.rmdir()
                    except Exception:
                        logger.exception("Cannot remove folder for slot %s", slot_id)
                    self._accounts.pop(i)
                    self._persist_health()
                    return True
            return False

    async def set_enabled(self, slot_id: str, enabled: bool) -> bool:
        async with self._lock:
            for acc in self._accounts:
                if acc.slot_id == slot_id:
                    acc.meta.enabled = enabled
                    acc.save_meta()
                    return True
            return False

    async def set_rotation_enabled(self, slot_id: str, enabled: bool) -> bool:
        async with self._lock:
            for acc in self._accounts:
                if acc.slot_id == slot_id:
                    acc.meta.rotation_enabled = enabled
                    acc.save_meta()
                    return True
            return False

    async def update_alias(self, slot_id: str, alias: str) -> bool:
        async with self._lock:
            for acc in self._accounts:
                if acc.slot_id == slot_id:
                    acc.meta.alias = alias
                    acc.save_meta()
                    return True
            return False

    async def update_tier(self, slot_id: str, tier: str) -> bool:
        async with self._lock:
            for acc in self._accounts:
                if acc.slot_id == slot_id:
                    acc.meta.tier = tier
                    acc.save_meta()
                    return True
            return False

    def get(self, slot_id: str) -> Optional[Account]:
        for acc in self._accounts:
            if acc.slot_id == slot_id:
                return acc
        return None

    def all_accounts(self) -> List[Account]:
        return list(self._accounts)

    def by_provider(self, provider: str) -> List[Account]:
        return [a for a in self._accounts if a.provider == provider]

    # -- selection ----------------------------------------------------------

    def available(self, provider: str) -> List[Account]:
        now = time.time()
        return [
            a for a in self._accounts
            if (
                a.provider == provider
                and a.meta.enabled
                and a.health.is_healthy(now)
            )
        ]

    def healthy(self, provider: str) -> List[Account]:
        # Backward-compatible name: health/enablement decides whether a slot
        # can run. Rotation is only a selection preference in acquire().
        return self.available(provider)

    async def acquire(
        self, provider: str, strategy: Optional[str] = None
    ) -> Optional[Account]:
        strategy = strategy or SETTINGS.pool_strategy
        async with self._lock:
            candidates = self.available(provider)
            if not candidates:
                return None

            if strategy == "round_robin":
                rotating = [a for a in candidates if a.meta.rotation_enabled]
                if rotating:
                    candidates = rotating
                idx = self._round_robin_idx.get(provider, 0) % len(candidates)
                self._round_robin_idx[provider] = idx + 1
                acc = candidates[idx]
            elif strategy == "random":
                rotating = [a for a in candidates if a.meta.rotation_enabled]
                if rotating:
                    candidates = rotating
                acc = random.choice(candidates)
            else:  # least_load
                acc = min(
                    candidates,
                    key=lambda a: (a.health.in_flight, a.health.last_used_at),
                )
            acc.health.in_flight += 1
            acc.health.last_used_at = time.time()
            return acc

    async def release(self, account: Account, success: bool) -> None:
        async with self._lock:
            account.health.in_flight = max(0, account.health.in_flight - 1)
            if success:
                account.health.consecutive_failures = 0
                now = time.time()
                account.health.last_success_at = now
                if account.health.exhausted_until <= now:
                    account.health.exhausted_until = 0
                    account.health.last_429_reason = ""
                    account.health.last_429_at = 0
                if not account.health.active_rate_limit(now):
                    account.health.rate_limit = {}
            else:
                account.health.consecutive_failures += 1
            self._persist_health()

    async def mark_exhausted(
        self,
        account: Account,
        resets_at: float,
        reason: str = "",
    ) -> None:
        async with self._lock:
            account.health.exhausted_until = max(account.health.exhausted_until, resets_at)
            account.health.last_429_reason = reason
            account.health.last_429_at = time.time()
            account.health.in_flight = max(0, account.health.in_flight - 1)
            self._persist_health()
        logger.warning(
            "Account %s (%s) exhausted until %s (in %ds): %s",
            account.slot_id, account.provider,
            time.strftime("%H:%M:%S", time.localtime(resets_at)) if resets_at else "?",
            max(0, int(resets_at - time.time())) if resets_at else 0,
            reason,
        )

    async def mark_invalid(self, account: Account) -> None:
        async with self._lock:
            account.health.invalid = True
            account.health.in_flight = max(0, account.health.in_flight - 1)
            self._persist_health()
        logger.warning("Account %s (%s) marked invalid", account.slot_id, account.provider)

    async def mark_valid(self, account: Account) -> None:
        async with self._lock:
            account.health.invalid = False
            account.health.exhausted_until = 0
            account.health.last_429_reason = ""
            account.health.last_429_at = 0
            self._persist_health()

    async def record_rate_limit(self, account: Account, info: Dict[str, Any]) -> None:
        async with self._lock:
            account.health.rate_limit = info
            self._persist_health()


# ---------------------------------------------------------------------------
# Migration from legacy single-account layout
# ---------------------------------------------------------------------------


def migrate_legacy_data() -> None:
    """Move old data/oauth.json -> data/accounts/chatgpt-default/oauth.json.

    Move old data/google_oauth.json -> data/accounts/google-default/oauth.json.
    Runs at most once (silent if no legacy files).
    """
    from core import DATA_DIR

    candidates = [
        (DATA_DIR / "oauth.json", "chatgpt", "ChatGPT (migrated)"),
        (DATA_DIR / "google_oauth.json", "google", "Google (migrated)"),
    ]
    for src, provider, alias in candidates:
        if not src.exists():
            continue
        slot = f"{provider}-default"
        target_folder = ACCOUNTS_DIR / slot
        target_oauth = target_folder / "oauth.json"
        target_meta = target_folder / "meta.json"
        if target_oauth.exists():
            # Already migrated. Remove legacy file to avoid re-migration ambiguity.
            try:
                src.unlink()
            except Exception:
                pass
            continue
        try:
            target_folder.mkdir(parents=True, exist_ok=True)
            target_oauth.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            target_meta.write_text(
                json.dumps(
                    AccountMeta(alias=alias, provider=provider).to_dict(),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            src.unlink()
            logger.info("Migrated legacy %s -> %s", src.name, target_folder)
        except Exception:
            logger.exception("Failed to migrate %s", src)


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

POOL = AccountPool()
