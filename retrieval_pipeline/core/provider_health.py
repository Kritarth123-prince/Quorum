"""
core/provider_health.py

Circuit breaker + status registry for every external provider, and a per-request
"notices" collector that surfaces provider problems to the end user.

Why: when Tavily runs out of credits (HTTP 432) the platform must not keep
hammering it on every sub-query. The registry remembers the failure, skips the
provider for a cooldown period, and the fallback chain moves straight on to the
next provider. The /providers endpoint and the UI read the same registry.

Cooldowns (configurable in .env):
  QUOTA       → quota_cooldown_s      (default 6 h)
  AUTH        → auth_cooldown_s       (default 24 h — fix the key and reset)
  RATE_LIMIT  → Retry-After or rate_limit_cooldown_s
  transient   → after failure_threshold consecutive failures, failure_cooldown_s
"""

from __future__ import annotations

import contextvars
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Iterator, Optional

from retrieval_pipeline.config import get_settings
from retrieval_pipeline.core.errors import ErrorKind, ProviderError, display_name

logger = logging.getLogger(__name__)
cfg = get_settings()


def _now_wall() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class ProviderState:
    name:                 str
    category:             str                     # search | academic | llm | infra
    configured:           bool          = True
    consecutive_failures: int           = 0
    total_calls:          int           = 0
    total_failures:       int           = 0
    disabled_until:       float         = 0.0     # time.monotonic() deadline
    disabled_until_wall:  Optional[str] = None
    last_error:           Optional[str] = None
    last_error_kind:      Optional[str] = None
    last_error_at:        Optional[str] = None
    last_success_at:      Optional[str] = None

    @property
    def status(self) -> str:
        if not self.configured:
            return "not_configured"
        if self.disabled_until > time.monotonic():
            return {
                ErrorKind.QUOTA.value:      "quota_exhausted",
                ErrorKind.AUTH.value:       "auth_failed",
                ErrorKind.RATE_LIMIT.value: "rate_limited",
            }.get(self.last_error_kind or "", "cooling_down")
        if self.consecutive_failures:
            return "degraded"
        return "ok"

    def to_dict(self) -> dict:
        remaining = max(0.0, self.disabled_until - time.monotonic())
        return {
            "provider":            self.name,
            "display_name":        display_name(self.name),
            "category":            self.category,
            "status":              self.status,
            "configured":          self.configured,
            "cooldown_remaining_s": round(remaining) if remaining else 0,
            "disabled_until":      self.disabled_until_wall if remaining else None,
            "consecutive_failures": self.consecutive_failures,
            "total_calls":         self.total_calls,
            "total_failures":      self.total_failures,
            "last_error":          self.last_error,
            "last_error_kind":     self.last_error_kind,
            "last_error_at":       self.last_error_at,
            "last_success_at":     self.last_success_at,
        }


class ProviderHealthRegistry:
    def __init__(self) -> None:
        self._states: dict[str, ProviderState] = {}

    def register(self, name: str, category: str, configured: bool = True) -> ProviderState:
        state = self._states.get(name)
        if state is None:
            state = ProviderState(name=name, category=category, configured=configured)
            self._states[name] = state
        else:
            state.configured = configured
            state.category = category
        return state

    def get(self, name: str) -> Optional[ProviderState]:
        return self._states.get(name)

    def check(self, name: str) -> Optional[ProviderError]:
        """Return a ProviderError explaining why `name` must be skipped, or None if usable."""
        state = self._states.get(name)
        if state is None:
            return None
        if not state.configured:
            return ProviderError(name, ErrorKind.NOT_CONFIGURED, "no API key configured")
        remaining = state.disabled_until - time.monotonic()
        if remaining > 0:
            reason = state.last_error_kind or "errors"
            return ProviderError(
                name, ErrorKind.UNAVAILABLE,
                f"paused after {reason.replace('_', ' ')} until {state.disabled_until_wall} "
                f"({int(remaining)}s left): {state.last_error or ''}".strip(),
                retry_after=remaining,
            )
        return None

    def is_available(self, name: str) -> bool:
        return self.check(name) is None

    def record_success(self, name: str) -> None:
        state = self._states.get(name) or self.register(name, "unknown")
        state.total_calls += 1
        state.consecutive_failures = 0
        state.last_success_at = _now_wall()
        if state.disabled_until and state.disabled_until <= time.monotonic():
            state.disabled_until = 0.0
            state.disabled_until_wall = None

    def record_failure(self, err: ProviderError) -> None:
        if err.kind in (ErrorKind.NOT_CONFIGURED, ErrorKind.UNAVAILABLE):
            return
        state = self._states.get(err.provider) or self.register(err.provider, "unknown")
        state.total_calls += 1
        state.total_failures += 1
        state.consecutive_failures += 1
        state.last_error = err.message
        state.last_error_kind = err.kind.value
        state.last_error_at = _now_wall()

        cooldown = 0.0
        if err.kind == ErrorKind.QUOTA:
            cooldown = cfg.quota_cooldown_s
        elif err.kind == ErrorKind.AUTH:
            cooldown = cfg.auth_cooldown_s
        elif err.kind == ErrorKind.RATE_LIMIT:
            cooldown = err.retry_after if err.retry_after else cfg.rate_limit_cooldown_s
        elif err.kind == ErrorKind.BAD_REQUEST and err.status_code == 404:
            cooldown = cfg.auth_cooldown_s       # unknown model / endpoint: configuration problem
        elif state.consecutive_failures >= cfg.failure_threshold:
            cooldown = cfg.failure_cooldown_s

        if cooldown > 0:
            state.disabled_until = time.monotonic() + cooldown
            state.disabled_until_wall = datetime.fromtimestamp(time.time() + cooldown).astimezone().strftime("%Y-%m-%d %H:%M")
            logger.warning(
                "Provider %s paused for %ds after %s: %s",
                err.provider, int(cooldown), err.kind.value, err.message,
            )

    def reset(self, name: Optional[str] = None) -> list[str]:
        targets = [name] if name else list(self._states)
        done = []
        for n in targets:
            state = self._states.get(n)
            if state is None:
                continue
            state.disabled_until = 0.0
            state.disabled_until_wall = None
            state.consecutive_failures = 0
            done.append(n)
        return done

    def snapshot(self, category: Optional[str] = None) -> list[dict]:
        return [
            s.to_dict() for s in self._states.values()
            if category is None or s.category == category
        ]


registry = ProviderHealthRegistry()


# ── Per-request notices ───────────────────────────────────────────────────────

@dataclass
class Notice:
    level:    str          # info | warning | error
    provider: str
    message:  str
    kind:     Optional[str] = None
    time:     str = field(default_factory=lambda: datetime.now().astimezone().strftime("%H:%M:%S"))

    def to_dict(self) -> dict:
        return {"level": self.level, "provider": self.provider, "message": self.message,
                "kind": self.kind, "time": self.time}


_notices: contextvars.ContextVar[Optional[list[Notice]]] = contextvars.ContextVar("research_notices", default=None)


@contextmanager
def collect_notices() -> Iterator[list[Notice]]:
    """Collect provider notices raised anywhere inside this context (including child tasks)."""
    bucket: list[Notice] = []
    token = _notices.set(bucket)
    try:
        yield bucket
    finally:
        _notices.reset(token)


def add_notice(level: str, provider: str, message: str, kind: Optional[str] = None) -> None:
    bucket = _notices.get()
    if bucket is None:
        return
    if any(n.provider == provider and n.message == message for n in bucket):
        return
    bucket.append(Notice(level=level, provider=provider, message=message, kind=kind))


def notice_from_error(err: ProviderError, fallback_note: str = "") -> None:
    """Turn a ProviderError into a user-facing notice (quota/auth are errors; the rest warnings)."""
    if err.kind == ErrorKind.NOT_CONFIGURED:
        return
    level = "error" if err.kind in (ErrorKind.QUOTA, ErrorKind.AUTH) else "warning"
    text = err.user_message + (f" {fallback_note}" if fallback_note else "")
    add_notice(level, err.provider, text.strip(), kind=err.kind.value)


def current_notices() -> list[Notice]:
    return list(_notices.get() or [])


@contextmanager
def notice_scope() -> Iterator["Callable[[], list[Notice]]"]:
    """
    Yield a function returning the notices added inside this block. Re-uses an
    outer collector when one is active (so a research run still sees them all),
    otherwise opens a fresh one.
    """
    existing = _notices.get()
    if existing is None:
        with collect_notices() as bucket:
            yield lambda: list(bucket)
    else:
        start = len(existing)
        yield lambda: list(existing[start:])
