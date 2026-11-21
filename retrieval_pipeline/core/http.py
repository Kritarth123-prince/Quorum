"""
core/http.py

One resilient HTTP helper used by every REST provider:
  - checks the circuit breaker before calling
  - classifies failures (quota, auth, rate limit, timeout, …)
  - retries transient failures with exponential backoff + jitter, honouring Retry-After
  - records success / failure in the provider registry
  - never leaks API keys into exceptions or logs

Also provides a per-event-loop-safe async rate limiter for APIs with strict
request-rate rules (arXiv: 1 request / 3 s, Semantic Scholar: 1 request / s).
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import weakref
from typing import Any, Callable, Optional

import httpx

from retrieval_pipeline.config import get_settings
from retrieval_pipeline.core.errors import (
    ErrorKind,
    ProviderError,
    classify_http_error,
    redact,
)
from retrieval_pipeline.core.provider_health import registry

logger = logging.getLogger(__name__)
cfg = get_settings()


def make_client(timeout: float, **kwargs: Any) -> httpx.AsyncClient:
    """Factory for HTTP clients — tests monkeypatch this to inject a MockTransport."""
    return httpx.AsyncClient(timeout=timeout, **kwargs)


def _backoff(attempt: int, retry_after: Optional[float]) -> float:
    if retry_after is not None:
        return min(retry_after, cfg.max_retry_wait_s)
    return min(cfg.max_retry_wait_s, (2 ** attempt) * 0.75 + random.uniform(0, 0.5))


async def request(
    provider: str,
    method: str,
    url: str,
    *,
    params: Optional[dict] = None,
    json: Any = None,
    headers: Optional[dict] = None,
    timeout: float = 20.0,
    max_retries: int = 2,
    follow_redirects: bool = True,
    rate_limiter: Optional["AsyncRateLimiter"] = None,
    track: bool = True,
) -> httpx.Response:
    """
    Perform an HTTP request on behalf of `provider` with classification and retries.
    Returns the successful httpx.Response; raises ProviderError otherwise.

    track=False skips the circuit-breaker check and health bookkeeping (for callers,
    such as the LLM provider chain, that do their own).
    """
    if track:
        skip = registry.check(provider)
        if skip is not None:
            raise skip

    last_err: Optional[ProviderError] = None
    for attempt in range(max_retries + 1):
        if rate_limiter is not None:
            await rate_limiter.acquire()
        try:
            async with make_client(timeout, follow_redirects=follow_redirects) as client:
                resp = await client.request(method, url, params=params, json=json, headers=headers)
        except httpx.TimeoutException as exc:
            last_err = ProviderError(provider, ErrorKind.TIMEOUT, f"timed out after {timeout:.0f}s ({type(exc).__name__})")
        except httpx.TransportError as exc:
            last_err = ProviderError(provider, ErrorKind.NETWORK, f"{type(exc).__name__}: {redact(exc)}")
        else:
            if resp.status_code < 400:
                if track:
                    registry.record_success(provider)
                return resp
            last_err = classify_http_error(provider, resp.status_code, resp.text, resp.headers)

        if not last_err.retryable or attempt >= max_retries:
            break
        if last_err.retry_after is not None and last_err.retry_after > cfg.max_retry_wait_s:
            break   # waiting that long is worse than falling back to another provider
        delay = _backoff(attempt, last_err.retry_after)
        logger.info("%s: %s — retry %d/%d in %.1fs", provider, last_err.kind.value, attempt + 1, max_retries, delay)
        await asyncio.sleep(delay)

    assert last_err is not None
    if track:
        registry.record_failure(last_err)
    logger.warning("%s request failed: %s", provider, last_err)
    raise last_err


async def request_json(provider: str, method: str, url: str, **kwargs: Any) -> Any:
    resp = await request(provider, method, url, **kwargs)
    try:
        return resp.json()
    except ValueError as exc:
        err = ProviderError(provider, ErrorKind.SERVER, f"invalid JSON in response: {exc}", status_code=resp.status_code)
        if kwargs.get("track", True):
            registry.record_failure(err)
        raise err


# ── Rate limiter (safe across multiple event loops) ───────────────────────────

class AsyncRateLimiter:
    """Guarantees at least `min_interval` seconds between successive acquire() calls."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0
        self._locks: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = weakref.WeakKeyDictionary()

    def _lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        lock = self._locks.get(loop)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[loop] = lock
        return lock

    async def acquire(self) -> None:
        async with self._lock():
            wait = self._last + self.min_interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


class KeyedRateLimiter:
    """One AsyncRateLimiter per key (e.g. per domain for the crawler)."""

    def __init__(self, min_interval: float, factory: Callable[[float], AsyncRateLimiter] = AsyncRateLimiter):
        self._min_interval = min_interval
        self._factory = factory
        self._limiters: dict[str, AsyncRateLimiter] = {}

    async def acquire(self, key: str) -> None:
        limiter = self._limiters.get(key)
        if limiter is None:
            limiter = self._factory(self._min_interval)
            self._limiters[key] = limiter
        await limiter.acquire()
