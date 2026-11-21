"""
retrieval/web_search.py

Multi-provider web search with a resilient fallback chain.

Providers (order configurable via WEB_SEARCH_PROVIDERS):
  1. Tavily     — LLM-ready snippets + full page content
  2. SerpAPI    — Google results
  3. DuckDuckGo — free, no key, rate-limited; last resort

Resilience:
  - Every failure is classified (quota / auth / rate limit / timeout / …).
  - Credit exhaustion (Tavily 432/433, SerpAPI "run out of searches") pauses that
    provider for QUOTA_COOLDOWN_S so later sub-queries go straight to the next one.
  - Users see a plain-English notice ("Tavily: API credits exhausted …").
  - Identical searches inside SEARCH_CACHE_TTL_S are served from cache (no credits spent).

All providers normalise their output to list[RetrievedDocument].
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from datetime import datetime
from typing import Optional

import httpx

from retrieval_pipeline.config import get_settings
from retrieval_pipeline.core import http
from retrieval_pipeline.core.cache import TTLCache
from retrieval_pipeline.core.errors import (
    ErrorKind,
    ProviderError,
    classify_body_error,
    display_name,
    redact,
)
from retrieval_pipeline.core.provider_health import add_notice, notice_from_error, registry
from retrieval_pipeline.core.text import best_passages, clean_text, domain_of, truncate
from retrieval_pipeline.models.schemas import (
    ContentType,
    DocumentMetadata,
    RetrievalMethod,
    RetrievedDocument,
)

logger = logging.getLogger(__name__)
cfg = get_settings()

try:
    from duckduckgo_search import AsyncDDGS
    from duckduckgo_search.exceptions import RatelimitException, TimeoutException as DDGTimeout
    _DDG_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    AsyncDDGS = None  # type: ignore[assignment]
    RatelimitException = DDGTimeout = Exception  # type: ignore[misc,assignment]
    _DDG_AVAILABLE = False

if sys.platform == "win32":
    # duckduckgo_search switches Windows to the Selector event-loop policy on import,
    # which cannot start subprocesses and therefore breaks Playwright. Its async client
    # runs in a thread pool, so the default Proactor policy is safe to restore.
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

registry.register("tavily",     "search", configured=bool(cfg.tavily_api_key))
registry.register("serpapi",    "search", configured=bool(cfg.serpapi_key))
registry.register("duckduckgo", "search", configured=_DDG_AVAILABLE)

_cache = TTLCache(ttl_s=cfg.search_cache_ttl_s)


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d", "%a, %d %b %Y %H:%M:%S %Z"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


def _make_doc(
    *,
    query: str,
    provider: str,
    url: str,
    title: Optional[str],
    snippet: str,
    raw: str = "",
    rank: int,
    score: Optional[float] = None,
    published: Optional[str] = None,
) -> Optional[RetrievedDocument]:
    snippet = (snippet or "").strip()
    raw_clean = clean_text(raw) if raw else ""
    if raw_clean:
        focused = best_passages(query, raw_clean, max_chars=cfg.max_doc_chars)
        content = focused if snippet and snippet[:80] in focused else (snippet + "\n\n" + focused if snippet else focused)
    else:
        content = snippet
    content = truncate(content.strip(), cfg.max_doc_chars + len(snippet))
    if len(content) < 20:
        return None
    return RetrievedDocument(
        content=content,
        metadata=DocumentMetadata(
            source_method=RetrievalMethod.WEB_SEARCH,
            domain=domain_of(url),
            url=url,
            title=(title or "").strip() or None,
            published_at=_parse_date(published),
            content_type=ContentType.MARKDOWN if raw_clean else ContentType.PLAIN_TEXT,
            word_count=len(content.split()),
            extra={
                "provider":  provider,
                "rank":      rank,
                "score":     score,
                "snippet":   truncate(snippet, 400),
                "published": published,
                "has_full_text": bool(raw_clean),
            },
        ),
        relevance_score=float(score) if score is not None else round(1.0 / (1 + rank), 4),
    )


# ── Tavily ────────────────────────────────────────────────────────────────────

async def _search_tavily(query: str, max_results: int) -> list[RetrievedDocument]:
    """Docs: https://docs.tavily.com/documentation/api-reference/endpoint/search"""
    if not cfg.tavily_api_key:
        raise ProviderError("tavily", ErrorKind.NOT_CONFIGURED, "TAVILY_API_KEY not set")

    payload = {
        "query":               query,
        "search_depth":        cfg.tavily_search_depth if cfg.tavily_search_depth in ("basic", "advanced", "fast", "ultra-fast") else "basic",
        "max_results":         max(1, min(max_results, 20)),
        "include_raw_content": "markdown" if cfg.tavily_include_raw_content else False,
        "include_answer":      False,
        "topic":               "general",
    }
    data = await http.request_json(
        "tavily", "POST", "https://api.tavily.com/search",
        json=payload,
        headers={"Authorization": f"Bearer {cfg.tavily_api_key}"},
        timeout=cfg.web_search_timeout_s + (10 if cfg.tavily_include_raw_content else 0),
    )

    docs = []
    for i, item in enumerate(data.get("results") or []):
        doc = _make_doc(
            query=query, provider="tavily", url=item.get("url") or "", title=item.get("title"),
            snippet=item.get("content") or "", raw=item.get("raw_content") or "",
            rank=i, score=item.get("score"), published=item.get("published_date"),
        )
        if doc:
            docs.append(doc)
    logger.info("Tavily returned %d results for %r", len(docs), query)
    return docs


# ── SerpAPI ───────────────────────────────────────────────────────────────────

async def _search_serpapi(query: str, max_results: int) -> list[RetrievedDocument]:
    """SerpAPI proxies Google Search. Docs: https://serpapi.com/search-api"""
    if not cfg.serpapi_key:
        raise ProviderError("serpapi", ErrorKind.NOT_CONFIGURED, "SERPAPI_KEY not set")

    params = {
        "engine":  "google",
        "q":       query,
        "api_key": cfg.serpapi_key,
        "num":     max(1, min(max_results, 20)),
        "hl":      "en",
        "gl":      "us",
    }
    data = await http.request_json(
        "serpapi", "GET", "https://serpapi.com/search.json",
        params=params, timeout=cfg.web_search_timeout_s,
    )

    if data.get("error"):
        message = str(data["error"])
        if "hasn't returned any results" in message.lower() or "no results" in message.lower():
            return []
        err = classify_body_error("serpapi", message)
        registry.record_failure(err)
        raise err

    docs: list[RetrievedDocument] = []
    answer_box = data.get("answer_box") or {}
    box_text = answer_box.get("answer") or answer_box.get("snippet") or ""
    if box_text and answer_box.get("link"):
        doc = _make_doc(query=query, provider="serpapi", url=answer_box["link"], title=answer_box.get("title"),
                        snippet=box_text, rank=0, score=1.0)
        if doc:
            docs.append(doc)

    for i, item in enumerate((data.get("organic_results") or [])[:max_results]):
        snippet = item.get("snippet") or ""
        if not snippet or not item.get("link"):
            continue
        doc = _make_doc(query=query, provider="serpapi", url=item["link"], title=item.get("title"),
                        snippet=snippet, rank=i, published=item.get("date"))
        if doc:
            docs.append(doc)

    logger.info("SerpAPI returned %d results for %r", len(docs), query)
    return docs


# ── DuckDuckGo ────────────────────────────────────────────────────────────────

async def _search_duckduckgo(query: str, max_results: int) -> list[RetrievedDocument]:
    """Free, no API key. Rate-limited; used as the last-resort fallback."""
    if not _DDG_AVAILABLE:
        raise ProviderError("duckduckgo", ErrorKind.NOT_CONFIGURED, "duckduckgo-search not installed")
    skip = registry.check("duckduckgo")
    if skip is not None:
        raise skip

    results: list[dict] = []
    last_exc: Optional[BaseException] = None
    for backend in ("lite", "html"):
        try:
            async with AsyncDDGS(timeout=cfg.web_search_timeout_s) as ddgs:
                results = await asyncio.wait_for(
                    ddgs.atext(query, max_results=max_results, backend=backend),
                    timeout=cfg.web_search_timeout_s + 5,
                ) or []
            last_exc = None
            break
        except RatelimitException as exc:
            last_exc = exc
            break
        except Exception as exc:  # timeouts, parser changes, network errors
            last_exc = exc

    if last_exc is not None:
        if isinstance(last_exc, RatelimitException):
            err = ProviderError("duckduckgo", ErrorKind.RATE_LIMIT, "rate limited by DuckDuckGo", retry_after=60)
        elif isinstance(last_exc, (asyncio.TimeoutError, DDGTimeout)):
            err = ProviderError("duckduckgo", ErrorKind.TIMEOUT, "search timed out")
        else:
            err = ProviderError("duckduckgo", ErrorKind.NETWORK, f"{type(last_exc).__name__}: {redact(last_exc)}")
        registry.record_failure(err)
        raise err

    registry.record_success("duckduckgo")
    docs = []
    for i, item in enumerate(results):
        doc = _make_doc(query=query, provider="duckduckgo", url=item.get("href") or "",
                        title=item.get("title"), snippet=item.get("body") or "", rank=i)
        if doc and doc.metadata.url:
            docs.append(doc)
    logger.info("DuckDuckGo returned %d results for %r", len(docs), query)
    return docs


# ── Provider chain ────────────────────────────────────────────────────────────

_PROVIDERS = {
    "tavily":     _search_tavily,
    "serpapi":    _search_serpapi,
    "duckduckgo": _search_duckduckgo,
}


def provider_order() -> list[str]:
    order = [p.strip().lower() for p in cfg.web_search_providers.split(",") if p.strip()]
    return [p for p in order if p in _PROVIDERS] or list(_PROVIDERS)


async def web_search(
    query: str,
    max_results: int = 10,
    providers: Optional[list[str]] = None,
) -> list[RetrievedDocument]:
    """
    Try each provider in order and return the first non-empty result.
    Pass `providers` to restrict which ones to attempt, e.g. ["tavily", "duckduckgo"].
    Never raises — returns [] (with user-facing notices) when every provider fails.
    """
    order = provider_order()
    if providers:
        allowed = {p.lower() for p in providers}
        order = [p for p in order if p in allowed]

    cache_key = (tuple(order), " ".join(query.lower().split()), max_results)
    cached = _cache.get(cache_key)
    if cached is not None:
        logger.info("web_search cache hit for %r", query)
        return [d.model_copy(deep=True) for d in cached]

    failures: list[ProviderError] = []
    for name in order:
        fn = _PROVIDERS[name]
        t0 = time.perf_counter()
        try:
            results = await fn(query, max_results)
        except ProviderError as err:
            if err.kind != ErrorKind.NOT_CONFIGURED:
                failures.append(err)
                remaining = [p for p in order[order.index(name) + 1:] if registry.is_available(p)]
                note = f"Falling back to {display_name(remaining[0])}." if remaining else ""
                notice_from_error(err, note)
            continue
        except Exception as exc:  # defensive: an unexpected bug in one provider
            err = ProviderError(name, ErrorKind.UNKNOWN, f"{type(exc).__name__}: {redact(exc)}")
            registry.record_failure(err)
            failures.append(err)
            logger.exception("web_search: %s crashed", name)
            continue

        if results:
            logger.info("web_search: %s → %d results in %.0fms", name, len(results), (time.perf_counter() - t0) * 1000)
            _cache.set(cache_key, [d.model_copy(deep=True) for d in results])
            return results
        logger.warning("web_search: %s returned no results for %r", name, query)

    if failures:
        add_notice("error", "web_search",
                   "All web search providers failed for this query. Check API credits/keys on the Providers panel.",
                   kind="all_failed")
    logger.error("web_search: no provider returned results for %r", query)
    return []


async def web_search_batch(
    queries: list[str],
    max_results_per_query: int = 10,
) -> dict[str, list[RetrievedDocument]]:
    """Run several searches concurrently. Returns {query: documents}."""
    results = await asyncio.gather(
        *(web_search(q, max_results_per_query) for q in queries), return_exceptions=True
    )
    output: dict[str, list[RetrievedDocument]] = {}
    for query, result in zip(queries, results):
        if isinstance(result, BaseException):
            logger.error("Batch search failed for %r: %s", query, redact(result))
            output[query] = []
        else:
            output[query] = result
    return output


# ── Credit / quota introspection ──────────────────────────────────────────────

_quota_cache = TTLCache(ttl_s=120, max_entries=4)


async def search_quota(force: bool = False) -> dict:
    """
    Remaining credits for the paid search providers (both endpoints are free to call).
    A provider found at 0 remaining credits is paused proactively.
    """
    if not force:
        cached = _quota_cache.get("quota")
        if cached is not None:
            return cached

    out: dict[str, dict] = {}

    async with httpx.AsyncClient(timeout=10) as client:
        if cfg.tavily_api_key:
            try:
                resp = await client.get("https://api.tavily.com/usage",
                                        headers={"Authorization": f"Bearer {cfg.tavily_api_key}"})
                if resp.status_code == 200:
                    acct = (resp.json() or {}).get("account") or {}
                    used, limit = acct.get("plan_usage"), acct.get("plan_limit")
                    remaining = (limit - used) if isinstance(used, (int, float)) and isinstance(limit, (int, float)) else None
                    out["tavily"] = {"plan": acct.get("current_plan"), "used": used, "limit": limit, "remaining": remaining}
                else:
                    out["tavily"] = {"error": f"HTTP {resp.status_code}"}
            except Exception as exc:
                out["tavily"] = {"error": redact(f"{type(exc).__name__}: {exc}")}

        if cfg.serpapi_key:
            try:
                resp = await client.get("https://serpapi.com/account.json", params={"api_key": cfg.serpapi_key})
                body = resp.json() if resp.status_code == 200 else {}
                if resp.status_code == 200 and not body.get("error"):
                    out["serpapi"] = {
                        "plan":      body.get("plan_name"),
                        "used":      body.get("this_month_usage"),
                        "limit":     body.get("searches_per_month"),
                        "remaining": body.get("total_searches_left", body.get("plan_searches_left")),
                    }
                else:
                    out["serpapi"] = {"error": body.get("error") or f"HTTP {resp.status_code}"}
            except Exception as exc:
                out["serpapi"] = {"error": redact(f"{type(exc).__name__}: {exc}")}

    for name, info in out.items():
        remaining, limit = info.get("remaining"), info.get("limit")
        if isinstance(remaining, (int, float)):
            if remaining <= 0 and registry.is_available(name):
                registry.record_failure(ProviderError(name, ErrorKind.QUOTA, "0 credits remaining (account check)"))
            info["low"] = bool(limit and remaining / max(limit, 1) < 0.1)

    _quota_cache.set("quota", out)
    return out
