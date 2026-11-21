"""
retrieval/web_crawl.py

Web page fetcher / crawler ("deep read").

Strategy:
  - Primary: httpx (fast, static HTML)
  - Fallback: Playwright Chromium for JS-rendered pages (optional)
  - Optional BFS link following up to a configurable depth (same domain only)
  - robots.txt respected (RFC 9309 status handling; if robots.txt cannot be
    fetched at all because of a network error, the site is treated as allow-all)
  - Per-domain politeness delay, response-size cap, boilerplate stripping
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import sys
from collections import deque
from typing import Optional
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

from retrieval_pipeline.config import get_settings
from retrieval_pipeline.core.http import KeyedRateLimiter
from retrieval_pipeline.core.text import best_passages, domain_of
from retrieval_pipeline.models.schemas import (
    ContentType,
    DocumentMetadata,
    RetrievalMethod,
    RetrievedDocument,
)

logger = logging.getLogger(__name__)
cfg = get_settings()

_MAX_RESPONSE_BYTES = 3_000_000
_HTML_TYPES = ("text/html", "application/xhtml+xml", "text/plain")


# ── URL normalisation ─────────────────────────────────────────────────────────

def normalise_url(url: str) -> str:
    """Strip fragments, lowercase scheme+host."""
    parsed = urlparse(url)
    return urlunparse(parsed._replace(scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower(), fragment=""))


def url_fingerprint(url: str) -> str:
    return hashlib.md5(normalise_url(url).encode()).hexdigest()


# ── robots.txt (RFC 9309) ─────────────────────────────────────────────────────

_robots_cache: dict[str, RobotFileParser] = {}


async def _can_fetch(url: str) -> bool:
    if not cfg.crawl_respect_robots:
        return True

    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    rp = _robots_cache.get(base)
    if rp is None:
        rp = RobotFileParser()
        rp.set_url(f"{base}/robots.txt")
        try:
            async with httpx.AsyncClient(timeout=8, follow_redirects=True,
                                         headers={"User-Agent": cfg.crawl_user_agent}) as client:
                resp = await client.get(f"{base}/robots.txt")
            if resp.status_code < 300:
                rp.parse(resp.text.splitlines())        # parse() also marks the file as read
            elif resp.status_code < 500:
                rp.allow_all = True                     # 4xx: no robots.txt → everything allowed
            else:
                rp.disallow_all = True                  # 5xx: unreachable → assume disallowed
        except Exception:
            # Network failure fetching robots.txt. NOTE: without this flag
            # RobotFileParser.can_fetch() returns False for *every* URL because the
            # file was never "read" — the previous version silently blocked all pages.
            rp.allow_all = True
        _robots_cache[base] = rp

    try:
        return rp.can_fetch(cfg.crawl_user_agent, url)
    except Exception:
        return True


# ── Content extraction ────────────────────────────────────────────────────────

_BOILERPLATE_TAGS = ["nav", "footer", "header", "aside", "script", "style", "form",
                     "iframe", "noscript", "svg", "button", "template"]

_CONTENT_SELECTORS = [
    "article", "main", '[role="main"]',
    ".post-content", ".entry-content", ".article-body", ".article-content",
    ".content", "#content", ".markdown-body", ".documentation", ".docs-content",
]


def _clean_lines(text: str) -> str:
    lines = []
    for line in (l.strip() for l in text.splitlines()):
        if not line:
            continue
        # keep sentences and headings; drop menu fragments ("Home", "Sign in", …)
        if len(line) >= 25 or len(line.split()) >= 4:
            lines.append(line)
    return "\n".join(lines).strip()


def extract_main_content(html: str, url: str = "") -> tuple[str, str]:
    """Returns (title, clean_text) from raw HTML, preferring the main article body."""
    soup = BeautifulSoup(html, "lxml") if _has_lxml() else BeautifulSoup(html, "html.parser")

    title = ""
    if soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(strip=True)
    elif soup.find("h1"):
        title = soup.find("h1").get_text(strip=True)

    for tag in soup.find_all(_BOILERPLATE_TAGS):
        tag.decompose()

    for selector in _CONTENT_SELECTORS:
        el = soup.select_one(selector)
        if el:
            text = _clean_lines(el.get_text(separator="\n", strip=True))
            if len(text) > 300:
                return title, text

    body = soup.find("body") or soup
    return title, _clean_lines(body.get_text(separator="\n", strip=True))


def _has_lxml() -> bool:
    try:
        import lxml  # noqa: F401
        return True
    except ImportError:
        return False


def extract_links(html: str, base_url: str) -> list[str]:
    """All same-domain http(s) links on a page."""
    soup = BeautifulSoup(html, "html.parser")
    base_domain = urlparse(base_url).netloc
    links = set()
    for a_tag in soup.find_all("a", href=True):
        full_url = urljoin(base_url, a_tag["href"])
        parsed = urlparse(full_url)
        if parsed.netloc == base_domain and parsed.scheme in ("http", "https"):
            links.add(normalise_url(full_url))
    return sorted(links)


# ── Page fetchers ─────────────────────────────────────────────────────────────

async def _fetch_with_httpx(url: str) -> Optional[str]:
    headers = {
        "User-Agent": cfg.crawl_user_agent,
        "Accept": "text/html,application/xhtml+xml;q=0.9,text/plain;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
    }
    try:
        async with httpx.AsyncClient(timeout=cfg.crawl_timeout_s, follow_redirects=True, headers=headers) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    logger.debug("httpx %s → HTTP %d", url, resp.status_code)
                    return None
                ctype = resp.headers.get("content-type", "").lower()
                if not any(t in ctype for t in _HTML_TYPES):
                    logger.debug("httpx %s → unsupported content-type %s", url, ctype)
                    return None
                chunks, size = [], 0
                async for chunk in resp.aiter_bytes():
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > _MAX_RESPONSE_BYTES:
                        break
                raw = b"".join(chunks)
                return raw.decode(resp.encoding or "utf-8", errors="replace")
    except Exception as exc:
        logger.debug("httpx fetch failed for %s: %s", url, exc)
    return None


_playwright_disabled_reason: Optional[str] = None
_playwright_slots: Optional[asyncio.Semaphore] = None


async def _fetch_with_playwright(url: str) -> Optional[str]:
    """JS-rendered fetch via Playwright Chromium. Disabled for the process after a hard failure."""
    global _playwright_disabled_reason, _playwright_slots
    if not cfg.crawl_use_playwright or _playwright_disabled_reason:
        return None
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        _playwright_disabled_reason = "playwright not installed"
        return None

    if _playwright_slots is None:
        _playwright_slots = asyncio.Semaphore(2)     # each browser is ~150 MB of RAM
    async with _playwright_slots:
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                try:
                    page = await browser.new_page(user_agent=cfg.crawl_user_agent)
                    await page.goto(url, wait_until="domcontentloaded", timeout=cfg.crawl_timeout_s * 1000)
                    await page.wait_for_timeout(1200)
                    return await page.content()
                finally:
                    await browser.close()
        except NotImplementedError:
            # Windows + SelectorEventLoop (e.g. uvicorn --reload) cannot spawn subprocesses.
            _playwright_disabled_reason = "event loop does not support subprocesses (run uvicorn without --reload on Windows)"
            logger.warning("Playwright disabled: %s", _playwright_disabled_reason)
        except Exception as exc:
            message = str(exc)
            if "Executable doesn't exist" in message or "playwright install" in message:
                _playwright_disabled_reason = "Chromium not installed — run: python -m playwright install chromium"
                logger.warning("Playwright disabled: %s", _playwright_disabled_reason)
            else:
                logger.debug("Playwright fetch failed for %s: %s", url, exc)
    return None


async def fetch_page(url: str, prefer_js: bool = False) -> Optional[str]:
    """httpx first (fast), Playwright fallback — or the reverse when prefer_js=True."""
    if prefer_js:
        html = await _fetch_with_playwright(url)
        if html:
            return html
    html = await _fetch_with_httpx(url)
    if html:
        return html
    if not prefer_js:
        html = await _fetch_with_playwright(url)
    return html


_domain_limiter = KeyedRateLimiter(min_interval=1.0)


# ── BFS crawler ───────────────────────────────────────────────────────────────

async def crawl_url(
    seed_url: str,
    max_depth: Optional[int] = None,
    max_pages: Optional[int] = None,
    follow_links: bool = True,
    prefer_js: bool = False,
    query: Optional[str] = None,
) -> list[RetrievedDocument]:
    """
    BFS crawl from seed_url (same domain only).

    max_depth=0 fetches just the seed page. When `query` is given, very long pages
    are trimmed to the passages most relevant to it.
    """
    max_depth = cfg.crawl_max_depth if max_depth is None else max_depth
    max_pages = cfg.crawl_max_pages if max_pages is None else max_pages

    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(normalise_url(seed_url), 0)])
    docs: list[RetrievedDocument] = []

    while queue and len(docs) < max_pages:
        url, depth = queue.popleft()
        fp = url_fingerprint(url)
        if fp in visited:
            continue
        visited.add(fp)

        if not await _can_fetch(url):
            logger.info("robots.txt disallows %s", url)
            continue

        domain = urlparse(url).netloc
        await _domain_limiter.acquire(domain)
        html = await fetch_page(url, prefer_js=prefer_js)
        if not html:
            continue

        title, text = extract_main_content(html, url)
        if len(text) < 200:
            continue
        if query and len(text) > cfg.max_doc_chars:
            text = best_passages(query, text, max_chars=cfg.max_doc_chars)
        elif len(text) > cfg.max_doc_chars * 2:
            text = text[: cfg.max_doc_chars * 2]

        docs.append(RetrievedDocument(
            content=text,
            metadata=DocumentMetadata(
                source_method=RetrievalMethod.WEB_CRAWL,
                domain=domain_of(url),
                url=url,
                title=title or None,
                content_type=ContentType.HTML,
                crawl_depth=depth,
                word_count=len(text.split()),
                extra={"provider": "crawler"},
            ),
            relevance_score=round(0.9 / (1 + depth), 4),
        ))

        if follow_links and depth < max_depth:
            for link in extract_links(html, url):
                if url_fingerprint(link) not in visited:
                    queue.append((link, depth + 1))

    logger.info("Crawl of %s complete: %d pages", seed_url, len(docs))
    return docs


async def crawl_batch(
    seed_urls: list[str],
    max_depth: int = 0,
    max_pages_per_seed: int = 1,
    query: Optional[str] = None,
) -> list[RetrievedDocument]:
    """Fetch several seed URLs concurrently (bounded by CRAWL_CONCURRENCY)."""
    semaphore = asyncio.Semaphore(max(1, cfg.crawl_concurrency))

    async def _one(url: str) -> list[RetrievedDocument]:
        async with semaphore:
            return await crawl_url(url, max_depth=max_depth, max_pages=max_pages_per_seed,
                                   follow_links=max_depth > 0, query=query)

    unique_seeds = list(dict.fromkeys(u for u in seed_urls if u and u.startswith(("http://", "https://"))))
    results = await asyncio.gather(*(_one(u) for u in unique_seeds), return_exceptions=True)

    all_docs: list[RetrievedDocument] = []
    for seed, result in zip(unique_seeds, results):
        if isinstance(result, BaseException):
            logger.warning("Crawl failed for %s: %s", seed, result)
        else:
            all_docs.extend(result)
    return all_docs


def playwright_status() -> str:
    if not cfg.crawl_use_playwright:
        return "disabled in config"
    if _playwright_disabled_reason:
        return f"disabled: {_playwright_disabled_reason}"
    if sys.platform == "win32":
        return "available (needs ProactorEventLoop — run uvicorn without --reload)"
    return "available"
