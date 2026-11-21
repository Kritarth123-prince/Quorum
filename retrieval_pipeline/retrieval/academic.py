"""
retrieval/academic.py

Academic paper retrieval from two sources, queried in parallel:

  1. Semantic Scholar Graph API — broad coverage, citation counts, abstracts, TL;DRs
  2. arXiv Atom API             — preprints (CS / AI / physics / maths focus)

Both APIs have strict request-rate rules (arXiv: one request every 3 s;
Semantic Scholar: 1 request/s with a key), so calls go through rate limiters
and the shared resilient HTTP layer (retries, quota/auth classification).
Results are de-duplicated by normalised title and interleaved by rank.
"""

from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Optional

from retrieval_pipeline.config import get_settings
from retrieval_pipeline.core import http
from retrieval_pipeline.core.errors import ErrorKind, ProviderError
from retrieval_pipeline.core.provider_health import notice_from_error, registry
from retrieval_pipeline.core.text import keywords
from retrieval_pipeline.models.schemas import (
    ContentType,
    DocumentMetadata,
    RetrievalMethod,
    RetrievedDocument,
)

logger = logging.getLogger(__name__)
cfg = get_settings()

_SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
_ARXIV_BASE            = "https://export.arxiv.org/api/query"
_ARXIV_NS              = {"atom": "http://www.w3.org/2005/Atom"}

registry.register("semantic_scholar", "academic", configured=True)   # works without a key (shared pool)
registry.register("arxiv",            "academic", configured=True)

_s2_limiter    = http.AsyncRateLimiter(1.1 if cfg.semantic_scholar_key else 3.0)
_arxiv_limiter = http.AsyncRateLimiter(3.0)


def _paper_content(title: str, authors: str, year: Optional[int], venue: str, body: str) -> str:
    byline = ", ".join(filter(None, [authors, venue, str(year) if year else ""]))
    parts = [f"# {title}"]
    if byline:
        parts.append(byline)
    if body:
        parts.append(body)
    return "\n\n".join(parts)


# ── Semantic Scholar ──────────────────────────────────────────────────────────

async def _semantic_scholar_search(query: str, max_results: int = 5) -> list[RetrievedDocument]:
    params = {
        "query":  query[:300],
        "limit":  max(1, min(max_results, 100)),
        "fields": "title,abstract,tldr,authors,year,venue,citationCount,externalIds,url,openAccessPdf,publicationDate",
    }
    headers = {"x-api-key": cfg.semantic_scholar_key} if cfg.semantic_scholar_key else None

    data = await http.request_json(
        "semantic_scholar", "GET", f"{_SEMANTIC_SCHOLAR_BASE}/paper/search",
        params=params, headers=headers, timeout=cfg.academic_timeout_s,
        rate_limiter=_s2_limiter, max_retries=2,
    )

    docs = []
    for rank, paper in enumerate(data.get("data") or []):
        title    = (paper.get("title") or "").strip()
        abstract = (paper.get("abstract") or "").strip()
        tldr     = ((paper.get("tldr") or {}).get("text") or "").strip()
        body     = abstract or (f"TL;DR: {tldr}" if tldr else "")
        if not title or not body:
            continue

        authors = ", ".join(a.get("name", "") for a in (paper.get("authors") or [])[:5] if a.get("name"))
        if len(paper.get("authors") or []) > 5:
            authors += " et al."
        year    = paper.get("year")
        ext     = paper.get("externalIds") or {}
        doi     = ext.get("DOI")
        url     = paper.get("url") or (f"https://doi.org/{doi}" if doi else None)
        pdf_url = (paper.get("openAccessPdf") or {}).get("url")
        published_at = None
        if paper.get("publicationDate"):
            try:
                published_at = datetime.strptime(paper["publicationDate"], "%Y-%m-%d")
            except ValueError:
                published_at = None
        elif year:
            published_at = datetime(int(year), 1, 1)

        citations = paper.get("citationCount") or 0
        content = _paper_content(title, authors, year, paper.get("venue") or "", body)
        docs.append(RetrievedDocument(
            content=content,
            metadata=DocumentMetadata(
                source_method=RetrievalMethod.ACADEMIC,
                domain="semanticscholar.org",
                url=url,
                title=title,
                author=authors or None,
                published_at=published_at,
                content_type=ContentType.PLAIN_TEXT,
                word_count=len(content.split()),
                extra={
                    "provider":       "semantic_scholar",
                    "paper_id":       paper.get("paperId"),
                    "citation_count": citations,
                    "doi":            doi,
                    "arxiv_id":       ext.get("ArXiv"),
                    "pdf_url":        pdf_url,
                    "venue":          paper.get("venue"),
                    "rank":           rank,
                },
            ),
            # Search rank is the relevance signal; citations add a small credibility bonus.
            relevance_score=round(1.0 / (1 + rank) + min(0.1, citations / 10000), 4),
        ))

    logger.info("Semantic Scholar: %d results for %r", len(docs), query)
    return docs


# ── arXiv ─────────────────────────────────────────────────────────────────────

def _parse_arxiv_date(date_str: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def build_arxiv_query(query: str, categories: Optional[list[str]] = None, mode: str = "and") -> str:
    """
    arXiv field queries: all:term AND all:term …  (httpx URL-encodes the result;
    the old code also called quote() first, which double-encoded spaces as %2520).
    """
    terms = [t for t in keywords(query, limit=5) if re.fullmatch(r"[a-z0-9][a-z0-9\-]*", t)]
    if not terms:
        terms = [re.sub(r"[^a-z0-9 ]", " ", query.lower()).strip() or "research"]
    joiner = " AND " if mode == "and" else " OR "
    search = joiner.join(f"all:{t}" for t in terms)
    if categories:
        cats = " OR ".join(f"cat:{c}" for c in categories)
        search = f"({search}) AND ({cats})"
    return search


def _parse_arxiv(xml_text: str) -> list[RetrievedDocument]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ProviderError("arxiv", ErrorKind.SERVER, f"invalid Atom XML: {exc}")

    docs = []
    for rank, entry in enumerate(root.findall("atom:entry", _ARXIV_NS)):
        title    = re.sub(r"\s+", " ", entry.findtext("atom:title", namespaces=_ARXIV_NS) or "").strip()
        abstract = re.sub(r"\s+", " ", entry.findtext("atom:summary", namespaces=_ARXIV_NS) or "").strip()
        if not title or not abstract or title.lower() == "error":
            continue

        author_names = [(a.findtext("atom:name", namespaces=_ARXIV_NS) or "").strip()
                        for a in entry.findall("atom:author", _ARXIV_NS)]
        authors = ", ".join(n for n in author_names[:5] if n) + (" et al." if len(author_names) > 5 else "")

        url, pdf_url = None, None
        for link in entry.findall("atom:link", _ARXIV_NS):
            if link.get("type") == "text/html" or link.get("rel") == "alternate":
                url = url or link.get("href")
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
        arxiv_id = None
        match = re.search(r"abs/([^/]+(?:/[^/]+)?)$", url or "")
        if match:
            arxiv_id = match.group(1)

        published_str = entry.findtext("atom:published", namespaces=_ARXIV_NS)
        published_at  = _parse_arxiv_date(published_str) if published_str else None
        cats = [c.get("term", "") for c in entry.findall("atom:category", _ARXIV_NS)]
        content = _paper_content(title, authors, published_at.year if published_at else None, "arXiv", abstract)

        docs.append(RetrievedDocument(
            content=content,
            metadata=DocumentMetadata(
                source_method=RetrievalMethod.ACADEMIC,
                domain="arxiv.org",
                url=url,
                title=title,
                author=authors or None,
                published_at=published_at,
                content_type=ContentType.PLAIN_TEXT,
                word_count=len(content.split()),
                extra={"provider": "arxiv", "arxiv_id": arxiv_id, "pdf_url": pdf_url, "categories": cats, "rank": rank},
            ),
            relevance_score=round(1.0 / (1 + rank), 4),
        ))
    return docs


async def _arxiv_search(
    query: str,
    max_results: int = 5,
    categories: Optional[list[str]] = None,
) -> list[RetrievedDocument]:
    docs: list[RetrievedDocument] = []
    for mode in ("and", "or"):           # strict first; broaden if nothing matched
        params = {
            "search_query": build_arxiv_query(query, categories, mode),
            "start":        0,
            "max_results":  max(1, min(max_results, 50)),
            "sortBy":       "relevance",
            "sortOrder":    "descending",
        }
        resp = await http.request(
            "arxiv", "GET", _ARXIV_BASE, params=params,
            timeout=cfg.academic_timeout_s, rate_limiter=_arxiv_limiter, max_retries=2,
        )
        docs = _parse_arxiv(resp.text)
        if docs:
            break
    logger.info("arXiv: %d results for %r", len(docs), query)
    return docs


# ── De-duplication ────────────────────────────────────────────────────────────

def _title_key(doc: RetrievedDocument) -> str:
    return re.sub(r"[^a-z0-9]", "", (doc.metadata.title or "").lower())


def _dedup_academic(docs: list[RetrievedDocument]) -> list[RetrievedDocument]:
    seen: set[str] = set()
    unique = []
    for doc in docs:
        key = _title_key(doc)
        if key and key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


# ── Public interface ──────────────────────────────────────────────────────────

async def academic_search(
    query: str,
    max_results: int = 5,
    providers: Optional[list[str]] = None,
    arxiv_categories: Optional[list[str]] = None,
) -> list[RetrievedDocument]:
    """
    Run Semantic Scholar + arXiv in parallel, de-duplicate, interleave by rank.
    Never raises — failing providers produce user-facing notices instead.
    """
    allowed = {p.lower() for p in providers} if providers else {"semantic_scholar", "arxiv"}
    names, tasks = [], []
    if "semantic_scholar" in allowed:
        names.append("semantic_scholar")
        tasks.append(_semantic_scholar_search(query, max_results))
    if "arxiv" in allowed:
        names.append("arxiv")
        tasks.append(_arxiv_search(query, max_results, categories=arxiv_categories))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    per_provider: list[list[RetrievedDocument]] = []
    for name, result in zip(names, results):
        if isinstance(result, ProviderError):
            notice_from_error(result)
        elif isinstance(result, BaseException):
            logger.warning("Academic provider %s crashed: %s", name, result)
        else:
            per_provider.append(result)

    interleaved: list[RetrievedDocument] = []
    for i in range(max((len(lst) for lst in per_provider), default=0)):
        for lst in per_provider:
            if i < len(lst):
                interleaved.append(lst[i])

    deduped = _dedup_academic(interleaved)
    logger.info("Academic search: %d unique results for %r", len(deduped), query)
    return deduped
