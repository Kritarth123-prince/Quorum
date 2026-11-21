"""
agents/sources.py

The numbered source list shared by every agent.

After retrieval, the top documents become sources [1]…[N]. The verifier, the
answer writer and the report all refer to these numbers, so a citation always
points at the document it came from.
"""

from __future__ import annotations

import re
from typing import Any

from retrieval_pipeline.core.text import best_passages, truncate
from retrieval_pipeline.models.schemas import RetrievalMethod, RetrievedDocument

# ── Domain credibility heuristics ─────────────────────────────────────────────

_HIGH_CRED = {
    "arxiv.org", "semanticscholar.org", "nature.com", "science.org", "acm.org", "ieee.org",
    "springer.com", "sciencedirect.com", "nih.gov", "ncbi.nlm.nih.gov", "who.int", "nist.gov",
    "github.com", "docs.python.org", "developer.mozilla.org", "huggingface.co", "openai.com",
    "anthropic.com", "aws.amazon.com", "cloud.google.com", "learn.microsoft.com", "postgresql.org",
    "kubernetes.io", "reuters.com", "apnews.com", "bbc.co.uk", "bbc.com", "ft.com", "economist.com",
}
_MEDIUM_HIGH = {"wikipedia.org", "stackoverflow.com", "nytimes.com", "theguardian.com", "techcrunch.com", "wired.com"}
_LOW_CRED = {
    "reddit.com", "quora.com", "medium.com", "substack.com", "twitter.com", "x.com", "facebook.com",
    "linkedin.com", "pinterest.com", "tiktok.com", "instagram.com", "youtube.com",
}


def domain_credibility(domain: str, method: str = "") -> float:
    """0–1 prior for how reliable a source domain usually is (a heuristic, not a verdict)."""
    d = (domain or "").lower()
    if d.startswith("www."):
        d = d[4:]

    def matches(pool: set[str]) -> bool:
        return d in pool or any(d.endswith("." + p) for p in pool)

    if method == RetrievalMethod.ACADEMIC.value or matches(_HIGH_CRED):
        return 0.9
    if d.endswith((".gov", ".gov.uk", ".edu", ".ac.uk", ".int")) or d.startswith(("docs.", "developer.")):
        return 0.85
    if matches(_MEDIUM_HIGH):
        return 0.75
    if matches(_LOW_CRED):
        return 0.4
    return 0.6


# ── Source list ───────────────────────────────────────────────────────────────

def _published(doc: RetrievedDocument) -> str | None:
    if doc.metadata.published_at:
        return doc.metadata.published_at.date().isoformat()
    return doc.metadata.extra.get("published")


def build_sources(query: str, docs: list[RetrievedDocument], max_sources: int, chars_per_source: int) -> list[dict]:
    sources = []
    for i, doc in enumerate(docs[:max_sources], start=1):
        meta = doc.metadata
        method = meta.source_method.value
        original = meta.extra.get("original_method") or method
        sources.append({
            "id":          i,
            "title":       (meta.title or meta.domain or "Untitled").strip(),
            "url":         meta.url or "",
            "domain":      meta.domain,
            "method":      method,
            "provider":    meta.extra.get("provider"),
            "authors":     meta.author,
            "published":   _published(doc),
            "relevance":   round(float(doc.rerank_score if doc.rerank_score is not None else (doc.relevance_score or 0.0)), 3),
            "credibility": domain_credibility(meta.domain, original),
            "snippet":     truncate(meta.extra.get("snippet") or doc.content, 300),
            "excerpt":     best_passages(query, doc.content, max_chars=chars_per_source),
            "citations":   meta.extra.get("citation_count"),
        })
    return sources


def format_sources_for_prompt(sources: list[dict]) -> str:
    blocks = []
    for s in sources:
        header = f"[{s['id']}] {s['title']} — {s['domain']}"
        if s.get("published"):
            header += f" ({s['published']})"
        if s.get("method") == "academic":
            header += " [academic paper]"
        blocks.append(f"{header}\n{s['excerpt']}")
    return "\n\n---\n\n".join(blocks)


def public_source(s: dict) -> dict:
    """Source as returned to API clients (without the long prompt excerpt)."""
    return {k: v for k, v in s.items() if k != "excerpt"}


# ── Citations in generated text ───────────────────────────────────────────────

_CITE_GROUP_RE = re.compile(r"\[(\s*\d+\s*(?:[,;–-]\s*\d+\s*)*)\]")


def _expand(group: str) -> list[int]:
    ids: list[int] = []
    for part in re.split(r"[,;]", group):
        part = part.strip()
        if re.fullmatch(r"\d+\s*[–-]\s*\d+", part):
            a, b = (int(x) for x in re.split(r"[–-]", part))
            if 0 < a <= b and b - a <= 20:
                ids.extend(range(a, b + 1))
        elif part.isdigit():
            ids.append(int(part))
    return ids


def sanitize_citations(text: str, n_sources: int) -> str:
    """Normalise [1, 2] / [1-3] to [1][2][3] and drop references to sources that don't exist."""
    def repl(match: re.Match) -> str:
        ids = [i for i in _expand(match.group(1)) if 1 <= i <= n_sources]
        seen: list[int] = []
        for i in ids:
            if i not in seen:
                seen.append(i)
        return "".join(f"[{i}]" for i in seen)
    cleaned = _CITE_GROUP_RE.sub(repl, text or "")
    return re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)


def cited_ids(text: str) -> list[int]:
    out: list[int] = []
    for match in _CITE_GROUP_RE.finditer(text or ""):
        for i in _expand(match.group(1)):
            if i not in out:
                out.append(i)
    return out


def citations_for(ids: list[Any], sources: list[dict]) -> list[dict]:
    by_id = {s["id"]: s for s in sources}
    out = []
    for raw in ids or []:
        try:
            sid = int(raw)
        except (TypeError, ValueError):
            continue
        s = by_id.get(sid)
        if s and not any(c["id"] == sid for c in out):
            out.append({"id": sid, "title": s["title"], "url": s["url"], "domain": s["domain"],
                        "relevance": s["relevance"], "excerpt": s["snippet"][:200]})
    return out
