"""
core/text.py

Dependency-free text helpers shared by retrieval, ranking and the agents:
tokenising, sentence/passage splitting, query-focused passage selection,
markdown/boilerplate cleaning and URL normalisation.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

STOPWORDS = frozenset({
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "for", "of", "and", "or", "but",
    "with", "this", "that", "are", "was", "were", "be", "been", "have", "has", "had", "not",
    "by", "from", "as", "if", "its", "which", "their", "what", "how", "why", "when", "where",
    "who", "whom", "does", "do", "did", "can", "could", "should", "would", "will", "about",
    "into", "than", "then", "there", "these", "those", "they", "them", "you", "your", "our",
    "we", "i", "me", "my", "vs", "versus", "between", "best", "most", "more", "also", "such",
    "any", "all", "some", "other", "use", "using", "used", "get", "via", "per", "so", "no",
})

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_\-\.]*[a-z0-9]|[a-z0-9]")


def tokenise(text: str, keep_stopwords: bool = False) -> list[str]:
    tokens = _TOKEN_RE.findall((text or "").lower())
    if keep_stopwords:
        return tokens
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def keywords(text: str, limit: int = 8) -> list[str]:
    """Distinct content words in first-seen order."""
    seen: list[str] = []
    for tok in tokenise(text):
        if tok not in seen:
            seen.append(tok)
        if len(seen) >= limit:
            break
    return seen


# ── Cleaning ──────────────────────────────────────────────────────────────────

_MD_IMAGE_RE  = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK_RE   = re.compile(r"\[([^\]]+)\]\((?:[^()]|\([^)]*\))*\)")
_BARE_URL_RE  = re.compile(r"https?://\S{60,}")
_MULTI_NL_RE  = re.compile(r"\n{3,}")
_MULTI_SP_RE  = re.compile(r"[ \t ]{2,}")


def clean_text(text: str) -> str:
    """Strip markdown images/links, very long URLs and menu-like fragments."""
    if not text:
        return ""
    text = _MD_IMAGE_RE.sub("", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _BARE_URL_RE.sub("", text)
    lines: list[str] = []
    for raw in text.splitlines():
        line = _MULTI_SP_RE.sub(" ", raw).strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        words = line.split()
        # Drop navigation fragments: very short lines that are not headings or list items.
        if len(words) < 4 and not line.startswith(("#", "-", "*", "|")) and not line.endswith((".", ":", "?")):
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    return _MULTI_NL_RE.sub("\n\n", cleaned).strip()


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    space = cut.rfind(" ")
    if space > max_chars * 0.8:
        cut = cut[:space]
    return cut.rstrip() + "…"


# ── Sentences & passages ──────────────────────────────────────────────────────

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for para in re.split(r"\n{2,}|\n(?=[#\-*|])", text or ""):
        para = para.strip()
        if not para:
            continue
        out.extend(s.strip() for s in _SENTENCE_RE.split(para) if s.strip())
    return out


def split_passages(text: str, size: int = 700, overlap: int = 120) -> list[str]:
    """Paragraph-aware windows of roughly `size` characters."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    passages: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > size:
            if current:
                passages.append(current)
                current = ""
            step = max(1, size - overlap)
            for start in range(0, len(para), step):
                chunk = para[start:start + size].strip()
                if chunk:
                    passages.append(chunk)
                if start + size >= len(para):
                    break
            continue
        if current and len(current) + len(para) + 2 > size:
            passages.append(current)
            tail = current[-overlap:] if overlap else ""
            current = (tail + "\n\n" + para) if tail else para
        else:
            current = (current + "\n\n" + para) if current else para
    if current:
        passages.append(current)
    return passages


def _score_passages(query_tokens: list[str], passages: list[str]) -> list[float]:
    """Light BM25-style scoring of passages against query tokens (no numpy needed)."""
    if not query_tokens or not passages:
        return [0.0] * len(passages)
    tokenised = [tokenise(p) for p in passages]
    n = len(passages)
    avg_len = sum(len(t) for t in tokenised) / max(n, 1) or 1.0
    df = Counter()
    for toks in tokenised:
        df.update(set(toks))
    scores = []
    k1, b = 1.4, 0.75
    for toks in tokenised:
        tf = Counter(toks)
        length = len(toks) or 1
        score = 0.0
        for q in set(query_tokens):
            if q not in tf:
                continue
            idf = math.log(1 + (n - df[q] + 0.5) / (df[q] + 0.5))
            freq = tf[q]
            score += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * length / avg_len))
        scores.append(score)
    return scores


def best_passages(query: str, text: str, max_chars: int = 1500, passage_size: int = 600) -> str:
    """
    Return the parts of `text` most relevant to `query`, up to `max_chars`,
    kept in their original order so the excerpt still reads naturally.
    """
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    passages = split_passages(text, size=passage_size, overlap=80)
    scores = _score_passages(tokenise(query), passages)
    # Always give the opening passage a small boost (titles / summaries live there).
    if scores:
        scores[0] += 0.5
    order = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)
    chosen: list[int] = []
    used = 0
    for idx in order:
        length = len(passages[idx]) + 5
        if used + length > max_chars and chosen:
            continue
        chosen.append(idx)
        used += length
        if used >= max_chars:
            break
    chosen.sort()
    return truncate("\n…\n".join(passages[i] for i in chosen), max_chars)


_BULLET_PREFIX_RE = re.compile(r"^[\s\-\*•·▪●◦>]+")
_ANSWER_STARTERS = ("yes,", "no,", "yes.", "no.", "yes ", "no ", "click ", "read more", "sign up", "subscribe")


_DATE_PREFIX_RE = re.compile(r"^(?:[A-Z][a-z]{2,8}\.? \d{1,2}, \d{4}|\d{1,2} [A-Z][a-z]{2,8} \d{4}|\d+ (?:days?|hours?) ago)\s*[·\-—|]\s*")


def clean_sentence(sentence: str) -> str:
    sentence = _BULLET_PREFIX_RE.sub("", sentence or "").strip()
    sentence = _DATE_PREFIX_RE.sub("", sentence)          # "Sep 22, 2026 · …" search-snippet prefixes
    sentence = sentence.replace("**", "").replace("__", "")
    return re.sub(r"\s+", " ", sentence)


def is_statement(sentence: str, min_words: int = 8) -> bool:
    """A self-contained declarative sentence (not a heading, question, fragment or call to action)."""
    words = sentence.split()
    if len(words) < min_words or len(sentence) > 450:
        return False
    if sentence.endswith(("?", ":")) or sentence.startswith(("#", "|", "(")):
        return False
    if sentence.lower().startswith(_ANSWER_STARTERS):
        return False
    capitalised = sum(1 for w in words if w[:1].isupper())
    return capitalised / len(words) < 0.6          # Title Case lines are headings


def relevant_sentences(query: str, text: str, limit: int = 2, min_len: int = 40) -> list[str]:
    """Declarative sentences from `text` that share the most content words with `query`."""
    q = set(tokenise(query))
    scored = []
    for idx, raw in enumerate(split_sentences(text)):
        sent = clean_sentence(raw)
        if len(sent) < min_len or not is_statement(sent):
            continue
        overlap = len(q & set(tokenise(sent)))
        scored.append((overlap, -idx, sent))
    scored.sort(reverse=True)
    return [s for overlap, _, s in scored[:limit] if overlap > 0] or [s for _, _, s in scored[:1]]


def bigrams(text: str) -> set[tuple[str, str]]:
    toks = tokenise(text)
    return set(zip(toks, toks[1:]))


# ── URLs ──────────────────────────────────────────────────────────────────────

_TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid", "ref", "ref_src"}


def normalise_url(url: str) -> str:
    """Canonical form for de-duplication: lowercase host, no fragment/tracking params/trailing slash."""
    if not url:
        return ""
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    query = urlencode([(k, v) for k, v in parse_qsl(parsed.query) if k.lower() not in _TRACKING_PARAMS])
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme.lower() or "https", host, path, "", query, ""))


def domain_of(url: str) -> str:
    host = urlparse(url or "").netloc.lower()
    return host[4:] if host.startswith("www.") else host
