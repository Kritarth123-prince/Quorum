"""
core/errors.py

A single error model for every external provider (search APIs, academic APIs, LLMs).

Every failure is classified into an ErrorKind so the platform can decide what to do:

  QUOTA        credits / plan limit exhausted   → stop using the provider for a long cooldown
  AUTH         invalid, revoked or unpermitted key → stop using it until the key is fixed
  RATE_LIMIT   too many requests                → short cooldown, honour Retry-After
  TIMEOUT / NETWORK / SERVER                    → retry with backoff, then fall back
  BAD_REQUEST  invalid parameters / unknown model → configuration problem, fall back
  REFUSAL      the model declined the request   → fall back to the next provider
"""

from __future__ import annotations

import json
import re
from email.utils import parsedate_to_datetime
from enum import Enum
from typing import Any, Iterable, Mapping, Optional


class ErrorKind(str, Enum):
    AUTH           = "auth"
    QUOTA          = "quota"
    RATE_LIMIT     = "rate_limit"
    TIMEOUT        = "timeout"
    NETWORK        = "network"
    SERVER         = "server"
    BAD_REQUEST    = "bad_request"
    REFUSAL        = "refusal"
    NOT_CONFIGURED = "not_configured"
    UNAVAILABLE    = "unavailable"     # skipped because the provider is cooling down
    EMPTY          = "empty"
    UNKNOWN        = "unknown"


RETRYABLE_KINDS = frozenset({ErrorKind.RATE_LIMIT, ErrorKind.TIMEOUT, ErrorKind.NETWORK, ErrorKind.SERVER})

_DISPLAY_NAMES = {
    "tavily": "Tavily", "serpapi": "SerpAPI", "duckduckgo": "DuckDuckGo",
    "semantic_scholar": "Semantic Scholar", "arxiv": "arXiv",
    "anthropic": "Anthropic (Claude)", "openai": "OpenAI", "groq": "Groq",
    "gemini": "Google Gemini", "openrouter": "OpenRouter", "deepseek": "DeepSeek",
    "ollama": "Ollama", "qdrant": "Qdrant", "redis": "Redis",
}


def display_name(provider: str) -> str:
    return _DISPLAY_NAMES.get(provider, provider)


class ProviderError(Exception):
    """A classified failure from an external provider."""

    def __init__(
        self,
        provider: str,
        kind: ErrorKind,
        message: str = "",
        status_code: Optional[int] = None,
        retry_after: Optional[float] = None,
    ):
        self.provider    = provider
        self.kind        = kind
        self.message     = redact(message or kind.value)[:400]
        self.status_code = status_code
        self.retry_after = retry_after
        super().__init__(self.__str__())

    @property
    def retryable(self) -> bool:
        return self.kind in RETRYABLE_KINDS

    def __str__(self) -> str:
        code = f" (HTTP {self.status_code})" if self.status_code else ""
        return f"{self.provider}: {self.kind.value}{code} — {self.message}"

    @property
    def user_message(self) -> str:
        """Plain-English explanation suitable for showing in the UI."""
        name = display_name(self.provider)
        code = f" (HTTP {self.status_code})" if self.status_code else ""
        if self.kind == ErrorKind.QUOTA:
            return (f"{name}: API credits or plan quota exhausted{code}. "
                    f"Top up / upgrade the plan or wait for the quota to reset.")
        if self.kind == ErrorKind.AUTH:
            return f"{name}: API key rejected{code}. Check the key in .env."
        if self.kind == ErrorKind.RATE_LIMIT:
            wait = f" Retry after ~{int(self.retry_after)}s." if self.retry_after else ""
            return f"{name}: rate limit reached{code}.{wait}"
        if self.kind == ErrorKind.TIMEOUT:
            return f"{name}: request timed out."
        if self.kind == ErrorKind.NETWORK:
            return f"{name}: network error — {self.message}"
        if self.kind == ErrorKind.SERVER:
            return f"{name}: service error{code} — temporarily unavailable."
        if self.kind == ErrorKind.BAD_REQUEST:
            return f"{name}: request rejected{code} — {self.message}"
        if self.kind == ErrorKind.REFUSAL:
            return f"{name}: the model declined this request."
        if self.kind == ErrorKind.UNAVAILABLE:
            return f"{name}: skipped — {self.message}"
        if self.kind == ErrorKind.NOT_CONFIGURED:
            return f"{name}: not configured."
        return f"{name}: {self.message}"


class AllProvidersFailedError(Exception):
    """Raised when every provider in a fallback chain failed or was skipped."""

    def __init__(self, category: str, errors: list[ProviderError]):
        self.category = category
        self.errors   = errors
        detail = "; ".join(str(e) for e in errors) or "no provider configured"
        super().__init__(f"All {category} providers failed: {detail}")


# ── Secret redaction ──────────────────────────────────────────────────────────

_SECRET_PARAM_RE = re.compile(
    r"(?i)((?:api[_-]?key|apikey|key|token|access_token|authorization)\s*[=:]\s*)([^\s&'\",;]+)"
)
_BEARER_RE = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}")
_KNOWN_KEY_RE = re.compile(r"\b(tvly-[A-Za-z0-9\-_]{6,}|sk-[A-Za-z0-9\-_]{10,}|gsk_[A-Za-z0-9]{10,}|AIza[0-9A-Za-z\-_]{20,})")

_extra_secrets: list[str] = []


def register_secrets(secrets: Iterable[str]) -> None:
    """Register literal secret values so redact() masks them anywhere they appear."""
    for s in secrets:
        if s and len(s) >= 6 and s not in _extra_secrets:
            _extra_secrets.append(s)


def redact(text: Any) -> str:
    out = str(text)
    for secret in _extra_secrets:
        out = out.replace(secret, "***")
    out = _SECRET_PARAM_RE.sub(r"\1***", out)
    out = _BEARER_RE.sub(r"\1***", out)
    out = _KNOWN_KEY_RE.sub("***", out)
    return out


# ── HTTP classification ───────────────────────────────────────────────────────

_QUOTA_PATTERNS = (
    "credit balance", "insufficient_quota", "insufficient quota", "exceeded your current quota",
    "run out of searches", "out of searches", "plan limit", "usage limit", "monthly limit",
    "insufficient balance", "insufficient credits", "not enough credits", "credits exhausted",
    "payment required", "billing", "upgrade your plan", "quota exceeded", "exceeds your plan",
    "pay-as-you-go limit", "spending limit", "budget",
)
_RATE_PATTERNS = ("rate limit", "rate_limit", "too many requests", "per minute", "hourly", "slow down", "throughput")
_AUTH_PATTERNS = ("invalid api key", "invalid_api_key", "incorrect api key", "api key not valid",
                  "unauthorized", "authentication", "invalid x-api-key", "api_key_invalid")


def extract_error_message(body: Any) -> str:
    """Pull a human-readable message out of the many error-body shapes APIs use."""
    if body is None:
        return ""
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8", errors="replace")
    if isinstance(body, str):
        stripped = body.strip()
        if not stripped:
            return ""
        try:
            body = json.loads(stripped)
        except (ValueError, TypeError):
            return re.sub(r"\s+", " ", stripped)[:400]

    if isinstance(body, list):                       # e.g. Gemini: [{"error": {...}}]
        return "; ".join(filter(None, (extract_error_message(item) for item in body[:3])))[:400]

    if isinstance(body, dict):
        for key in ("error", "detail", "message", "errors", "error_message", "msg"):
            if key not in body:
                continue
            value = body[key]
            if isinstance(value, str):
                return value[:400]
            if isinstance(value, dict):
                inner = value.get("message") or value.get("error") or value.get("detail")
                code  = value.get("code") or value.get("type") or value.get("status")
                text  = inner if isinstance(inner, str) else json.dumps(value)[:300]
                return f"{text} [{code}]"[:400] if code and str(code) not in text else text[:400]
            if isinstance(value, list):
                return extract_error_message(value)
        return json.dumps(body)[:400]
    return str(body)[:400]


def parse_retry_after(headers: Optional[Mapping[str, str]], message: str = "") -> Optional[float]:
    if headers:
        lowered = {k.lower(): v for k, v in headers.items()}
        for key in ("retry-after", "x-ratelimit-reset-after", "x-ratelimit-reset-requests"):
            raw = lowered.get(key)
            if not raw:
                continue
            try:
                return max(0.0, float(raw))
            except ValueError:
                pass
            match = re.match(r"^\s*(?:(\d+)m)?\s*([\d.]+)s\s*$", raw)        # "1m30s" / "12.5s"
            if match:
                return float(match.group(1) or 0) * 60 + float(match.group(2))
            try:
                dt = parsedate_to_datetime(raw)
                from datetime import datetime, timezone
                return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError):
                pass
    match = re.search(r"retry (?:in|after) ([\d.]+)\s*s", message, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _contains(text: str, patterns: Iterable[str]) -> bool:
    return any(p in text for p in patterns)


def classify_http_error(
    provider: str,
    status_code: int,
    body: Any = None,
    headers: Optional[Mapping[str, str]] = None,
) -> ProviderError:
    """Map an HTTP error response to a ProviderError with the right ErrorKind."""
    message     = extract_error_message(body) or f"HTTP {status_code}"
    lowered     = message.lower()
    retry_after = parse_retry_after(headers, message)

    if status_code in (402, 432, 433):                 # 432/433 = Tavily plan / pay-as-you-go limit
        kind = ErrorKind.QUOTA
    elif status_code == 429:
        if _contains(lowered, _QUOTA_PATTERNS) and not (retry_after is not None and retry_after <= 120):
            kind = ErrorKind.QUOTA
        else:
            kind = ErrorKind.RATE_LIMIT
    elif status_code in (401, 407):
        kind = ErrorKind.AUTH
    elif status_code == 403:
        kind = ErrorKind.QUOTA if _contains(lowered, _QUOTA_PATTERNS) else ErrorKind.AUTH
    elif status_code in (408, 504, 524):
        kind = ErrorKind.TIMEOUT
    elif status_code == 434 or status_code >= 500:     # 434 = Tavily flex capacity; 529 = overloaded
        kind = ErrorKind.SERVER
    elif status_code in (400, 404, 405, 409, 413, 415, 422):
        if _contains(lowered, _QUOTA_PATTERNS):        # e.g. Anthropic: "Your credit balance is too low"
            kind = ErrorKind.QUOTA
        elif _contains(lowered, _AUTH_PATTERNS):
            kind = ErrorKind.AUTH
        else:
            kind = ErrorKind.BAD_REQUEST
    else:
        kind = ErrorKind.UNKNOWN

    return ProviderError(provider, kind, message, status_code=status_code, retry_after=retry_after)


def classify_body_error(provider: str, message: str) -> ProviderError:
    """Classify an error reported inside an HTTP 200 body (SerpAPI does this)."""
    lowered = (message or "").lower()
    if _contains(lowered, _QUOTA_PATTERNS):
        kind = ErrorKind.QUOTA
    elif _contains(lowered, _AUTH_PATTERNS):
        kind = ErrorKind.AUTH
    elif _contains(lowered, _RATE_PATTERNS):
        kind = ErrorKind.RATE_LIMIT
    else:
        kind = ErrorKind.UNKNOWN
    return ProviderError(provider, kind, message)
