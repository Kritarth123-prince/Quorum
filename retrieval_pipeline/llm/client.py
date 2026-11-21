"""
llm/client.py

Provider-agnostic async LLM client with an automatic fallback chain.

Providers (order: LLM_PROVIDER_ORDER; only those with a key — or a reachable
Ollama — take part):
  anthropic   Claude via the official Anthropic SDK (Messages API)
  openai      OpenAI Chat Completions
  groq        OpenAI-compatible
  gemini      Google's OpenAI-compatible endpoint
  openrouter  OpenAI-compatible
  deepseek    OpenAI-compatible
  ollama      local models via Ollama's OpenAI-compatible endpoint

Error handling (shared with the search providers):
  - "credit balance too low" / insufficient_quota / HTTP 402 → QUOTA: provider
    paused for QUOTA_COOLDOWN_S, request moves to the next provider, the user
    sees a notice ("Anthropic (Claude): API credits … exhausted. Switched to Groq.")
  - invalid key → AUTH; rate limits honour Retry-After; 5xx/timeouts are retried.
  - If every provider fails, AllProvidersFailedError is raised and the agents
    fall back to their rule-based / extractive logic, so a research run never dies.

Public API:
  get_llm().complete(prompt, system=…)          → LLMResponse
  get_llm().complete_json(prompt, schema=…)     → dict
  get_llm().stream(prompt, system=…)            → async iterator of StreamChunk
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator, Optional

import httpx

from retrieval_pipeline.config import get_settings
from retrieval_pipeline.core import http
from retrieval_pipeline.core.errors import (
    AllProvidersFailedError,
    ErrorKind,
    ProviderError,
    classify_http_error,
    display_name,
    redact,
)
from retrieval_pipeline.core.provider_health import notice_from_error, registry

logger = logging.getLogger(__name__)
cfg = get_settings()

_FALLBACK_BETA = "server-side-fallback-2026-07-01"


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    text:        str
    provider:    str
    model:       str
    stop_reason: Optional[str] = None
    usage:       dict = field(default_factory=dict)
    latency_ms:  float = 0.0


@dataclass
class StreamChunk:
    text:     str = ""
    provider: str = ""
    model:    str = ""
    reset:    bool = False   # True: the previous provider failed mid-stream; discard partial text


class LLMUnavailableError(AllProvidersFailedError):
    """No LLM provider is configured or currently usable."""


# ── Per-request LLM call log (for the agent trace) ────────────────────────────

_llm_calls: contextvars.ContextVar[Optional[list[dict]]] = contextvars.ContextVar("llm_calls", default=None)


@contextmanager
def collect_llm_calls() -> Iterator[list[dict]]:
    bucket: list[dict] = []
    token = _llm_calls.set(bucket)
    try:
        yield bucket
    finally:
        _llm_calls.reset(token)


def last_call_provider(task: str) -> Optional[str]:
    """Provider that served the most recent successful call for `task` in this request."""
    for call in reversed(_llm_calls.get() or []):
        if call["task"] == task and call["ok"]:
            return call["provider"]
    return None


def _log_call(task: str, provider: str, model: str, latency_ms: float, usage: dict, ok: bool, error: str = "") -> None:
    bucket = _llm_calls.get()
    if bucket is not None:
        bucket.append({
            "task": task, "provider": provider, "model": model, "ok": ok,
            "latency_ms": round(latency_ms, 1), "usage": usage, "error": error,
        })


# ── JSON extraction ───────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> Optional[Any]:
    """Parse JSON from model output, tolerating code fences and surrounding prose."""
    if not text:
        return None
    candidate = _FENCE_RE.sub("", text.strip())
    try:
        return json.loads(candidate)
    except ValueError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        while start != -1:
            depth, in_str, escape = 0, False, False
            for i in range(start, len(candidate)):
                ch = candidate[i]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(candidate[start:i + 1])
                        except ValueError:
                            break
            start = candidate.find(opener, start + 1)
    return None


# ── Providers ─────────────────────────────────────────────────────────────────

class BaseProvider:
    name: str = ""
    model: str = ""
    token_floor: int = 1024
    token_cap: int = 16000

    def tokens(self, requested: int) -> int:
        return max(self.token_floor, min(requested, self.token_cap))

    async def reachable(self) -> bool:
        return True

    async def complete(self, system: str, messages: list[dict], max_tokens: int,
                       json_schema: Optional[dict], effort: Optional[str]) -> LLMResponse:
        raise NotImplementedError

    def stream(self, system: str, messages: list[dict], max_tokens: int,
               effort: Optional[str]) -> AsyncIterator[str]:
        raise NotImplementedError


class AnthropicProvider(BaseProvider):
    """
    Claude via the official Anthropic SDK.
    - Opus 5.5 always uses adaptive thinking, so `thinking` is omitted and depth is
      controlled with output_config.effort (set explicitly; the model default is medium).
    - Structured outputs (output_config.format) guarantee valid JSON for agent calls.
    - Server-side refusal fallbacks ("default" mode) are opted in by default; set
      ANTHROPIC_FALLBACKS=false to disable.
    - The SDK retries 408/409/429/5xx itself (LLM_MAX_RETRIES).
    """

    name = "anthropic"
    token_floor = 16000      # thinking tokens count towards max_tokens
    token_cap = 64000

    def __init__(self, api_key: str, model: str):
        import anthropic
        self._anthropic = anthropic
        self.model = model
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key, timeout=float(cfg.llm_timeout_s), max_retries=cfg.llm_max_retries,
        )
        self._use_fallbacks = cfg.anthropic_fallbacks
        self._use_effort = True
        self._use_schema = True

    def _params(self, system: str, messages: list[dict], max_tokens: int,
                json_schema: Optional[dict], effort: Optional[str]) -> dict:
        params: dict[str, Any] = {
            "model":      self.model,
            "max_tokens": self.tokens(max_tokens),
            "messages":   messages,
        }
        if system:
            params["system"] = system
        output_config: dict[str, Any] = {}
        if self._use_effort:
            output_config["effort"] = effort or cfg.anthropic_effort
        if json_schema and self._use_schema:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        if output_config:
            params["output_config"] = output_config
        if self._use_fallbacks:
            params["betas"] = [_FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    def _to_provider_error(self, exc: Exception) -> ProviderError:
        a = self._anthropic
        if isinstance(exc, a.APITimeoutError):
            return ProviderError(self.name, ErrorKind.TIMEOUT, "request timed out")
        if isinstance(exc, a.APIConnectionError):
            return ProviderError(self.name, ErrorKind.NETWORK, redact(exc))
        if isinstance(exc, a.APIStatusError):
            headers = exc.response.headers if getattr(exc, "response", None) is not None else None
            body = getattr(exc, "body", None) or getattr(exc, "message", "") or str(exc)
            return classify_http_error(self.name, exc.status_code, body, headers)
        return ProviderError(self.name, ErrorKind.UNKNOWN, f"{type(exc).__name__}: {redact(exc)}")

    def _adapt(self, err: ProviderError) -> bool:
        """On a 400 caused by an optional feature, switch it off and retry."""
        if err.kind != ErrorKind.BAD_REQUEST:
            return False
        msg = err.message.lower()
        if self._use_fallbacks and ("fallback" in msg or "beta" in msg):
            self._use_fallbacks = False
        elif self._use_effort and "effort" in msg:
            self._use_effort = False
        elif self._use_schema and ("output_config" in msg or "format" in msg or "schema" in msg):
            self._use_schema = False
        else:
            return False
        logger.info("Anthropic: retrying without an unsupported option (%s)", err.message[:120])
        return True

    @staticmethod
    def _text(message: Any) -> str:
        return "".join(getattr(b, "text", "") for b in message.content if getattr(b, "type", "") == "text").strip()

    async def complete(self, system, messages, max_tokens, json_schema, effort) -> LLMResponse:
        for _ in range(4):
            t0 = time.perf_counter()
            try:
                msg = await self._client.beta.messages.create(**self._params(system, messages, max_tokens, json_schema, effort))
            except Exception as exc:
                err = self._to_provider_error(exc)
                if self._adapt(err):
                    continue
                raise err
            if msg.stop_reason == "refusal":       # check before reading content
                raise ProviderError(self.name, ErrorKind.REFUSAL, "the model declined the request")
            text = self._text(msg)
            if not text:
                raise ProviderError(self.name, ErrorKind.EMPTY, f"empty response (stop_reason={msg.stop_reason})")
            usage = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
            return LLMResponse(text=text, provider=self.name, model=getattr(msg, "model", self.model),
                               stop_reason=msg.stop_reason, usage=usage,
                               latency_ms=(time.perf_counter() - t0) * 1000)
        raise ProviderError(self.name, ErrorKind.BAD_REQUEST, "request rejected after removing optional features")

    async def stream(self, system, messages, max_tokens, effort) -> AsyncIterator[str]:
        for _ in range(4):
            emitted = False
            try:
                async with self._client.beta.messages.stream(**self._params(system, messages, max_tokens, None, effort)) as stream:
                    async for text in stream.text_stream:
                        emitted = True
                        yield text
                    final = await stream.get_final_message()
                if final.stop_reason == "refusal":
                    raise ProviderError(self.name, ErrorKind.REFUSAL, "the model declined the request")
                return
            except ProviderError:
                raise
            except Exception as exc:
                err = self._to_provider_error(exc)
                if not emitted and self._adapt(err):
                    continue
                raise err


@dataclass
class _CompatSpec:
    name:          str
    base_url:      str
    api_key:       Optional[str]
    model:         str
    max_param:     str = "max_tokens"
    token_floor:   int = 1024
    token_cap:     int = 16000
    json_mode:     bool = True
    extra_headers: dict = field(default_factory=dict)


class OpenAICompatibleProvider(BaseProvider):
    """Any OpenAI-style /chat/completions endpoint (OpenAI, Groq, Gemini, OpenRouter, DeepSeek, Ollama)."""

    def __init__(self, spec: _CompatSpec):
        self.spec = spec
        self.name = spec.name
        self.model = spec.model
        self.token_floor = spec.token_floor
        self.token_cap = spec.token_cap
        self._json_mode = spec.json_mode
        self._reachable_cache: tuple[float, bool] = (0.0, False)

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json", **self.spec.extra_headers}
        if self.spec.api_key:
            headers["Authorization"] = f"Bearer {self.spec.api_key}"
        return headers

    def _body(self, system: str, messages: list[dict], max_tokens: int, json_mode: bool, stream: bool) -> dict:
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        body: dict[str, Any] = {"model": self.model, "messages": msgs, self.spec.max_param: self.tokens(max_tokens)}
        # No custom temperature: reasoning models (e.g. GPT-5 family) reject non-default values.
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if stream:
            body["stream"] = True
        return body

    async def reachable(self) -> bool:
        if self.name != "ollama":
            return True
        checked_at, ok = self._reachable_cache
        if time.monotonic() - checked_at < 60:
            return ok
        base = self.spec.base_url.rsplit("/v1", 1)[0]
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                resp = await client.get(f"{base}/api/tags")
                ok = resp.status_code == 200 and any(
                    m.get("name", "").split(":")[0] == self.model.split(":")[0] for m in resp.json().get("models", [])
                )
        except Exception:
            ok = False
        self._reachable_cache = (time.monotonic(), ok)
        return ok

    async def complete(self, system, messages, max_tokens, json_schema, effort) -> LLMResponse:
        json_mode = bool(json_schema) and self._json_mode
        t0 = time.perf_counter()
        try:
            resp = await http.request(
                self.name, "POST", f"{self.spec.base_url}/chat/completions",
                json=self._body(system, messages, max_tokens, json_mode, stream=False),
                headers=self._headers(), timeout=cfg.llm_timeout_s, max_retries=cfg.llm_max_retries, track=False,
            )
        except ProviderError as err:
            if json_mode and err.kind == ErrorKind.BAD_REQUEST and "response_format" in err.message.lower():
                self._json_mode = False
                return await self.complete(system, messages, max_tokens, json_schema, effort)
            raise
        try:
            data = resp.json()
        except ValueError:
            raise ProviderError(self.name, ErrorKind.SERVER, "invalid JSON from provider")
        if isinstance(data, dict) and data.get("error"):
            raise classify_http_error(self.name, 400, data)
        choice = (data.get("choices") or [{}])[0]
        text = ((choice.get("message") or {}).get("content") or "").strip()
        finish = choice.get("finish_reason")
        if finish == "content_filter" and not text:
            raise ProviderError(self.name, ErrorKind.REFUSAL, "blocked by the provider's content filter")
        if not text:
            raise ProviderError(self.name, ErrorKind.EMPTY, f"empty response (finish_reason={finish})")
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text, provider=self.name, model=data.get("model") or self.model, stop_reason=finish,
            usage={"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")},
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    async def stream(self, system, messages, max_tokens, effort) -> AsyncIterator[str]:
        body = self._body(system, messages, max_tokens, json_mode=False, stream=True)
        timeout = httpx.Timeout(cfg.llm_timeout_s, connect=15.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", f"{self.spec.base_url}/chat/completions",
                                         json=body, headers=self._headers()) as resp:
                    if resp.status_code >= 400:
                        raw = await resp.aread()
                        raise classify_http_error(self.name, resp.status_code, raw, resp.headers)
                    emitted = False
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            event = json.loads(payload)
                        except ValueError:
                            continue
                        if event.get("error"):
                            raise classify_http_error(self.name, 500, event)
                        for choice in event.get("choices") or []:
                            token = (choice.get("delta") or {}).get("content")
                            if token:
                                emitted = True
                                yield token
                            if choice.get("finish_reason") == "content_filter" and not emitted:
                                raise ProviderError(self.name, ErrorKind.REFUSAL, "blocked by content filter")
                    if not emitted:
                        raise ProviderError(self.name, ErrorKind.EMPTY, "stream ended without content")
        except httpx.TimeoutException:
            raise ProviderError(self.name, ErrorKind.TIMEOUT, "stream timed out")
        except httpx.TransportError as exc:
            raise ProviderError(self.name, ErrorKind.NETWORK, f"{type(exc).__name__}: {redact(exc)}")


def _build_providers() -> list[BaseProvider]:
    specs: dict[str, Optional[BaseProvider]] = {}

    if cfg.anthropic_api_key:
        try:
            specs["anthropic"] = AnthropicProvider(cfg.anthropic_api_key, cfg.anthropic_model)
        except ImportError:
            logger.warning("ANTHROPIC_API_KEY set but the 'anthropic' package is not installed")
    if cfg.openai_api_key:
        specs["openai"] = OpenAICompatibleProvider(_CompatSpec(
            "openai", cfg.openai_base_url.rstrip("/"), cfg.openai_api_key, cfg.openai_model,
            max_param="max_completion_tokens", token_floor=8000, token_cap=32000))
    if cfg.groq_api_key:
        specs["groq"] = OpenAICompatibleProvider(_CompatSpec(
            "groq", "https://api.groq.com/openai/v1", cfg.groq_api_key, cfg.groq_model,
            max_param="max_completion_tokens", token_floor=4000, token_cap=32000))
    if cfg.gemini_api_key:
        specs["gemini"] = OpenAICompatibleProvider(_CompatSpec(
            "gemini", "https://generativelanguage.googleapis.com/v1beta/openai", cfg.gemini_api_key, cfg.gemini_model,
            token_floor=4000, token_cap=32000))
    if cfg.openrouter_api_key:
        specs["openrouter"] = OpenAICompatibleProvider(_CompatSpec(
            "openrouter", "https://openrouter.ai/api/v1", cfg.openrouter_api_key, cfg.openrouter_model,
            token_floor=2000, token_cap=16000, json_mode=False,
            extra_headers={"HTTP-Referer": "http://localhost", "X-Title": "Quorum"}))
    if cfg.deepseek_api_key:
        specs["deepseek"] = OpenAICompatibleProvider(_CompatSpec(
            "deepseek", "https://api.deepseek.com/v1", cfg.deepseek_api_key, cfg.deepseek_model,
            token_floor=2000, token_cap=8000))
    if cfg.ollama_enabled:
        specs["ollama"] = OpenAICompatibleProvider(_CompatSpec(
            "ollama", cfg.ollama_base_url.rstrip("/") + "/v1", None, cfg.ollama_model,
            token_floor=1024, token_cap=8000))

    order = [p.strip().lower() for p in cfg.llm_provider_order.split(",") if p.strip()]
    order += [name for name in specs if name not in order]
    providers = [specs[name] for name in order if specs.get(name) is not None]

    for name in ("anthropic", "openai", "groq", "gemini", "openrouter", "deepseek", "ollama"):
        registry.register(name, "llm", configured=name in specs)
    return providers


# ── Client with fallback chain ────────────────────────────────────────────────

class LLMClient:
    def __init__(self):
        self.providers = _build_providers()
        logger.info("LLM providers configured: %s", [p.name for p in self.providers] or "none")

    async def _candidates(self, skip: frozenset[str] = frozenset()) -> list[BaseProvider]:
        out = []
        for p in self.providers:
            if p.name in skip or not registry.is_available(p.name):
                continue
            if not await p.reachable():
                continue
            out.append(p)
        return out

    async def is_available(self) -> bool:
        return bool(await self._candidates())

    async def active_provider(self) -> Optional[str]:
        cands = await self._candidates()
        return cands[0].name if cands else None

    @property
    def model(self) -> str:
        return self.providers[0].model if self.providers else "none"

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4000,
        effort: Optional[str] = None,
        json_schema: Optional[dict] = None,
        task: str = "generate",
        skip: frozenset[str] = frozenset(),
    ) -> LLMResponse:
        candidates = await self._candidates(skip)
        if not candidates:
            raise LLMUnavailableError("llm", [])
        errors: list[ProviderError] = []
        messages = [{"role": "user", "content": prompt}]
        for idx, provider in enumerate(candidates):
            t0 = time.perf_counter()
            try:
                resp = await provider.complete(system or "", messages, max_tokens, json_schema, effort)
            except ProviderError as err:
                registry.record_failure(err)
                errors.append(err)
                _log_call(task, provider.name, provider.model, (time.perf_counter() - t0) * 1000, {}, False, err.message)
                nxt = candidates[idx + 1].name if idx + 1 < len(candidates) else None
                notice_from_error(err, f"Switched to {display_name(nxt)}." if nxt else "Using the built-in fallback instead.")
                logger.warning("LLM %s failed for %s: %s", provider.name, task, err)
                continue
            except Exception as exc:     # defensive: unexpected SDK behaviour
                err = ProviderError(provider.name, ErrorKind.UNKNOWN, f"{type(exc).__name__}: {redact(exc)}")
                registry.record_failure(err)
                errors.append(err)
                logger.exception("LLM %s crashed", provider.name)
                continue
            registry.record_success(provider.name)
            _log_call(task, resp.provider, resp.model, resp.latency_ms, resp.usage, True)
            return resp
        raise AllProvidersFailedError("llm", errors)

    async def complete_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        schema: Optional[dict] = None,
        max_tokens: int = 4000,
        effort: Optional[str] = None,
        task: str = "json",
    ) -> dict:
        """Structured output; tries the next provider if a reply isn't valid JSON."""
        json_system = (system or "") + (
            "\n\nRespond with a single valid JSON object only — no markdown fences, no commentary."
        )
        if schema:
            json_system += "\nThe JSON must follow this JSON Schema:\n" + json.dumps(schema)
        tried: set[str] = set()
        last_error: Optional[Exception] = None
        for _ in range(max(1, len(self.providers))):
            try:
                resp = await self.complete(prompt, system=json_system, max_tokens=max_tokens, effort=effort,
                                           json_schema=schema, task=task, skip=frozenset(tried))
            except AllProvidersFailedError as exc:
                if last_error is None or exc.errors:
                    last_error = exc
                break
            data = extract_json(resp.text)
            if isinstance(data, dict):
                return data
            logger.warning("LLM %s returned invalid JSON for %s — trying next provider", resp.provider, task)
            tried.add(resp.provider)
            last_error = ProviderError(resp.provider, ErrorKind.BAD_REQUEST, "reply was not valid JSON")
        if isinstance(last_error, AllProvidersFailedError):
            raise last_error
        raise AllProvidersFailedError("llm", [last_error] if isinstance(last_error, ProviderError) else [])

    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4000,
        effort: Optional[str] = None,
        task: str = "stream",
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream tokens. If a provider fails before producing output the next one is
        tried transparently; if it fails mid-stream a StreamChunk(reset=True) tells the
        consumer to discard the partial text before the next provider starts.
        """
        candidates = await self._candidates()
        if not candidates:
            raise LLMUnavailableError("llm", [])
        errors: list[ProviderError] = []
        messages = [{"role": "user", "content": prompt}]
        for idx, provider in enumerate(candidates):
            t0 = time.perf_counter()
            started = False
            try:
                async for token in provider.stream(system or "", messages, max_tokens, effort):
                    started = True
                    yield StreamChunk(text=token, provider=provider.name, model=provider.model)
                registry.record_success(provider.name)
                _log_call(task, provider.name, provider.model, (time.perf_counter() - t0) * 1000, {}, True)
                return
            except ProviderError as err:
                registry.record_failure(err)
                errors.append(err)
                _log_call(task, provider.name, provider.model, (time.perf_counter() - t0) * 1000, {}, False, err.message)
                nxt = candidates[idx + 1].name if idx + 1 < len(candidates) else None
                notice_from_error(err, f"Switched to {display_name(nxt)}." if nxt else "Using the built-in fallback instead.")
                if started:
                    yield StreamChunk(reset=True, provider=provider.name)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                err = ProviderError(provider.name, ErrorKind.UNKNOWN, f"{type(exc).__name__}: {redact(exc)}")
                registry.record_failure(err)
                errors.append(err)
                logger.exception("LLM stream %s crashed", provider.name)
                if started:
                    yield StreamChunk(reset=True, provider=provider.name)
        raise AllProvidersFailedError("llm", errors)

    def describe(self) -> list[dict]:
        return [{"provider": p.name, "display_name": display_name(p.name), "model": p.model} for p in self.providers]


_llm_instance: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMClient()
    return _llm_instance


def reset_llm() -> None:
    """Rebuild providers (e.g. after editing .env and clearing the settings cache)."""
    global _llm_instance
    _llm_instance = None
