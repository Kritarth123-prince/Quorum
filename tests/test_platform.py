"""Offline tests (no network, no credits): error handling, fallbacks, ranking helpers."""

import asyncio
import json

import httpx
import pytest

from retrieval_pipeline.core import http as core_http
from retrieval_pipeline.core.errors import ErrorKind, classify_http_error, redact, register_secrets
from retrieval_pipeline.core.provider_health import collect_notices, registry


@pytest.fixture(autouse=True)
def _reset_registry():
    registry.reset()
    for state in registry.snapshot():
        s = registry.get(state["provider"])
        s.consecutive_failures = s.total_failures = s.total_calls = 0
    yield
    registry.reset()


def _mock_transport(monkeypatch, handler):
    monkeypatch.setattr(core_http, "make_client",
                        lambda timeout, **kw: httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=timeout, **kw))


# ── Error classification ──────────────────────────────────────────────────────

@pytest.mark.parametrize("provider,status,body,kind", [
    ("tavily", 432, {"detail": {"error": "This request exceeds your plan's set usage limit."}}, ErrorKind.QUOTA),
    ("tavily", 433, {"detail": {"error": "Pay-as-you-go limit exceeded"}}, ErrorKind.QUOTA),
    ("serpapi", 429, {"error": "Your account has run out of searches."}, ErrorKind.QUOTA),
    ("serpapi", 429, {"error": "You have exceeded the hourly throughput limit."}, ErrorKind.RATE_LIMIT),
    ("openai", 429, {"error": {"message": "You exceeded your current quota, please check your plan and billing details.", "code": "insufficient_quota"}}, ErrorKind.QUOTA),
    ("openai", 429, {"error": {"message": "Rate limit reached for requests", "code": "rate_limit_exceeded"}}, ErrorKind.RATE_LIMIT),
    ("anthropic", 400, {"type": "error", "error": {"type": "invalid_request_error", "message": "Your credit balance is too low to access the Anthropic API."}}, ErrorKind.QUOTA),
    ("anthropic", 401, {"error": {"type": "authentication_error", "message": "invalid x-api-key"}}, ErrorKind.AUTH),
    ("anthropic", 529, {"error": {"type": "overloaded_error", "message": "Overloaded"}}, ErrorKind.SERVER),
    ("openrouter", 402, {"error": {"message": "Insufficient credits"}}, ErrorKind.QUOTA),
    ("gemini", 400, [{"error": {"code": 400, "message": "API key not valid.", "status": "INVALID_ARGUMENT"}}], ErrorKind.AUTH),
    ("openai", 404, {"error": {"message": "The model does not exist"}}, ErrorKind.BAD_REQUEST),
])
def test_classification(provider, status, body, kind):
    assert classify_http_error(provider, status, json.dumps(body)).kind == kind


def test_redaction_hides_keys():
    register_secrets(["fake-test-secret-value"])
    text = redact("GET https://serpapi.com/search?q=x&api_key=abcdef123456 token fake-test-secret-value")
    assert "abcdef123456" not in text and "fake-test-secret-value" not in text


# ── Search fallback on exhausted credits ──────────────────────────────────────

def test_tavily_quota_falls_back_to_serpapi_and_pauses_tavily(monkeypatch):
    from retrieval_pipeline.retrieval import web_search as ws
    monkeypatch.setattr(ws.cfg, "tavily_api_key", "tvly-test", raising=False)
    monkeypatch.setattr(ws.cfg, "serpapi_key", "serp-test", raising=False)
    registry.register("tavily", "search", configured=True)
    registry.register("serpapi", "search", configured=True)
    ws._cache.clear()
    calls = {"tavily": 0, "serpapi": 0}

    def handler(request: httpx.Request):
        if "tavily" in request.url.host:
            calls["tavily"] += 1
            return httpx.Response(432, json={"detail": {"error": "Plan usage limit exceeded"}})
        calls["serpapi"] += 1
        return httpx.Response(200, json={"organic_results": [
            {"link": "https://example.org/a", "title": "A", "snippet": "Retrieval augmented generation grounds answers in sources."}]})

    _mock_transport(monkeypatch, handler)

    async def run():
        with collect_notices() as notices:
            first = await ws.web_search("rag basics", max_results=3)
            ws._cache.clear()
            second = await ws.web_search("rag basics again", max_results=3)
        return first, second, notices

    first, second, notices = asyncio.run(run())
    assert first and first[0].metadata.extra["provider"] == "serpapi"
    assert second and calls["tavily"] == 1                    # paused: not called again
    assert registry.get("tavily").to_dict()["status"] == "quota_exhausted"
    assert any("credits" in n.message.lower() for n in notices)


# ── LLM provider fallback ─────────────────────────────────────────────────────

def test_llm_falls_back_when_first_provider_has_no_credits(monkeypatch):
    from retrieval_pipeline.llm import client as llm_client

    def handler(request: httpx.Request):
        if "first.test" in request.url.host:
            return httpx.Response(429, json={"error": {"message": "You exceeded your current quota", "code": "insufficient_quota"}})
        return httpx.Response(200, json={"model": "m2", "choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}]})

    _mock_transport(monkeypatch, handler)
    client = llm_client.LLMClient.__new__(llm_client.LLMClient)
    client.providers = [
        llm_client.OpenAICompatibleProvider(llm_client._CompatSpec("first", "https://first.test/v1", "k1", "m1")),
        llm_client.OpenAICompatibleProvider(llm_client._CompatSpec("second", "https://second.test/v1", "k2", "m2")),
    ]
    registry.register("first", "llm")
    registry.register("second", "llm")

    async def run():
        with collect_notices() as notices:
            data = await client.complete_json("hi", schema={"type": "object"})
        return data, notices

    data, notices = asyncio.run(run())
    assert data == {"ok": True}
    assert registry.get("first").to_dict()["status"] == "quota_exhausted"
    assert any("Switched to" in n.message for n in notices)


def test_extract_json_handles_fences_and_prose():
    from retrieval_pipeline.llm.client import extract_json
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": {"b": "x}"}} thanks') == {"a": {"b": "x}"}}
    assert extract_json("not json") is None


# ── Text, chunking, citations ─────────────────────────────────────────────────

def test_chunker_always_advances():
    from retrieval_pipeline.retrieval.vector_rag import DocumentChunker
    text = "Short. " + ("word " * 400) + ". End."
    spans = DocumentChunker(chunk_size=200, overlap=150).split(text)
    starts = [s for s, _, _ in spans]
    assert starts == sorted(set(starts)) and spans[-1][1] == len(text)


def test_citation_sanitising():
    from retrieval_pipeline.agents.sources import cited_ids, sanitize_citations
    out = sanitize_citations("Fact [1, 2]. Other [9]. Range [2-3].", n_sources=3)
    assert out == "Fact [1][2]. Other. Range [2][3]."
    assert cited_ids(out) == [1, 2, 3]


def test_hybrid_rank_without_models_prefers_matching_doc(monkeypatch):
    from retrieval_pipeline.models.schemas import DocumentMetadata, RetrievalMethod, RetrievedDocument
    from retrieval_pipeline.retrieval.hybrid_search import HybridRetriever

    def doc(text, score):
        return RetrievedDocument(content=text, relevance_score=score,
                                 metadata=DocumentMetadata(source_method=RetrievalMethod.WEB_SEARCH, domain="x.org"))

    docs = [doc("Bananas are yellow fruit grown in the tropics.", 0.9),
            doc("Qdrant is a vector database written in Rust for similarity search.", 0.1)]
    ranked = asyncio.run(HybridRetriever().rank("qdrant vector database", docs, use_dense=False))
    assert "Qdrant" in ranked[0].content
