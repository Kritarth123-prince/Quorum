"""
api.py — Quorum API + web UI

  GET  /                       web UI (Perplexity-style threads with live progress)
  POST /research               full pipeline → JSON report
  POST /research/stream        full pipeline → Server-Sent Events (progress, tokens, report)
  POST /retrieve               retrieval only (no LLM)
  POST /retrieve/batch         multi-query retrieval fused and ranked once
  POST /index                  add documents to the local knowledge base
  POST /index/urls             fetch URLs and add them to the knowledge base
  GET  /sessions               recent research threads
  GET  /sessions/{id}          one thread (turns + reports)
  DELETE /sessions/{id}        delete a thread
  GET  /providers              provider health + remaining search credits
  POST /providers/reset        re-enable a paused provider (e.g. after topping up credits)
  GET  /health                 component status
  GET  /metrics                Prometheus text format

Run (from the project root, with the venv active):
  uvicorn retrieval_pipeline.api:app --host 127.0.0.1 --port 8000
On Windows do not add --reload: it switches to an event loop that cannot start
Playwright's browser (static page fetching still works).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from retrieval_pipeline import __version__
from retrieval_pipeline.config import PACKAGE_DIR, get_settings
from retrieval_pipeline.core import ml
from retrieval_pipeline.core.errors import redact
from retrieval_pipeline.core.logging_setup import setup_logging
from retrieval_pipeline.core.provider_health import registry
from retrieval_pipeline.llm.client import get_llm
from retrieval_pipeline.memory import short_term
from retrieval_pipeline.memory.short_term import SessionMemory, list_sessions
from retrieval_pipeline.models.schemas import RetrievalMethod, RetrievalQuery, RetrievalResult, RetrievedDocument
from retrieval_pipeline.orchestration.graph import get_router, run_research, stream_research
from retrieval_pipeline.retrieval import web_search as web_search_module
from retrieval_pipeline.retrieval.vector_store import backend_info, close_qdrant, get_qdrant
from retrieval_pipeline.retrieval.web_crawl import crawl_batch, playwright_status

cfg = get_settings()
setup_logging(cfg.log_level)
logger = logging.getLogger(__name__)

STATIC_DIR = PACKAGE_DIR / "static"
SESSION_ID_PATTERN = r"^[A-Za-z0-9_\-]{1,64}$"

_start_time = time.monotonic()
_metrics: dict[str, float] = {
    "research_requests_total":   0,
    "research_errors_total":     0,
    "research_timeouts_total":   0,
    "research_latency_ms_total": 0,
    "retrieval_requests_total":  0,
    "docs_retrieved_total":      0,
}
_research_slots: Optional[asyncio.Semaphore] = None
_tasks: set[asyncio.Task] = set()


def _background(coro) -> None:
    task = asyncio.create_task(coro)
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _research_slots
    ml.BLOCKING_LOAD = False                  # never make a request wait for a cold model
    _research_slots = asyncio.Semaphore(max(1, cfg.max_concurrent_research))
    get_router()
    get_llm()
    _background(get_qdrant())
    _background(web_search_module.search_quota())
    if cfg.warmup_models:
        _background(ml.warm_up())
    logger.info("Quorum v%s ready — open http://127.0.0.1:8000", __version__)
    yield
    for task in list(_tasks):
        task.cancel()
    await close_qdrant()
    logger.info("Shut down cleanly")


app = FastAPI(
    title="Quorum",
    version=__version__,
    description="Research your sources agree on — multi-agent planning, multi-source retrieval, "
                "cross-source verification and cited synthesis.",
    lifespan=lifespan,
)

if cfg.cors_origins.strip():
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in cfg.cors_origins.split(",") if o.strip()],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"error": "internal_error", "message": redact(exc)})


# ── Schemas ───────────────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query:        str           = Field(..., min_length=2, max_length=2000, description="Research question")
    max_results:  int           = Field(8, ge=1, le=20, description="Results per search query")
    use_academic: bool          = Field(False, description="Always include Semantic Scholar + arXiv")
    use_crawl:    bool          = Field(False, description="Deep-read the top result pages")
    session_id:   Optional[str] = Field(None, pattern=SESSION_ID_PATTERN, description="Continue an existing thread")


class RetrieveRequest(BaseModel):
    query:       str                   = Field(..., min_length=1, max_length=2000)
    methods:     list[RetrievalMethod] = Field(default=[RetrievalMethod.WEB_SEARCH, RetrievalMethod.HYBRID])
    max_results: int                   = Field(10, ge=1, le=50)
    filters:     dict                  = Field(default_factory=dict)
    use_mmr:     bool                  = False
    seed_urls:   Optional[list[str]]   = None


class BatchRetrieveRequest(BaseModel):
    queries:     list[str]             = Field(..., min_length=1, max_length=10)
    methods:     list[RetrievalMethod] = Field(default=[RetrievalMethod.WEB_SEARCH, RetrievalMethod.HYBRID])
    max_results: int                   = Field(10, ge=1, le=50)
    rank_query:  Optional[str]         = Field(None, description="Question to rank the pooled results against")


class IndexRequest(BaseModel):
    documents: list[RetrievedDocument] = Field(..., min_length=1, max_length=500)


class IndexUrlsRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1, max_length=20)


class ResetRequest(BaseModel):
    provider: Optional[str] = Field(None, description="Provider name; omit to reset all")


def _doc_view(d: RetrievedDocument) -> dict:
    return {
        "title":        d.metadata.title,
        "url":          d.metadata.url,
        "domain":       d.metadata.domain,
        "method":       d.metadata.source_method.value,
        "provider":     d.metadata.extra.get("provider"),
        "rerank_score": d.rerank_score,
        "relevance":    d.relevance_score,
        "published":    d.metadata.published_at.isoformat() if d.metadata.published_at else None,
        "snippet":      d.content[:400],
    }


def _retrieval_view(result: RetrievalResult) -> dict:
    return {
        "query":              result.query.text,
        "total_fetched":      result.total_fetched,
        "total_after_rerank": result.total_after_rerank,
        "methods_used":       [m.value for m in result.methods_used],
        "latency_ms":         result.latency_ms,
        "warnings":           result.warnings,
        "errors":             result.errors,
        "documents":          [_doc_view(d) for d in result.documents],
    }


# ── UI ────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def index():
    page = STATIC_DIR / "index.html"
    if not page.exists():
        return JSONResponse({"message": "UI not found — use /docs for the API."})
    return FileResponse(page)


# ── Research ──────────────────────────────────────────────────────────────────

@app.post("/research")
async def research(req: ResearchRequest):
    """Run the full pipeline and return the report (answer, findings, sources, confidence, warnings)."""
    _metrics["research_requests_total"] += 1
    t0 = time.perf_counter()
    assert _research_slots is not None
    async with _research_slots:
        try:
            state = await run_research(req.query, req.max_results, req.use_academic, req.use_crawl, req.session_id)
        except Exception as exc:
            _metrics["research_errors_total"] += 1
            logger.exception("Research failed")
            raise HTTPException(500, f"Research failed: {redact(exc)}")
    report = state.get("report", {})
    if str(report.get("error", "")).startswith("Research timed out"):
        _metrics["research_timeouts_total"] += 1
    _metrics["research_latency_ms_total"] += (time.perf_counter() - t0) * 1000
    _metrics["docs_retrieved_total"] += state.get("total_fetched", 0) or 0
    return {"session_id": state.get("session_id"), "report": report, "warnings": report.get("warnings", [])}


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str, ensure_ascii=False)}\n\n"


@app.post("/research/stream")
async def research_stream(req: ResearchRequest):
    """
    Server-Sent Events. Events: start, queued, status, plan, plan_update, sources,
    verification, token, answer_reset, answer, notice, warning, heartbeat, report, error, done.
    The final "report" event is always sent, even after errors or a timeout.
    """
    _metrics["research_requests_total"] += 1

    async def events():
        assert _research_slots is not None
        t0 = time.perf_counter()
        if _research_slots.locked():
            yield _sse("queued", {"message": "Waiting for another research run to finish…"})
        async with _research_slots:
            try:
                async for ev in stream_research(req.query, req.max_results, req.use_academic, req.use_crawl, req.session_id):
                    if ev["event"] == "heartbeat":
                        yield ": keep-alive\n\n"
                    if ev["event"] == "error":
                        _metrics["research_errors_total"] += 1
                        if "timed out" in ev["data"].get("message", ""):
                            _metrics["research_timeouts_total"] += 1
                    yield _sse(ev["event"], ev["data"])
            except asyncio.CancelledError:
                logger.info("Client disconnected; research cancelled")
                raise
            except Exception as exc:
                _metrics["research_errors_total"] += 1
                logger.exception("Streaming research failed")
                yield _sse("error", {"message": f"Research failed: {redact(exc)}"})
        _metrics["research_latency_ms_total"] += (time.perf_counter() - t0) * 1000

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Retrieval ─────────────────────────────────────────────────────────────────

@app.post("/retrieve")
async def retrieve(req: RetrieveRequest):
    """Retrieval only — web / academic / crawl / knowledge base, fused and reranked. No LLM."""
    _metrics["retrieval_requests_total"] += 1
    result = await get_router().retrieve(
        RetrievalQuery(text=req.query, methods=req.methods, max_results=req.max_results, filters=req.filters),
        crawl_seed_urls=req.seed_urls, use_mmr=req.use_mmr,
    )
    _metrics["docs_retrieved_total"] += result.total_fetched
    return _retrieval_view(result)


@app.post("/retrieve/batch")
async def retrieve_batch(req: BatchRetrieveRequest):
    """Multi-query retrieval: sub-queries are searched in parallel, pooled and ranked once."""
    _metrics["retrieval_requests_total"] += 1
    result = await get_router().retrieve_multi_query(
        req.queries, methods=req.methods, max_results=req.max_results, rank_query=req.rank_query,
    )
    _metrics["docs_retrieved_total"] += result.total_fetched
    return _retrieval_view(result)


# ── Knowledge base ────────────────────────────────────────────────────────────

@app.post("/index", status_code=202)
async def index_documents(req: IndexRequest, background_tasks: BackgroundTasks):
    """Index documents into the local knowledge base (runs in the background)."""
    router = get_router()

    async def _index():
        try:
            n = await router._vector_retriever.index_documents(req.documents)
            logger.info("Manual indexing complete: %d chunks", n)
        except Exception as exc:
            logger.error("Manual indexing failed: %s", redact(exc))

    background_tasks.add_task(_index)
    return {"status": "indexing", "document_count": len(req.documents)}


@app.post("/index/urls", status_code=202)
async def index_urls(req: IndexUrlsRequest, background_tasks: BackgroundTasks):
    """Fetch pages and add them to the knowledge base (runs in the background)."""
    urls = [u for u in req.urls if u.startswith(("http://", "https://"))]
    if not urls:
        raise HTTPException(422, "Provide http(s) URLs")
    router = get_router()

    async def _crawl_and_index():
        try:
            docs = await crawl_batch(urls, max_depth=0, max_pages_per_seed=1)
            n = await router._vector_retriever.index_documents(docs)
            logger.info("Indexed %d chunks from %d/%d URLs", n, len(docs), len(urls))
        except Exception as exc:
            logger.error("URL indexing failed: %s", redact(exc))

    background_tasks.add_task(_crawl_and_index)
    return {"status": "indexing", "url_count": len(urls)}


# ── Sessions ──────────────────────────────────────────────────────────────────

@app.get("/sessions")
async def sessions(limit: int = Query(30, ge=1, le=200)):
    return {"sessions": await list_sessions(limit)}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    if not re.fullmatch(SESSION_ID_PATTERN, session_id):
        raise HTTPException(422, "Invalid session id")
    memory = SessionMemory(session_id)
    turns = await memory.get_turns()
    reports = await memory.get_reports()
    if not turns and not reports:
        raise HTTPException(404, f"Session {session_id} not found")
    return {"session_id": session_id, "turns": turns, "reports": reports,
            "report": reports[-1] if reports else None}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if not re.fullmatch(SESSION_ID_PATTERN, session_id):
        raise HTTPException(422, "Invalid session id")
    await SessionMemory(session_id).clear()
    return {"deleted": session_id}


# ── Providers / health / metrics ──────────────────────────────────────────────

@app.get("/providers")
async def providers(refresh: bool = False):
    """Health of every external provider, plus remaining credits for Tavily / SerpAPI."""
    quota = await web_search_module.search_quota(force=refresh)
    llm = get_llm()
    states = registry.snapshot()
    for s in states:
        if s["provider"] in quota:
            s["quota"] = quota[s["provider"]]
        if s["category"] == "llm":
            match = next((p for p in llm.providers if p.name == s["provider"]), None)
            if match is not None:
                s["model"] = match.model
                if match.name == "ollama" and not await match.reachable():
                    s["status"] = "not_running"
    return {"providers": states, "llm_order": [p.name for p in llm.providers],
            "search_order": web_search_module.provider_order()}


@app.post("/providers/reset")
async def reset_providers(req: ResetRequest):
    """Clear a provider's cooldown — e.g. after topping up credits or fixing a key in .env (restart for new keys)."""
    reset = registry.reset(req.provider)
    if req.provider and not reset:
        raise HTTPException(404, f"Unknown provider {req.provider}")
    web_search_module._quota_cache.clear()
    return {"reset": reset}


@app.get("/health")
async def health():
    llm = get_llm()
    active_llm = await llm.active_provider()
    search_ok = [p for p in web_search_module.provider_order() if registry.is_available(p)]
    qdrant = await get_qdrant()
    kb_size = await get_router().knowledge_base_size() if qdrant is not None and ml.embedding_model.loaded else None
    status = "ok" if search_ok else "degraded"
    return {
        "status":   status,
        "version":  __version__,
        "uptime_s": round(time.monotonic() - _start_time, 1),
        "components": {
            "llm":            {"active_provider": active_llm, "configured": llm.describe(),
                               "mode": "llm" if active_llm else "extractive"},
            "web_search":     {"available": search_ok, "order": web_search_module.provider_order()},
            "vector_store":   {**backend_info(), "knowledge_base_chunks": kb_size},
            "session_store":  await short_term.backend_name(),
            "embedding_model": ml.embedding_model.status(),
            "reranker_model": ml.reranker_model.status(),
            "playwright":     playwright_status(),
        },
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    lines = []
    for key, value in _metrics.items():
        kind = "counter"
        lines += [f"# HELP {key} Quorum {kind}", f"# TYPE {key} {kind}", f"{key} {value}"]
    lines += ["# HELP provider_failures_total Failures per external provider", "# TYPE provider_failures_total counter"]
    for s in registry.snapshot():
        lines.append(f'provider_failures_total{{provider="{s["provider"]}"}} {s["total_failures"]}')
    lines += ["# HELP provider_available 1 if the provider is currently usable", "# TYPE provider_available gauge"]
    for s in registry.snapshot():
        lines.append(f'provider_available{{provider="{s["provider"]}",status="{s["status"]}"}} {1 if s["status"] in ("ok", "degraded") else 0}')
    lines.append(f"platform_uptime_seconds {time.monotonic() - _start_time:.1f}")
    return "\n".join(lines) + "\n"
