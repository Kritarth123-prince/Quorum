"""
orchestration/graph.py

LangGraph research pipeline.

    START
      │
      ▼
   preflight ─► planner ─► retriever ──(too few sources?)──► refine ─┐
                               ▲                                     │
                               └─────────────────────────────────────┘
                               │ enough sources
                               ▼
                           verifier ─► writer ─► reporter ─► memory_writer ─► END

  preflight      checks LLM / search / vector-store availability, loads thread history
  planner        standalone query, research goal, search queries, academic/crawl decisions
  retriever      multi-query, multi-method retrieval ranked once against the question
  refine         autonomous second pass with broader queries when too little was found
  verifier       claim extraction + cross-source confidence scoring
  writer         streamed, cited Markdown answer
  reporter       structured report (summary, findings, trade-offs, related questions)
  memory_writer  thread history (SQLite/Redis) + verified findings (Qdrant)

Entry points:
  run_research(...)     → final state dict (state["report"] is the report)
  stream_research(...)  → async iterator of {"event": ..., "data": ...} for SSE / CLI
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from typing import Any, AsyncIterator, Awaitable, Callable, Optional
from uuid import uuid4

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from retrieval_pipeline.agents.planner import PlannerAgent, PlannerOutput
from retrieval_pipeline.agents.report_generator import ReportGeneratorAgent
from retrieval_pipeline.agents.sources import build_sources, public_source, sanitize_citations
from retrieval_pipeline.agents.verifier import VerifierAgent, VerifierOutput
from retrieval_pipeline.config import get_settings
from retrieval_pipeline.core.errors import display_name, redact
from retrieval_pipeline.core.provider_health import collect_notices, registry
from retrieval_pipeline.llm.client import collect_llm_calls, get_llm
from retrieval_pipeline.memory.long_term import store_research_session
from retrieval_pipeline.memory.short_term import SessionMemory
from retrieval_pipeline.models.schemas import RetrievalMethod
from retrieval_pipeline.orchestration.state import ResearchState
from retrieval_pipeline.retrieval.router import RetrievalRouter
from retrieval_pipeline.retrieval.vector_store import backend_info, get_qdrant
from retrieval_pipeline.retrieval.web_search import provider_order

logger = logging.getLogger(__name__)
cfg = get_settings()


# ── Shared singletons ─────────────────────────────────────────────────────────

_router:   Optional[RetrievalRouter] = None
_planner:  Optional[PlannerAgent] = None
_verifier: Optional[VerifierAgent] = None
_reporter: Optional[ReportGeneratorAgent] = None
_background: set[asyncio.Task] = set()


def get_router() -> RetrievalRouter:
    global _router
    if _router is None:
        _router = RetrievalRouter()
    return _router


def _get_planner() -> PlannerAgent:
    global _planner
    if _planner is None:
        _planner = PlannerAgent()
    return _planner


def _get_verifier() -> VerifierAgent:
    global _verifier
    if _verifier is None:
        _verifier = VerifierAgent()
    return _verifier


def _get_reporter() -> ReportGeneratorAgent:
    global _reporter
    if _reporter is None:
        _reporter = ReportGeneratorAgent()
    return _reporter


# ── Event helpers ─────────────────────────────────────────────────────────────

def _emit(event: str, data: Any) -> None:
    """Send a custom event to stream_research() consumers (no-op under ainvoke)."""
    try:
        writer = get_stream_writer()
    except Exception:
        return
    try:
        writer({"event": event, "data": data})
    except Exception:
        pass


_NODE_LABELS = {
    "preflight":     "Checking services",
    "planner":       "Planning the research",
    "retriever":     "Searching and reading sources",
    "refine":        "Too few sources — broadening the search",
    "verifier":      "Cross-checking claims",
    "writer":        "Writing the answer",
    "reporter":      "Building the research report",
    "memory_writer": "Saving to memory",
}


def _node(name: str) -> Callable[[Callable[[ResearchState], Awaitable[dict]]], Callable[[ResearchState], Awaitable[dict]]]:
    """Time a node and announce its start/end to streaming consumers."""
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(state: ResearchState) -> dict:
            _emit("status", {"node": name, "state": "running", "message": _NODE_LABELS.get(name, name)})
            t0 = time.perf_counter()
            update = await fn(state) or {}
            elapsed = round((time.perf_counter() - t0) * 1000, 1)
            update["timings"] = {**state.get("timings", {}), name: elapsed}
            _emit("status", {"node": name, "state": "done", "ms": elapsed})
            return update
        return wrapper
    return decorator


def _plan_from_state(state: ResearchState) -> PlannerOutput:
    plan = state.get("plan") or {}
    return PlannerOutput(
        original_query=state["query"],
        standalone_query=state.get("standalone_query") or state["query"],
        research_goal=state.get("research_goal") or state["query"],
        query_type=plan.get("query_type", "other"),
        subtasks=state.get("subtasks", []),
        search_queries=state.get("search_queries", []),
        needs_academic=state.get("needs_academic", False),
        needs_crawl=state.get("needs_crawl", False),
    )


# ── Nodes ─────────────────────────────────────────────────────────────────────

@_node("preflight")
async def node_preflight(state: ResearchState) -> dict:
    warnings = list(state.get("warnings", []))

    llm = get_llm()
    llm_provider = await llm.active_provider()
    keyed = [p for p in llm.providers if p.name != "ollama"]
    if llm_provider is None and not keyed:
        warnings.append("No LLM API key configured (and Ollama is not running) — answers are extractive summaries. "
                        "Add ANTHROPIC_API_KEY or another provider key to .env for AI-written answers.")
    elif llm_provider is None:
        warnings.append("All configured LLM providers are unavailable right now (credits, keys or rate limits) — "
                        "using extractive answers. See the Providers panel for details.")

    search_ready = [p for p in provider_order() if registry.is_available(p)]
    if not search_ready:
        warnings.append("All web search providers are unavailable (credits exhausted or keys invalid). "
                        "Results will come from academic search and the local knowledge base only.")
    elif search_ready[0] != provider_order()[0]:
        warnings.append(f"Primary search provider {display_name(provider_order()[0])} is paused — "
                        f"using {display_name(search_ready[0])}.")

    # Don't block the run on opening the vector store (slow on a cold start);
    # retrieval awaits it only when the knowledge base is actually queried.
    backend = backend_info().get("backend")
    if backend == "uninitialised":
        task = asyncio.create_task(get_qdrant())
        _background.add(task)
        task.add_done_callback(_background.discard)
    qdrant_ok = backend not in ("unavailable",)
    history = []
    try:
        history = await SessionMemory(state["session_id"]).get_turns()
    except Exception as exc:
        logger.warning("Could not load session history: %s", exc)

    return {
        "llm_available":    llm_provider is not None,
        "llm_provider":     llm_provider,
        "qdrant_available": qdrant_ok,
        "history":          history,
        "warnings":         warnings,
        "refinements":      0,
    }


@_node("planner")
async def node_planner(state: ResearchState) -> dict:
    query = state["query"]
    try:
        plan = await _get_planner().plan(query, state.get("history"))
    except Exception as exc:
        logger.exception("Planner failed")
        plan = PlannerOutput(original_query=query, research_goal=query, standalone_query=query,
                             search_queries=[query], fallback_used=True, fallback_reason=redact(exc))
    needs_academic = plan.needs_academic or bool(state.get("use_academic"))
    needs_crawl = plan.needs_crawl or bool(state.get("use_crawl"))
    plan_dict = {**plan.to_dict(), "needs_academic": needs_academic, "needs_crawl": needs_crawl}
    _emit("plan", plan_dict)
    return {
        "plan":             plan_dict,
        "standalone_query": plan.standalone_query or query,
        "research_goal":    plan.research_goal,
        "subtasks":         plan.subtasks,
        "search_queries":   plan.search_queries or [query],
        "needs_academic":   needs_academic,
        "needs_crawl":      needs_crawl,
        "prior_findings":   plan.prior_findings,
    }


@_node("retriever")
async def node_retriever(state: ResearchState) -> dict:
    router = get_router()
    question = state.get("standalone_query") or state["query"]
    methods = [RetrievalMethod.WEB_SEARCH, RetrievalMethod.HYBRID]
    if state.get("needs_academic"):
        methods.append(RetrievalMethod.ACADEMIC)
    if state.get("needs_crawl"):
        methods.append(RetrievalMethod.WEB_CRAWL)

    try:
        result = await router.retrieve_multi_query(
            queries=state.get("search_queries") or [question],
            methods=methods,
            max_results=state.get("max_results", 10),
            rank_query=question,
            use_mmr=True,
            top_k=cfg.synthesis_max_sources,
        )
        docs = result.documents
        if state.get("refinements") and state.get("retrieved_docs"):
            # Second pass: merge with what the first pass found and rank the union.
            docs, _, _ = await router.rank(question, list(state["retrieved_docs"]) + docs,
                                           top_k=cfg.synthesis_max_sources, use_hybrid=False, use_mmr=True)
        latency, methods_used, fetched = result.latency_ms, [m.value for m in result.methods_used], result.total_fetched
    except Exception as exc:
        logger.exception("Retriever failed")
        docs, latency, methods_used, fetched = list(state.get("retrieved_docs", [])), {}, [], 0
        _emit("warning", {"message": f"Retrieval error: {redact(exc)}"})

    sources = build_sources(question, docs, cfg.synthesis_max_sources, cfg.synthesis_chars_per_source)
    _emit("sources", {"sources": [public_source(s) for s in sources], "total_fetched": fetched, "methods": methods_used})
    return {
        "retrieved_docs":    docs,
        "sources":           sources,
        "retrieval_methods": methods_used,
        "retrieval_latency": latency,
        "total_fetched":     (state.get("total_fetched") or 0) + fetched,
    }


def route_after_retrieval(state: ResearchState) -> str:
    if len(state.get("retrieved_docs") or []) < cfg.min_docs_before_refine and not state.get("refinements"):
        return "refine"
    return "verifier"


@_node("refine")
async def node_refine(state: ResearchState) -> dict:
    found = len(state.get("retrieved_docs") or [])
    try:
        queries = await _get_planner().refine(_plan_from_state(state), found)
    except Exception as exc:
        logger.warning("Refine failed: %s", exc)
        queries = [state.get("standalone_query") or state["query"]]
    _emit("plan_update", {"search_queries": queries, "reason": f"only {found} sources found"})
    return {"search_queries": queries, "refinements": (state.get("refinements") or 0) + 1, "needs_academic": True}


@_node("verifier")
async def node_verifier(state: ResearchState) -> dict:
    sources = state.get("sources") or []
    question = state.get("standalone_query") or state["query"]
    try:
        output = await _get_verifier().verify(question, sources)
    except Exception as exc:
        logger.exception("Verifier failed")
        output = VerifierOutput(query=question, fallback_used=True)
        _emit("warning", {"message": f"Verification error: {redact(exc)}"})
    _emit("verification", {
        "claims": len(output.claims), "high_confidence": output.high_confidence_count,
        "contradictions": output.contradictions_found, "hallucination_flags": output.hallucination_flags,
        "mode": "heuristic" if output.fallback_used else "llm",
    })
    return {
        "verification":          output,
        "verified_claims":       output.claims,
        "hallucination_flags":   output.hallucination_flags,
        "high_confidence_count": output.high_confidence_count,
        "contradictions_found":  output.contradictions_found,
    }


@_node("writer")
async def node_writer(state: ResearchState) -> dict:
    sources = state.get("sources") or []
    question = state.get("standalone_query") or state["query"]
    parts: list[str] = []
    provider = "extractive"
    try:
        async for chunk in _get_reporter().stream_answer(question, sources, state.get("verification"), state.get("history")):
            if chunk.reset:
                parts = []
                _emit("answer_reset", {"provider": chunk.provider})
                continue
            if chunk.text:
                parts.append(chunk.text)
                provider = chunk.provider or provider
                _emit("token", {"text": chunk.text})
    except Exception as exc:
        logger.exception("Writer failed")
        _emit("warning", {"message": f"Answer generation error: {redact(exc)}"})
    answer = sanitize_citations("".join(parts).strip(), len(sources))
    if not answer:
        from retrieval_pipeline.agents.report_generator import extractive_answer
        answer, provider = extractive_answer(question, sources, state.get("verification")), "extractive"
    _emit("answer", {"text": answer, "provider": provider})
    return {"answer": answer, "answer_provider": provider}


@_node("reporter")
async def node_reporter(state: ResearchState) -> dict:
    sources = state.get("sources") or []
    question = state.get("standalone_query") or state["query"]
    reporter = _get_reporter()
    try:
        report = await reporter.build_report(question, sources, state.get("verification"), state.get("plan"),
                                             state.get("answer", ""), state.get("answer_provider", "extractive"))
    except Exception as exc:
        logger.exception("Reporter failed — offline report")
        report = reporter.build_report_offline(question, sources, state.get("verification"), state.get("plan"),
                                               state.get("answer", ""), state.get("answer_provider", "extractive"))
        report["error"] = redact(exc)
    return {"report": report}


@_node("memory_writer")
async def node_memory_writer(state: ResearchState) -> dict:
    report = state.get("report") or {}
    try:
        # The finalised report itself is saved by stream_research() once warnings are merged.
        await SessionMemory(state["session_id"]).add_turn(
            state["query"], state.get("answer", ""), report.get("report_id"), state.get("standalone_query"))
    except Exception as exc:
        logger.warning("Session memory write failed (non-fatal): %s", exc)

    findings = report.get("key_findings") or []
    if findings:
        task = asyncio.create_task(store_research_session(
            state["session_id"], state.get("standalone_query") or state["query"], findings,
            report.get("executive_summary", ""),
        ))
        _background.add(task)
        task.add_done_callback(_background.discard)
    return {}


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_research_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("preflight", node_preflight)
    graph.add_node("planner", node_planner)
    graph.add_node("retriever", node_retriever)
    graph.add_node("refine", node_refine)
    graph.add_node("verifier", node_verifier)
    graph.add_node("writer", node_writer)
    graph.add_node("reporter", node_reporter)
    graph.add_node("memory_writer", node_memory_writer)

    graph.add_edge(START, "preflight")
    graph.add_edge("preflight", "planner")
    graph.add_edge("planner", "retriever")
    graph.add_conditional_edges("retriever", route_after_retrieval, {"refine": "refine", "verifier": "verifier"})
    graph.add_edge("refine", "retriever")
    graph.add_edge("verifier", "writer")
    graph.add_edge("writer", "reporter")
    graph.add_edge("reporter", "memory_writer")
    graph.add_edge("memory_writer", END)
    return graph.compile()


_compiled = None


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_research_graph()
    return _compiled


# ── Finalisation ──────────────────────────────────────────────────────────────

def _finalise_report(state: dict, report: dict, notices: list, llm_calls: list, elapsed_ms: Optional[float]) -> dict:
    report = dict(report)
    warnings: list[str] = []
    for w in list(state.get("warnings", [])) + [n.message for n in notices]:
        if w and w not in warnings:
            warnings.append(w)
    report["warnings"] = warnings
    report["notices"] = [n.to_dict() for n in notices]
    report["session_id"] = state.get("session_id")
    report["query"] = state.get("query", report.get("query"))
    report["retrieval"] = {
        "methods":       state.get("retrieval_methods", []),
        "latency_ms":    state.get("retrieval_latency", {}),
        "total_fetched": state.get("total_fetched", 0),
        "search_queries": state.get("search_queries", []),
        "refinements":   state.get("refinements", 0),
        "vector_store":  backend_info().get("detail"),
    }
    trace = dict(report.get("agent_trace") or {})
    trace["timings_ms"] = state.get("timings", {})
    if llm_calls:
        trace["llm_calls"] = llm_calls
    report["agent_trace"] = trace
    if elapsed_ms is not None:
        report["total_latency_ms"] = round(elapsed_ms, 1)
    return report


def _initial_state(query: str, max_results: int, use_academic: bool, use_crawl: bool,
                   session_id: Optional[str]) -> ResearchState:
    return {
        "query":        query.strip(),
        "max_results":  max(1, min(int(max_results or 10), 20)),
        "use_academic": bool(use_academic),
        "use_crawl":    bool(use_crawl),
        "session_id":   session_id or f"sess_{uuid4().hex[:12]}",
        "warnings":     [],
        "timings":      {},
    }


def _report_from_partial(state: dict, reason: str) -> dict:
    reporter = _get_reporter()
    question = state.get("standalone_query") or state.get("query", "")
    sources = state.get("sources") or []
    if sources:
        report = reporter.build_report_offline(question, sources, state.get("verification"), state.get("plan"),
                                               state.get("answer", ""), state.get("answer_provider", "extractive"))
    else:
        report = reporter.empty_report(question, reason)
    report["error"] = reason
    return report


# ── Entry points ──────────────────────────────────────────────────────────────

async def run_research(
    query: str,
    max_results: int = 10,
    use_academic: bool = False,
    use_crawl: bool = False,
    session_id: Optional[str] = None,
) -> dict:
    """Run the full pipeline and return the final state (state["report"] holds the report)."""
    final: dict = {}
    async for event in stream_research(query, max_results, use_academic, use_crawl, session_id, _collect_state=final):
        if event["event"] == "report":
            final["report"] = event["data"]
    return final


async def stream_research(
    query: str,
    max_results: int = 10,
    use_academic: bool = False,
    use_crawl: bool = False,
    session_id: Optional[str] = None,
    _collect_state: Optional[dict] = None,
) -> AsyncIterator[dict]:
    """
    Run the pipeline and yield events as they happen:
      start, status, plan, plan_update, sources, verification, token, answer_reset,
      answer, warning, heartbeat, report, error, done
    A "report" event is ALWAYS produced (partial if the run fails or times out).
    """
    initial = _initial_state(query, max_results, use_academic, use_crawl, session_id)
    state: dict = dict(initial)
    queue: asyncio.Queue = asyncio.Queue()
    t0 = time.perf_counter()
    graph = get_graph()

    async def producer() -> None:
        async for mode, chunk in graph.astream(initial, stream_mode=["updates", "custom"]):
            if mode == "custom":
                await queue.put(chunk)
            else:
                for node_name, update in (chunk or {}).items():
                    if update:
                        state.update(update)

    with collect_notices() as notices, collect_llm_calls() as llm_calls:
        task = asyncio.create_task(producer())
        deadline = time.monotonic() + cfg.research_timeout_s
        error: Optional[str] = None
        yield {"event": "start", "data": {"session_id": initial["session_id"], "query": initial["query"]}}
        try:
            seen_notices = 0
            while True:
                if task.done() and queue.empty():
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    task.cancel()
                    error = f"Research timed out after {cfg.research_timeout_s}s — returning partial results."
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=min(10.0, remaining))
                except asyncio.TimeoutError:
                    item = {"event": "heartbeat", "data": {"elapsed_s": round(time.perf_counter() - t0, 1)}}
                yield item
                while seen_notices < len(notices):           # surface provider problems live
                    yield {"event": "notice", "data": notices[seen_notices].to_dict()}
                    seen_notices += 1

            if not task.cancelled() and task.done() and task.exception() is not None:
                exc = task.exception()
                logger.error("Research pipeline crashed: %s", exc, exc_info=exc)
                error = f"Pipeline error: {redact(exc)}"
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        if error:
            yield {"event": "error", "data": {"message": error}}
        report = state.get("report") or _report_from_partial(state, error or "Pipeline ended without a report")
        elapsed = (time.perf_counter() - t0) * 1000
        report = _finalise_report(state, report, list(notices), list(llm_calls), elapsed)
        state["report"] = report
        try:
            memory = SessionMemory(initial["session_id"])
            if "memory_writer" not in state.get("timings", {}):      # run ended early: still record the turn
                await memory.add_turn(initial["query"], report.get("answer", ""), report.get("report_id"),
                                      state.get("standalone_query"))
            await memory.save_report(report)
        except Exception as exc:
            logger.warning("Could not save report to session memory: %s", exc)
        if _collect_state is not None:
            _collect_state.update(state)
        yield {"event": "report", "data": report}
        yield {"event": "done", "data": {"session_id": initial["session_id"], "elapsed_ms": round(elapsed, 1)}}


async def drain_background(timeout: float = 60.0) -> None:
    """Wait for background memory/indexing tasks (CLI calls this before exiting)."""
    pending = set(_background)
    if _router is not None:
        await _router.drain_background(timeout)
    if pending:
        await asyncio.wait(pending, timeout=timeout)
