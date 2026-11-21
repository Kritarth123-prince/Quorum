"""
orchestration/state.py

Shared LangGraph state. Nodes return partial updates; LangGraph merges them.
"""

from __future__ import annotations

from typing import Any, Optional

from typing_extensions import TypedDict

from retrieval_pipeline.models.schemas import RetrievedDocument


class ResearchState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────────
    query:            str
    session_id:       str
    max_results:      int
    use_academic:     bool
    use_crawl:        bool

    # ── Runtime ───────────────────────────────────────────────────────────────
    llm_available:    bool
    llm_provider:     Optional[str]
    qdrant_available: bool
    history:          list[dict]          # earlier turns of this thread
    timings:          dict[str, float]    # node → milliseconds
    refinements:      int

    # ── Planner ───────────────────────────────────────────────────────────────
    plan:             dict
    standalone_query: str
    research_goal:    str
    subtasks:         list[str]
    search_queries:   list[str]
    needs_academic:   bool
    needs_crawl:      bool
    prior_findings:   list[dict]

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieved_docs:    list[RetrievedDocument]
    sources:           list[dict]
    retrieval_methods: list[str]
    retrieval_latency: dict[str, float]
    total_fetched:     int

    # ── Verification ──────────────────────────────────────────────────────────
    verification:          Any               # VerifierOutput
    verified_claims:       list[Any]
    hallucination_flags:   int
    high_confidence_count: int
    contradictions_found:  int

    # ── Synthesis ─────────────────────────────────────────────────────────────
    answer:           str
    answer_provider:  str
    report:           dict

    # ── Diagnostics ───────────────────────────────────────────────────────────
    error:            Optional[str]
    warnings:         list[str]
