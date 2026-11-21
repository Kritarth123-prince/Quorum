"""
agents/planner.py

Planner Agent — first reasoning step of the research graph.

  1. Resolve follow-up questions into a standalone query using the thread history
  2. Decompose the question into research angles (subtasks)
  3. Write focused web-search queries (each one costs a search credit, so few and distinct)
  4. Decide whether academic papers and full-page reading are needed
  5. Recall related findings from long-term memory

Uses one structured LLM call; falls back to rule-based planning when no LLM is usable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from retrieval_pipeline.config import get_settings
from retrieval_pipeline.core.errors import AllProvidersFailedError
from retrieval_pipeline.core.text import keywords
from retrieval_pipeline.llm.client import get_llm, last_call_provider
from retrieval_pipeline.memory.long_term import recall

logger = logging.getLogger(__name__)
cfg = get_settings()


@dataclass
class PlannerOutput:
    original_query:   str
    research_goal:    str
    standalone_query: str = ""
    query_type:       str = "other"
    subtasks:         list[str] = field(default_factory=list)
    search_queries:   list[str] = field(default_factory=list)
    needs_academic:   bool = False
    needs_crawl:      bool = False
    prior_findings:   list[dict] = field(default_factory=list)
    fallback_used:    bool = False
    fallback_reason:  str = ""
    llm_provider:     Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query, "standalone_query": self.standalone_query,
            "research_goal": self.research_goal, "query_type": self.query_type,
            "subtasks": self.subtasks, "search_queries": self.search_queries,
            "needs_academic": self.needs_academic, "needs_crawl": self.needs_crawl,
            "prior_findings": len(self.prior_findings), "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason, "llm_provider": self.llm_provider,
        }


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "standalone_query": {"type": "string", "description": "The question rewritten to be self-contained"},
        "research_goal":    {"type": "string", "description": "One sentence: what the answer must establish"},
        "query_type":       {"type": "string", "enum": ["factual", "comparison", "how_to", "explanatory",
                                                        "current_events", "opinion", "other"]},
        "subtasks":         {"type": "array", "items": {"type": "string"}},
        "search_queries":   {"type": "array", "items": {"type": "string"}},
        "needs_academic":   {"type": "boolean"},
        "needs_crawl":      {"type": "boolean"},
    },
    "required": ["standalone_query", "research_goal", "query_type", "subtasks",
                 "search_queries", "needs_academic", "needs_crawl"],
    "additionalProperties": False,
}

_PLANNER_SYSTEM = """You plan research for an AI research assistant that answers questions from live web search, academic papers and a local knowledge base, then writes a cited answer.

Today's date is {today}.

Produce a research plan for the user's latest question:
- standalone_query: if the question depends on earlier turns ("what about its pricing?"), rewrite it so it can be understood alone; otherwise repeat it.
- research_goal: one sentence stating what a complete answer must establish.
- subtasks: {max_subtasks} or fewer distinct angles the answer should cover.
- search_queries: between 2 and {max_queries} short, keyword-style web search queries. The first is the most direct query for the question; the others cover different angles (specific entities, comparisons, recent developments, limitations). Avoid near-duplicates — each query costs a paid search. Only add a year when recency matters.
- needs_academic: true when peer-reviewed or preprint research would materially improve the answer (science, medicine, ML methods, engineering research).
- needs_crawl: true only when the answer requires reading long documents in full (detailed documentation, specifications, long reports)."""


_COMPARISON_RE = re.compile(r"\s+(?:vs\.?|versus|or|compared (?:to|with))\s+", re.IGNORECASE)
_FOLLOW_UP_RE  = re.compile(r"\b(it|its|they|them|their|this|that|these|those|he|she|his|her|the same)\b", re.IGNORECASE)


def _fallback_plan(query: str, history: list[dict], reason: str) -> PlannerOutput:
    """Rule-based plan used when no LLM is available."""
    standalone = query.strip()
    if history and (len(query.split()) <= 7 or _FOLLOW_UP_RE.search(query)):
        prev = history[-1].get("standalone_query") or history[-1].get("query", "")
        context = " ".join(keywords(prev, limit=5))
        if context and context.lower() not in standalone.lower():
            standalone = f"{standalone} ({context})"

    lowered = standalone.lower()
    is_comparison = bool(_COMPARISON_RE.search(standalone)) or any(w in lowered for w in ("compare", "comparison", "difference between", "better than"))
    is_how_to     = lowered.startswith(("how to", "how do i", "how can i", "steps to"))
    is_academic   = any(w in lowered for w in ("paper", "study", "studies", "research", "algorithm", "model", "clinical", "theorem", "evidence"))
    is_recent     = any(w in lowered for w in ("latest", "recent", "news", "today", "this year", str(date.today().year)))

    core = " ".join(keywords(standalone, limit=8)) or standalone
    queries = [standalone if len(standalone) <= 120 else core]
    if is_comparison:
        parts = [p.strip(" ?.") for p in _COMPARISON_RE.split(standalone) if p.strip(" ?.")]
        if len(parts) >= 2:
            queries.append(f"{parts[0]} vs {parts[1]} differences")
        queries.append(f"{core} pros and cons")
    elif is_how_to:
        queries.append(f"{core} step by step guide")
        queries.append(f"{core} common mistakes")
    else:
        queries.append(f"{core} explained")
        queries.append(f"{core} limitations criticism")
    if is_recent:
        queries.append(f"{core} {date.today().year}")

    subtasks = [f"Core facts and definitions: {core}"]
    if is_comparison:
        subtasks += ["Side-by-side comparison of the options", "Trade-offs and when to choose each"]
    elif is_how_to:
        subtasks += ["Step-by-step procedure", "Pitfalls and best practices"]
    else:
        subtasks += ["Key evidence and examples", "Limitations, risks and open questions"]

    unique_queries = list(dict.fromkeys(q.strip() for q in queries if q.strip()))
    return PlannerOutput(
        original_query=query,
        standalone_query=standalone,
        research_goal=f"Give a well-sourced answer to: {standalone}",
        query_type="comparison" if is_comparison else "how_to" if is_how_to else "current_events" if is_recent else "explanatory",
        subtasks=subtasks[:cfg.planner_max_subtasks],
        search_queries=unique_queries[:max(1, cfg.planner_max_queries)],
        needs_academic=is_academic,
        needs_crawl=False,
        fallback_used=True,
        fallback_reason=reason,
    )


def _clean_queries(raw: list, standalone: str) -> list[str]:
    out: list[str] = []
    for q in raw or []:
        if not isinstance(q, str):
            continue
        q = re.sub(r"\s+", " ", q).strip().strip('"')
        if 3 <= len(q) <= 200 and q.lower() not in (x.lower() for x in out):
            out.append(q)
    if not out:
        out = [standalone]
    return out[:max(1, cfg.planner_max_queries)]


class PlannerAgent:
    def __init__(self):
        self.llm = get_llm()

    async def plan(self, query: str, history: Optional[list[dict]] = None) -> PlannerOutput:
        history = history or []
        logger.info("Planner: %r (history turns: %d)", query, len(history))

        prior = await recall(query, top_k=3, min_confidence="medium")

        if not await self.llm.is_available():
            output = _fallback_plan(query, history, "no LLM provider available")
            output.prior_findings = prior
            return output

        context = ""
        if history:
            turns = "\n".join(
                f"Q{i}: {t.get('standalone_query') or t.get('query')}\nA{i}: {(t.get('answer') or '')[:400]}"
                for i, t in enumerate(history[-3:], start=1)
            )
            context = f"Earlier turns in this conversation:\n{turns}\n\n"
        if prior:
            context += "Findings from earlier research (may be outdated):\n" + "\n".join(
                f"- {p.get('content', '')[:200]}" for p in prior
            ) + "\n\n"
        prompt = f"{context}Latest question: {query}"

        system = _PLANNER_SYSTEM.format(
            today=date.today().strftime("%d %B %Y"),
            max_subtasks=cfg.planner_max_subtasks,
            max_queries=max(2, cfg.planner_max_queries),
        )
        try:
            data = await self.llm.complete_json(prompt, system=system, schema=PLAN_SCHEMA,
                                                max_tokens=2000, effort="low", task="planner")
        except AllProvidersFailedError as exc:
            logger.warning("Planner LLM unavailable (%s) — rule-based plan", exc)
            output = _fallback_plan(query, history, "LLM call failed")
            output.prior_findings = prior
            return output

        standalone = str(data.get("standalone_query") or query).strip() or query
        output = PlannerOutput(
            original_query=query,
            standalone_query=standalone,
            research_goal=str(data.get("research_goal") or standalone),
            query_type=str(data.get("query_type") or "other"),
            subtasks=[s for s in (data.get("subtasks") or []) if isinstance(s, str)][:cfg.planner_max_subtasks],
            search_queries=_clean_queries(data.get("search_queries"), standalone),
            needs_academic=bool(data.get("needs_academic")),
            needs_crawl=bool(data.get("needs_crawl")),
            prior_findings=prior,
            fallback_used=False,
            llm_provider=last_call_provider("planner"),
        )
        logger.info("Planner: %d queries, academic=%s, crawl=%s", len(output.search_queries),
                    output.needs_academic, output.needs_crawl)
        return output

    async def refine(self, plan: PlannerOutput, found: int) -> list[str]:
        """Broader queries for an autonomous second retrieval pass when too little was found."""
        base = plan.standalone_query or plan.original_query
        if await self.llm.is_available():
            try:
                data = await self.llm.complete_json(
                    f"Question: {base}\nQueries already tried: {plan.search_queries}\n"
                    f"They found only {found} usable sources. Suggest broader or differently-worded queries.",
                    system="You improve web search queries. Return JSON {\"search_queries\": [...]} with 2-3 short keyword queries.",
                    schema={"type": "object", "properties": {"search_queries": {"type": "array", "items": {"type": "string"}}},
                            "required": ["search_queries"], "additionalProperties": False},
                    max_tokens=1000, effort="low", task="planner_refine",
                )
                queries = _clean_queries(data.get("search_queries"), base)
                tried = {q.lower() for q in plan.search_queries}
                fresh = [q for q in queries if q.lower() not in tried]
                if fresh:
                    return fresh[:3]
            except AllProvidersFailedError:
                pass
        words = keywords(base, limit=6)
        broader = [" ".join(words[:3]), " ".join(words[:5])]
        tried = {q.lower() for q in plan.search_queries}
        return [q for q in dict.fromkeys(broader) if q and q.lower() not in tried][:2] or [base]
