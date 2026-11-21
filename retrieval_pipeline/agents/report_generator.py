"""
agents/report_generator.py

Report Generator Agent — the final synthesis steps.

  1. answer   — a direct, cited Markdown answer streamed token by token
                (citations like [1][3] refer to the numbered sources)
  2. report   — a structured research report: executive summary, key findings with
                evidence-checked confidence, trade-offs, related questions,
                confidence breakdown and the full citation list

Both steps fall back to extractive synthesis (built from verified claims and
source sentences) when no LLM provider is usable, so a run always produces output.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Optional
from uuid import uuid4

from retrieval_pipeline.config import get_settings
from retrieval_pipeline.core.errors import AllProvidersFailedError
from retrieval_pipeline.core.text import keywords, relevant_sentences, truncate
from retrieval_pipeline.llm.client import StreamChunk, get_llm, last_call_provider
from retrieval_pipeline.agents.sources import (
    cited_ids,
    citations_for,
    format_sources_for_prompt,
    public_source,
    sanitize_citations,
)
from retrieval_pipeline.agents.verifier import VerifierOutput, score_claim

logger = logging.getLogger(__name__)
cfg = get_settings()

_RANK = {"low": 0, "medium": 1, "high": 2}

_ANSWER_SYSTEM = """You are a meticulous research analyst. Answer the user's question using only the numbered sources provided.

- Open with a direct answer in one to three sentences, then give the supporting detail.
- Cite every factual statement with the source number(s) in square brackets right after it, e.g. "… under 10 ms [2][5]." Use only the numbers shown.
- When sources disagree, say so and cite both sides. When the sources do not cover part of the question, say that plainly instead of filling the gap.
- Use Markdown: short paragraphs, bullet points where they help, "###" headings for longer answers, and a table for comparisons.
- Do not add a references or sources list at the end — the interface shows the sources.
- Write in the same language as the question."""

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "executive_summary": {"type": "string"},
        "key_findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "headline":   {"type": "string"},
                    "detail":     {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "source_ids": {"type": "array", "items": {"type": "integer"}},
                    "tags":       {"type": "array", "items": {"type": "string"}},
                },
                "required": ["headline", "detail", "confidence", "source_ids", "tags"],
                "additionalProperties": False,
            },
        },
        "tradeoffs":         {"type": "string"},
        "related_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["executive_summary", "key_findings", "tradeoffs", "related_questions"],
    "additionalProperties": False,
}

_REPORT_SYSTEM = """You turn research results into a structured report.

Return:
- executive_summary: 3–5 sentences for a non-specialist; lead with the most important conclusion and mention notable uncertainty or disagreement.
- key_findings: up to {max_findings} findings that group related evidence. Each has a one-sentence headline, a 2–4 sentence detail with specifics, the source numbers it relies on (source_ids), 1–3 short topic tags, and a confidence of high / medium / low that reflects how strong and consistent the evidence is.
- tradeoffs: a short Markdown paragraph on pros/cons or trade-offs if the question involves a choice; otherwise an empty string.
- related_questions: 3–4 natural follow-up questions a curious reader would ask next.

Use only the information in the sources and verified claims."""


def _verified_block(verified: Optional[VerifierOutput], limit: int = 12) -> str:
    if not verified or not verified.claims:
        return "No claims were extracted."
    return "\n".join(
        f"- [{c.confidence.upper()}] {c.claim} (sources: {', '.join(str(i) for i in c.supporting_sources) or 'none'})"
        + (f" — note: {c.note}" if c.note else "")
        for c in verified.claims[:limit]
    )


def _history_block(history: list[dict]) -> str:
    if not history:
        return ""
    turns = "\n".join(f"- Q: {t.get('standalone_query') or t.get('query')}" for t in history[-3:])
    return f"Earlier questions in this conversation:\n{turns}\n\n"


# ── Extractive fallbacks (no LLM) ─────────────────────────────────────────────

def extractive_answer(query: str, sources: list[dict], verified: Optional[VerifierOutput]) -> str:
    """A cited bullet summary assembled from verified claims / source sentences."""
    if not sources:
        return ("I couldn't retrieve any sources for this question. Check the provider status panel — "
                "search API credits may be exhausted or the network may be unavailable.")
    bullets: list[str] = []
    used: set[str] = set()
    for claim in (verified.claims if verified else []):
        if claim.confidence == "low" or not claim.supporting_sources:
            continue
        key = claim.claim[:60].lower()
        if key in used:
            continue
        used.add(key)
        refs = "".join(f"[{i}]" for i in claim.supporting_sources[:3])
        bullets.append(f"- {claim.claim.rstrip('.')}. {refs}")
        if len(bullets) >= 6:
            break
    if len(bullets) < 3:
        for s in sources[:6]:
            for sentence in relevant_sentences(query, s["excerpt"], limit=1):
                key = sentence[:60].lower()
                if key not in used:
                    used.add(key)
                    bullets.append(f"- {sentence.rstrip('.')}. [{s['id']}]")
    lead = (f"**Key points from {len(sources)} sources** "
            f"({', '.join(sorted({s['domain'] for s in sources})[:4])}{', …' if len({s['domain'] for s in sources}) > 4 else ''}):")
    note = ("\n\n_This is an extractive summary. Add an LLM API key (e.g. ANTHROPIC_API_KEY) in `.env` "
            "for an AI-written, synthesised answer._")
    return lead + "\n\n" + "\n".join(bullets[:8]) + note


def _fallback_report_parts(query: str, sources: list[dict], verified: Optional[VerifierOutput],
                           plan: Optional[dict], answer: str) -> dict:
    findings = []
    for claim in (verified.claims if verified else [])[:cfg.report_max_findings * 2]:
        if claim.confidence == "low":
            continue
        findings.append({
            "headline":   truncate(claim.claim, 160),
            "detail":     claim.claim + (f" ({claim.note})" if claim.note else ""),
            "confidence": claim.confidence,
            "source_ids": claim.supporting_sources,
            "tags":       [],
        })
        if len(findings) >= cfg.report_max_findings:
            break
    domains = sorted({s["domain"] for s in sources})
    lead = " ".join(f["detail"].rstrip(".") + "." for f in findings[:2])
    summary = (
        (lead + " " if lead else "")
        + f"This report draws on {len(sources)} sources from {len(domains)} domains"
        + (f"; {len(findings)} findings were cross-checked, and confidence reflects how many "
           f"independent sources state the same thing." if findings else ".")
    ).strip()
    return {"executive_summary": summary, "key_findings": findings, "tradeoffs": "",
            "related_questions": _fallback_related(query, plan)}


def _fallback_related(query: str, plan: Optional[dict]) -> list[str]:
    """Template follow-up questions (used only when no LLM is available)."""
    generic = {"main", "key", "benefits", "risks", "pros", "cons", "effective", "compared", "advantages",
               "disadvantages", "better", "good", "bad", "important", "difference", "differences"}
    words = [w for w in keywords((plan or {}).get("standalone_query") or query, limit=10) if w not in generic]
    topic = " ".join(words[:4]) or query.rstrip("?")
    qtype = (plan or {}).get("query_type", "")
    if qtype == "comparison":
        return [f"What are the main trade-offs in {topic}?", f"When is each option the better choice ({topic})?",
                f"What do independent benchmarks or studies say about {topic}?"]
    if qtype == "how_to":
        return [f"What are common mistakes with {topic}?", f"What tools help with {topic}?",
                f"How long does {topic} usually take?"]
    return [f"What are the limitations or risks of {topic}?", f"What does recent research say about {topic}?",
            f"How is {topic} applied in practice?"]


# ── Agent ─────────────────────────────────────────────────────────────────────

class ReportGeneratorAgent:
    def __init__(self):
        self.llm = get_llm()

    async def stream_answer(
        self,
        query: str,
        sources: list[dict],
        verified: Optional[VerifierOutput],
        history: Optional[list[dict]] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Yields StreamChunk tokens. A chunk with reset=True means "discard what you have
        so far" (a provider failed mid-answer and the next one restarts). The extractive
        fallback is emitted as a single chunk with provider="extractive".
        """
        if sources and await self.llm.is_available():
            prompt = (
                f"{_history_block(history or [])}Question: {query}\n\n"
                f"Verified claims (with confidence):\n{_verified_block(verified)}\n\n"
                f"Sources:\n\n{format_sources_for_prompt(sources)}"
            )
            emitted = False
            try:
                async for chunk in self.llm.stream(prompt, system=_ANSWER_SYSTEM, max_tokens=6000,
                                                   effort=cfg.anthropic_effort, task="answer"):
                    emitted = emitted or bool(chunk.text)
                    yield chunk
                return
            except AllProvidersFailedError as exc:
                logger.warning("Answer synthesis failed on every provider (%s) — extractive answer", exc)
                if emitted:
                    yield StreamChunk(reset=True, provider="extractive")
        yield StreamChunk(text=extractive_answer(query, sources, verified), provider="extractive", model="extractive")

    async def generate_answer(self, query: str, sources: list[dict], verified: Optional[VerifierOutput],
                              history: Optional[list[dict]] = None) -> tuple[str, str]:
        """Non-streaming convenience wrapper. Returns (answer_markdown, provider)."""
        parts: list[str] = []
        provider = "extractive"
        async for chunk in self.stream_answer(query, sources, verified, history):
            if chunk.reset:
                parts = []
                continue
            parts.append(chunk.text)
            provider = chunk.provider or provider
        return sanitize_citations("".join(parts).strip(), len(sources)), provider

    async def build_report(
        self,
        query: str,
        sources: list[dict],
        verified: Optional[VerifierOutput],
        plan: Optional[dict],
        answer: str,
        answer_provider: str,
    ) -> dict:
        parts: Optional[dict] = None
        report_provider = None
        fallback = True
        if sources and await self.llm.is_available():
            prompt = (
                f"Question: {query}\n"
                f"Research goal: {(plan or {}).get('research_goal', query)}\n\n"
                f"Verified claims:\n{_verified_block(verified)}\n\n"
                f"Answer already given to the user:\n{truncate(answer, 3000)}\n\n"
                f"Sources:\n\n{format_sources_for_prompt(sources)}"
            )
            try:
                parts = await self.llm.complete_json(
                    prompt, system=_REPORT_SYSTEM.format(max_findings=cfg.report_max_findings),
                    schema=REPORT_SCHEMA, max_tokens=5000, effort="medium", task="report",
                )
                report_provider = last_call_provider("report")
                fallback = False
            except AllProvidersFailedError as exc:
                logger.warning("Report LLM unavailable (%s) — extractive report", exc)
        if parts is None:
            parts = _fallback_report_parts(query, sources, verified, plan, answer)
        return self._assemble(query, sources, verified, plan, answer, answer_provider, parts, fallback, report_provider)

    def _assemble(self, query, sources, verified, plan, answer, answer_provider, parts, fallback, report_provider) -> dict:
        findings = []
        for raw in (parts.get("key_findings") or [])[:cfg.report_max_findings]:
            if not isinstance(raw, dict) or not raw.get("headline"):
                continue
            ids = [int(i) for i in raw.get("source_ids") or [] if str(i).isdigit() and 1 <= int(i) <= len(sources)]
            stated = raw.get("confidence") if raw.get("confidence") in _RANK else "medium"
            # The model's confidence is capped by what the cited evidence can support.
            _, evidence = score_claim("supported", ids, [], sources)
            confidence = stated if _RANK[stated] <= _RANK[evidence] else evidence
            findings.append({
                "headline":   str(raw["headline"]).strip(),
                "detail":     sanitize_citations(str(raw.get("detail") or ""), len(sources)),
                "confidence": confidence,
                "tags":       [str(t) for t in (raw.get("tags") or [])][:4],
                "citations":  citations_for(ids, sources),
            })

        claims = verified.claims if verified else []
        total = len(claims)
        high = sum(1 for c in claims if c.confidence == "high")
        medium = sum(1 for c in claims if c.confidence == "medium")
        low = total - high - medium
        overall = round((high * 1.0 + medium * 0.6 + low * 0.2) / total, 2) if total else 0.0

        used_ids = cited_ids(answer)
        all_citations = [
            {**public_source(s), "cited_in_answer": s["id"] in used_ids}
            for s in sources[:cfg.report_max_citations]
        ]
        related = [q.strip() for q in (parts.get("related_questions") or []) if isinstance(q, str) and q.strip()][:4]
        tradeoffs = (parts.get("tradeoffs") or "").strip() or None

        return {
            "report_id":         f"rpt_{uuid4().hex[:10]}",
            "query":             query,
            "standalone_query":  (plan or {}).get("standalone_query") or query,
            "research_goal":     (plan or {}).get("research_goal") or query,
            "created_at":        datetime.now(timezone.utc).isoformat(),
            "answer":            answer,
            "executive_summary": str(parts.get("executive_summary") or "").strip(),
            "key_findings":      findings,
            "tradeoffs":         sanitize_citations(tradeoffs, len(sources)) if tradeoffs else None,
            "related_questions": related,
            "next_steps":        related,             # backwards-compatible name
            "all_citations":     all_citations,
            "sources":           all_citations,
            "verified_claims":   [c.to_dict() for c in claims],
            "confidence_summary": {
                "overall_score":         overall,
                "high_confidence_pct":   round(100 * high / total, 1) if total else 0.0,
                "medium_confidence_pct": round(100 * medium / total, 1) if total else 0.0,
                "low_confidence_pct":    round(100 * low / total, 1) if total else 0.0,
                "total_claims":          total,
                "verified_citations":    len(all_citations),
                "cited_in_answer":       len([i for i in used_ids if i <= len(sources)]),
                "hallucination_flags":   verified.hallucination_flags if verified else 0,
                "contradictions_found":  verified.contradictions_found if verified else 0,
            },
            "agent_trace": {
                "llm_used":         answer_provider != "extractive" or not fallback,
                "answer_provider":  answer_provider,
                "report_provider":  report_provider or "extractive",
                "verifier_mode":    "heuristic" if (verified is None or verified.fallback_used) else "llm",
                "docs_retrieved":   len(sources),
                "claims_checked":   total,
                "plan_subtasks":    len((plan or {}).get("subtasks", [])),
            },
            "fallback_used": fallback and answer_provider == "extractive",
        }

    def build_report_offline(self, query: str, sources: list[dict], verified: Optional[VerifierOutput],
                             plan: Optional[dict], answer: str = "", answer_provider: str = "extractive") -> dict:
        """Report without any LLM call (used when a run times out or the reporter fails)."""
        answer = answer or extractive_answer(query, sources, verified)
        parts = _fallback_report_parts(query, sources, verified, plan, answer)
        return self._assemble(query, sources, verified, plan, answer, answer_provider, parts, True, None)

    def empty_report(self, query: str, reason: str) -> dict:
        return self._assemble(query, [], None, None, extractive_answer(query, [], None), "extractive",
                              {"executive_summary": reason, "key_findings": [], "tradeoffs": "", "related_questions": []},
                              True, None)
