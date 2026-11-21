"""
agents/verifier.py

Verifier Agent — cross-checks the key claims against the numbered sources.

  1. One LLM call extracts the claims that answer the question and maps each claim
     to the sources that support or contradict it.
  2. Confidence is then computed deterministically from the evidence — number of
     independent supporting domains, their credibility, and contradictions — so the
     model cannot simply declare a claim "high confidence".
  3. Claims citing non-existent sources, or with no support at all, are flagged as
     hallucination risks.

Falls back to a heuristic cross-reference (sentence extraction + term overlap)
when no LLM is available. (The previous version made up to 26 sequential LLM calls.)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Optional

from retrieval_pipeline.config import get_settings
from retrieval_pipeline.core.errors import AllProvidersFailedError
from retrieval_pipeline.core.text import bigrams, relevant_sentences, truncate
from retrieval_pipeline.llm.client import get_llm, last_call_provider
from retrieval_pipeline.agents.sources import format_sources_for_prompt

logger = logging.getLogger(__name__)
cfg = get_settings()


@dataclass
class VerifiedClaim:
    claim:               str
    confidence:          str            # high | medium | low
    confidence_score:    float          # 0–1
    assessment:          str = "supported"
    supporting_sources:  list[int] = field(default_factory=list)
    contradicting_sources: list[int] = field(default_factory=list)
    supporting_urls:     list[str] = field(default_factory=list)
    source_domains:      list[str] = field(default_factory=list)
    contradictions:      list[str] = field(default_factory=list)
    note:                str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VerifierOutput:
    query:                 str
    claims:                list[VerifiedClaim] = field(default_factory=list)
    hallucination_flags:   int = 0
    high_confidence_count: int = 0
    contradictions_found:  int = 0
    fallback_used:         bool = False
    llm_provider:          Optional[str] = None


CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim":                 {"type": "string"},
                    "supporting_sources":    {"type": "array", "items": {"type": "integer"}},
                    "contradicting_sources": {"type": "array", "items": {"type": "integer"}},
                    "assessment":            {"type": "string", "enum": ["supported", "partially_supported", "contested", "unsupported"]},
                    "note":                  {"type": "string"},
                },
                "required": ["claim", "supporting_sources", "contradicting_sources", "assessment", "note"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["claims"],
    "additionalProperties": False,
}

_VERIFY_SYSTEM = """You are a careful fact-checker for a research assistant.

From the numbered sources, extract up to {max_claims} specific, checkable factual claims that matter for answering the question (numbers, names, capabilities, dates, causal statements — not opinions or marketing).

For every claim:
- supporting_sources: the numbers of ALL sources that state or directly support it.
- contradicting_sources: the numbers of sources that disagree with it (empty if none).
- assessment: supported (clearly stated), partially_supported (implied or only partly stated), contested (sources disagree) or unsupported.
- note: a short explanation when a claim is contested or only partially supported; otherwise an empty string.

Use only the source numbers shown. Do not add knowledge that is not in the sources."""

_BASE_SCORE = {"supported": 0.5, "partially_supported": 0.35, "contested": 0.2, "unsupported": 0.05}


def score_claim(assessment: str, support: list[int], contra: list[int], sources: list[dict]) -> tuple[float, str]:
    """Evidence-based confidence: independent domains + credibility − contradictions."""
    by_id = {s["id"]: s for s in sources}
    supporting = [by_id[i] for i in support if i in by_id]
    domains = {s["domain"] for s in supporting}
    credibility = max((s["credibility"] for s in supporting), default=0.0)

    score = _BASE_SCORE.get(assessment, 0.3)
    score += 0.15 * min(max(len(domains) - 1, 0), 2)
    score += 0.2 * credibility
    if contra:
        score -= 0.15
    if not supporting:
        score = min(score, 0.15)
    if len(domains) < cfg.verifier_min_sources:
        score = min(score, 0.7)                 # single-source claims can't be "high"
    score = round(max(0.0, min(1.0, score)), 3)
    label = "high" if score >= 0.72 else "medium" if score >= 0.45 else "low"
    return score, label


def _build_claim(text: str, assessment: str, support: list[int], contra: list[int], note: str,
                 sources: list[dict]) -> VerifiedClaim:
    by_id = {s["id"]: s for s in sources}
    score, label = score_claim(assessment, support, contra, sources)
    return VerifiedClaim(
        claim=truncate(text.strip(), 400),
        confidence=label,
        confidence_score=score,
        assessment=assessment,
        supporting_sources=[i for i in support if i in by_id],
        contradicting_sources=[i for i in contra if i in by_id],
        supporting_urls=[by_id[i]["url"] for i in support if i in by_id and by_id[i]["url"]],
        source_domains=sorted({by_id[i]["domain"] for i in support if i in by_id}),
        contradictions=[f"Source [{i}] ({by_id[i]['domain']}) disagrees" for i in contra if i in by_id],
        note=note or "",
    )


def _summarise(query: str, claims: list[VerifiedClaim], fallback: bool, hallucinations: int = 0,
               provider: Optional[str] = None) -> VerifierOutput:
    claims.sort(key=lambda c: c.confidence_score, reverse=True)
    return VerifierOutput(
        query=query,
        claims=claims,
        hallucination_flags=hallucinations,
        high_confidence_count=sum(1 for c in claims if c.confidence == "high"),
        contradictions_found=sum(1 for c in claims if c.contradicting_sources or c.assessment == "contested"),
        fallback_used=fallback,
        llm_provider=provider,
    )


def heuristic_verify(query: str, sources: list[dict]) -> VerifierOutput:
    """
    No-LLM fallback: query-relevant statements from each source, cross-referenced by
    phrase (bigram) overlap — shared phrasing is far stronger evidence of "the same
    claim" than shared topic words, which every source on the topic has.
    """
    claims: list[VerifiedClaim] = []
    seen: set[str] = set()
    source_bigrams = {s["id"]: bigrams(s["excerpt"]) for s in sources}

    for source in sources:
        for sentence in relevant_sentences(query, source["excerpt"], limit=2):
            key = sentence[:80].lower()
            if key in seen or len(sentence) < 50:
                continue
            seen.add(key)
            claim_bigrams = bigrams(sentence)
            support = [source["id"]]
            if len(claim_bigrams) >= 4:
                for other in sources:
                    if other["id"] == source["id"]:
                        continue
                    shared = len(claim_bigrams & source_bigrams[other["id"]]) / len(claim_bigrams)
                    if shared >= 0.35:
                        support.append(other["id"])
            claims.append(_build_claim(sentence, "supported", support, [], "", sources))
            if len(claims) >= cfg.verifier_max_claims:
                return _summarise(query, claims, fallback=True)
    return _summarise(query, claims, fallback=True)


class VerifierAgent:
    def __init__(self):
        self.llm = get_llm()

    async def verify(self, query: str, sources: list[dict]) -> VerifierOutput:
        logger.info("Verifier: %d sources", len(sources))
        if not sources:
            return VerifierOutput(query=query)
        if not await self.llm.is_available():
            return heuristic_verify(query, sources)

        prompt = f"Question: {query}\n\nSources:\n\n{format_sources_for_prompt(sources)}"
        try:
            data = await self.llm.complete_json(
                prompt,
                system=_VERIFY_SYSTEM.format(max_claims=cfg.verifier_max_claims),
                schema=CLAIMS_SCHEMA, max_tokens=4000, effort="medium", task="verifier",
            )
        except AllProvidersFailedError as exc:
            logger.warning("Verifier LLM unavailable (%s) — heuristic verification", exc)
            return heuristic_verify(query, sources)

        valid_ids = {s["id"] for s in sources}
        claims: list[VerifiedClaim] = []
        hallucinations = 0
        for raw in (data.get("claims") or [])[:cfg.verifier_max_claims]:
            if not isinstance(raw, dict) or not str(raw.get("claim", "")).strip():
                continue
            support = [int(i) for i in raw.get("supporting_sources") or [] if str(i).lstrip("-").isdigit()]
            contra  = [int(i) for i in raw.get("contradicting_sources") or [] if str(i).lstrip("-").isdigit()]
            assessment = raw.get("assessment") if raw.get("assessment") in _BASE_SCORE else "partially_supported"
            invalid = [i for i in support + contra if i not in valid_ids]
            if invalid or not [i for i in support if i in valid_ids] or assessment == "unsupported":
                hallucinations += 1
            claims.append(_build_claim(str(raw["claim"]), assessment, support, contra, str(raw.get("note") or ""), sources))

        if not claims:
            logger.warning("Verifier: LLM returned no claims — heuristic verification")
            return heuristic_verify(query, sources)

        out = _summarise(query, claims, fallback=False, hallucinations=hallucinations,
                         provider=last_call_provider("verifier"))
        logger.info("Verifier: %d claims — %d high, %d contested, %d flagged",
                    len(out.claims), out.high_confidence_count, out.contradictions_found, out.hallucination_flags)
        return out
