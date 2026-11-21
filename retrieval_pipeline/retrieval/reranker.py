"""
retrieval/reranker.py

Cross-encoder reranking (second-stage ranking) with graceful degradation.

  - The model loads lazily on the dedicated ML thread (or during API warm-up),
    so a missing model never blocks start-up or crashes a request.
  - Each document is scored on its passage most relevant to the query instead of
    its first characters (which, for web pages, are often navigation text).
  - If the model is unavailable, documents are ordered by their hybrid /
    provider relevance score instead (rerank_score = that score).
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from retrieval_pipeline.config import get_settings
from retrieval_pipeline.core.ml import model_ready, reranker_model, run_ml
from retrieval_pipeline.core.text import best_passages
from retrieval_pipeline.models.schemas import DocumentStatus, RetrievedDocument

logger = logging.getLogger(__name__)
cfg = get_settings()


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


def _fallback_rank(docs: list[RetrievedDocument], top_k: int) -> list[RetrievedDocument]:
    """Order by existing relevance_score when the cross-encoder is unavailable."""
    scored = []
    for doc in docs:
        d = doc.model_copy(deep=True)
        d.rerank_score = round(float(d.relevance_score or 0.0), 4)
        d.status = DocumentStatus.RERANKED
        scored.append(d)
    scored.sort(key=lambda d: d.rerank_score or 0.0, reverse=True)
    return scored[:top_k]


class CrossEncoderReranker:
    def __init__(self, model_name: Optional[str] = None):
        if model_name and model_name != reranker_model.model_name:
            logger.warning("Custom reranker model %s ignored; configure RERANKER_MODEL instead", model_name)

    @property
    def using_fallback(self) -> bool:
        return not cfg.reranker_enabled or not reranker_model.loaded

    def _score_batch(self, query: str, docs: list[RetrievedDocument]) -> Optional[list[float]]:
        model = reranker_model.get()
        if model is None:
            return None
        pairs = [(query, best_passages(query, d.content, max_chars=cfg.reranker_max_chars)) for d in docs]
        scores: list[float] = []
        for i in range(0, len(pairs), cfg.reranker_batch_size):
            batch = pairs[i:i + cfg.reranker_batch_size]
            try:
                scores.extend(float(s) for s in model.predict(batch, show_progress_bar=False))
            except Exception as exc:
                logger.warning("Reranker batch %d failed: %s — scoring as neutral", i, exc)
                scores.extend([0.0] * len(batch))
        return scores

    async def rerank(
        self,
        query: str,
        docs: list[RetrievedDocument],
        top_k: Optional[int] = None,
        score_threshold: float = -10.0,
    ) -> list[RetrievedDocument]:
        if not docs:
            return []
        top_k = top_k or cfg.reranker_top_k

        raw_scores = None
        if cfg.reranker_enabled and await model_ready(reranker_model):
            raw_scores = await run_ml(self._score_batch, query, docs)
        if raw_scores is None:
            result = _fallback_rank(docs, top_k)
            logger.info("Fallback rerank (no cross-encoder): %d → %d docs", len(docs), len(result))
            return result

        scored = []
        for doc, raw in zip(docs, raw_scores):
            d = doc.model_copy(deep=True)
            # Temperature keeps the 0–1 score informative (raw sigmoid saturates near 1.0).
            d.rerank_score = round(_sigmoid(raw / max(cfg.reranker_temperature, 1e-6)), 4)
            d.metadata.extra = {**d.metadata.extra, "rerank_logit": round(raw, 3)}
            d.status = DocumentStatus.RERANKED
            scored.append((raw, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        kept = [d for raw, d in scored if raw >= score_threshold]
        if len(kept) < min(3, len(scored)):          # never return (almost) nothing
            kept = [d for _, d in scored[:min(3, len(scored))]]
        result = kept[:top_k]
        logger.info("CrossEncoder rerank: %d → %d docs (top %.3f)", len(docs), len(result),
                    result[0].rerank_score if result else 0.0)
        return result

    async def rerank_with_diversity(
        self,
        query: str,
        docs: list[RetrievedDocument],
        top_k: Optional[int] = None,
        diversity_penalty: float = 0.15,
    ) -> list[RetrievedDocument]:
        """Rerank, then greedily pick documents while penalising repeated domains."""
        top_k = top_k or cfg.reranker_top_k
        candidates = await self.rerank(query, docs, top_k=max(top_k * 3, top_k))
        selected: list[RetrievedDocument] = []
        domain_counts: dict[str, int] = {}
        remaining = list(candidates)
        while remaining and len(selected) < top_k:
            best = max(
                remaining,
                key=lambda d: (d.rerank_score or 0.0) - diversity_penalty * domain_counts.get(d.metadata.domain, 0),
            )
            remaining.remove(best)
            selected.append(best)
            domain_counts[best.metadata.domain] = domain_counts.get(best.metadata.domain, 0) + 1
        return selected


_reranker_instance: Optional[CrossEncoderReranker] = None


def get_reranker() -> CrossEncoderReranker:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance
