"""
retrieval/hybrid_search.py

Hybrid ranking: BM25 (sparse) + BGE embeddings (dense) fused with
Reciprocal Rank Fusion (RRF).

FIX (v4): the previous version kept ONE shared BM25 index that was rebuilt
*after* each request and searched *during* the next one, so HYBRID returned the
previous query's documents (and nothing on the first query; concurrent requests
overwrote each other's index). Hybrid ranking is now stateless and runs on the
candidate documents of the current request — first-stage ranking before the
cross-encoder reranker. HybridRetriever.retrieve() additionally searches the
persistent knowledge base for knowledge-base-only queries.

Graceful degradation:
  - embedding model unavailable → BM25-only
  - rank_bm25 unavailable       → dense-only
  - both unavailable            → original order preserved
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from retrieval_pipeline.config import get_settings
from retrieval_pipeline.core.ml import run_ml
from retrieval_pipeline.core.text import _score_passages, best_passages, tokenise
from retrieval_pipeline.models.schemas import RetrievalMethod, RetrievedDocument

logger = logging.getLogger(__name__)
cfg = get_settings()

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except Exception as e:  # pragma: no cover
    logger.warning("rank_bm25 unavailable (%s) — sparse ranking disabled", e)
    _BM25_AVAILABLE = False


def _tokenise(text: str) -> list[str]:          # kept for backwards compatibility
    return tokenise(text)


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    doc_map: dict[str, RetrievedDocument],
    k: Optional[int] = None,
    weights: Optional[list[float]] = None,
) -> list[RetrievedDocument]:
    """
    Fuse ranked lists of doc_ids:  score(d) = Σ weight_i / (k + rank_i(d))
    Documents keep their original source_method; the fused score is stored in
    relevance_score (normalised to 0–1) and extra["hybrid_score"].
    """
    k = k or cfg.rrf_k
    weights = weights or [1.0] * len(ranked_lists)

    scores: dict[str, float] = defaultdict(float)
    for ranked, weight in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] += weight / (k + rank)

    max_score = max(scores.values(), default=1.0) or 1.0
    fused = []
    for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        if doc_id not in doc_map:
            continue
        doc = doc_map[doc_id].model_copy(deep=True)
        doc.relevance_score = round(score / max_score, 4)
        doc.metadata.extra = {**doc.metadata.extra, "hybrid_score": round(score, 6)}
        fused.append(doc)
    return fused


# ── BM25 index (per call) ─────────────────────────────────────────────────────

class BM25Index:
    """Wraps BM25Okapi. search() returns [] gracefully if BM25 is unavailable."""

    def __init__(self, docs: Optional[list[RetrievedDocument]] = None, texts: Optional[list[str]] = None):
        self._corpus: list[RetrievedDocument] = []
        self._bm25 = None
        if docs:
            self.build(docs, texts)

    def build(self, docs: list[RetrievedDocument], texts: Optional[list[str]] = None) -> None:
        # Smoothed IDF (log(1 + …)) instead of rank_bm25's Okapi IDF, which is zero or
        # negative for terms in half or more of a small candidate set — i.e. exactly the
        # per-request case — and silently disabled the sparse signal.
        self._corpus = list(docs)
        self._texts = list(texts or [d.content for d in docs])
        self._bm25 = True if docs else None

    def search(self, query: str, top_k: int) -> list[tuple[RetrievedDocument, float]]:
        if self._bm25 is None:
            return []
        tokens = tokenise(query)
        if not tokens:
            return []
        try:
            scores = _score_passages(tokens, self._texts)
        except Exception as e:
            logger.warning("BM25 search failed: %s", e)
            return []
        ranked = sorted(zip(self._corpus, scores), key=lambda x: x[1], reverse=True)
        return [(doc, float(score)) for doc, score in ranked[:top_k] if score > 0]

    @property
    def size(self) -> int:
        return len(self._corpus)


# ── Hybrid ranking ────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    rank(): fuse BM25 + dense similarity over a candidate set (stateless, per request).
    retrieve(): knowledge-base search + hybrid re-ranking of the hits.
    """

    def __init__(self):
        self._embedder = None
        self._vector = None

    def _get_embedder(self):
        if self._embedder is None:
            from retrieval_pipeline.retrieval.vector_rag import EmbeddingService
            self._embedder = EmbeddingService()
        return self._embedder

    def _get_vector(self):
        if self._vector is None:
            from retrieval_pipeline.retrieval.vector_rag import VectorRetriever
            self._vector = VectorRetriever()
        return self._vector

    def build_bm25_index(self, docs: list[RetrievedDocument]) -> None:
        """Deprecated no-op: hybrid ranking now builds its index per request."""

    async def _dense_order(self, query: str, docs: list[RetrievedDocument], texts: list[str]) -> list[str]:
        embedder = self._get_embedder()
        if not await embedder.available():
            return []

        def _score() -> list[float]:
            import numpy as np
            q = np.asarray(embedder.embed_query(query))
            d = np.asarray(embedder.embed_documents(texts))
            return (d @ q).tolist()

        try:
            sims = await run_ml(_score)
        except Exception as exc:
            logger.warning("Dense scoring failed: %s — BM25 only", exc)
            return []
        order = sorted(range(len(docs)), key=lambda i: sims[i], reverse=True)
        for i, sim in enumerate(sims):
            docs[i].metadata.extra["dense_similarity"] = round(float(sim), 4)
        return [docs[i].doc_id for i in order]

    async def rank(
        self,
        query: str,
        docs: list[RetrievedDocument],
        top_n: Optional[int] = None,
        use_dense: bool = True,
    ) -> list[RetrievedDocument]:
        """First-stage ranking of candidates. Returns at most top_n documents."""
        if not docs:
            return []
        top_n = top_n or cfg.hybrid_candidates
        docs = [d.model_copy(deep=True) for d in docs]
        doc_map = {d.doc_id: d for d in docs}
        # ~800 characters per document keeps dense scoring fast on CPU (cost grows with length).
        texts = [best_passages(query, d.content, max_chars=800) for d in docs]

        bm25_ranked = [d.doc_id for d, _ in BM25Index(docs, texts).search(query, len(docs))]
        dense_ranked = await self._dense_order(query, docs, texts) if use_dense else []

        ranked_lists, weights = [], []
        if dense_ranked:
            ranked_lists.append(dense_ranked)
            weights.append(cfg.hybrid_alpha)
        if bm25_ranked:
            ranked_lists.append(bm25_ranked)
            weights.append(1.0 - cfg.hybrid_alpha)
        # Prior order (provider ranking) acts as a weak third signal and keeps every doc in play.
        prior = [d.doc_id for d in sorted(docs, key=lambda d: d.relevance_score or 0.0, reverse=True)]
        ranked_lists.append(prior)
        weights.append(0.15)

        fused = reciprocal_rank_fusion(ranked_lists, doc_map, weights=weights)
        logger.info("Hybrid rank: %d candidates (dense=%s, bm25=%d) → keep %d",
                    len(docs), bool(dense_ranked), len(bm25_ranked), min(top_n, len(fused)))
        return fused[:top_n]

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None,
        **_: object,
    ) -> list[RetrievedDocument]:
        """Knowledge-base retrieval with hybrid re-ranking. Never raises."""
        top_k = top_k or cfg.qdrant_top_k
        try:
            hits = await self._get_vector().retrieve(query, top_k=top_k * 2, filters=filters, min_score=0.3)
        except Exception as exc:
            logger.warning("Hybrid knowledge-base search failed: %s", exc)
            return []
        ranked = await self.rank(query, hits, top_n=top_k)
        for doc in ranked:
            doc.metadata = doc.metadata.model_copy(update={"source_method": RetrievalMethod.HYBRID})
        return ranked

    async def retrieve_multi_query(self, queries: list[str], top_k: Optional[int] = None) -> list[RetrievedDocument]:
        import asyncio
        top_k = top_k or cfg.qdrant_top_k
        results = await asyncio.gather(*(self.retrieve(q, top_k=top_k) for q in queries), return_exceptions=True)
        doc_map: dict[str, RetrievedDocument] = {}
        lists: list[list[str]] = []
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("Multi-query hybrid sub-task failed: %s", result)
                continue
            lists.append([d.doc_id for d in result])
            doc_map.update({d.doc_id: d for d in result})
        return reciprocal_rank_fusion(lists, doc_map)[:top_k] if lists else []
