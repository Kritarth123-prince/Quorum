"""
retrieval/router.py

Central Retrieval Router — the single entry point for agent nodes and the API.

Two stages:

  1. GATHER (concurrent, each method isolated — one failing never blocks the others)
        web_search ─┐
        academic   ─┼─► candidates ──► web_crawl (deep-read top result pages / seed URLs)
        knowledge  ─┘     (vector_rag / hybrid → local Qdrant knowledge base)

  2. RANK (once, against the user's question)
        dedup ──► hybrid fusion (BM25 + dense, RRF) ──► cross-encoder rerank ──► top-k

Freshly fetched pages are indexed into the knowledge base in the background
(AUTO_INDEX), so VECTOR_RAG / HYBRID get better over time.

FIXES vs v3:
  - HYBRID no longer returns the previous request's documents (stateless ranking).
  - Multi-query fuses the pooled results and ranks them ONCE against the original
    question (previously: reranked per sub-query, then against queries[0] only).
  - Crawl seeds come from the search results already fetched (no extra paid search).
  - Background index tasks are kept referenced so they can't be garbage-collected.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Optional

from retrieval_pipeline.config import get_settings
from retrieval_pipeline.core.errors import redact
from retrieval_pipeline.core.provider_health import notice_scope
from retrieval_pipeline.core.text import normalise_url
from retrieval_pipeline.models.schemas import (
    RetrievalMethod,
    RetrievalQuery,
    RetrievalResult,
    RetrievedDocument,
)
from retrieval_pipeline.retrieval.academic import academic_search
from retrieval_pipeline.retrieval.hybrid_search import HybridRetriever
from retrieval_pipeline.retrieval.reranker import get_reranker
from retrieval_pipeline.retrieval.vector_rag import VectorRetriever
from retrieval_pipeline.retrieval.web_crawl import crawl_batch
from retrieval_pipeline.retrieval.web_search import web_search

logger = logging.getLogger(__name__)
cfg = get_settings()

_EXTERNAL_METHODS = (RetrievalMethod.WEB_SEARCH, RetrievalMethod.WEB_CRAWL, RetrievalMethod.ACADEMIC)


# ── De-duplication ────────────────────────────────────────────────────────────

def _content_fingerprint(doc: RetrievedDocument) -> str:
    snippet = " ".join(doc.content[:500].lower().split())
    return hashlib.md5(snippet.encode()).hexdigest()


# Fresh, full content wins over stored knowledge-base chunks of the same page.
_DEDUP_PRIORITY = {
    RetrievalMethod.WEB_CRAWL:  0,
    RetrievalMethod.WEB_SEARCH: 1,
    RetrievalMethod.ACADEMIC:   2,
    RetrievalMethod.HYBRID:     3,
    RetrievalMethod.VECTOR_RAG: 4,
}


def _dedup_documents(docs: list[RetrievedDocument]) -> list[RetrievedDocument]:
    ordered = sorted(
        enumerate(docs),
        key=lambda pair: (_DEDUP_PRIORITY.get(pair[1].metadata.source_method, 9), -len(pair[1].content), pair[0]),
    )
    seen_fps: set[str] = set()
    seen_urls: set[str] = set()
    keep: list[tuple[int, RetrievedDocument]] = []
    for idx, doc in ordered:
        fp  = _content_fingerprint(doc)
        url = normalise_url(doc.metadata.url or "")
        if fp in seen_fps or (url and url in seen_urls):
            continue
        seen_fps.add(fp)
        if url:
            seen_urls.add(url)
        keep.append((idx, doc))
    keep.sort(key=lambda pair: pair[0])         # restore original (provider rank) order
    return [doc for _, doc in keep]


def _interleave_urls(groups: list[list[RetrievedDocument]], limit: int) -> list[str]:
    urls: list[str] = []
    for i in range(max((len(g) for g in groups), default=0)):
        for group in groups:
            if i < len(group) and group[i].metadata.url and group[i].metadata.url not in urls:
                urls.append(group[i].metadata.url)
            if len(urls) >= limit:
                return urls
    return urls


# ── Router ────────────────────────────────────────────────────────────────────

class RetrievalRouter:
    """
    Orchestrates every retrieval method with graceful degradation:
      no Qdrant server   → embedded knowledge base (or none) — no crash
      no search credits  → next provider in the chain (Tavily → SerpAPI → DuckDuckGo)
      no ML models       → BM25 / provider-score ranking
    """

    def __init__(self):
        self._vector_retriever = VectorRetriever()
        self._hybrid_retriever = HybridRetriever()
        self._reranker         = get_reranker()
        self._bg_tasks: set[asyncio.Task] = set()
        logger.info("RetrievalRouter initialised")

    # ── Stage 1: gather ──────────────────────────────────────────────────────

    async def _timed(self, method: RetrievalMethod, coro) -> tuple[RetrievalMethod, list[RetrievedDocument], float, Optional[str]]:
        t0 = time.perf_counter()
        try:
            docs = await coro
            error = None
        except Exception as exc:                 # isolation: one method never sinks the others
            logger.exception("%s failed", method.value)
            docs, error = [], f"{method.value}: {redact(exc)}"
        return method, docs or [], (time.perf_counter() - t0) * 1000, error

    async def gather_candidates(
        self,
        text: str,
        methods: list[RetrievalMethod],
        max_results: int = 10,
        filters: Optional[dict] = None,
        include_academic: bool = True,
        include_crawl: bool = True,
        crawl_seed_urls: Optional[list[str]] = None,
    ) -> tuple[list[RetrievedDocument], dict[str, float], list[RetrievalMethod], list[str], list[RetrievedDocument]]:
        """Returns (docs, latencies_ms, methods_used, errors, web_docs)."""
        tasks = []
        if RetrievalMethod.WEB_SEARCH in methods:
            tasks.append(self._timed(RetrievalMethod.WEB_SEARCH, web_search(text, max_results=max_results)))
        if RetrievalMethod.ACADEMIC in methods and include_academic:
            tasks.append(self._timed(RetrievalMethod.ACADEMIC,
                                     academic_search(text, max_results=min(max_results, cfg.arxiv_max_results))))
        if RetrievalMethod.VECTOR_RAG in methods or RetrievalMethod.HYBRID in methods:
            kb_method = RetrievalMethod.VECTOR_RAG if RetrievalMethod.VECTOR_RAG in methods else RetrievalMethod.HYBRID
            tasks.append(self._timed(kb_method, self._vector_retriever.retrieve(text, top_k=max_results, filters=filters or None)))

        results = await asyncio.gather(*tasks)

        docs: list[RetrievedDocument] = []
        web_docs: list[RetrievedDocument] = []
        latencies: dict[str, float] = {}
        used: list[RetrievalMethod] = []
        errors: list[str] = []
        for method, found, latency, error in results:
            key = "knowledge_base" if method in (RetrievalMethod.VECTOR_RAG, RetrievalMethod.HYBRID) else method.value
            latencies[key] = round(latency, 1)
            used.append(method)
            docs.extend(found)
            if method == RetrievalMethod.WEB_SEARCH:
                web_docs = found
            if error:
                errors.append(error)
            logger.info("%-11s → %3d docs (%.0fms)", method.value, len(found), latency)

        if RetrievalMethod.WEB_CRAWL in methods and include_crawl:
            seeds = list(crawl_seed_urls or [])
            if not seeds:
                if not web_docs and RetrievalMethod.WEB_SEARCH not in methods:
                    web_docs = await web_search(text, max_results=cfg.crawl_seed_count)
                seeds = [d.metadata.url for d in web_docs if d.metadata.url][:cfg.crawl_seed_count]
            depth = cfg.crawl_max_depth if crawl_seed_urls else 0
            pages = cfg.crawl_max_pages if crawl_seed_urls else 1
            method, found, latency, error = await self._timed(
                RetrievalMethod.WEB_CRAWL, crawl_batch(seeds, max_depth=depth, max_pages_per_seed=pages, query=text)
            )
            latencies[method.value] = round(latency, 1)
            used.append(method)
            docs.extend(found)
            if error:
                errors.append(error)
            logger.info("%-11s → %3d docs (%.0fms)", method.value, len(found), latency)

        return docs, latencies, used, errors, web_docs

    # ── Stage 2: rank ────────────────────────────────────────────────────────

    async def rank(
        self,
        query: str,
        docs: list[RetrievedDocument],
        top_k: Optional[int] = None,
        use_hybrid: bool = True,
        use_mmr: bool = False,
    ) -> tuple[list[RetrievedDocument], dict[str, float], list[str]]:
        """Dedup → hybrid fusion → cross-encoder. Returns (ranked, latencies_ms, errors)."""
        latencies: dict[str, float] = {}
        errors: list[str] = []
        top_k = top_k or cfg.reranker_top_k

        unique = _dedup_documents(docs)
        logger.info("Dedup: %d → %d unique docs", len(docs), len(unique))
        if not unique:
            return [], latencies, errors

        candidates = unique
        if use_hybrid and len(unique) > 1:
            t0 = time.perf_counter()
            try:
                candidates = await self._hybrid_retriever.rank(query, unique, top_n=max(cfg.hybrid_candidates, top_k))
            except Exception as exc:
                logger.exception("Hybrid ranking failed")
                errors.append(f"hybrid: {redact(exc)}")
            latencies["hybrid_rank"] = round((time.perf_counter() - t0) * 1000, 1)

        t0 = time.perf_counter()
        try:
            if use_mmr:
                ranked = await self._reranker.rerank_with_diversity(query, candidates, top_k=top_k)
            else:
                ranked = await self._reranker.rerank(query, candidates, top_k=top_k)
        except Exception as exc:
            logger.exception("Reranker failed — returning hybrid order")
            errors.append(f"reranker: {redact(exc)}")
            ranked = candidates[:top_k]
        latencies["reranker"] = round((time.perf_counter() - t0) * 1000, 1)
        return ranked, latencies, errors

    # ── Background indexing ──────────────────────────────────────────────────

    def schedule_indexing(self, docs: list[RetrievedDocument]) -> None:
        if not cfg.auto_index:
            return
        indexable = [
            d for d in docs
            if d.metadata.source_method in _EXTERNAL_METHODS and len(d.content) >= 300
        ][:cfg.auto_index_max_docs]
        if not indexable:
            return

        async def _index_bg():
            try:
                await self._vector_retriever.index_documents(indexable)
            except Exception as exc:
                logger.warning("Background indexing failed: %s", redact(exc))

        task = asyncio.create_task(_index_bg())
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def drain_background(self, timeout: float = 60.0) -> None:
        """Wait for pending background indexing (used by CLI before exit and by tests)."""
        if self._bg_tasks:
            await asyncio.wait(set(self._bg_tasks), timeout=timeout)

    # ── Public API ───────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: RetrievalQuery,
        crawl_seed_urls: Optional[list[str]] = None,
        use_mmr: bool = False,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        t_start = time.perf_counter()
        with notice_scope() as new_notices:
            docs, latencies, used, errors, _ = await self.gather_candidates(
                query.text, query.methods, max_results=query.max_results,
                filters=query.filters, crawl_seed_urls=crawl_seed_urls,
            )
            use_hybrid = RetrievalMethod.HYBRID in query.methods or len(docs) > cfg.hybrid_candidates
            ranked, rank_lat, rank_errors = await self.rank(
                query.text, docs, top_k=top_k or query.max_results,
                use_hybrid=use_hybrid, use_mmr=use_mmr,
            )
            self.schedule_indexing(_dedup_documents(docs))
            warnings = [n.message for n in new_notices()]

        latencies.update(rank_lat)
        latencies["total"] = round((time.perf_counter() - t_start) * 1000, 1)
        logger.info("Retrieval done — %d final docs in %.0fms", len(ranked), latencies["total"])
        return RetrievalResult(
            query=query,
            documents=ranked,
            total_fetched=len(docs),
            total_after_rerank=len(ranked),
            methods_used=used,
            latency_ms=latencies,
            errors=errors + rank_errors,
            warnings=warnings,
        )

    async def retrieve_multi_query(
        self,
        queries: list[str],
        methods: Optional[list[RetrievalMethod]] = None,
        max_results: int = 10,
        rank_query: Optional[str] = None,
        use_mmr: bool = False,
        top_k: Optional[int] = None,
        crawl_seed_urls: Optional[list[str]] = None,
    ) -> RetrievalResult:
        """
        Gather candidates for every sub-query concurrently, pool them, then rank the
        pool ONCE against `rank_query` (the user's original question).
        """
        t_start = time.perf_counter()
        queries = [q for q in dict.fromkeys(q.strip() for q in queries) if q] or [rank_query or ""]
        methods = methods or [RetrievalMethod.WEB_SEARCH, RetrievalMethod.HYBRID]
        rank_query = rank_query or queries[0]

        with notice_scope() as new_notices:
            per_query = await asyncio.gather(*(
                self.gather_candidates(
                    q, methods, max_results=max_results,
                    include_academic=i < cfg.academic_max_queries,   # academic APIs are strictly rate-limited
                    include_crawl=False,                             # crawl once, after pooling
                )
                for i, q in enumerate(queries)
            ), return_exceptions=True)

            pooled: list[RetrievedDocument] = []
            web_groups: list[list[RetrievedDocument]] = []
            latencies: dict[str, float] = {}
            used: list[RetrievalMethod] = []
            errors: list[str] = []
            for q, result in zip(queries, per_query):
                if isinstance(result, BaseException):
                    errors.append(f"{q}: {redact(result)}")
                    continue
                docs, lat, m_used, errs, web_docs = result
                pooled.extend(docs)
                web_groups.append(web_docs)
                errors.extend(errs)
                for m in m_used:
                    if m not in used:
                        used.append(m)
                for k, v in lat.items():
                    latencies[k] = max(latencies.get(k, 0.0), v)   # sub-queries run in parallel

            if RetrievalMethod.WEB_CRAWL in methods:
                seeds = crawl_seed_urls or _interleave_urls(web_groups, cfg.crawl_seed_count)
                if seeds:
                    method, found, latency, error = await self._timed(
                        RetrievalMethod.WEB_CRAWL, crawl_batch(seeds, max_depth=0, max_pages_per_seed=1, query=rank_query)
                    )
                    pooled.extend(found)
                    used.append(method)
                    latencies[method.value] = round(latency, 1)
                    if error:
                        errors.append(error)

            ranked, rank_lat, rank_errors = await self.rank(
                rank_query, pooled, top_k=top_k or cfg.reranker_top_k, use_hybrid=True, use_mmr=use_mmr,
            )
            self.schedule_indexing(_dedup_documents(pooled))
            warnings = [n.message for n in new_notices()]

        latencies.update(rank_lat)
        latencies["total"] = round((time.perf_counter() - t_start) * 1000, 1)
        return RetrievalResult(
            query=RetrievalQuery(text=rank_query, methods=methods, max_results=max(1, min(max_results, 50))),
            documents=ranked,
            total_fetched=len(pooled),
            total_after_rerank=len(ranked),
            methods_used=used,
            latency_ms=latencies,
            errors=errors + rank_errors,
            warnings=warnings,
        )

    async def knowledge_base_size(self) -> int:
        return await self._vector_retriever.count()
