"""
memory/long_term.py

Long-term knowledge memory backed by the shared Qdrant client.

Stores verified findings (not raw pages — those live in the document index) so
the planner can recall what earlier research established. Uses the same
embedding model and backend selection as the rest of the platform
(Qdrant server → embedded on-disk store → in-memory).
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import NAMESPACE_URL, uuid5

from retrieval_pipeline.config import get_settings
from retrieval_pipeline.core.ml import embedding_model, model_ready, run_ml
from retrieval_pipeline.retrieval.vector_store import ensure_collection, get_qdrant

logger = logging.getLogger(__name__)
cfg = get_settings()

_KNOWLEDGE_BASE_NAME = "research_knowledge"
_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


class KnowledgeEntry:
    """A verified research finding stored in long-term memory."""

    def __init__(
        self,
        content:      str,
        source_query: str,
        confidence:   str = "medium",
        sources:      Optional[list] = None,
        tags:         Optional[list] = None,
        session_id:   Optional[str] = None,
    ):
        digest = hashlib.sha256(content.strip().lower().encode()).hexdigest()
        self.entry_id     = str(uuid5(NAMESPACE_URL, f"knowledge:{digest}"))   # valid Qdrant ID, de-duplicates
        self.content      = content
        self.source_query = source_query
        self.confidence   = confidence
        self.sources      = sources or []
        self.tags         = tags or []
        self.session_id   = session_id
        self.created_at   = datetime.now(timezone.utc).isoformat()


async def _collection(wait: bool) -> Optional[str]:
    if not await model_ready(embedding_model, wait):
        return None
    model = embedding_model.get()
    dim = cfg.embedding_dim or int(await run_ml(model.get_sentence_embedding_dimension))
    return await ensure_collection(_KNOWLEDGE_BASE_NAME, dim)


async def store_finding(entry: KnowledgeEntry) -> bool:
    """Embed and store a finding. Returns False if the store is unavailable."""
    try:
        name = await _collection(wait=True)
        client = await get_qdrant()
        if not name or client is None:
            return False
        from qdrant_client.models import PointStruct
        vectors = await run_ml(lambda: embedding_model.get().encode(
            [entry.content], normalize_embeddings=True, show_progress_bar=False).tolist())
        await client.upsert(collection_name=name, points=[PointStruct(
            id=entry.entry_id,
            vector=vectors[0],
            payload={
                "content":      entry.content,
                "source_query": entry.source_query,
                "confidence":   entry.confidence,
                "sources":      entry.sources,
                "tags":         entry.tags,
                "session_id":   entry.session_id,
                "created_at":   entry.created_at,
            },
        )])
        return True
    except Exception as exc:
        logger.warning("store_finding failed: %s", exc)
        return False


async def recall(
    query: str,
    top_k: int = 5,
    min_confidence: str = "low",
    min_similarity: float = 0.6,
) -> list[dict]:
    """Past findings relevant to `query` (never blocks on a cold model in the API)."""
    try:
        name = await _collection(wait=False)
        client = await get_qdrant()
        if not name or client is None:
            return []
        from retrieval_pipeline.retrieval.vector_rag import EmbeddingService
        vec = await EmbeddingService().embed_query_async(query)
        hits = await client.search(collection_name=name, query_vector=vec, limit=top_k * 2,
                                   with_payload=True, with_vectors=False)
        floor = _CONFIDENCE_RANK.get(min_confidence, 0)
        results = []
        for hit in hits:
            payload = dict(hit.payload or {})
            if hit.score < min_similarity or _CONFIDENCE_RANK.get(payload.get("confidence", "low"), 0) < floor:
                continue
            payload["similarity_score"] = round(float(hit.score), 4)
            results.append(payload)
            if len(results) >= top_k:
                break
        if results:
            logger.info("Recalled %d past findings for %r", len(results), query)
        return results
    except Exception as exc:
        logger.warning("recall failed: %s", exc)
        return []


async def store_research_session(session_id: str, query: str, findings: list[dict], summary: str) -> int:
    """Store the medium/high-confidence findings of a research run. Returns how many were stored."""
    stored = 0
    for finding in findings:
        confidence = finding.get("confidence", "medium")
        if confidence == "low":
            continue
        text = finding.get("detail") or finding.get("headline") or ""
        if len(text) < 30:
            continue
        entry = KnowledgeEntry(
            content=f"{finding.get('headline', '')}: {text}" if finding.get("headline") and finding.get("headline") not in text else text,
            source_query=query,
            confidence=confidence,
            sources=[c.get("url", "") for c in finding.get("citations", []) if c.get("url")],
            tags=finding.get("tags", []),
            session_id=session_id,
        )
        if await store_finding(entry):
            stored += 1
    logger.info("Long-term memory: stored %d/%d findings", stored, len(findings))
    return stored
