"""
retrieval/vector_rag.py

Dense vector retrieval over the local knowledge base (Qdrant + BGE embeddings).

  1. EmbeddingService  — encodes queries and passages (lazy model, dedicated ML thread)
  2. DocumentChunker   — sentence-aware sliding windows with overlap
  3. QdrantStore       — collection lifecycle, upsert, filtered ANN search
  4. VectorRetriever   — index_documents() / retrieve() used by the router

Every page the platform reads is (optionally) indexed, so later questions can be
answered partly from the accumulated knowledge base.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import NAMESPACE_URL, uuid5

from retrieval_pipeline.config import get_settings
from retrieval_pipeline.core.ml import embedding_model, model_ready, run_ml
from retrieval_pipeline.models.schemas import (
    ContentType,
    DocumentChunk,
    DocumentMetadata,
    DocumentStatus,
    RetrievalMethod,
    RetrievedDocument,
)
from retrieval_pipeline.retrieval.vector_store import ensure_collection, get_qdrant

logger = logging.getLogger(__name__)
cfg = get_settings()

_KEYWORD_FIELDS = ("metadata.domain", "metadata.source_method", "metadata.language", "metadata.url")


# ── Embedding service ─────────────────────────────────────────────────────────

class EmbeddingService:
    """
    Thin wrapper over the shared lazily-loaded SentenceTransformer.
    BGE v1.5 models use an instruction prefix for queries (not for passages).
    """

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    @property
    def model_name(self) -> str:
        return embedding_model.model_name

    def _model(self):
        model = embedding_model.get()
        if model is None:
            raise RuntimeError(f"embedding model unavailable: {embedding_model.error}")
        return model

    def _prefix(self) -> str:
        return self.QUERY_PREFIX if "bge" in self.model_name.lower() and "-en" in self.model_name.lower() else ""

    @property
    def dim(self) -> int:
        if cfg.embedding_dim:
            return cfg.embedding_dim
        return int(self._model().get_sentence_embedding_dimension())

    def embed_query(self, text: str) -> list[float]:
        vec = self._model().encode(self._prefix() + text, normalize_embeddings=True, show_progress_bar=False)
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._model().encode(
            texts, normalize_embeddings=True, batch_size=cfg.embedding_batch_size, show_progress_bar=False,
        )
        return vecs.tolist()

    async def available(self, wait: Optional[bool] = None) -> bool:
        """wait=None follows core.ml.BLOCKING_LOAD; background indexing passes wait=True."""
        return await model_ready(embedding_model, wait)

    async def embed_query_async(self, text: str) -> list[float]:
        return await run_ml(self.embed_query, text)

    async def embed_documents_async(self, texts: list[str]) -> list[list[float]]:
        return await run_ml(self.embed_documents, texts)

    async def dim_async(self) -> int:
        return await run_ml(lambda: self.dim)


# ── Document chunker ──────────────────────────────────────────────────────────

class DocumentChunker:
    """
    Fixed-size sliding window with overlap that prefers to end on a sentence boundary.

    FIX: the old version could move the window backwards (or not at all) when a
    sentence boundary was found near the start of the window, looping forever.
    The window now always advances, and boundaries are only accepted in the
    second half of the window.
    """

    _BOUNDARIES = ("\n\n", ". ", ".\n", "! ", "? ", "\n")

    def __init__(self, chunk_size: Optional[int] = None, overlap: Optional[int] = None, max_chunks: int = 24):
        self.chunk_size = chunk_size or cfg.chunk_size
        self.overlap    = min(overlap if overlap is not None else cfg.chunk_overlap, self.chunk_size // 2)
        self.max_chunks = max_chunks

    def split(self, text: str) -> list[tuple[int, int, str]]:
        text = text or ""
        if len(text) <= self.chunk_size:
            return [(0, len(text), text.strip())] if text.strip() else []
        spans: list[tuple[int, int, str]] = []
        start = 0
        while start < len(text) and len(spans) < self.max_chunks:
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                floor = start + self.chunk_size // 2
                for boundary in self._BOUNDARIES:
                    pos = text.rfind(boundary, floor, end)
                    if pos != -1:
                        end = pos + len(boundary)
                        break
            chunk = text[start:end].strip()
            if chunk:
                spans.append((start, end, chunk))
            if end >= len(text):
                break
            start = max(end - self.overlap, start + 1)
        return spans

    def chunk(self, doc: RetrievedDocument) -> list[DocumentChunk]:
        return [
            DocumentChunk(
                parent_doc_id=doc.doc_id,
                content=chunk_text,
                chunk_index=idx,
                start_char=start,
                end_char=end,
                metadata=doc.metadata,
            )
            for idx, (start, end, chunk_text) in enumerate(self.split(doc.content))
        ]


def _point_id(chunk: DocumentChunk) -> str:
    """
    Qdrant point IDs must be unsigned integers or UUIDs — the old 'chk_xxxx' IDs were
    rejected, so nothing was ever indexed. A deterministic UUID also means re-reading
    the same page overwrites its chunks instead of duplicating them.
    """
    key = f"{chunk.metadata.url or chunk.parent_doc_id}#{chunk.chunk_index}"
    return str(uuid5(NAMESPACE_URL, key))


# ── Qdrant vector store ───────────────────────────────────────────────────────

class QdrantStore:
    def __init__(self, collection: Optional[str] = None, embedder: Optional[EmbeddingService] = None):
        self.base_collection = collection or cfg.qdrant_collection
        self.embedder = embedder or EmbeddingService()

    async def _collection(self) -> Optional[str]:
        if not embedding_model.loaded and not await self.embedder.available():
            return None
        dim = await self.embedder.dim_async()
        return await ensure_collection(self.base_collection, dim, _KEYWORD_FIELDS)

    async def ensure_collection(self) -> Optional[str]:
        return await self._collection()

    async def upsert_chunks(self, chunks: list[DocumentChunk]) -> int:
        from qdrant_client.models import PointStruct

        name = await self._collection()
        client = await get_qdrant()
        if not name or client is None:
            return 0

        points = []
        indexed_at = datetime.now(timezone.utc).isoformat()
        for chunk in chunks:
            if chunk.embedding is None:
                continue
            original = chunk.metadata.extra.get("original_method") or chunk.metadata.source_method.value
            points.append(PointStruct(
                id=_point_id(chunk),
                vector=chunk.embedding,
                payload={
                    "content":       chunk.content,
                    "parent_doc_id": chunk.parent_doc_id,
                    "chunk_index":   chunk.chunk_index,
                    "metadata": {
                        "domain":         chunk.metadata.domain,
                        "url":            chunk.metadata.url,
                        "title":          chunk.metadata.title,
                        "source_method":  original,
                        "provider":       chunk.metadata.extra.get("provider"),
                        "language":       chunk.metadata.language,
                        "content_type":   chunk.metadata.content_type.value,
                        "crawl_depth":    chunk.metadata.crawl_depth,
                        "published_at":   chunk.metadata.published_at.isoformat() if chunk.metadata.published_at else None,
                        "indexed_at":     indexed_at,
                    },
                },
            ))
        for i in range(0, len(points), 128):
            await client.upsert(collection_name=name, points=points[i:i + 128])
        logger.debug("Upserted %d points into %s", len(points), name)
        return len(points)

    async def search(self, query_vector: list[float], top_k: Optional[int] = None, filters: Optional[dict] = None):
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        name = await self._collection()
        client = await get_qdrant()
        if not name or client is None:
            return []

        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                field = key if key.startswith("metadata.") else f"metadata.{key}"
                match = MatchAny(any=list(value)) if isinstance(value, (list, tuple, set)) else MatchValue(value=value)
                conditions.append(FieldCondition(key=field, match=match))
            qdrant_filter = Filter(must=conditions)

        return await client.search(
            collection_name=name,
            query_vector=query_vector,
            limit=top_k or cfg.qdrant_top_k,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=False,
        )

    async def delete_by_domain(self, domain: str) -> None:
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

        name = await self._collection()
        client = await get_qdrant()
        if not name or client is None:
            return
        await client.delete(
            collection_name=name,
            points_selector=FilterSelector(filter=Filter(must=[
                FieldCondition(key="metadata.domain", match=MatchValue(value=domain))
            ])),
        )

    async def count(self) -> int:
        name = await self._collection()
        client = await get_qdrant()
        if not name or client is None:
            return 0
        return (await client.count(collection_name=name, exact=True)).count


# ── Vector retriever ──────────────────────────────────────────────────────────

class VectorRetriever:
    """Query the knowledge base; index newly fetched documents into it."""

    def __init__(self):
        self.embedder = EmbeddingService()
        self.store    = QdrantStore(embedder=self.embedder)
        self.chunker  = DocumentChunker()

    async def index_documents(self, docs: list[RetrievedDocument]) -> int:
        """Chunk, embed and upsert documents. Returns the number of chunks indexed."""
        if not docs or not await self.embedder.available(wait=True):
            return 0
        all_chunks: list[DocumentChunk] = []
        for doc in docs:
            if doc.is_usable():
                all_chunks.extend(self.chunker.chunk(doc))
        if not all_chunks:
            return 0

        embeddings = await self.embedder.embed_documents_async([c.content for c in all_chunks])
        for chunk, emb in zip(all_chunks, embeddings):
            chunk.embedding = emb
        count = await self.store.upsert_chunks(all_chunks)
        logger.info("Indexed %d chunks from %d documents", count, len(docs))
        return count

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None,
        min_score: float = 0.35,
    ) -> list[RetrievedDocument]:
        """ANN search over indexed chunks; one document per source URL."""
        if not await self.embedder.available():
            return []
        query_vec = await self.embedder.embed_query_async(query)
        hits = await self.store.search(query_vec, top_k=(top_k or cfg.qdrant_top_k) * 2, filters=filters)

        docs: list[RetrievedDocument] = []
        seen_urls: set[str] = set()
        for hit in hits:
            payload = hit.payload or {}
            content = (payload.get("content") or "").strip()
            meta = payload.get("metadata") or {}
            url = meta.get("url")
            if not content or hit.score < min_score or (url and url in seen_urls):
                continue
            if url:
                seen_urls.add(url)
            try:
                ctype = ContentType(meta.get("content_type", "plain_text"))
            except ValueError:
                ctype = ContentType.PLAIN_TEXT
            docs.append(RetrievedDocument(
                content=content,
                metadata=DocumentMetadata(
                    source_method=RetrievalMethod.VECTOR_RAG,
                    domain=meta.get("domain") or "",
                    url=url,
                    title=meta.get("title"),
                    content_type=ctype,
                    crawl_depth=meta.get("crawl_depth") or 0,
                    extra={
                        "provider":        "knowledge_base",
                        "qdrant_id":       str(hit.id),
                        "parent_doc_id":   payload.get("parent_doc_id"),
                        "original_method": meta.get("source_method"),
                        "indexed_at":      meta.get("indexed_at"),
                    },
                ),
                relevance_score=round(float(hit.score), 4),
                status=DocumentStatus.EMBEDDED,
            ))
            if len(docs) >= (top_k or cfg.qdrant_top_k):
                break

        logger.info("Knowledge-base search returned %d docs for %r", len(docs), query)
        return docs

    async def count(self) -> int:
        try:
            return await self.store.count()
        except Exception as exc:
            logger.debug("count failed: %s", exc)
            return 0
