"""
models/schemas.py

Shared Pydantic v2 data models for the retrieval pipeline.
Every retrieval method returns a RetrievedDocument that flows
into the reranker, then into the knowledge store.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Enumerations ──────────────────────────────────────────────────────────────

class RetrievalMethod(str, Enum):
    WEB_SEARCH   = "web_search"    # Tavily / SerpAPI / DDG
    VECTOR_RAG   = "vector_rag"    # Dense ANN from Qdrant
    HYBRID       = "hybrid"        # BM25 + dense fusion
    WEB_CRAWL    = "web_crawl"     # Playwright/httpx deep crawl
    ACADEMIC     = "academic"      # Semantic Scholar / arXiv


class DocumentStatus(str, Enum):
    RAW         = "raw"            # just fetched, not processed
    CHUNKED     = "chunked"        # split into chunks
    EMBEDDED    = "embedded"       # vector stored
    RERANKED    = "reranked"       # scored by reranker
    VERIFIED    = "verified"       # passed verifier agent


class ContentType(str, Enum):
    HTML        = "html"
    MARKDOWN    = "markdown"
    PDF_TEXT    = "pdf_text"
    PLAIN_TEXT  = "plain_text"
    JSON        = "json"


# ── Core document model ───────────────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    """Rich metadata attached to every retrieved document."""

    source_method:  RetrievalMethod
    domain:         str                      = Field(..., description="e.g. arxiv.org")
    url:            Optional[str]            = None
    title:          Optional[str]            = None
    author:         Optional[str]            = None
    published_at:   Optional[datetime]       = None
    language:       str                      = "en"
    word_count:     Optional[int]            = None
    content_type:   ContentType              = ContentType.PLAIN_TEXT
    crawl_depth:    int                      = 0   # 0 = direct fetch, N = followed links
    extra:          dict                     = Field(default_factory=dict)


class RetrievedDocument(BaseModel):
    """
    Universal document model returned by every retrieval method.
    The retrieval router normalises all sources into this shape.
    """

    doc_id:         str             = Field(default_factory=lambda: f"doc_{uuid4().hex[:10]}")
    content:        str             = Field(..., min_length=1)
    metadata:       DocumentMetadata
    status:         DocumentStatus  = DocumentStatus.RAW

    # Scoring — populated downstream
    relevance_score:   Optional[float] = None   # cosine / BM25 / RRF
    rerank_score:      Optional[float] = None   # cross-encoder score
    credibility_score: Optional[float] = None   # verifier output

    retrieved_at: datetime = Field(default_factory=_utcnow)

    def is_usable(self) -> bool:
        """Quick check before expensive processing."""
        return bool(self.content.strip()) and len(self.content) > 100


class DocumentChunk(BaseModel):
    """
    A fixed-size window slice of a RetrievedDocument.
    This is what gets embedded and stored in Qdrant.
    """

    chunk_id:       str     = Field(default_factory=lambda: f"chk_{uuid4().hex[:10]}")
    parent_doc_id:  str
    content:        str
    chunk_index:    int
    start_char:     int
    end_char:       int
    metadata:       DocumentMetadata
    embedding:      Optional[list[float]] = None   # populated by EmbeddingService


class RetrievalQuery(BaseModel):
    """Input to the retrieval router."""

    query_id:       str             = Field(default_factory=lambda: f"q_{uuid4().hex[:8]}")
    text:           str             = Field(..., min_length=1)
    methods:        list[RetrievalMethod] = Field(
        default_factory=lambda: [
            RetrievalMethod.WEB_SEARCH,
            RetrievalMethod.VECTOR_RAG,
            RetrievalMethod.HYBRID,
        ]
    )
    max_results:    int             = Field(10, ge=1, le=50)
    filters:        dict            = Field(default_factory=dict)
    language:       str             = "en"


class RetrievalResult(BaseModel):
    """Final output of the retrieval pipeline for one query."""

    query:          RetrievalQuery
    documents:      list[RetrievedDocument]
    chunks:         list[DocumentChunk]     = Field(default_factory=list)
    total_fetched:  int                     = 0
    total_after_rerank: int                 = 0
    methods_used:   list[RetrievalMethod]   = Field(default_factory=list)
    latency_ms:     dict[str, float]        = Field(default_factory=dict)
    errors:         list[str]               = Field(default_factory=list)
    warnings:       list[str]               = Field(default_factory=list)   # provider fallbacks, quota issues
    completed_at:   datetime                = Field(default_factory=_utcnow)
