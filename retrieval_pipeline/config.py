"""
config.py

Platform-wide configuration using pydantic-settings.
Every value can be overridden through environment variables or the .env file.

The .env file is looked up in this order (first match wins):
  1. $RESEARCH_ENV_FILE                (explicit override)
  2. Project root                      (parent of retrieval_pipeline/)
  3. retrieval_pipeline/
  4. Current working directory
  5. Parent of the current working directory
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent


def _find_env_file() -> str:
    override = os.environ.get("RESEARCH_ENV_FILE")
    if override and Path(override).exists():
        return str(Path(override).resolve())

    candidates = [PROJECT_ROOT, PACKAGE_DIR, Path.cwd(), Path.cwd().parent]
    for directory in candidates:
        env_path = directory / ".env"
        if env_path.exists():
            return str(env_path.resolve())
    return str(PROJECT_ROOT / ".env")   # pydantic-settings tolerates a missing file


_ENV_FILE = _find_env_file()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── General ───────────────────────────────────────────────────────────────
    data_dir:               Path          = PROJECT_ROOT / "data"
    log_level:              str           = "INFO"
    cors_origins:           str           = ""      # comma-separated; empty = same-origin only
    hf_endpoint:            Optional[str] = None    # e.g. https://hf-mirror.com if huggingface.co is blocked
    warmup_models:          bool          = True    # load embedding/reranker models at API start-up

    # ── Web search ────────────────────────────────────────────────────────────
    tavily_api_key:         Optional[str] = None
    serpapi_key:            Optional[str] = None
    web_search_providers:   str           = "tavily,serpapi,duckduckgo"   # fallback order
    tavily_search_depth:    str           = "basic"   # basic = 1 credit, advanced = 2 credits
    tavily_include_raw_content: bool      = True
    ddg_max_results:        int           = 10
    web_search_timeout_s:   int           = 20
    search_cache_ttl_s:     int           = 1800      # identical searches reuse results (saves credits)
    max_doc_chars:          int           = 8000      # per-document content cap

    # ── Web crawl ─────────────────────────────────────────────────────────────
    crawl_max_depth:        int           = 1
    crawl_max_pages:        int           = 10
    crawl_concurrency:      int           = 4
    crawl_timeout_s:        int           = 15
    crawl_respect_robots:   bool          = True
    crawl_use_playwright:   bool          = True
    crawl_seed_count:       int           = 4         # pages deep-read when "deep read" is on
    crawl_user_agent:       str           = "Mozilla/5.0 (compatible; QuorumBot/1.0; local research assistant)"

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model:        str           = "BAAI/bge-small-en-v1.5"
    embedding_batch_size:   int           = 32
    embedding_device:       str           = "cpu"     # cpu | cuda | auto
    embedding_dim:          Optional[int] = None      # auto-detected from the model

    # ── Qdrant (vector store + long-term memory) ──────────────────────────────
    qdrant_mode:            str           = "auto"    # auto | server | local | memory
    qdrant_host:            str           = "localhost"
    qdrant_port:            int           = 6333
    qdrant_api_key:         Optional[str] = None
    qdrant_path:            Optional[Path] = None     # default: <data_dir>/qdrant
    qdrant_collection:      str           = "research_docs"
    qdrant_top_k:           int           = 20

    # ── Redis (short-term session memory; SQLite is used when Redis is absent) ─
    redis_enabled:          bool          = True
    redis_host:             str           = "localhost"
    redis_port:             int           = 6379
    redis_db:               int           = 0
    redis_ttl_s:            int           = 7 * 24 * 3600

    # ── BM25 / hybrid ─────────────────────────────────────────────────────────
    bm25_top_k:             int           = 20
    hybrid_alpha:           float         = 0.6       # weight of the dense list in RRF
    rrf_k:                  int           = 60
    hybrid_candidates:      int           = 30        # docs kept after hybrid fusion, before reranking

    # ── Reranker ─────────────────────────────────────────────────────────────
    reranker_enabled:       bool          = True
    reranker_model:         str           = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k:         int           = 8
    reranker_batch_size:    int           = 16
    reranker_max_chars:     int           = 1200
    reranker_temperature:   float         = 4.0       # score = sigmoid(logit / T)

    # ── Chunking / indexing ───────────────────────────────────────────────────
    chunk_size:             int           = 800
    chunk_overlap:          int           = 100
    auto_index:             bool          = True      # index fetched pages into the local knowledge base
    auto_index_max_docs:    int           = 15

    # ── Academic ──────────────────────────────────────────────────────────────
    semantic_scholar_key:   Optional[str] = None
    arxiv_max_results:      int           = 5
    academic_timeout_s:     int           = 25
    academic_max_queries:   int           = 2         # sub-queries sent to academic APIs per research run

    # ── LLM providers (tried in order; only configured ones are used) ─────────
    llm_provider_order:     str           = "anthropic,openai,groq,gemini,openrouter,deepseek,ollama"
    llm_timeout_s:          int           = 180
    llm_max_retries:        int           = 2

    anthropic_api_key:      Optional[str] = None
    anthropic_model:        str           = "claude-opus-5-5"
    anthropic_effort:       str           = "high"    # low | medium | high | xhigh | max
    anthropic_fallbacks:    bool          = True      # server-side refusal fallbacks ("default" mode)

    openai_api_key:         Optional[str] = None
    openai_model:           str           = "gpt-5-mini"
    openai_base_url:        str           = "https://api.openai.com/v1"

    groq_api_key:           Optional[str] = None
    groq_model:             str           = "openai/gpt-oss-120b"

    gemini_api_key:         Optional[str] = None
    gemini_model:           str           = "gemini-3.6-flash"

    openrouter_api_key:     Optional[str] = None
    openrouter_model:       str           = "openrouter/auto"

    deepseek_api_key:       Optional[str] = None
    deepseek_model:         str           = "deepseek-chat"

    ollama_enabled:         bool          = True
    ollama_base_url:        str           = "http://localhost:11434"
    ollama_model:           str           = "qwen2.5:7b"

    # ── Agents ────────────────────────────────────────────────────────────────
    planner_max_subtasks:   int           = 5
    planner_max_queries:    int           = 4       # each sub-query costs one search credit
    verifier_min_sources:   int           = 2
    verifier_max_claims:    int           = 12
    report_max_findings:    int           = 6
    report_max_citations:   int           = 12
    synthesis_max_sources:  int           = 10
    synthesis_chars_per_source: int       = 1800
    min_docs_before_refine: int           = 3       # fewer docs than this triggers an autonomous re-plan
    research_timeout_s:     int           = 420
    max_concurrent_research: int          = 2       # protects small machines and API rate limits

    # ── Provider resilience (circuit breaker) ─────────────────────────────────
    quota_cooldown_s:       int           = 6 * 3600   # credits exhausted
    auth_cooldown_s:        int           = 24 * 3600  # invalid / revoked key
    rate_limit_cooldown_s:  int           = 60
    failure_threshold:      int           = 3          # consecutive transient failures before cooling down
    failure_cooldown_s:     int           = 120
    max_retry_wait_s:       float         = 20.0       # never sleep longer than this for a Retry-After

    @field_validator(
        "tavily_api_key", "serpapi_key", "semantic_scholar_key", "qdrant_api_key",
        "anthropic_api_key", "openai_api_key", "groq_api_key", "gemini_api_key",
        "openrouter_api_key", "deepseek_api_key", "hf_endpoint",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, value):
        """Treat empty strings and template placeholders as 'not configured'."""
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower().startswith(("your_", "your-", "tvly-your")) or text in {"...", "changeme"}:
            return None
        return text

    @property
    def qdrant_storage_path(self) -> Path:
        return Path(self.qdrant_path) if self.qdrant_path else Path(self.data_dir) / "qdrant"

    @property
    def sessions_db_path(self) -> Path:
        return Path(self.data_dir) / "sessions.db"

    def secret_values(self) -> list[str]:
        """All configured secrets — used to redact them from logs and error messages."""
        names = (
            "tavily_api_key", "serpapi_key", "semantic_scholar_key", "qdrant_api_key",
            "anthropic_api_key", "openai_api_key", "groq_api_key", "gemini_api_key",
            "openrouter_api_key", "deepseek_api_key",
        )
        return [v for v in (getattr(self, n) for n in names) if v and len(v) >= 6]


def _mask(secret: Optional[str]) -> str:
    if not secret:
        return "NOT SET"
    return f"set (…{secret[-4:]})"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()

    from retrieval_pipeline.core.errors import register_secrets
    register_secrets(settings.secret_values())

    # huggingface_hub reads HF_ENDPOINT when it is first imported, so export it early.
    if settings.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)

    try:
        Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logging.getLogger(__name__).warning("Cannot create data dir %s: %s", settings.data_dir, exc)

    log = logging.getLogger(__name__)
    log.info("Config loaded from: %s", _ENV_FILE)
    log.info("Tavily key: %s | SerpAPI key: %s", _mask(settings.tavily_api_key), _mask(settings.serpapi_key))
    return settings


# Backwards-compatible alias (older code imported RetrieverSettings)
RetrieverSettings = Settings
