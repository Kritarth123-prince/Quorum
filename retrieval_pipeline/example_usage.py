"""
example_usage.py  v3

HOW TO RUN (from any directory — .env is found automatically):

  Option A — from research_platform/ root (recommended):
    cd research_platform
    python retrieval_pipeline/example_usage.py

  Option B — from inside retrieval_pipeline/:
    cd retrieval_pipeline
    python example_usage.py

  Option C — absolute path:
    python "F:/Projects/.../research_platform/retrieval_pipeline/example_usage.py"

SETUP:
  1. pip install "numpy<2" --force-reinstall
  2. pip install -e .
  3. pip install -r retrieval_pipeline/requirements.txt
  4. playwright install chromium
  5. Add TAVILY_API_KEY to .env  (get free key at tavily.com — takes 30 sec)
"""

from __future__ import annotations

import sys, os

# ── Path fix: always works regardless of run location ─────────────────────────
_THIS = os.path.abspath(__file__)
_PKG  = os.path.dirname(_THIS)           # .../retrieval_pipeline/
_ROOT = os.path.dirname(_PKG)            # .../research_platform/
for _p in (_ROOT, _PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s  %(name)-40s  %(message)s",
)
for _noisy in ("httpx", "httpcore", "sentence_transformers",
               "huggingface_hub", "filelock", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def _print_startup_diagnostics() -> None:
    """Print exactly which .env was loaded and what keys are active."""
    from retrieval_pipeline.config import _ENV_FILE, get_settings
    cfg = get_settings()

    import os
    env_exists = os.path.exists(_ENV_FILE)

    print()
    print("  ── Configuration ─────────────────────────────────────")
    print(f"  .env file found : {'✓  ' + _ENV_FILE if env_exists else '✗  NOT FOUND — create .env from .env.example'}")
    print()
    print("  Provider / service status:")

    # Tavily
    if cfg.tavily_api_key:
        masked = cfg.tavily_api_key[:8] + "..." + cfg.tavily_api_key[-4:]
        print(f"    Tavily API key    : ✓  {masked}")
    else:
        print( "    Tavily API key    : ✗  NOT SET  → add TAVILY_API_KEY=tvly-... to .env")

    # SerpAPI
    print(f"    SerpAPI key       : {'✓  set' if cfg.serpapi_key else '✗  not set'}")

    # Semantic Scholar
    print(f"    Semantic Scholar  : {'✓  set' if cfg.semantic_scholar_key else '✗  not set (rate limited without key)'}")

    # Qdrant
    print(f"    Qdrant            : {cfg.qdrant_host}:{cfg.qdrant_port}  (probed on first request)")

    # numpy / reranker
    try:
        import numpy as np
        np_ver = tuple(int(x) for x in np.__version__.split(".")[:2])
        if np_ver >= (2, 0):
            print(f"    numpy             : ✗  {np.__version__}  — NEEDS DOWNGRADE")
            print( "                        Run:  pip install 'numpy<2' --force-reinstall")
            print( "                        Reranker will use fallback scorer until fixed.")
        else:
            print(f"    numpy             : ✓  {np.__version__}")
    except ImportError:
        print( "    numpy             : ✗  not installed")

    try:
        import torch
        print(f"    torch             : ✓  {torch.__version__}")
    except Exception:
        print( "    torch             : ✗  not installed / broken")

    print("  ─────────────────────────────────────────────────────")
    print()


def _print_result(result, label: str) -> None:
    print(f"\n{'─' * 54}")
    print(f"  {label}")
    print(f"{'─' * 54}")
    print(f"  Total fetched   : {result.total_fetched}")
    print(f"  After rerank    : {result.total_after_rerank}")
    print(f"  Methods used    : {[m.value for m in result.methods_used]}")
    print(f"  Latencies (ms)  : {result.latency_ms}")

    if result.errors:
        print(f"  Errors          : {result.errors}")

    if not result.documents:
        print()
        print("  ⚠  No documents returned.")
        print("     Likely causes:")
        print("     1. TAVILY_API_KEY not in .env — get free key at tavily.com")
        print("     2. DuckDuckGo rate-limited — wait 60 sec and retry")
        print("     3. Network blocked — check firewall/VPN")
        return

    print(f"\n  Top {min(3, len(result.documents))} documents:")
    for i, doc in enumerate(result.documents[:3], 1):
        rerank = f"{doc.rerank_score:.3f}" if doc.rerank_score is not None else "N/A"
        print(f"\n    [{i}] {doc.metadata.title or '(no title)'}")
        print(f"         Method  : {doc.metadata.source_method.value}")
        print(f"         Domain  : {doc.metadata.domain}")
        print(f"         Score   : rerank={rerank}  relevance={doc.relevance_score}")
        snippet = doc.content[:180].replace("\n", " ").strip()
        print(f"         Snippet : {snippet}…")


async def main() -> None:
    print()
    print("=" * 54)
    print("  Quorum — Retrieval Pipeline")
    print("=" * 54)

    _print_startup_diagnostics()

    from retrieval_pipeline.models.schemas import RetrievalMethod, RetrievalQuery
    from retrieval_pipeline.retrieval.router import RetrievalRouter

    router = RetrievalRouter()

    # ── Test 1: Web search + Academic ────────────────────────────────────────
    print("[1/3] Web search + Academic  (Qdrant not required)")
    result1 = await router.retrieve(
        RetrievalQuery(
            text="Qdrant vs Weaviate vs Milvus for production RAG",
            methods=[RetrievalMethod.WEB_SEARCH, RetrievalMethod.ACADEMIC],
            max_results=6,
        )
    )
    _print_result(result1, "Web Search + Academic")

    # ── Test 2: Hybrid ───────────────────────────────────────────────────────
    print("\n[2/3] Hybrid  (BM25 + Qdrant, degrades if Qdrant offline)")
    result2 = await router.retrieve(
        RetrievalQuery(
            text="vector database query latency benchmark 2024",
            methods=[RetrievalMethod.WEB_SEARCH, RetrievalMethod.HYBRID],
            max_results=6,
        )
    )
    _print_result(result2, "Web Search + Hybrid")

    # ── Test 3: Multi-query ──────────────────────────────────────────────────
    print("\n[3/3] Multi-query planner sub-tasks fused with RRF")
    result3 = await router.retrieve_multi_query(
        queries=[
            "Qdrant HNSW indexing latency",
            "Weaviate hybrid BM25 vector search",
            "Milvus distributed horizontal scaling",
        ],
        methods=[RetrievalMethod.WEB_SEARCH],
        max_results=5,
    )
    _print_result(result3, "Multi-query Fusion")

    print()
    print("=" * 54)
    print("  Done.")
    print("=" * 54)
    print()


if __name__ == "__main__":
    asyncio.run(main())
