"""
retrieval/vector_store.py

One shared Qdrant client for the whole process (vector RAG + long-term memory).

Backend selection (QDRANT_MODE):
  auto    → Qdrant server if reachable, else embedded on-disk store (<data_dir>/qdrant),
            else in-memory if the on-disk store is locked by another process
  server  → Qdrant server only
  local   → embedded on-disk store only (no Docker needed)
  memory  → in-memory store (lost on restart)

The embedded store allows only ONE client per folder, which is why every
component must go through get_qdrant() instead of constructing its own client.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from typing import Optional

import httpx

from retrieval_pipeline.config import get_settings
from retrieval_pipeline.core.provider_health import registry

logger = logging.getLogger(__name__)
cfg = get_settings()

registry.register("qdrant", "infra", configured=True)

_client = None
_client_loop: Optional[asyncio.AbstractEventLoop] = None
_backend: str = "uninitialised"
_backend_detail: str = ""
_init_error: Optional[str] = None
_ensured: dict[str, int] = {}
_locks: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = weakref.WeakKeyDictionary()


def _init_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _locks[loop] = lock
    return lock


async def _server_reachable() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            resp = await client.get(f"http://{cfg.qdrant_host}:{cfg.qdrant_port}/healthz")
            return resp.status_code == 200
    except Exception:
        return False


async def _open_server():
    from qdrant_client import AsyncQdrantClient
    client = AsyncQdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port, api_key=cfg.qdrant_api_key, timeout=10)
    await client.get_collections()
    return client


async def _open_local():
    from qdrant_client import AsyncQdrantClient
    path = cfg.qdrant_storage_path
    path.mkdir(parents=True, exist_ok=True)
    client = AsyncQdrantClient(path=str(path))
    await client.get_collections()
    return client


async def _open_memory():
    from qdrant_client import AsyncQdrantClient
    client = AsyncQdrantClient(location=":memory:")
    await client.get_collections()
    return client


async def get_qdrant():
    """Return the shared AsyncQdrantClient, or None if no backend could be opened."""
    global _client, _client_loop, _backend, _backend_detail, _init_error

    loop = asyncio.get_running_loop()
    if _client is not None and (_backend != "server" or _client_loop is loop):
        return _client
    if _init_error is not None and _client is None:
        return None

    async with _init_lock():
        if _client is not None and (_backend != "server" or _client_loop is loop):
            return _client

        mode = (cfg.qdrant_mode or "auto").lower()
        attempts: list[str] = []
        if mode in ("auto", "server"):
            attempts.append("server")
        if mode in ("auto", "local"):
            attempts.append("local")
        if mode in ("auto", "memory", "local"):
            attempts.append("memory")

        errors = []
        for backend in attempts:
            try:
                if backend == "server":
                    if mode == "auto" and not await _server_reachable():
                        errors.append("server: not reachable")
                        continue
                    _client = await _open_server()
                    _backend_detail = f"server {cfg.qdrant_host}:{cfg.qdrant_port}"
                elif backend == "local":
                    _client = await _open_local()
                    _backend_detail = f"embedded ({cfg.qdrant_storage_path})"
                else:
                    _client = await _open_memory()
                    _backend_detail = "in-memory (not persisted)"
                _backend, _client_loop, _init_error = backend, loop, None
                _ensured.clear()
                registry.record_success("qdrant")
                logger.info("Vector store backend: %s", _backend_detail)
                if errors:
                    logger.info("Vector store fallbacks tried: %s", "; ".join(errors))
                return _client
            except Exception as exc:
                errors.append(f"{backend}: {type(exc).__name__}: {str(exc)[:160]}")
                logger.warning("Qdrant %s backend unavailable: %s", backend, errors[-1])

        _init_error = "; ".join(errors) or "no backend allowed by QDRANT_MODE"
        _backend, _backend_detail = "unavailable", _init_error
        logger.warning("Vector store disabled — %s", _init_error)
        return None


def collection_name(base: str, dim: int) -> str:
    """Vector size is part of the name, so switching embedding models never corrupts a collection."""
    return f"{base}_{dim}d"


async def ensure_collection(base: str, dim: int, keyword_fields: tuple[str, ...] = ()) -> Optional[str]:
    """Create the collection if needed; returns its actual name (or None without a backend)."""
    client = await get_qdrant()
    if client is None:
        return None
    name = collection_name(base, dim)
    if _ensured.get(name) == dim:
        return name

    from qdrant_client.models import Distance, VectorParams
    existing = {c.name for c in (await client.get_collections()).collections}
    if name not in existing:
        await client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        if _backend == "server":            # payload indexes are a no-op in embedded mode
            for field in keyword_fields:
                try:
                    await client.create_payload_index(collection_name=name, field_name=field, field_schema="keyword")
                except Exception as exc:
                    logger.debug("Payload index %s on %s failed: %s", field, name, exc)
        logger.info("Created Qdrant collection %s", name)
    _ensured[name] = dim
    return name


def backend_info() -> dict:
    return {"backend": _backend, "detail": _backend_detail}


async def close_qdrant() -> None:
    global _client, _backend
    if _client is not None:
        try:
            await _client.close()
        except Exception:
            pass
    _client = None
    _backend = "uninitialised"
    _ensured.clear()
