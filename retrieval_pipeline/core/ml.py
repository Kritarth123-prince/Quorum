"""
core/ml.py

Lazy, thread-safe loading of the local ML models (sentence embeddings and the
cross-encoder reranker).

  - Models load on first use (or during API warm-up), never at import time,
    so a missing model or a broken torch install can't crash the platform.
  - The local Hugging Face cache is tried first with no network access; only if
    the model isn't cached is it downloaded.
  - All inference runs on ONE dedicated worker thread so the two models never
    fight over CPU cores (important on small laptops) and the event loop stays free.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, TypeVar

from retrieval_pipeline.config import get_settings

logger = logging.getLogger(__name__)
cfg = get_settings()

warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)

T = TypeVar("T")

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ml-worker")

# When False (API server), requests never wait for a cold model load: they use the
# BM25 / score-based fallbacks while the model loads in the background.
# When True (CLI, scripts), callers wait so a one-shot run gets full quality.
BLOCKING_LOAD = True
_background_loads: set[asyncio.Task] = set()


async def run_ml(fn: Callable[..., T], *args: Any) -> T:
    """Run a blocking model call on the dedicated ML thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, fn, *args)


def resolve_device(requested: str) -> str:
    if requested and requested.lower() != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover - torch missing or broken
        pass
    return "cpu"


def _numpy_ok() -> Optional[str]:
    try:
        import numpy as np
        major = int(np.__version__.split(".")[0])
        if major >= 2:
            import torch  # noqa: F401  (torch 2.3 wheels were built against numpy 1.x)
            if torch.__version__.startswith(("2.0", "2.1", "2.2", "2.3")):
                return f"numpy {np.__version__} is incompatible with torch {torch.__version__}; run: pip install \"numpy<2\""
    except Exception as exc:
        return f"numpy/torch import failed: {exc}"
    return None


class _LazyModel:
    """Loads a model once, remembers failures so we don't retry on every request."""

    kind = "model"

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model: Any = None
        self._error: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def error(self) -> Optional[str]:
        return self._error

    def _build(self, local_only: bool) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    def get(self) -> Any:
        """Return the loaded model, or None if it can't be loaded. Blocking — call via run_ml()."""
        if self._model is not None or self._error is not None:
            return self._model
        with self._lock:
            if self._model is not None or self._error is not None:
                return self._model
            problem = _numpy_ok()
            if problem:
                self._error = problem
                logger.warning("%s disabled: %s", self.kind, problem)
                return None
            try:
                try:
                    self._model = self._build(local_only=True)
                except Exception:
                    logger.info("%s %s not in local cache — downloading…", self.kind, self.model_name)
                    self._model = self._build(local_only=False)
                logger.info("%s ready: %s", self.kind, self.model_name)
            except Exception as exc:
                self._error = f"{type(exc).__name__}: {str(exc)[:300]}"
                logger.warning("%s %s could not be loaded (%s) — using fallback", self.kind, self.model_name, self._error)
        return self._model

    async def aget(self) -> Any:
        return await run_ml(self.get)

    def status(self) -> dict:
        return {
            "model":  self.model_name,
            "status": "loaded" if self.loaded else ("failed" if self._error else "not_loaded"),
            "error":  self._error,
        }


class _EmbeddingModel(_LazyModel):
    kind = "Embedding model"

    def _build(self, local_only: bool) -> Any:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(
            self.model_name,
            device=resolve_device(cfg.embedding_device),
            local_files_only=local_only,
        )
        model.max_seq_length = min(512, model.max_seq_length or 512)
        return model


class _RerankerModel(_LazyModel):
    kind = "Reranker model"

    def _build(self, local_only: bool) -> Any:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(
            self.model_name,
            device=resolve_device(cfg.embedding_device),
            max_length=512,
            local_files_only=local_only,
        )


embedding_model = _EmbeddingModel(cfg.embedding_model)
reranker_model  = _RerankerModel(cfg.reranker_model)


async def model_ready(model: _LazyModel, wait: Optional[bool] = None) -> bool:
    """
    True if `model` can be used right now. In non-blocking mode a cold model is
    scheduled to load in the background and False is returned immediately.
    """
    if model.loaded:
        return True
    if model.error is not None:
        return False
    if BLOCKING_LOAD if wait is None else wait:
        return (await model.aget()) is not None
    if not any(getattr(t, "_ml_model", None) is model for t in _background_loads):
        task = asyncio.create_task(model.aget())
        task._ml_model = model  # type: ignore[attr-defined]
        _background_loads.add(task)
        task.add_done_callback(_background_loads.discard)
        logger.info("%s still loading — using fallback ranking for this request", model.kind)
    return False


async def warm_up() -> None:
    """Load both models in the background so the first research request is fast."""
    await embedding_model.aget()
    if cfg.reranker_enabled:
        await reranker_model.aget()
