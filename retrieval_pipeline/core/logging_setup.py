"""
core/logging_setup.py

Logging configuration shared by the API server and the CLI.
Windows consoles default to a legacy code page, so stdout/stderr are switched
to UTF-8 (with replacement) to avoid UnicodeEncodeError on symbols like "→".
"""

from __future__ import annotations

import logging
import sys

_NOISY = ("httpx", "httpcore", "sentence_transformers", "huggingface_hub", "filelock", "urllib3",
          "transformers", "asyncio", "langsmith", "primp", "duckduckgo_search", "playwright")


def force_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def setup_logging(level: str = "INFO") -> None:
    force_utf8_stdio()
    root = logging.getLogger()
    if not any(getattr(h, "_research_handler", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)-34s %(message)s", "%H:%M:%S"))
        handler._research_handler = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
