"""
core/cache.py

Tiny in-process TTL cache. Used to avoid paying for the same web search twice
within a short window (each Tavily/SerpAPI call costs credits).
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Hashable, Optional


class TTLCache:
    def __init__(self, ttl_s: float, max_entries: int = 512):
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self._data: "OrderedDict[Hashable, tuple[float, Any]]" = OrderedDict()

    def get(self, key: Hashable) -> Optional[Any]:
        if self.ttl_s <= 0:
            return None
        item = self._data.get(key)
        if item is None:
            return None
        expires, value = item
        if expires < time.monotonic():
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: Hashable, value: Any) -> None:
        if self.ttl_s <= 0:
            return
        self._data[key] = (time.monotonic() + self.ttl_s, value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
