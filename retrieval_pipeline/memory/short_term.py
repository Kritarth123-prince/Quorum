"""
memory/short_term.py

Session memory: conversation turns, intermediate agent outputs and final reports.

Backends (first available wins):
  1. Redis   — if REDIS_ENABLED and a server answers PING
  2. SQLite  — <data_dir>/sessions.db (default on a laptop; survives restarts)
  3. In-process dict — last resort

Each session is a "thread" (like a Perplexity thread): follow-up questions reuse
the session_id, and the planner reads the previous turns to resolve references
such as "what about its pricing?".
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from retrieval_pipeline.config import get_settings
from retrieval_pipeline.core.provider_health import registry

logger = logging.getLogger(__name__)
cfg = get_settings()

registry.register("redis", "infra", configured=cfg.redis_enabled)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Backends ──────────────────────────────────────────────────────────────────

class _MemoryBackend:
    name = "memory"

    def __init__(self):
        self._kv: dict[str, str] = {}
        self._sessions: dict[str, dict] = {}

    async def set(self, key: str, value: str) -> None:
        self._kv[key] = value

    async def get(self, key: str) -> Optional[str]:
        return self._kv.get(key)

    async def delete_prefix(self, prefix: str) -> None:
        for k in [k for k in self._kv if k.startswith(prefix)]:
            del self._kv[k]

    async def touch_session(self, session_id: str, title: Optional[str], turns: int) -> None:
        s = self._sessions.setdefault(session_id, {"session_id": session_id, "title": title, "created_at": _now_iso()})
        if title and not s.get("title"):
            s["title"] = title
        s["turn_count"] = turns
        s["updated_at"] = _now_iso()

    async def list_sessions(self, limit: int) -> list[dict]:
        return sorted(self._sessions.values(), key=lambda s: s.get("updated_at", ""), reverse=True)[:limit]

    async def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


class _SQLiteBackend:
    name = "sqlite"

    def __init__(self, path):
        self.path = str(path)
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at REAL)")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, title TEXT, "
                "created_at TEXT, updated_at TEXT, turn_count INTEGER DEFAULT 0)"
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def _run(self, fn):
        def wrapped():
            conn = self._conn()
            try:
                with conn:
                    return fn(conn)
            finally:
                conn.close()
        return asyncio.to_thread(wrapped)

    async def set(self, key: str, value: str) -> None:
        await self._run(lambda c: c.execute(
            "INSERT INTO kv(key, value, updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, value, time.time()),
        ))

    async def get(self, key: str) -> Optional[str]:
        row = await self._run(lambda c: c.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone())
        return row[0] if row else None

    async def delete_prefix(self, prefix: str) -> None:
        await self._run(lambda c: c.execute("DELETE FROM kv WHERE key LIKE ? ESCAPE '\\'",
                                            (prefix.replace("%", "\\%").replace("_", "\\_") + "%",)))

    async def touch_session(self, session_id: str, title: Optional[str], turns: int) -> None:
        now = _now_iso()
        await self._run(lambda c: c.execute(
            "INSERT INTO sessions(session_id, title, created_at, updated_at, turn_count) VALUES(?,?,?,?,?) "
            "ON CONFLICT(session_id) DO UPDATE SET updated_at=excluded.updated_at, turn_count=excluded.turn_count, "
            "title=COALESCE(sessions.title, excluded.title)",
            (session_id, title, now, now, turns),
        ))

    async def list_sessions(self, limit: int) -> list[dict]:
        rows = await self._run(lambda c: c.execute(
            "SELECT session_id, title, created_at, updated_at, turn_count FROM sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall())
        return [dict(zip(("session_id", "title", "created_at", "updated_at", "turn_count"), r)) for r in rows]

    async def delete_session(self, session_id: str) -> None:
        await self._run(lambda c: c.execute("DELETE FROM sessions WHERE session_id=?", (session_id,)))


class _RedisBackend:
    name = "redis"
    _INDEX = "sessions:index"

    def __init__(self, client):
        self.client = client

    async def set(self, key: str, value: str) -> None:
        await self.client.set(key, value, ex=cfg.redis_ttl_s)

    async def get(self, key: str) -> Optional[str]:
        return await self.client.get(key)

    async def delete_prefix(self, prefix: str) -> None:
        keys = [k async for k in self.client.scan_iter(f"{prefix}*")]
        if keys:
            await self.client.delete(*keys)

    async def touch_session(self, session_id: str, title: Optional[str], turns: int) -> None:
        meta_key = f"sessionmeta:{session_id}"
        raw = await self.client.get(meta_key)
        meta = json.loads(raw) if raw else {"session_id": session_id, "title": title, "created_at": _now_iso()}
        if title and not meta.get("title"):
            meta["title"] = title
        meta.update(turn_count=turns, updated_at=_now_iso())
        await self.client.set(meta_key, json.dumps(meta), ex=cfg.redis_ttl_s)
        await self.client.zadd(self._INDEX, {session_id: time.time()})

    async def list_sessions(self, limit: int) -> list[dict]:
        ids = await self.client.zrevrange(self._INDEX, 0, limit - 1)
        out = []
        for sid in ids:
            raw = await self.client.get(f"sessionmeta:{sid}")
            if raw:
                out.append(json.loads(raw))
            else:
                await self.client.zrem(self._INDEX, sid)
        return out

    async def delete_session(self, session_id: str) -> None:
        await self.client.delete(f"sessionmeta:{session_id}")
        await self.client.zrem(self._INDEX, session_id)


_backend = None
_backend_lock: Optional[asyncio.Lock] = None


async def _get_backend():
    global _backend, _backend_lock
    if _backend is not None:
        return _backend
    if _backend_lock is None:
        _backend_lock = asyncio.Lock()
    async with _backend_lock:
        if _backend is not None:
            return _backend
        if cfg.redis_enabled:
            try:
                import redis.asyncio as aioredis
                client = aioredis.Redis(host=cfg.redis_host, port=cfg.redis_port, db=cfg.redis_db,
                                        decode_responses=True, socket_connect_timeout=1.5)
                await client.ping()
                _backend = _RedisBackend(client)
                registry.record_success("redis")
                logger.info("Session memory: Redis at %s:%s", cfg.redis_host, cfg.redis_port)
                return _backend
            except Exception as exc:
                registry.register("redis", "infra", configured=False)    # shown as "not in use"
                logger.info("Redis not available (%s) — using SQLite session store", type(exc).__name__)
        try:
            _backend = _SQLiteBackend(cfg.sessions_db_path)
            logger.info("Session memory: SQLite (%s)", cfg.sessions_db_path)
        except Exception as exc:
            logger.warning("SQLite session store unavailable (%s) — using in-process memory", exc)
            _backend = _MemoryBackend()
        return _backend


async def backend_name() -> str:
    return (await _get_backend()).name


def reset_backend() -> None:
    """For tests: forget the chosen backend."""
    global _backend, _backend_lock
    _backend, _backend_lock = None, None


# ── Session memory ────────────────────────────────────────────────────────────

class SessionMemory:
    """
    Per-session memory; all keys are namespaced under the session id.

        mem = SessionMemory("sess_abc")
        await mem.add_turn(query, answer, report_id)
        turns = await mem.get_turns()
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or f"sess_{uuid4().hex[:12]}"

    def _key(self, suffix: str) -> str:
        return f"session:{self.session_id}:{suffix}"

    async def _load_json(self, suffix: str, default: Any) -> Any:
        raw = await (await _get_backend()).get(self._key(suffix))
        if not raw:
            return default
        try:
            return json.loads(raw)
        except ValueError:
            return default

    # ── Conversation history ──────────────────────────────────────────────────

    async def add_message(self, role: str, content: str) -> None:
        history = await self.get_history()
        history.append({"role": role, "content": content, "timestamp": _now_iso()})
        await (await _get_backend()).set(self._key("history"), json.dumps(history[-50:]))

    async def get_history(self) -> list[dict]:
        return await self._load_json("history", [])

    async def get_messages_for_llm(self) -> list[dict[str, str]]:
        return [{"role": m["role"], "content": m["content"]} for m in await self.get_history()]

    # ── Turns (question → answer) ─────────────────────────────────────────────

    async def add_turn(self, query: str, answer: str, report_id: Optional[str] = None,
                       standalone_query: Optional[str] = None) -> None:
        turns = await self.get_turns()
        turns.append({
            "query":            query,
            "standalone_query": standalone_query or query,
            "answer":           answer,
            "report_id":        report_id,
            "timestamp":        _now_iso(),
        })
        turns = turns[-20:]
        backend = await _get_backend()
        await backend.set(self._key("turns"), json.dumps(turns))
        await backend.touch_session(self.session_id, turns[0]["query"][:120], len(turns))

    async def get_turns(self) -> list[dict]:
        return await self._load_json("turns", [])

    # ── Intermediate state / reports ──────────────────────────────────────────

    async def save(self, key: str, value: Any) -> None:
        await (await _get_backend()).set(self._key(key), json.dumps(value, default=str))

    async def load(self, key: str) -> Optional[Any]:
        return await self._load_json(key, None)

    async def save_report(self, report: dict) -> None:
        reports = await self._load_json("reports", [])
        reports.append(report)
        await self.save("reports", reports[-20:])
        await self.save("final_report", report)

    async def get_reports(self) -> list[dict]:
        return await self._load_json("reports", [])

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def clear(self) -> None:
        backend = await _get_backend()
        await backend.delete_prefix(f"session:{self.session_id}:")
        await backend.delete_session(self.session_id)

    async def get_summary(self) -> dict:
        turns = await self.get_turns()
        return {"session_id": self.session_id, "turn_count": len(turns), "has_report": bool(await self.load("final_report"))}


async def list_sessions(limit: int = 30) -> list[dict]:
    return await (await _get_backend()).list_sessions(limit)
