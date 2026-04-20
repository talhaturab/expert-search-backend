"""In-memory session store for conversational context.

Holds the last SessionTurn per conversation_id. Entries expire after `ttl_seconds`.
Eviction is lazy (performed on every get/put). Thread-safety via a single Lock —
good enough for the single-worker uvicorn default; swap for Redis if we ever
scale horizontally.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from app.models import ChatResponse


@dataclass
class SessionTurn:
    query: str
    response: ChatResponse
    timestamp: float  # time.monotonic() at write time


class SessionStore:
    def __init__(self, ttl_seconds: int = 1800):
        self._ttl = ttl_seconds
        self._data: dict[str, SessionTurn] = {}
        self._lock = threading.Lock()

    def get(self, conversation_id: str) -> SessionTurn | None:
        with self._lock:
            self._evict_expired_locked()
            return self._data.get(conversation_id)

    def put(self, conversation_id: str, turn: SessionTurn) -> None:
        with self._lock:
            self._evict_expired_locked()
            self._data[conversation_id] = turn

    def _evict_expired_locked(self) -> None:
        now = time.monotonic()
        stale = [cid for cid, t in self._data.items() if now - t.timestamp > self._ttl]
        for cid in stale:
            del self._data[cid]
