"""
Thread-safe in-memory conversation history manager with TTL expiry.

Stores Q&A turns per conversation_id. Auto-cleanup on access.
Zero external dependencies — no database, no Redis, no files.
Conversations are ephemeral (lost on restart), which is fine for
a demo/research assistant. Swap for SQLite if persistence is needed later.

Usage:
    from .conversation import conversation_manager

    conv_id = conversation_manager.create()
    conversation_manager.add_message(conv_id, "user", "What is X?")
    conversation_manager.add_message(conv_id, "assistant", "X is ...")
    history = conversation_manager.get_history(conv_id)
"""
import threading
import time
import uuid
from collections import OrderedDict
from typing import Dict, List, Optional

from .config import settings


class ConversationManager:
    """Manages chat conversation history with TTL expiry and LRU eviction."""

    def __init__(self, ttl: int = 3600, max_convs: int = 100, max_history: int = 10):
        self._ttl = ttl
        self._max_convs = max_convs
        self._max_history = max_history
        self._conversations: Dict[str, List[Dict]] = {}
        # OrderedDict for LRU: most recently used at the end
        self._timestamps: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()

    def create(self) -> str:
        """Create a new conversation and return its ID."""
        conv_id = str(uuid.uuid4())
        with self._lock:
            self._conversations[conv_id] = []
            self._timestamps[conv_id] = time.monotonic()
            self._evict_locked()
        return conv_id

    def add_message(self, conv_id: str, role: str, content: str):
        """Append a message to a conversation. Silently no-ops if expired/invalid."""
        with self._lock:
            msgs = self._conversations.get(conv_id)
            if msgs is None:
                return
            msgs.append({"role": role, "content": content})
            self._timestamps[conv_id] = time.monotonic()

    def get_history(self, conv_id: str) -> List[Dict]:
        """Return recent history for a conversation (newest-last order).

        Returns empty list if conversation doesn't exist or expired.
        """
        with self._lock:
            ts = self._timestamps.get(conv_id)
            if ts is None or (time.monotonic() - ts > self._ttl):
                # Expired or invalid — clean up
                self._conversations.pop(conv_id, None)
                self._timestamps.pop(conv_id, None)
                return []
            self._timestamps.move_to_end(conv_id)
            return list(self._conversations[conv_id][-self._max_history:])

    def clear(self, conv_id: str):
        """Delete a conversation entirely."""
        with self._lock:
            self._conversations.pop(conv_id, None)
            self._timestamps.pop(conv_id, None)

    def _evict_locked(self):
        """Evict the oldest conversations when over the limit."""
        while len(self._conversations) > self._max_convs:
            oldest_id, _ = self._timestamps.popitem(last=False)
            del self._conversations[oldest_id]


# Singleton instance — one per process, shared across all endpoints
conversation_manager = ConversationManager(
    ttl=settings.CONVERSATION_TTL,
    max_history=settings.CONVERSATION_MAX_HISTORY,
)
