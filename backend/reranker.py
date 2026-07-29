"""
Optional re-ranking of retrieved chunks using cross-encoder models.

Improves precision by scoring query-chunk relevance with a dedicated
model rather than relying on embedding cosine similarity alone.

Two modes:
  - "none" (default): pass-through, no re-ranking
  - "cross-encoder": uses sentence-transformers CrossEncoder

The cross-encoder model is ~80MB and loaded on first use. Off by default
to keep free-tier hosts happy. Enable with RERANKER=cross-encoder in .env.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from .config import settings


class ReRanker(ABC):
    """Abstract re-ranker interface."""

    @abstractmethod
    def rerank(
        self, query: str, candidates: List[Tuple[Dict, float]], top_k: int = 5
    ) -> List[Tuple[Dict, float]]:
        ...


class NullReRanker(ReRanker):
    """Pass-through: no re-ranking applied. Returns top_k as-is."""

    def rerank(
        self, query: str, candidates: List[Tuple[Dict, float]], top_k: int = 5
    ) -> List[Tuple[Dict, float]]:
        return candidates[:top_k]


class CrossEncoderReRanker(ReRanker):
    """Cross-encoder based re-ranker using sentence-transformers.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB)
    Lazy-loaded on first use.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)

    def rerank(
        self, query: str, candidates: List[Tuple[Dict, float]], top_k: int = 5
    ) -> List[Tuple[Dict, float]]:
        if not candidates:
            return candidates

        self._load()
        pairs = [(query, chunk["text"]) for chunk, _ in candidates]
        scores = self._model.predict(pairs)
        scored = [
            (chunk, float(score))
            for (chunk, _), score in zip(candidates, scores)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def _create_reranker() -> ReRanker:
    """Factory: return the appropriate re-ranker based on settings."""
    mode = settings.RERANKER
    if mode == "cross-encoder":
        try:
            return CrossEncoderReRanker()
        except ImportError:
            return NullReRanker()
    return NullReRanker()


# Singleton — one re-ranker per process
reranker: ReRanker = _create_reranker()
