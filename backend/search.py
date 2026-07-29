"""
Hybrid search combining dense (FAISS) and sparse (BM25) retrieval
via Reciprocal Rank Fusion (RRF).

Dependencies:
  - rank-bm25 (pure Python, ~50KB, no native deps) — pip install rank-bm25

The BM25 index is rebuilt after every ingest/delete. For small document sets
(<1000 chunks) this is sub-millisecond. The HybridSearch wraps the existing
VectorStore — no changes needed to FAISS.
"""
import threading
from typing import Dict, List, Tuple

import numpy as np

from .config import settings
from .vector_store import VectorStore

RRF_K = 60  # RRF constant


class HybridSearch:
    """Wraps VectorStore with BM25 keyword search + RRF fusion.

    Usage:
        hybrid = HybridSearch(vector_store)
        results = hybrid.search(query_vector, query_text, top_k=5)
        # Returns (chunk_dict, fusion_score) tuples
    """

    def __init__(self, vector_store: VectorStore):
        self._vector_store = vector_store
        self._bm25 = None
        self._tokenized: List[List[str]] = []
        self._lock = threading.Lock()
        self._build_index()

    def _build_index(self):
        """(Re)build the BM25 index from the current vector store records."""
        from rank_bm25 import BM25Okapi

        with self._lock:
            records = self._vector_store.records
            if records:
                self._tokenized = [rec["text"].split() for rec in records]
                self._bm25 = BM25Okapi(self._tokenized)
            else:
                self._bm25 = None
                self._tokenized = []

    def search(
        self, query_vector: np.ndarray, query_text: str, top_k: int = 5
    ) -> List[Tuple[Dict, float]]:
        """Hybrid search: FAISS + BM25 fused with RRF.

        Returns top_k results sorted by RRF score descending.
        """
        n_total = len(self._vector_store.records)
        if n_total == 0:
            return []

        top_k = min(top_k, n_total)
        k_for_each = min(top_k * 3, n_total)

        # Vector search
        vector_results = self._vector_store.search(query_vector, top_k=k_for_each)

        # BM25 scores
        bm25_scores = None
        if self._bm25 is not None:
            bm25_scores = self._bm25.get_scores(query_text.split())

        # RRF fusion
        alpha = settings.HYBRID_SEARCH_ALPHA
        doc_scores: Dict[str, dict] = {}

        for rank, (chunk, score) in enumerate(vector_results):
            doc_scores[chunk["id"]] = {"chunk": chunk, "rrf": 0.0}
            doc_scores[chunk["id"]]["rrf"] += alpha * (1.0 / (RRF_K + rank + 1))

        if bm25_scores is not None:
            bm25_ranks = np.argsort(bm25_scores)[::-1]
            for rank, idx in enumerate(bm25_ranks[:k_for_each]):
                chunk = self._vector_store.records[idx]
                if chunk["id"] not in doc_scores:
                    doc_scores[chunk["id"]] = {"chunk": chunk, "rrf": 0.0}
                doc_scores[chunk["id"]]["rrf"] += (1 - alpha) * (1.0 / (RRF_K + rank + 1))

        # Sort by RRF score descending
        sorted_results = sorted(
            doc_scores.values(), key=lambda x: x["rrf"], reverse=True
        )
        return [(r["chunk"], r["rrf"]) for r in sorted_results[:top_k]]

    def rebuild_if_needed(self):
        """Call after ingest/delete to keep BM25 in sync."""
        self._build_index()


# Singleton
hybrid_search: HybridSearch | None = None


def hybrid_search_init(vector_store: VectorStore):
    """Initialize or re-initialize the global hybrid search singleton."""
    global hybrid_search
    hybrid_search = HybridSearch(vector_store)
