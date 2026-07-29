"""
Orchestrates the full RAG flow:
  ingest:  file -> extract text -> chunk -> embed -> store
  query:   question -> embed -> retrieve top-k -> build prompt -> LLM -> answer + sources

Enhanced with:
  - Conversation memory for multi-turn follow-up questions
  - Query rewriting for ambiguous follow-up references
  - Hybrid search (FAISS + BM25) when enabled
  - Cross-encoder re-ranking when configured
  - TTL-based LRU response cache
"""
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Generator

from .config import settings
from .document_loader import load_text
from .chunking import split_text
from .embeddings import embed_texts, embed_query
from .vector_store import VectorStore
from .llm import generate_answer, generate_answer_stream
from .conversation import conversation_manager

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
store = VectorStore()

# ---------------------------------------------------------------------------
# Response cache (TTL + LRU)
# ---------------------------------------------------------------------------
class _ResponseCache:
    """TTL-based LRU response cache. Cleared on document changes."""

    def __init__(self, maxsize: int = 128, ttl: int = 1800):
        self._cache: OrderedDict[str, Tuple[Dict, float]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl

    def get(self, key: str) -> Dict | None:
        if key in self._cache:
            entry, ts = self._cache[key]
            if time.monotonic() - ts < self._ttl:
                self._cache.move_to_end(key)
                return entry
            del self._cache[key]
        return None

    def put(self, key: str, entry: Dict):
        if key in self._cache:
            del self._cache[key]
        self._cache[key] = (entry, time.monotonic())
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()


_response_cache = _ResponseCache()
_stream_cache = _ResponseCache(maxsize=64, ttl=1800)


# ---------------------------------------------------------------------------
# Retrieve — determines which search backend to use
# ---------------------------------------------------------------------------
def _retrieve(query: str, top_k: int) -> Tuple[List[Dict], List[float]]:
    """Embed the query and search the vector store (or hybrid search if enabled).

    Returns (context_chunks, scores).
    """
    query_vector = embed_query(query)

    if settings.HYBRID_SEARCH_ENABLED:
        # Lazy-import hybrid search so the rank_bm25 dependency is optional
        from .search import hybrid_search
        if hybrid_search is not None:
            results = hybrid_search.search(query_vector, query, top_k=top_k)
        else:
            results = store.search(query_vector, top_k=top_k)
    else:
        results = store.search(query_vector, top_k=top_k)

    context_chunks: List[Dict] = [r[0] for r in results]
    scores: List[float] = [r[1] for r in results]

    # Optional re-ranking
    if settings.RERANKER == "cross-encoder":
        from .reranker import reranker
        candidates = list(zip(context_chunks, scores))
        reranked = reranker.rerank(query, candidates, top_k=top_k)
        context_chunks = [c for c, _ in reranked]
        scores = [s for _, s in reranked]

    return context_chunks, scores


# ---------------------------------------------------------------------------
# Resolve query — handles conversation context and rewriting
# ---------------------------------------------------------------------------
def _resolve_query(
    question: str, conversation_id: str | None
) -> Tuple[str, List[Dict]]:
    """Resolve the effective query and conversation history.

    If conversation_id is provided and has history, the question may be
    rewritten for multi-turn disambiguation. Returns (resolved_query, history).
    """
    history = []
    if conversation_id:
        history = conversation_manager.get_history(conversation_id)

    if history:
        from .llm import rewrite_query
        rewritten = rewrite_query(history, question)
        return rewritten, history

    return question, history


# ---------------------------------------------------------------------------
# Sources builder
# ---------------------------------------------------------------------------
def _build_sources(context_chunks: List[Dict], scores: List[float]) -> List[Dict]:
    return [
        {
            "rank": i + 1,
            "filename": chunk["filename"],
            "doc_id": chunk["doc_id"],
            "text_preview": chunk["text"][:300],
            "relevance_score": round(score, 4),
        }
        for i, (chunk, score) in enumerate(zip(context_chunks, scores))
    ]


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
def ingest_document(file_path: Path, original_filename: str) -> Dict:
    """Process one uploaded file end-to-end and add it to the vector store."""
    text = load_text(file_path)
    if not text.strip():
        raise ValueError("No extractable text found in this document.")

    chunks = split_text(
        text, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
    )
    if not chunks:
        raise ValueError("Document produced no chunks after splitting.")

    vectors = embed_texts(chunks)
    doc_id = str(uuid.uuid4())
    added = store.add(vectors, chunks, {"doc_id": doc_id, "filename": original_filename})

    # Rebuild hybrid search index if enabled
    if settings.HYBRID_SEARCH_ENABLED:
        from .search import hybrid_search_init
        hybrid_search_init(store)

    _response_cache.clear()
    _stream_cache.clear()
    return {
        "doc_id": doc_id,
        "filename": original_filename,
        "chunks_added": added,
    }


# ---------------------------------------------------------------------------
# Answer (non-streaming)
# ---------------------------------------------------------------------------
def answer_question(
    question: str,
    top_k: int = None,
    conversation_id: str | None = None,
) -> Dict:
    """Retrieve relevant chunks and generate a grounded answer with citations."""
    top_k = top_k or settings.TOP_K
    has_context = conversation_id is not None

    # Only cache standalone (non-conversation) queries
    if not has_context:
        cache_key = f"{question}||{top_k}"
        cached = _response_cache.get(cache_key)
        if cached is not None:
            return cached

    # Resolve query with conversation context
    resolved_query, history = _resolve_query(question, conversation_id)

    # Retrieve
    context_chunks, scores = _retrieve(resolved_query, top_k)

    # Generate
    answer = generate_answer(question, context_chunks, history)
    sources = _build_sources(context_chunks, scores)

    result = {"answer": answer, "sources": sources}

    # Save to conversation history
    if has_context:
        conversation_manager.add_message(conversation_id, "user", question)
        conversation_manager.add_message(conversation_id, "assistant", answer)

    # Cache (standalone only)
    if not has_context:
        _response_cache.put(cache_key, result)

    return result


# ---------------------------------------------------------------------------
# Answer (streaming)
# ---------------------------------------------------------------------------
def answer_question_stream(
    question: str,
    top_k: int = None,
    conversation_id: str | None = None,
) -> Generator:
    """Retrieve relevant chunks, then yield answer tokens via streaming.

    Yields (event_type, payload) tuples:
      - ("token", str) for each answer fragment
      - ("sources", list) once streaming is complete
    """
    top_k = top_k or settings.TOP_K
    has_context = conversation_id is not None

    # Resolve query with conversation context
    resolved_query, history = _resolve_query(question, conversation_id)

    # Retrieve
    context_chunks, scores = _retrieve(resolved_query, top_k)

    # Stream the answer and collect for caching
    full_answer = ""
    for token in generate_answer_stream(question, context_chunks, history):
        full_answer += token
        yield ("token", token)

    # Fallback if Gemini returned nothing
    if not full_answer:
        full_answer = (
            "I couldn't generate an answer from the retrieved documents. "
            "This may happen if the content was filtered or the model "
            "returned an empty response. Try rephrasing your question."
        )
        for token in full_answer.split(" "):
            yield ("token", token + " ")

    sources = _build_sources(context_chunks, scores)

    # Save to conversation history
    if has_context:
        conversation_manager.add_message(conversation_id, "user", question)
        conversation_manager.add_message(conversation_id, "assistant", full_answer)

    # Cache (standalone only)
    if not has_context:
        cache_key = f"{question}||{top_k}"
        _stream_cache.put(cache_key, {"answer": full_answer, "sources": sources})
        _response_cache.put(cache_key, {"answer": full_answer, "sources": sources})

    yield ("sources", sources)


# ---------------------------------------------------------------------------
# Document management
# ---------------------------------------------------------------------------
def list_documents() -> List[Dict]:
    return store.list_documents()


def delete_document(doc_id: str) -> bool:
    deleted = store.delete_document(doc_id)
    if deleted and settings.HYBRID_SEARCH_ENABLED:
        from .search import hybrid_search_init
        hybrid_search_init(store)
    return deleted


def get_stats() -> Dict:
    return store.stats()
