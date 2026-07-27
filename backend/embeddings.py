"""
Embedding backend, selectable via EMBEDDING_PROVIDER:

  "local"  - sentence-transformers, runs on CPU, no API key needed.
             Pulls in torch, so it needs real RAM (~1GB+) -- fine on a
             laptop or a proper VM, too heavy for many free-tier hosts.
  "openai" - OpenAI's embeddings API. Lightweight install (no torch),
             good fit for small/low-memory hosts. Needs OPENAI_API_KEY
             and a funded OpenAI account (no free API quota).
  "voyage" - Voyage AI's embeddings API. Lightweight install (no torch).
             Needs VOYAGE_API_KEY. Comes with a genuinely free tier
             (200M tokens on the current model generation, no card
             required to start) -- good default for low-memory hosts
             where you don't want to pay for embeddings.

All paths expose the same interface: embed_texts(), embed_query(),
embedding_dim().
"""
from functools import lru_cache
from typing import List
import numpy as np

from .config import settings


# ---------------------------------------------------------------------------
# Local (sentence-transformers) backend
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_local_model():
    # Imported lazily so the (large) torch dependency is only ever loaded
    # if EMBEDDING_PROVIDER=local is actually selected.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.EMBEDDING_MODEL)


def _embed_local(texts: List[str]) -> np.ndarray:
    model = _get_local_model()
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.astype("float32")


def _local_dim() -> int:
    return _get_local_model().get_sentence_embedding_dimension()


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------
def _get_openai_client():
    from openai import OpenAI

    if not settings.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Required when EMBEDDING_PROVIDER=openai."
        )
    return OpenAI(api_key=settings.OPENAI_API_KEY)


def _embed_openai(texts: List[str]) -> np.ndarray:
    client = _get_openai_client()
    response = client.embeddings.create(
        model=settings.OPENAI_EMBEDDING_MODEL, input=texts
    )
    vectors = np.array([item.embedding for item in response.data], dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _openai_dim() -> int:
    return settings.OPENAI_EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Voyage AI backend
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_voyage_client():
    import voyageai

    if not settings.VOYAGE_API_KEY:
        raise RuntimeError(
            "VOYAGE_API_KEY is not set. Required when EMBEDDING_PROVIDER=voyage."
        )
    return voyageai.Client(api_key=settings.VOYAGE_API_KEY)


def _embed_voyage(texts: List[str], input_type: str) -> np.ndarray:
    client = _get_voyage_client()
    result = client.embed(
        texts, model=settings.VOYAGE_EMBEDDING_MODEL, input_type=input_type
    )
    vectors = np.array(result.embeddings, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _voyage_dim() -> int:
    return settings.VOYAGE_EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
def embed_texts(texts: List[str], is_query: bool = False) -> np.ndarray:
    """Embed a list of strings. Returns a (n, dim) float32 numpy array, L2-normalized.

    `is_query` matters only for providers (like Voyage) that distinguish
    query vs. document embeddings for better retrieval quality.
    """
    if not texts:
        return np.zeros((0, embedding_dim()), dtype="float32")

    if settings.EMBEDDING_PROVIDER == "local":
        return _embed_local(texts)
    elif settings.EMBEDDING_PROVIDER == "openai":
        return _embed_openai(texts)
    elif settings.EMBEDDING_PROVIDER == "voyage":
        return _embed_voyage(texts, input_type="query" if is_query else "document")
    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER '{settings.EMBEDDING_PROVIDER}'. "
            "Use 'local', 'openai', or 'voyage'."
        )


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query], is_query=True)[0]


def embedding_dim() -> int:
    if settings.EMBEDDING_PROVIDER == "local":
        return _local_dim()
    elif settings.EMBEDDING_PROVIDER == "openai":
        return _openai_dim()
    elif settings.EMBEDDING_PROVIDER == "voyage":
        return _voyage_dim()
    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER '{settings.EMBEDDING_PROVIDER}'. "
            "Use 'local', 'openai', or 'voyage'."
        )
