"""
Embedding backend, selectable via EMBEDDING_PROVIDER:

  "local"  - sentence-transformers, runs on CPU, no API key needed.
             Pulls in torch, so it needs real RAM (~1GB+) -- fine on a
             laptop or a proper VM, too heavy for many free-tier hosts.
  "openai" - OpenAI's embeddings API. Lightweight install (no torch),
             good fit for small/low-memory hosts. Needs OPENAI_API_KEY.

Both paths expose the same interface: embed_texts(), embed_query(),
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
# Public interface
# ---------------------------------------------------------------------------
def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of strings. Returns a (n, dim) float32 numpy array, L2-normalized."""
    if not texts:
        return np.zeros((0, embedding_dim()), dtype="float32")

    if settings.EMBEDDING_PROVIDER == "local":
        return _embed_local(texts)
    elif settings.EMBEDDING_PROVIDER == "openai":
        return _embed_openai(texts)
    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER '{settings.EMBEDDING_PROVIDER}'. "
            "Use 'local' or 'openai'."
        )


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]


def embedding_dim() -> int:
    if settings.EMBEDDING_PROVIDER == "local":
        return _local_dim()
    elif settings.EMBEDDING_PROVIDER == "openai":
        return _openai_dim()
    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER '{settings.EMBEDDING_PROVIDER}'. "
            "Use 'local' or 'openai'."
        )
