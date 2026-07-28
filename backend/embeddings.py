"""
Embedding backend, selectable via EMBEDDING_PROVIDER:

  "gemini" - Google's Gemini embeddings API. Lightweight install (no
             torch). Needs GEMINI_API_KEY - same key used for chat
             generation (see llm.py). Genuinely free tier (1,500
             requests/day), no card required at all. Default.
  "local"  - sentence-transformers, runs on CPU, no API key needed, no
             rate limits at any scale. Pulls in torch, so it needs real
             RAM (~1GB+) -- too heavy for small free-tier hosts like
             Render's free plan, but the right choice for a self-hosted
             VM (see ORACLE_DEPLOY.md) where you have RAM to spare and
             want zero external API calls.

Both paths expose the same interface: embed_texts(), embed_query(),
embedding_dim().
"""
from functools import lru_cache
from typing import List
import numpy as np

from .config import settings


# ---------------------------------------------------------------------------
# Gemini backend
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_gemini_client():
    from google import genai

    if not settings.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Required for embeddings. Get a free "
            "key (no card required) at https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=settings.GEMINI_API_KEY)


def _embed_gemini(texts: List[str], task_type: str) -> np.ndarray:
    from google.genai import types

    client = _get_gemini_client()
    response = client.models.embed_content(
        model=settings.GEMINI_EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=settings.GEMINI_EMBEDDING_DIM,
        ),
    )
    vectors = np.array([e.values for e in response.embeddings], dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _gemini_dim() -> int:
    return settings.GEMINI_EMBEDDING_DIM


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
# Public interface
# ---------------------------------------------------------------------------
def embed_texts(texts: List[str], is_query: bool = False) -> np.ndarray:
    """Embed a list of strings. Returns a (n, dim) float32 numpy array, L2-normalized.

    `is_query` matters only for Gemini, which distinguishes query vs.
    document embeddings for better retrieval quality.
    """
    if not texts:
        return np.zeros((0, embedding_dim()), dtype="float32")

    if settings.EMBEDDING_PROVIDER == "gemini":
        return _embed_gemini(
            texts, task_type="RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        )
    elif settings.EMBEDDING_PROVIDER == "local":
        return _embed_local(texts)
    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER '{settings.EMBEDDING_PROVIDER}'. "
            "Use 'gemini' or 'local'."
        )


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query], is_query=True)[0]


def embedding_dim() -> int:
    if settings.EMBEDDING_PROVIDER == "gemini":
        return _gemini_dim()
    elif settings.EMBEDDING_PROVIDER == "local":
        return _local_dim()
    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER '{settings.EMBEDDING_PROVIDER}'. "
            "Use 'gemini' or 'local'."
        )
