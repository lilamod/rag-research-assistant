"""
Local embedding model wrapper (sentence-transformers).
Runs on CPU, no external API calls or API keys required.
"""
from functools import lru_cache
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import settings


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    # Cached so the (relatively large) model is loaded into memory only once.
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of strings. Returns a (n, dim) float32 numpy array, L2-normalized."""
    if not texts:
        return np.zeros((0, embedding_dim()), dtype="float32")
    model = _get_model()
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine similarity via inner product
        show_progress_bar=False,
    )
    return vectors.astype("float32")


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]


def embedding_dim() -> int:
    return _get_model().get_sentence_embedding_dimension()
