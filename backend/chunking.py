"""
Splits long documents into overlapping chunks suitable for embedding.

Strategies (configurable via CHUNKING_STRATEGY env var):
  "recursive" (default) — tries paragraph → sentence → character separators.
  "sliding" (legacy)    — fixed-size sliding window with whitespace snap.

The recursive strategy preserves semantic boundaries much better than
a pure sliding window, especially for structured documents.
"""
from typing import List

from .config import settings


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def split_text(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 150
) -> List[str]:
    """Dispatch to the configured chunking strategy."""
    strategy = settings.CHUNKING_STRATEGY
    if strategy == "recursive":
        return split_text_recursive(text, chunk_size, chunk_overlap)
    elif strategy == "sliding":
        return split_text_sliding(text, chunk_size, chunk_overlap)
    raise ValueError(
        f"Unknown CHUNKING_STRATEGY '{strategy}'. "
        "Use 'recursive' or 'sliding'."
    )


# ---------------------------------------------------------------------------
# Recursive chunking (recommended)
# ---------------------------------------------------------------------------
def split_text_recursive(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[str]:
    """
    Recursive text splitter. Splits on paragraph boundaries first, then
    sentence boundaries, then falls back to character-level for remnants
    that still exceed chunk_size.

    This preserves semantic units (paragraphs, sentences) much better
    than a pure sliding-window character split.
    """
    separators = ["\n\n", "\n", ". ", "! ", "? ", " "]
    return _recursive_split(text, separators, 0, chunk_size, chunk_overlap)


def _recursive_split(
    text: str,
    separators: List[str],
    sep_idx: int,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """Recursively split using increasingly granular separators."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    # Try current separator
    sep = separators[sep_idx] if sep_idx < len(separators) else None

    if sep is None:
        # No more separators — fall back to character-level
        return _split_by_chars(text, chunk_size, chunk_overlap)

    parts = text.split(sep)
    # Merge small parts up to chunk_size
    merged = _merge_parts(parts, sep, chunk_size)

    # If the first separator produced only one big chunk OR didn't help,
    # recurse with the next separator
    if len(merged) == 1 and len(merged[0]) > chunk_size:
        return _recursive_split(text, separators, sep_idx + 1, chunk_size, chunk_overlap)

    chunks = []
    for part in merged:
        if len(part) <= chunk_size:
            chunks.append(part)
        else:
            chunks.extend(
                _recursive_split(part, separators, sep_idx + 1, chunk_size, chunk_overlap)
            )
    return chunks


def _merge_parts(parts: List[str], sep: str, chunk_size: int) -> List[str]:
    """Merge consecutive small parts up to chunk_size."""
    merged = []
    current = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        candidate = (current + sep + part) if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                merged.append(current)
            current = part
    if current:
        merged.append(current)
    return merged


def _split_by_chars(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Fallback: fixed-size character-level sliding window.

    Same algorithm as split_text_sliding but without whitespace snapping
    (no separators left to snap to by this point).
    """
    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        next_start = end - chunk_overlap
        start = next_start if next_start > start else end

    return chunks


# ---------------------------------------------------------------------------
# Sliding-window chunking (legacy)
# ---------------------------------------------------------------------------
def split_text_sliding(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 150
) -> List[str]:
    """
    Split `text` into chunks of at most chunk_size characters, with
    chunk_overlap characters of shared context between consecutive chunks.

    Snaps each cut point to the nearest preceding whitespace so words
    aren't split mid-token. Falls back to exact character boundary if
    no whitespace is found in the window.
    """
    text = text.strip()
    if not text:
        return []
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 4

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Snap to last whitespace to avoid mid-word splits
        if end < text_len:
            snap = text.rfind(" ", start, end)
            if snap > start:
                end = snap

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        next_start = end - chunk_overlap
        start = next_start if next_start > start else end

    return chunks
