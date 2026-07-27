"""
Splits long documents into overlapping chunks suitable for embedding.
Uses a recursive splitter that prefers breaking on paragraph/sentence
boundaries before falling back to hard character cuts.
"""
from typing import List

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def split_text(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 150
) -> List[str]:
    """
    Recursively split `text` into chunks of ~chunk_size characters,
    with chunk_overlap characters shared between consecutive chunks.
    """
    text = text.strip()
    if not text:
        return []

    chunks = _recursive_split(text, chunk_size, SEPARATORS)
    return _add_overlap(chunks, chunk_overlap, chunk_size)


def _recursive_split(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separator = separators[0] if separators else ""
    remaining_separators = separators[1:] if len(separators) > 1 else []

    if separator == "":
        # Hard cut as last resort
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(separator)
    chunks: List[str] = []
    current = ""

    for part in parts:
        candidate = (current + separator + part) if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size:
                chunks.extend(_recursive_split(part, chunk_size, remaining_separators))
                current = ""
            else:
                current = part

    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]


def _add_overlap(chunks: List[str], overlap: int, chunk_size: int) -> List[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        merged = (prev_tail + " " + chunks[i]).strip()
        # Guard against runaway growth if overlap is large relative to chunk_size
        overlapped.append(merged[: chunk_size + overlap])
    return overlapped
