"""
Splits long documents into overlapping chunks suitable for embedding.
Uses a sliding window over the text, snapping each boundary to the nearest
preceding whitespace where possible (so words aren't split mid-token),
while still guaranteeing every chunk is <= chunk_size characters.
"""
from typing import List


def split_text(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 150
) -> List[str]:
    """
    Split `text` into chunks of at most chunk_size characters, with
    chunk_overlap characters of shared context between consecutive chunks.
    """
    text = text.strip()
    if not text:
        return []
    if chunk_overlap >= chunk_size:
        # Prevent an infinite loop / zero forward progress.
        chunk_overlap = chunk_size // 4

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # If we're not at the very end of the text, try to snap the cut
        # point back to the last whitespace so we don't split mid-word.
        if end < text_len:
            snap = text.rfind(" ", start, end)
            if snap > start:
                end = snap

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        # Advance by a full window minus the overlap, guaranteeing forward
        # progress even when `end` got snapped back by whitespace.
        next_start = end - chunk_overlap
        start = next_start if next_start > start else end

    return chunks
