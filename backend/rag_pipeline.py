"""
Orchestrates the full RAG flow:
  ingest:  file -> extract text -> chunk -> embed -> store
  query:   question -> embed -> retrieve top-k -> build prompt -> LLM -> answer + sources
"""
import uuid
from pathlib import Path
from typing import Dict, List

from .config import settings
from .document_loader import load_text
from .chunking import split_text
from .embeddings import embed_texts, embed_query
from .vector_store import VectorStore
from .llm import generate_answer

# Single shared vector store instance for the app's lifetime
store = VectorStore()


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

    return {
        "doc_id": doc_id,
        "filename": original_filename,
        "chunks_added": added,
    }


def answer_question(question: str, top_k: int = None) -> Dict:
    """Retrieve relevant chunks and generate a grounded answer with citations."""
    top_k = top_k or settings.TOP_K
    query_vector = embed_query(question)
    results = store.search(query_vector, top_k=top_k)

    context_chunks: List[Dict] = [r[0] for r in results]
    scores = [r[1] for r in results]

    answer = generate_answer(question, context_chunks)

    sources = [
        {
            "rank": i + 1,
            "filename": chunk["filename"],
            "doc_id": chunk["doc_id"],
            "text_preview": chunk["text"][:300],
            "relevance_score": round(score, 4),
        }
        for i, (chunk, score) in enumerate(zip(context_chunks, scores))
    ]

    return {"answer": answer, "sources": sources}


def list_documents() -> List[Dict]:
    return store.list_documents()


def delete_document(doc_id: str) -> bool:
    return store.delete_document(doc_id)


def get_stats() -> Dict:
    return store.stats()
