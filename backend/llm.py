"""
LLM generation via Gemini only. Reuses the same GEMINI_API_KEY used for
embeddings (see embeddings.py) - one key covers both.
"""
from typing import List, Dict
from .config import settings

SYSTEM_PROMPT = """You are a careful research assistant. Answer the user's question \
using ONLY the information in the provided context excerpts. Each excerpt is \
labeled with a source number like [1], [2], etc.

Rules:
- Cite sources inline using their bracket number, e.g. "The results improved by 12% [2]."
- If the context does not contain enough information to answer, say so plainly \
instead of guessing.
- Be concise and precise. Do not fabricate facts or citations.
"""


def _build_user_message(question: str, context_chunks: List[Dict]) -> str:
    context_block = "\n\n".join(
        f"[{i + 1}] (source: {c['filename']})\n{c['text']}"
        for i, c in enumerate(context_chunks)
    )
    return (
        f"Context excerpts:\n\n{context_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer the question using the context above, citing sources like [1]."
    )


def generate_answer(question: str, context_chunks: List[Dict]) -> str:
    if not context_chunks:
        return (
            "I couldn't find any relevant documents to answer that question. "
            "Try uploading source material first, or rephrase your question."
        )

    user_message = _build_user_message(question, context_chunks)
    return _call_gemini(user_message)


def _call_gemini(user_message: str) -> str:
    from google import genai
    from google.genai import types

    if not settings.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Required for both chat generation and "
            "embeddings. Get a free key (no card required) at "
            "https://aistudio.google.com/apikey"
        )

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=1024,
        ),
    )
    return response.text
