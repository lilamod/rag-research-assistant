"""
Thin abstraction over LLM providers so the rest of the app doesn't care
whether we're calling Anthropic or OpenAI. Selected via LLM_PROVIDER env var.
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

    if settings.LLM_PROVIDER == "anthropic":
        return _call_anthropic(user_message)
    elif settings.LLM_PROVIDER == "openai":
        return _call_openai(user_message)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{settings.LLM_PROVIDER}'. Use 'anthropic' or 'openai'."
        )


def _call_anthropic(user_message: str) -> str:
    import anthropic

    if not settings.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _call_openai(user_message: str) -> str:
    from openai import OpenAI

    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content
