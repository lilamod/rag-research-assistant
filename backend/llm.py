"""
LLM generation via Gemini only. Reuses the same GEMINI_API_KEY used for
embeddings (see embeddings.py) - one key covers both.

Enhanced with:
- Flexible system prompt that follows formatting instructions
- Conversation history injection for multi-turn context
- Query rewriting for follow-up question disambiguation
- Formatting instruction extraction from user queries
"""
import threading
from typing import List, Dict, Generator

from .config import settings

SYSTEM_PROMPT = """You are a helpful research assistant. Answer the user's question \
using ONLY the information in the provided context excerpts. Each excerpt is \
labeled with a source number like [1], [2], etc.

Rules:
- Cite sources inline using their bracket number, e.g. "The results improved by 12% [2]."
- If the context does not contain enough information to answer, say so plainly \
instead of guessing.
- Do not fabricate facts or citations.
- You MAY follow the user's formatting or style instructions (e.g. "use bullet points", \
"write a table", "in one sentence", "explain like I'm a beginner") as long as you \
remain factually accurate to the provided context.
"""


# ---------------------------------------------------------------------------
# Greeting detection — avoids pointless API calls for casual greetings
# ---------------------------------------------------------------------------
_GREETING_KEYWORDS = {
    "hello", "hi", "hey", "greetings", "good morning", "good afternoon",
    "good evening", "howdy", "what's up", "sup", "yo",
}


def _is_greeting(question: str) -> bool:
    """Detect if the question is purely a greeting with no actual query.

    Returns True if the message is just a greeting (e.g. "hello", "hi").
    """
    q = question.lower().strip().rstrip("?!.,")
    if q in _GREETING_KEYWORDS:
        return True
    # Also check short phrases that start with a greeting
    words = q.split()
    if len(words) <= 3 and words[0] in _GREETING_KEYWORDS:
        return True
    return False


_GREETING_RESPONSE = (
    "Hello! I'm your research assistant. I can answer questions based on the "
    "documents you've uploaded. Try asking me something specific about your "
    "sources — for example, \"What are the key findings in the report?\" or "
    "\"Summarize the methodology section.\""
)


def _extract_format_instructions(question: str) -> str:
    """Detect formatting/style requests in the user's question.

    Returns a directive string to append to the prompt, or empty string.
    """
    q_lower = question.lower()
    hints = []

    format_map = {
        "bullet": "Use bullet points for your answer.",
        "list": "Present your answer as a list.",
        "table": "Use a table to present structured data.",
        "numbered": "Present your answer as a numbered list.",
        "step": "Present your answer as numbered steps.",
        "summar": "Provide a concise summary.",
        "tldr": "Start with a one-sentence TL;DR summary.",
        "tl;dr": "Start with a one-sentence TL;DR summary.",
        "short": "Be brief and concise.",
        "simple": "Explain in simple terms.",
        "explain like": "Explain in simple, accessible language.",
        "detailed": "Provide a thorough, detailed explanation.",
        "paragraph": "Write your answer as a single coherent paragraph.",
    }

    for keyword, instruction in format_map.items():
        if keyword in q_lower:
            hints.append(instruction)

    if hints:
        return "Format instruction: " + " ".join(hints)
    return ""


def _build_user_message(
    question: str,
    context_chunks: List[Dict],
    conversation_history: List[Dict] | None = None,
) -> str:
    """Build the full user prompt with context, history, and formatting hints."""
    context_block = "\n\n".join(
        f"[{i + 1}] (source: {c['filename']})\n{c['text']}"
        for i, c in enumerate(context_chunks)
    )

    parts = [f"Context excerpts:\n\n{context_block}"]

    if conversation_history:
        history_lines = []
        for msg in conversation_history:
            label = "User" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{label}: {msg['content']}")
        parts.append("Conversation history:\n" + "\n".join(history_lines))

    format_hint = _extract_format_instructions(question)
    answer_prompt = f"Question: {question}"
    if format_hint:
        answer_prompt += f"\n\n{format_hint}"
    answer_prompt += (
        "\n\nAnswer the question using the context above, citing sources like [1]. "
        "Follow any formatting requests in the question."
    )
    parts.append(answer_prompt)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Thread-safe Gemini client singleton
# ---------------------------------------------------------------------------
_llm_client = None
_llm_lock = threading.Lock()


def _get_llm_client():
    """Thread-safe singleton for the Gemini client."""
    global _llm_client
    if _llm_client is not None:
        return _llm_client
    with _llm_lock:
        if _llm_client is not None:
            return _llm_client
        from google import genai

        if not settings.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Required for both chat generation and "
                "embeddings. Get a free key (no card required) at "
                "https://aistudio.google.com/apikey"
            )
        _llm_client = genai.Client(api_key=settings.GEMINI_API_KEY)
        return _llm_client


# ---------------------------------------------------------------------------
# Query rewriting for multi-turn conversations
# ---------------------------------------------------------------------------
def rewrite_query(conversation_history: List[Dict], current_question: str) -> str:
    """Reformulate a follow-up question as a standalone query for better retrieval.

    Uses a lightweight Gemini call to resolve ambiguous references like
    "what about the other method?" or "explain that last point in detail."

    Returns the original question unchanged if rewriting is disabled or
    there is no history to work with.
    """
    if not conversation_history or not settings.QUERY_REWRITE_ENABLED:
        return current_question

    prompt_parts = [
        "Rewrite the user's latest question as a standalone search query "
        "that would retrieve the relevant documents. If the question is "
        "already self-contained, return it verbatim.\n",
        "Conversation:",
    ]
    for msg in conversation_history[-6:]:  # last 3 exchanges
        label = "Human" if msg["role"] == "user" else "Assistant"
        prompt_parts.append(f"{label}: {msg['content']}")
    prompt_parts.append(f"\nLatest question: {current_question}")
    prompt_parts.append("\nStandalone query:")

    prompt = "\n".join(prompt_parts)
    try:
        rewritten = _call_gemini_lite(prompt)
        return rewritten if rewritten else current_question
    except Exception:
        # If rewriting fails (rate limit, etc.), fall back to original
        return current_question


def _call_gemini_lite(user_message: str) -> str:
    """Lightweight Gemini call for query rewriting / auxiliary tasks.

    Uses a low token limit and low temperature for fast, deterministic output.
    """
    from google.genai import types

    client = _get_llm_client()
    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            max_output_tokens=128,
            temperature=0.2,
        ),
    )
    return response.text.strip()


# ---------------------------------------------------------------------------
# Main generation functions
# ---------------------------------------------------------------------------
def generate_answer(
    question: str,
    context_chunks: List[Dict],
    conversation_history: List[Dict] | None = None,
) -> str:
    """Generate a grounded answer. Optionally includes conversation context."""
    # Handle greetings without hitting the API
    if _is_greeting(question):
        return _GREETING_RESPONSE

    if not context_chunks:
        return (
            "I couldn't find any relevant documents to answer that question. "
            "Try uploading source material first, or rephrase your question."
        )

    user_message = _build_user_message(question, context_chunks, conversation_history)
    return _call_gemini(user_message)


def generate_answer_stream(
    question: str,
    context_chunks: List[Dict],
    conversation_history: List[Dict] | None = None,
) -> Generator[str, None, None]:
    """Yield answer tokens one at a time from the Gemini streaming API."""
    # Handle greetings without hitting the API
    if _is_greeting(question):
        yield _GREETING_RESPONSE
        return

    if not context_chunks:
        yield (
            "I couldn't find any relevant documents to answer that question. "
            "Try uploading source material first, or rephrase your question."
        )
        return

    user_message = _build_user_message(question, context_chunks, conversation_history)
    yield from _call_gemini_stream(user_message)


# ---------------------------------------------------------------------------
# Gemini API wrappers
# ---------------------------------------------------------------------------
def _call_gemini(user_message: str) -> str:
    from google.genai import types

    client = _get_llm_client()
    try:
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=2048,
                temperature=settings.LLM_TEMPERATURE,
                top_p=settings.LLM_TOP_P,
            ),
        )
        return response.text
    except Exception as exc:
        error_msg = str(exc)
        if "404" in error_msg or "is no longer available" in error_msg:
            raise RuntimeError(
                f"Gemini model '{settings.GEMINI_MODEL}' is no longer available. "
                "Update the GEMINI_MODEL environment variable to a supported model "
                "(e.g. 'gemini-2.5-flash' or 'gemini-2.5-pro')."
            ) from exc
        raise


def _call_gemini_stream(user_message: str) -> Generator[str, None, None]:
    from google.genai import types

    client = _get_llm_client()
    try:
        for chunk in client.models.generate_content_stream(
            model=settings.GEMINI_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=2048,
                temperature=settings.LLM_TEMPERATURE,
                top_p=settings.LLM_TOP_P,
            ),
        ):
            if chunk.text:
                yield chunk.text
    except Exception as exc:
        error_msg = str(exc)
        if "404" in error_msg or "is no longer available" in error_msg:
            raise RuntimeError(
                f"Gemini model '{settings.GEMINI_MODEL}' is no longer available. "
                "Update the GEMINI_MODEL environment variable to a supported model "
                "(e.g. 'gemini-2.5-flash' or 'gemini-2.5-pro')."
            ) from exc
        raise
