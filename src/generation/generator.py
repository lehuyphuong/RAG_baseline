"""
Generates a grounded answer from retrieved chunks using Ollama Mistral.

The prompt is structured as:
  [System]  You are a factual assistant …
  [Context] Passage 1: …
            Passage 2: …
  [User]    <question>

The LLM is instructed to answer only from the provided context.
"""

from __future__ import annotations

import logging

import httpx

from configs.settings import (
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
_HTTP_TIMEOUT = 180.0


def build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block.

    Each chunk corresponds to one slide; the label shows which deck and
    slide index the passage came from, e.g. "(My Presentation — slide 4)".
    """
    lines = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("deck_title", "Unknown")
        slide_index = chunk.get("slide_index", "?")
        text = chunk.get("text", "").strip()
        lines.append(f"[{i}] ({title} — slide {slide_index})\n{text}")
    return "\n\n".join(lines)


def generate(question: str, chunks: list[dict]) -> dict:
    """
    Call the LLM and return a dict:
      {
        "answer":      str,
        "model":       str,
        "prompt_tokens":    int,
        "completion_tokens": int,
      }

    Args:
        question: The user's raw question string.
        chunks:   Retrieved chunk dicts from retriever.retrieve().
    """
    context = build_context_block(chunks)
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    payload = {
        "model": LLM_MODEL,
        "options": {
            "temperature": LLM_TEMPERATURE,
            "num_predict": LLM_MAX_TOKENS,
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
    }

    logger.debug("Sending request to Ollama (model=%s)", LLM_MODEL)

    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        response = client.post(_CHAT_URL, json=payload)
        response.raise_for_status()

    data = response.json()
    answer = data.get("message", {}).get("content", "").strip()
    usage = data.get("usage", {})

    logger.info(
        "Generated answer (%d chars, prompt_tokens=%s)",
        len(answer),
        usage.get("prompt_tokens", "?"),
    )

    return {
        "answer": answer,
        "model": LLM_MODEL,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }
