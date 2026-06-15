"""
Retrieves the top-k most relevant slide-chunks from Qdrant for a query.

Steps:
  1. Embed the query string via Ollama (same model as ingestion's "text"
     named-vector).
  2. Run cosine ANN search against the "text" named-vector of the
     slides_baseline collection.
  3. Return ranked list of chunk dicts with scores.

The collection stores TWO named vectors per point ("text" and "image").
The baseline retrieval path searches "text" only — image-vector search
is exposed via search_by_image() for use by the dedup pipeline's
semantic+visual gate (Step 2), not by the standard query/chat scripts.
"""

from __future__ import annotations

import logging
import time

from src.ingestion.embedder import embed_texts
from src.utils.qdrant_client import get_client
from configs.settings import COLLECTION_NAME, TEXT_VECTOR_NAME, TOP_K

logger = logging.getLogger(__name__)


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embed the query and return the top-k matching slide-chunks.

    Returns a list of dicts:
      {
        "chunk_id":    str,
        "deck_id":     str,
        "deck_title":  str,
        "slide_index": int,
        "text":        str,
        "has_image":   bool,
        "score":       float,   # cosine similarity in [0, 1]
        "latency_ms":  float,   # only on the first item
      }
    """
    t0 = time.perf_counter()

    vectors = embed_texts([query])
    query_vector = vectors[0]

    response = get_client().query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using=TEXT_VECTOR_NAME,
        limit=top_k,
        with_payload=True,
    )
    results = response.points

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info("Retrieved %d chunks in %.1f ms", len(results), latency_ms)

    chunks = []
    for i, hit in enumerate(results):
        payload = hit.payload or {}
        chunk = {
            "chunk_id": payload.get("chunk_id", ""),
            "deck_id": payload.get("deck_id", ""),
            "deck_title": payload.get("deck_title", ""),
            "slide_index": payload.get("slide_index", -1),
            "text": payload.get("text", ""),
            "has_image": payload.get("has_image", False),
            "score": round(hit.score, 4),
        }
        if i == 0:
            chunk["latency_ms"] = round(latency_ms, 2)
        chunks.append(chunk)

    return chunks