"""
Embeds slide chunks with TWO independent vectors (late fusion):

  - "text"  : nomic-embed-text-v1.5 via Ollama  (768-dim)
  - "image" : CLIP ViT-B/32 (local inference)    (512-dim)

This matches the late-fusion / named-vector representation recommended
for the dedup pipeline (RQ4): Qdrant stores both vectors per point under
named-vector keys, so each modality can be queried/thresholded independently.

embed_texts() handles the text side via the Ollama HTTP API (same pattern
as the original wiki-rag embedder). embed_images() lazily loads CLIP on
first use and embeds a batch of PIL images.

If image rendering failed for a slide (image_path is None) or CLIP is
unavailable, a zero-vector is used and the chunk's payload is flagged with
"has_image": false so downstream analysis can account for it.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Generator

import httpx

from configs.settings import (
    CLIP_DEVICE,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    EMBED_BATCH_SIZE,
    EMBED_MODEL,
    IMAGE_EMBED_DIM,
    OLLAMA_BASE_URL,
)

logger = logging.getLogger(__name__)

_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embed"
_HTTP_TIMEOUT = 120.0  # seconds; large batches can be slow on CPU

# Text embedding (Ollama / nomic-embed-text-v1.5)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings via Ollama. Returns one 768-dim vector per string.

    Empty strings are sent as a single space (" ") so Ollama still returns
    a valid vector — slides with no extractable text are not dropped, they
    are embedded as "near-empty" text, which is itself a useful signal
    (e.g. an image-only slide).
    """
    if not texts:
        return []

    safe_texts = [t if t.strip() else " " for t in texts]
    payload = {"model": EMBED_MODEL, "input": safe_texts}

    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        response = client.post(_EMBED_URL, json=payload)
        response.raise_for_status()

    data = response.json()
    embeddings = data.get("embeddings")
    if embeddings is None or len(embeddings) != len(texts):
        raise ValueError(
            f"Unexpected Ollama response: expected {len(texts)} embeddings, "
            f"got {len(embeddings) if embeddings else 'None'}"
        )

    return embeddings


# Image embedding (CLIP, local inference)

_clip_model = None
_clip_preprocess = None
_clip_device = None


def _load_clip():
    """Lazily load CLIP on first use. Falls back to CPU if CUDA unavailable."""
    global _clip_model, _clip_preprocess, _clip_device

    if _clip_model is not None:
        return _clip_model, _clip_preprocess, _clip_device

    import torch
    import open_clip

    device = CLIP_DEVICE if torch.cuda.is_available() else "cpu"
    if device != CLIP_DEVICE:
        logger.warning(
            "CUDA not available — CLIP will run on CPU (slower). "
            "Set CLIP_DEVICE='cpu' in settings to silence this warning."
        )

    logger.info(
        "Loading CLIP model=%s pretrained=%s device=%s",
        CLIP_MODEL_NAME,
        CLIP_PRETRAINED,
        device,
    )
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    model = model.to(device).eval()

    _clip_model, _clip_preprocess, _clip_device = model, preprocess, device
    return _clip_model, _clip_preprocess, _clip_device


def embed_images(image_paths: list[Path | None]) -> list[list[float]]:
    """
    Embed a batch of slide-render PNGs via CLIP.

    Entries that are None (render failed) produce a zero-vector of
    dimension IMAGE_EMBED_DIM — these are flagged via payload["has_image"]
    by the caller, not silently dropped, so the point still gets indexed.
    """
    if not image_paths:
        return []

    valid_indices = [i for i, p in enumerate(image_paths) if p is not None]
    results: list[list[float]] = [[0.0] * IMAGE_EMBED_DIM for _ in image_paths]

    if not valid_indices:
        return results

    import torch
    from PIL import Image

    model, preprocess, device = _load_clip()

    batch_tensors = []
    for i in valid_indices:
        path = image_paths[i]
        try:
            img = Image.open(path).convert("RGB")
            batch_tensors.append(preprocess(img))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load image %s for CLIP: %s", path, exc)
            valid_indices.remove(i)

    if not batch_tensors:
        return results

    with torch.no_grad():
        batch = torch.stack(batch_tensors).to(device)
        features = model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
        features = features.cpu().numpy()

    for vec, idx in zip(features, valid_indices):
        results[idx] = vec.tolist()

    return results


# Combined batched embedding


def embed_chunks_batched(
    chunks: Generator[dict, None, None],
    batch_size: int = EMBED_BATCH_SIZE,
) -> Generator[tuple[dict, list[float], list[float], bool], None, None]:
    """
    Lazy generator that yields (chunk_dict, text_vector, image_vector, has_image).

    Accumulates chunks into batches, embeds text (Ollama) and image (CLIP)
    for each batch, then yields the tuples one at a time.
    """
    buffer: list[dict] = []
    total = 0
    t_start = time.perf_counter()

    for chunk in chunks:
        buffer.append(chunk)

        if len(buffer) >= batch_size:
            yield from _flush(buffer)
            total += len(buffer)
            buffer = []

            if total % 500 == 0:
                elapsed = time.perf_counter() - t_start
                rate = total / elapsed if elapsed > 0 else 0
                logger.info("Embedded %d slide-chunks (%.1f chunks/s)", total, rate)

    if buffer:
        yield from _flush(buffer)
        total += len(buffer)

    logger.info("Embedding complete: %d slide-chunks", total)


def _flush(
    batch: list[dict],
) -> Generator[tuple[dict, list[float], list[float], bool], None, None]:
    texts = [c["text"] for c in batch]
    image_paths = [c["image_path"] for c in batch]

    text_vectors = embed_texts(texts)
    image_vectors = embed_images(image_paths)

    for chunk, tvec, ivec, img_path in zip(batch, text_vectors, image_vectors, image_paths):
        yield chunk, tvec, ivec, img_path is not None
