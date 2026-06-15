"""
Streams .pptx presentation files from the Forceless/Zenodo10K dataset
(https://huggingface.co/datasets/Forceless/Zenodo10K).

The HF dataset only contains METADATA (filename, size, download url,
license, title, ...) — the actual .pptx bytes live on Zenodo and are
fetched on demand via the "url" column.

Each yielded item is a dict:
  {
    "deck_id":  str,   # stable id derived from the Zenodo DOI/checksum
    "title":    str,
    "filename": str,
    "path":     Path,  # local path to the downloaded .pptx
    "size_bytes": int,
  }

Files larger than MAX_PPTX_SIZE_MB are skipped (configurable) to keep
capped runs fast and predictable — some Zenodo decks embed large
video/image assets and are not representative for a baseline run.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Generator

import httpx
from datasets import load_dataset

from configs.settings import (
    DATASET_NAME,
    DATASET_SPLIT,
    MAX_DECKS,
    MAX_PPTX_SIZE_MB,
    RAW_PPTX_DIR,
)

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 120.0
_MAX_BYTES = MAX_PPTX_SIZE_MB * 1024 * 1024


def _safe_filename(name: str, deck_id: str) -> str:
    """Sanitize a Zenodo filename so it is safe to use on the local filesystem."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    if not stem.lower().endswith(".pptx"):
        stem += ".pptx"
    # Prefix with a short deck id to avoid collisions between decks that
    # happen to share the same uploaded filename.
    return f"{deck_id}_{stem}"[:200]


def stream_decks(
    max_decks: int | None = MAX_DECKS,
) -> Generator[dict, None, None]:
    """
    Yields downloaded .pptx decks one at a time.

    Args:
        max_decks: Cap the stream at this many decks.
                   None means the full dataset (~10,448 files).
    """
    logger.info(
        "Opening Zenodo10K stream: dataset=%s split=%s max=%s max_size=%sMB",
        DATASET_NAME,
        DATASET_SPLIT,
        max_decks or "unlimited",
        MAX_PPTX_SIZE_MB,
    )

    dataset = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
        streaming=True,
    )

    count = 0
    skipped_size = 0
    skipped_download = 0

    with httpx.Client(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
        for row in dataset:
            size_bytes = row.get("size", 0) or 0
            if size_bytes > _MAX_BYTES:
                skipped_size += 1
                continue

            checksum = (row.get("checksum") or "").replace(":", "_")
            deck_id = checksum[:16] if checksum else f"deck{count:06d}"
            filename = row.get("filename", f"{deck_id}.pptx")
            local_name = _safe_filename(filename, deck_id)
            local_path = RAW_PPTX_DIR / local_name

            if not local_path.exists():
                url = row.get("url")
                try:
                    resp = client.get(url)
                    resp.raise_for_status()
                    local_path.write_bytes(resp.content)
                except Exception as exc:  # noqa: BLE001 — log and skip bad files
                    logger.warning("Download failed for %s: %s", filename, exc)
                    skipped_download += 1
                    continue

            yield {
                "deck_id": deck_id,
                "title": row.get("title", filename),
                "filename": filename,
                "path": local_path,
                "size_bytes": local_path.stat().st_size,
            }

            count += 1
            if count % 10 == 0:
                logger.info(
                    "Downloaded %d decks so far (skipped: %d too-large, %d failed)",
                    count,
                    skipped_size,
                    skipped_download,
                )

            if max_decks is not None and count >= max_decks:
                logger.info("Reached max_decks cap (%d)", max_decks)
                break

    logger.info(
        "Stream finished: %d decks downloaded "
        "(skipped: %d too-large, %d download failures)",
        count,
        skipped_size,
        skipped_download,
    )
