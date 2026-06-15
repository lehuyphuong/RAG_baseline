"""
Ingestion pipeline: Zenodo10K .pptx decks -> per-slide chunks ->
(text + image) embeddings -> Qdrant.

This is the BASELINE pipeline: 1 slide = 1 chunk, no deduplication or
merging. Every slide becomes one Qdrant point with two named vectors
("text" from nomic-embed-text-v1.5, "image" from CLIP ViT-B/32).

The script reports, at the end of the run:
  - total decks / slides ingested
  - wall-clock time for the whole run, and a breakdown by stage
    (download+render, embed, upsert)
  - Qdrant collection stats (points_count, disk_data_size_bytes)
  - disk usage of the raw .pptx files and rendered PNGs (for reference)

These numbers are the reference point for later comparing against the
dedup pipeline (Step 1-5): same dataset, same cap, same hardware ->
delta in vector count / disk size / ingest time directly attributable
to deduplication.

Usage:
    python scripts/ingest.py                     # MAX_DECKS from settings (default 30)
    python scripts/ingest.py --max-decks 50
    python scripts/ingest.py --recreate           # drop + rebuild collection
    python scripts/ingest.py --stats-only         # print collection stats and exit
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client.http.models import PointStruct

from configs.settings import (
    COLLECTION_NAME,
    IMAGE_VECTOR_NAME,
    RAW_PPTX_DIR,
    RENDER_DIR,
    TEXT_VECTOR_NAME,
    UPSERT_BATCH_SIZE,
)
from src.ingestion.chunker import chunk_decks
from src.ingestion.embedder import embed_chunks_batched
from src.ingestion.loader import stream_decks
from src.utils.logger import configure_logging
from src.utils.qdrant_client import collection_stats, ensure_collection, get_client

logger = logging.getLogger(__name__)


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def ingest(max_decks: int | None, recreate: bool) -> None:
    t_total_start = time.perf_counter()

    ensure_collection(recreate=recreate)
    client = get_client()

    decks = stream_decks(max_decks=max_decks)
    chunks = chunk_decks(decks)
    embedded = embed_chunks_batched(chunks)

    upsert_buffer: list[PointStruct] = []
    total_upserted = 0
    total_with_image = 0
    point_id = 0

    t_embed_upsert_start = time.perf_counter()

    for chunk, text_vector, image_vector, has_image in embedded:
        upsert_buffer.append(
            PointStruct(
                id=point_id,
                vector={
                    TEXT_VECTOR_NAME: text_vector,
                    IMAGE_VECTOR_NAME: image_vector,
                },
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "deck_id": chunk["deck_id"],
                    "deck_title": chunk["deck_title"],
                    "slide_index": chunk["slide_index"],
                    "text": chunk["text"],
                    "has_image": has_image,
                },
            )
        )
        point_id += 1
        if has_image:
            total_with_image += 1

        if len(upsert_buffer) >= UPSERT_BATCH_SIZE:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=upsert_buffer,
                wait=False,
            )
            total_upserted += len(upsert_buffer)
            upsert_buffer = []

            if total_upserted % 500 == 0:
                logger.info("Upserted %d points so far", total_upserted)

    if upsert_buffer:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=upsert_buffer,
            wait=True,
        )
        total_upserted += len(upsert_buffer)

    t_embed_upsert_end = time.perf_counter()
    t_total_end = time.perf_counter()

    # Report
    total_time_s = t_total_end - t_total_start
    embed_upsert_time_s = t_embed_upsert_end - t_embed_upsert_start

    stats = collection_stats()
    raw_pptx_size = _dir_size_bytes(RAW_PPTX_DIR)
    render_size = _dir_size_bytes(RENDER_DIR)

    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info("Total slide-chunks (points) upserted : %d", total_upserted)
    logger.info("  of which with rendered image        : %d", total_with_image)
    logger.info("Total wall-clock time                 : %.1f s", total_time_s)
    logger.info("  embed + upsert stage                : %.1f s", embed_upsert_time_s)
    if total_upserted > 0:
        logger.info(
            "  avg per slide-chunk                  : %.1f ms",
            (total_time_s / total_upserted) * 1000,
        )
    logger.info("-" * 60)
    logger.info("Qdrant collection stats (%s):", COLLECTION_NAME)
    for k, v in stats.items():
        logger.info("  %-24s: %s", k, v)
    if stats.get("disk_data_size_bytes") is not None:
        logger.info(
            "  %-24s: %s",
            "disk_data_size (human)",
            _human_bytes(stats["disk_data_size_bytes"]),
        )
    logger.info("-" * 60)
    logger.info("On-disk auxiliary data (not part of Qdrant index):")
    logger.info("  raw .pptx files (%s)      : %s", RAW_PPTX_DIR, _human_bytes(raw_pptx_size))
    logger.info("  rendered slide PNGs (%s)  : %s", RENDER_DIR, _human_bytes(render_size))
    logger.info("=" * 60)


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Ingest Zenodo10K .pptx decks into Qdrant (slide-rag baseline)."
    )
    parser.add_argument(
        "--max-decks",
        type=int,
        default=None,
        help="Cap the number of decks to ingest (default: settings.MAX_DECKS).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the Qdrant collection before ingesting.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print collection stats and exit without ingesting.",
    )
    args = parser.parse_args()

    if args.stats_only:
        stats = collection_stats()
        for k, v in stats.items():
            print(f"{k}: {v}")
        return

    from configs.settings import MAX_DECKS

    max_decks = args.max_decks if args.max_decks is not None else MAX_DECKS

    ingest(max_decks=max_decks, recreate=args.recreate)


if __name__ == "__main__":
    main()
