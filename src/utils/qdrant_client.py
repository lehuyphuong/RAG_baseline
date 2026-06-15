"""
Shared Qdrant connection and collection management.

The baseline collection uses LATE FUSION named vectors (RQ4):
  - "text"  : 768-dim, cosine distance  (nomic-embed-text-v1.5)
  - "image" : 512-dim, cosine distance  (CLIP ViT-B/32)

Storing both modalities as separate named vectors on the same point lets
each be queried/thresholded independently — required for the dedup
pipeline's Step 1/2 gates, and supported natively by Qdrant without any
custom infrastructure.

Import get_client() wherever a QdrantClient is needed.
Call ensure_collection() once at ingestion startup.
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    HnswConfigDiff,
    VectorParams,
)

from configs.settings import (
    COLLECTION_NAME,
    IMAGE_EMBED_DIM,
    IMAGE_VECTOR_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    TEXT_EMBED_DIM,
    TEXT_VECTOR_NAME,
)

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Return a module-level singleton QdrantClient."""
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        logger.debug("Qdrant client connected to %s:%s", QDRANT_HOST, QDRANT_PORT)
    return _client


def ensure_collection(recreate: bool = False) -> None:
    """
    Create the Qdrant collection (with named text+image vectors) if it
    does not already exist.

    Args:
        recreate: If True, drop and recreate the collection.
                  Use this to start a fresh ingestion.
    """
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        if recreate:
            logger.warning("Dropping existing collection '%s'", COLLECTION_NAME)
            client.delete_collection(COLLECTION_NAME)
        else:
            logger.info(
                "Collection '%s' already exists — skipping creation.",
                COLLECTION_NAME,
            )
            return

    logger.info(
        "Creating collection '%s' (text=%d-dim, image=%d-dim, distance=Cosine)",
        COLLECTION_NAME,
        TEXT_EMBED_DIM,
        IMAGE_EMBED_DIM,
    )
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            TEXT_VECTOR_NAME: VectorParams(
                size=TEXT_EMBED_DIM,
                distance=Distance.COSINE,
            ),
            IMAGE_VECTOR_NAME: VectorParams(
                size=IMAGE_EMBED_DIM,
                distance=Distance.COSINE,
            ),
        },
        hnsw_config=HnswConfigDiff(
            m=16,             # HNSW connectivity; higher = better recall, more RAM
            ef_construct=100, # build-time search depth; higher = better index quality
        ),
    )
    logger.info("Collection created.")


def collection_stats() -> dict:
    """Return a snapshot of storage and vector counts.

    Note: older qdrant-client versions exposed a top-level
    `vectors_count` field on CollectionInfo; this was removed in
    qdrant-client 1.18 (use `points_count` * number of named vectors
    if a total vector count is needed).
    """
    client = get_client()
    info = client.get_collection(COLLECTION_NAME)
    return {
        "points_count": info.points_count,
        "indexed_vectors_count": info.indexed_vectors_count,
        "disk_data_size_bytes": getattr(info, "disk_data_size", None),
        "status": str(info.status),
    }