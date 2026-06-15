"""
Central configuration for slide-rag.
All tuneable parameters live here — never scattered in source files.

This is the "baseline" configuration: 1 slide = 1 chunk, with BOTH a text
named-vector (nomic-embed-text-v1.5) and an image named-vector (CLIP),
stored late-fusion style in Qdrant.
"""

from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

RAW_PPTX_DIR = DATA_DIR / "raw_pptx"          # downloaded .pptx files
RENDER_DIR = DATA_DIR / "rendered_slides"     # per-slide PNG renders (scratch)
RAW_PPTX_DIR.mkdir(exist_ok=True)
RENDER_DIR.mkdir(exist_ok=True)

# Dataset
# PPTAgent/Zenodo10K: ~10,448 real-world .pptx files crawled from Zenodo,
# CC-BY family licenses. https://huggingface.co/datasets/Forceless/Zenodo10K
DATASET_NAME = "Forceless/Zenodo10K"
DATASET_SPLIT = "pptx"

# None = all 10,448 files. Set an int for a capped run, e.g. 30.
MAX_DECKS: int | None = 30

# Skip files larger than this to keep a capped run fast and predictable
# (some Zenodo decks are >100MB with embedded video/large images).
MAX_PPTX_SIZE_MB = 30

# Rendering (pptx -> per-slide PNG via LibreOffice + PyMuPDF)
LIBREOFFICE_BIN = "libreoffice"     # must be on PATH
RENDER_DPI = 100                    # resolution for rasterizing slide PDF pages
LIBREOFFICE_TIMEOUT_S = 180         # per-deck conversion timeout

# Chunking
# Baseline chunking strategy: 1 slide = 1 chunk (no RecursiveCharacter split).
# Each chunk carries both its extracted text and its rendered slide image.
MAX_TEXT_CHARS_PER_SLIDE = 2000   # safety cap; very text-heavy slides are truncated

# Embedding: text (nomic-embed-text-v1.5 via Ollama)
EMBED_MODEL = "nomic-embed-text"   # Ollama model tag
TEXT_EMBED_DIM = 768               # nomic-embed-text-v1.5 output dimension
OLLAMA_BASE_URL = "http://localhost:11434"

# Embedding: image (CLIP, local inference)
# ViT-B/32 chosen for the 4GB VRAM constraint (RTX 1050).
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
IMAGE_EMBED_DIM = 512              # CLIP ViT-B/32 output dimension
CLIP_DEVICE = "cuda"               # falls back to "cpu" automatically if unavailable

# Slides with no extractable text still get embedded (empty-string -> nomic
# still returns a vector); slides that fail to render fall back to a
# zero-vector for the "image" field (flagged via payload "has_image": false).

# Vector store
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "slides_baseline"

TEXT_VECTOR_NAME = "text"
IMAGE_VECTOR_NAME = "image"

# Batch sizes — tune down if RAM is tight
EMBED_BATCH_SIZE = 32     # slides sent to Ollama / CLIP per batch
UPSERT_BATCH_SIZE = 128   # points upserted to Qdrant per batch

# Retrieval
TOP_K = 10                # number of chunks returned per query

# Generation
LLM_MODEL = "mistral"     # Ollama model tag; swap for "llama3.1" etc.
LLM_TEMPERATURE = 0.0     # deterministic for reproducibility
LLM_MAX_TOKENS = 512

SYSTEM_PROMPT = """\
You are a factual assistant. Answer the user's question using ONLY the \
context passages provided. Each passage corresponds to one slide from a \
presentation. If the answer is not contained in the context, \
say "I don't have enough information to answer that." Do not speculate.\
"""
