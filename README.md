# slide-rag (baseline)

End-to-end local RAG system over real-world PowerPoint presentations
(`Forceless/Zenodo10K` — ~10,448 `.pptx` files crawled from Zenodo).

This is the **baseline** configuration for the slide-deduplication research:
**1 slide = 1 chunk**, no deduplication or merging applied. Every slide
becomes one Qdrant point carrying two named vectors (late fusion):

- `"text"`  — 768-dim, `nomic-embed-text-v1.5` via Ollama
- `"image"` — 512-dim, CLIP ViT-B/32 (local inference)

This baseline exists to measure **index size** and **ingestion time** as a
reference point. The dedup pipeline (Step 1–5: lexical gate → semantic/
visual gate → containment check → extractive merge → index) will later be
run on the same dataset/cap, and compared against these numbers.

All components run fully offline except for the initial dataset/file
download (Hugging Face + Zenodo).

## Stack

| Layer | Tool |
|---|---|
| Data source | `Forceless/Zenodo10K` (HF dataset, metadata + Zenodo download links) |
| Parsing | `python-pptx` (text) + LibreOffice → PDF → PyMuPDF (per-slide PNG render) |
| Chunking | 1 slide = 1 chunk (no further splitting) |
| Text embedding | `nomic-embed-text-v1.5` via Ollama (768-dim) |
| Image embedding | CLIP ViT-B/32, local inference (512-dim) |
| Vector store | Qdrant v1.18.0 (local Docker, persistent), named vectors `text`+`image` |
| Retrieval | Qdrant cosine ANN on `"text"`, top-k=10 |
| LLM | Mistral 7B via Ollama |
| Orchestration | Python, no framework lock-in |

## Requirements

- Docker (for Qdrant)
- [Ollama](https://ollama.com) installed and running
- **LibreOffice** installed and on `PATH` (`libreoffice` / `soffice`) — used to
  rasterize each `.pptx` deck to a per-slide PNG for CLIP embedding
- Python 3.10+
- `qdrant-client==1.18.0` (must match the Docker image version)
- A working `torch` install (CPU is fine for small capped runs; CUDA
  recommended for larger runs — CLIP is sized for a 4GB-VRAM GPU)
- A HuggingFace account and read token (free) for dataset streaming

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# Install torch separately if you need a specific backend, e.g. CPU-only:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Set your HuggingFace token (avoids rate-limit warnings)
export HF_TOKEN=hf_your_token_here
# Get a free Read token at https://huggingface.co/settings/tokens

# 3. Create Qdrant data directories with correct ownership
mkdir -p ./data/qdrant_storage ./data/qdrant_snapshots
sudo chown -R 1000:1000 ./data/qdrant_storage ./data/qdrant_snapshots

# 4. Start Qdrant
docker compose up -d
curl http://localhost:6333/   # should return version JSON

# 5. Pull Ollama models
curl -fsSL https://ollama.com/install.sh | sh # install first
ollama pull nomic-embed-text
ollama pull mistral

# 6. Ingest decks (capped run — see configs/settings.py MAX_DECKS)
python scripts/ingest.py --max-decks 30      # small baseline run
python scripts/ingest.py --max-decks 30 --recreate

# 7. Query
python scripts/query.py --question "What was the revenue in Q2?"

# 8. Interactive CLI
python scripts/chat.py
```

## Project layout

```
slide-rag/
├── configs/
│   └── settings.py          # all tuneable parameters in one place
├── src/
│   ├── ingestion/
│   │   ├── loader.py        # streams Zenodo10K metadata + downloads .pptx
│   │   ├── chunker.py        # 1 slide = 1 chunk: text + rendered PNG
│   │   └── embedder.py       # nomic (text) + CLIP (image) embeddings
│   ├── retrieval/
│   │   └── retriever.py      # Qdrant ANN search on "text" named vector
│   ├── generation/
│   │   └── generator.py      # Ollama Mistral + prompt construction
│   └── utils/
│       ├── qdrant_client.py  # collection w/ named vectors text+image
│       └── logger.py         # structured logging
├── scripts/
│   ├── ingest.py             # CLI: run full ingestion pipeline, report size/time
│   ├── query.py              # CLI: single question
│   └── chat.py               # CLI: interactive loop
├── data/
│   ├── raw_pptx/             # downloaded .pptx decks (gitignored)
│   ├── rendered_slides/      # per-slide PNG renders, one subdir per deck (gitignored)
│   ├── qdrant_storage/        # Qdrant vector index (persistent)
│   └── qdrant_snapshots/      # Qdrant snapshots and temp files
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Configuration

All parameters live in `configs/settings.py`. Key knobs:

| Parameter | Default | Notes |
|---|---|---|
| `MAX_DECKS` | `30` | None = all ~10,448 decks; set int for capped runs |
| `MAX_PPTX_SIZE_MB` | `30` | skip decks larger than this (some Zenodo files embed video) |
| `RENDER_DPI` | `100` | resolution for per-slide PNG rasterization |
| `CLIP_MODEL_NAME` / `CLIP_PRETRAINED` | `ViT-B-32` / `openai` | sized for 4GB VRAM |
| `TOP_K` | `10` | chunks retrieved per query |
| `COLLECTION_NAME` | `slides_baseline` | Qdrant collection name |
| `EMBED_MODEL` | `nomic-embed-text` | must match ingestion model |
| `LLM_MODEL` | `mistral` | Ollama model tag |

## Measuring storage and ingestion time

`scripts/ingest.py` prints a full report at the end of every run:

- total slide-chunks (Qdrant points) and how many had a successful image render
- total wall-clock time, and the embed+upsert sub-stage time
- average time per slide-chunk
- Qdrant `collection_stats()` — `points_count`, `disk_data_size_bytes`
- on-disk size of `data/raw_pptx/` (downloaded decks) and
  `data/rendered_slides/` (PDF + PNG renders — scratch data, not part of
  the vector index but useful context for total pipeline footprint)

These numbers, captured for a fixed `--max-decks` cap, are the baseline
against which the dedup pipeline's output (fewer points after merging,
different disk size, different ingest time due to the extra gates) will
be compared.

```python
from src.utils.qdrant_client import get_client

client = get_client()
info = client.get_collection("slides_baseline")
print(info.points_count)
print(info.disk_data_size)
```

## Resetting the vector store

```bash
docker compose down
rm -rf ./data/qdrant_storage ./data/qdrant_snapshots
mkdir -p ./data/qdrant_storage ./data/qdrant_snapshots
docker compose up -d
python scripts/ingest.py --recreate --max-decks 30
```

To also re-download/re-render decks from scratch:

```bash
rm -rf ./data/raw_pptx ./data/rendered_slides
```

## Notes on the dataset

`Forceless/Zenodo10K` provides metadata (filename, size, license, title,
Zenodo download URL) — the actual `.pptx` bytes are fetched on demand from
Zenodo on first ingest and cached under `data/raw_pptx/`. Files above
`MAX_PPTX_SIZE_MB` are skipped to keep capped runs predictable; this is
logged at the end of the run.

## Version compatibility

The `qdrant-client` Python package and the Qdrant Docker image **must stay
on the same minor version**.

| Component | Pinned version |
|---|---|
| Docker image | `qdrant/qdrant:v1.18.0` |
| Python client | `qdrant-client>=1.18.0` |
