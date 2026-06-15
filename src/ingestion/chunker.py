"""
Splits .pptx decks into one chunk per slide.

Each chunk carries:
  - extracted text (via python-pptx, all text-frame runs concatenated)
  - a rendered PNG of the slide (via LibreOffice -> PDF -> PyMuPDF rasterize)

This is the BASELINE chunking strategy for the slide-rag system:
1 slide = 1 chunk, no further splitting, no deduplication / merging.

Each yielded chunk is a dict:
  {
    "chunk_id":    str,   # "{deck_id}_{slide_index}"
    "deck_id":     str,
    "deck_title":  str,
    "slide_index": int,
    "text":        str,
    "image_path":  Path | None,   # rendered PNG, or None if render failed
  }

Rendering is done once per deck (convert whole deck -> PDF -> N PNGs),
not once per slide, to avoid spawning LibreOffice N times.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF
from pptx import Presentation

from configs.settings import (
    LIBREOFFICE_BIN,
    LIBREOFFICE_TIMEOUT_S,
    MAX_TEXT_CHARS_PER_SLIDE,
    RENDER_DIR,
    RENDER_DPI,
)

logger = logging.getLogger(__name__)


def _extract_slide_text(slide) -> str:
    """Concatenate all text runs from all shapes on a slide."""
    lines: list[str] = []
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        for para in shape.text_frame.paragraphs:
            text = "".join(run.text for run in para.runs)
            text = text.strip()
            if text:
                lines.append(text)

        # Tables: python-pptx text frames don't cover table cells
        if shape.has_table:
            for row in shape.table.rows:
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    if cell_text:
                        lines.append(cell_text)

    full_text = "\n".join(lines)
    if len(full_text) > MAX_TEXT_CHARS_PER_SLIDE:
        full_text = full_text[:MAX_TEXT_CHARS_PER_SLIDE]
    return full_text


def _render_deck_to_pngs(pptx_path: Path, deck_id: str) -> list[Path | None]:
    """
    Render every slide of a deck to a separate PNG.

    Pipeline: pptx --(LibreOffice)--> pdf --(PyMuPDF)--> png per page.

    Returns a list of paths (or None for pages that failed to rasterize),
    in slide order. Returns an empty list if the LibreOffice conversion
    itself fails (e.g. corrupted file).
    """
    deck_render_dir = RENDER_DIR / deck_id
    deck_render_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = deck_render_dir / f"{pptx_path.stem}.pdf"
    if not pdf_path.exists():
        try:
            subprocess.run(
                [
                    LIBREOFFICE_BIN,
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    str(deck_render_dir),
                    str(pptx_path),
                ],
                check=True,
                capture_output=True,
                timeout=LIBREOFFICE_TIMEOUT_S,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            logger.warning("LibreOffice conversion failed for %s: %s", pptx_path.name, exc)
            return []

    if not pdf_path.exists():
        logger.warning("Expected PDF not found after conversion: %s", pdf_path)
        return []

    image_paths: list[Path | None] = []
    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            png_path = deck_render_dir / f"slide_{i}.png"
            if not png_path.exists():
                pix = page.get_pixmap(dpi=RENDER_DPI)
                pix.save(str(png_path))
            image_paths.append(png_path)
        doc.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("PDF rasterization failed for %s: %s", pdf_path.name, exc)
        return []

    return image_paths


def chunk_deck(deck: dict) -> list[dict]:
    """
    Split one deck into per-slide chunks. Returns a list of chunk dicts.

    Args:
        deck: dict from loader.stream_decks(), must contain
              "deck_id", "title", "path".
    """
    pptx_path: Path = deck["path"]
    deck_id = deck["deck_id"]

    try:
        prs = Presentation(str(pptx_path))
    except Exception as exc:  # noqa: BLE001 — corrupted/unsupported file
        logger.warning("Failed to open %s with python-pptx: %s", pptx_path.name, exc)
        return []

    slides = list(prs.slides)
    image_paths = _render_deck_to_pngs(pptx_path, deck_id)

    chunks = []
    for i, slide in enumerate(slides):
        text = _extract_slide_text(slide)
        img = image_paths[i] if i < len(image_paths) else None

        chunks.append(
            {
                "chunk_id": f"{deck_id}_{i}",
                "deck_id": deck_id,
                "deck_title": deck.get("title", pptx_path.stem),
                "slide_index": i,
                "text": text,
                "image_path": img,
            }
        )

    return chunks


def chunk_decks(
    decks: Generator[dict, None, None],
) -> Generator[dict, None, None]:
    """
    Lazy generator: consumes a deck stream and yields per-slide chunk dicts.
    """
    deck_count = 0
    chunk_count = 0
    empty_text_count = 0
    failed_render_count = 0

    for deck in decks:
        chunks = chunk_deck(deck)
        for chunk in chunks:
            if not chunk["text"]:
                empty_text_count += 1
            if chunk["image_path"] is None:
                failed_render_count += 1
            yield chunk
            chunk_count += 1

        deck_count += 1
        if deck_count % 5 == 0:
            logger.info(
                "Chunked %d decks -> %d slide-chunks so far "
                "(%d empty-text, %d failed-render)",
                deck_count,
                chunk_count,
                empty_text_count,
                failed_render_count,
            )

    logger.info(
        "Chunking complete: %d decks -> %d slide-chunks "
        "(%d empty-text, %d failed-render)",
        deck_count,
        chunk_count,
        empty_text_count,
        failed_render_count,
    )
