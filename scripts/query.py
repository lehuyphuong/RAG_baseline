"""
Ask a single question against the slide-rag system.

Usage:
    python scripts/query.py --question "What was the revenue in Q2?"
    python scripts/query.py -q "Who is on the engineering team?" --top-k 5
    python scripts/query.py -q "..." --no-answer   # retrieval only
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

from src.generation.generator import generate
from src.retrieval.retriever import retrieve
from src.utils.logger import configure_logging

logger = logging.getLogger(__name__)


def run_query(question: str, top_k: int, no_answer: bool) -> None:
    print(f"\nQuestion: {question}\n")
    print("─" * 60)

    chunks = retrieve(question, top_k=top_k)

    print(f"Retrieved {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        latency = chunk.get("latency_ms")
        latency_str = f"  [{latency:.1f} ms]" if latency else ""
        label = f"{chunk['deck_title']} — slide {chunk['slide_index']}"
        print(f"  [{i}] score={chunk['score']:.4f}{latency_str}  — {label}")
        print(f"      {chunk['text'][:120].strip()}…\n")

    if no_answer:
        return

    print("─" * 60)
    result = generate(question=question, chunks=chunks)
    print(f"\nAnswer ({result['model']}):\n")
    print(result["answer"])
    print()


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description="Query the slide-rag system.")
    parser.add_argument(
        "--question", "-q",
        required=True,
        help="The question to ask.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of chunks to retrieve (default: settings.TOP_K).",
    )
    parser.add_argument(
        "--no-answer",
        action="store_true",
        help="Only retrieve; skip LLM generation.",
    )
    args = parser.parse_args()

    from configs.settings import TOP_K
    top_k = args.top_k if args.top_k is not None else TOP_K

    run_query(
        question=args.question,
        top_k=top_k,
        no_answer=args.no_answer,
    )


if __name__ == "__main__":
    main()
