"""
Interactive REPL for the slide-rag system.

Usage:
    python scripts/chat.py
    python scripts/chat.py --top-k 5

Commands at the prompt:
    /quit or /exit   — exit
    /stats           — print collection stats
    /chunks          — toggle showing retrieved chunks
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging

from src.generation.generator import generate
from src.retrieval.retriever import retrieve
from src.utils.logger import configure_logging
from src.utils.qdrant_client import collection_stats

logger = logging.getLogger(__name__)


def chat_loop(top_k: int) -> None:
    show_chunks = True

    print("\nslide-rag  —  type your question, /stats, /chunks, or /quit\n")

    stats = collection_stats()
    print(f"  Collection: {stats.get('points_count', '?')} points  "
          f"| status: {stats.get('status', '?')}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit"):
            print("Bye.")
            break

        if user_input.lower() == "/stats":
            for k, v in collection_stats().items():
                print(f"  {k}: {v}")
            print()
            continue

        if user_input.lower() == "/chunks":
            show_chunks = not show_chunks
            print(f"  Chunk display: {'on' if show_chunks else 'off'}\n")
            continue

        chunks = retrieve(user_input, top_k=top_k)

        if show_chunks:
            print(f"\n  Retrieved {len(chunks)} chunks:")
            for i, c in enumerate(chunks, 1):
                latency = c.get("latency_ms")
                lat_str = f" [{latency:.0f} ms]" if latency else ""
                label = f"{c['deck_title']} — slide {c['slide_index']}"
                print(f"  [{i}] {c['score']:.3f}{lat_str} — {label}")
                print(f"      {c['text'][:100].strip()}…")
            print()

        result = generate(question=user_input, chunks=chunks)
        print(f"Assistant: {result['answer']}\n")


def main() -> None:
    configure_logging(level="WARNING")  # quieter in interactive mode

    parser = argparse.ArgumentParser(description="Interactive slide-rag chat.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of chunks to retrieve per query.",
    )
    args = parser.parse_args()

    from configs.settings import TOP_K
    top_k = args.top_k if args.top_k is not None else TOP_K

    chat_loop(top_k=top_k)


if __name__ == "__main__":
    main()