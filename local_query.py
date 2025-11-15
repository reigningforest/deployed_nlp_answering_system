"""Simple CLI for querying the local RAG service without FastAPI."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.rag.service import QAService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the local RAG service")
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask. If omitted, you'll be prompted interactively.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to configuration YAML (defaults to config/config.yaml).",
    )
    return parser.parse_args()


def _sanitize_question(value: str) -> str:
    """Trim leading prompt markers like '?' before sending downstream."""
    cleaned = value.lstrip(" ?!.,:;-\t\r\n")
    return cleaned or value


def main() -> int:
    args = parse_args()

    try:
        if args.config:
            os.environ["QA_CONFIG_PATH"] = str(args.config)

        service = QAService()
    except Exception as exc:
        print(f"Failed to initialise QA service: {exc}", file=sys.stderr)
        return 1

    print(f"Answering using Groq model '{service.groq_model}'.")

    if args.question:
        question = _sanitize_question(args.question)
        print(service.get_answer(question))
        return 0

    print("Enter questions (empty line to quit).")
    while True:
        try:
            question = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not question:
            return 0

        try:
            normalized_question = _sanitize_question(question)
            answer = service.get_answer(normalized_question)
        except Exception as exc:
            print(f"Error: {exc}")
        else:
            print(answer)


if __name__ == "__main__":
    sys.exit(main())
