"""CLI wrapper for the RAG OpenAI utilities."""

from __future__ import annotations

import argparse
import sys

from src.features.rag import service


def build_parser() -> argparse.ArgumentParser:
    """Create the command parser for RAG OpenAI utilities."""

    parser = argparse.ArgumentParser(description="RAG batch utilities for OpenAI")
    sub = parser.add_subparsers(dest="command", required=True)

    test_cmd = sub.add_parser("test", help="Tester un fichier vec.json avec un modèle OpenAI")
    test_cmd.add_argument("file")
    test_cmd.add_argument("--model", default="gpt-4o-mini")
    test_cmd.set_defaults(handler=service.cmd_test)

    batch_cmd = sub.add_parser("batch", help="Générer des JSONL et créer des batchs")
    batch_cmd.add_argument("--in-dir", required=True)
    batch_cmd.add_argument("--out-root", required=True)
    batch_cmd.add_argument("--prompt", default="Nom du fichier: {REL_PATH}\nContenu:")
    batch_cmd.add_argument("--glob", default="**/*.vec.json")
    batch_cmd.add_argument("--model", default="gpt-4o-mini")
    batch_cmd.add_argument("--max-tokens-per-jsonl", type=int, default=None)
    batch_cmd.add_argument("--max-items-per-jsonl", type=int, default=None)
    batch_cmd.add_argument("--max-total-tokens", type=int, default=None)
    batch_cmd.add_argument("--max-items", type=int, default=None)
    batch_cmd.add_argument("--dry-run", action="store_true")
    batch_cmd.set_defaults(handler=service.cmd_batch)

    fetch_cmd = sub.add_parser("fetch", help="Récupérer les résultats des batchs")
    fetch_cmd.add_argument("--out-root", required=True)
    fetch_cmd.set_defaults(handler=service.cmd_fetch)

    fetch_local_cmd = sub.add_parser(
        "fetch-local",
        help="Appliquer des résultats batch JSONL déjà téléchargés",
    )
    fetch_local_cmd.add_argument("--in-dir", required=True)
    fetch_local_cmd.add_argument("--target-root", required=True)
    fetch_local_cmd.add_argument("--glob", default="**/*.jsonl")
    fetch_local_cmd.set_defaults(handler=service.cmd_fetch_local)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the RAG OpenAI CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except service.RagServiceError as exc:  # type: ignore[attr-defined]
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover
        print(f"[UNEXPECTED] {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
