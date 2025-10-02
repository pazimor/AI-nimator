"""CLI wrapper for the LMF extractor feature."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.features.lmf import extractor
from src.features.lmf.bulk import run_bulk


def build_single_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract LMF descriptors from animation.json")
    parser.add_argument("input", help="Path to animation.json")
    parser.add_argument("-o", "--output", default=None, help="Output .lmf.json path")
    parser.add_argument("--fps-internal", type=float, default=15.0)
    return parser


def build_bulk_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bulk convert animation.json files into animation.lmf.json",
        prog="lmf_extractor bulk",
    )
    parser.add_argument("root", nargs="?", default=".", help="Root directory to search")
    parser.add_argument("--pattern", default="animation.json", help="Filename pattern (case-insensitive)")
    parser.add_argument("--fps-internal", type=float, default=15.0, help="Internal FPS for extraction")
    parser.add_argument(
        "--follow-links",
        action="store_true",
        help="Follow symbolic links while traversing the directory tree",
    )
    parser.add_argument(
        "--skip-up-to-date",
        dest="skip_up_to_date",
        action="store_true",
        default=True,
        help="Skip conversion if animation.lmf.json is newer than the source",
    )
    parser.add_argument(
        "--no-skip-up-to-date",
        dest="skip_up_to_date",
        action="store_false",
        help="Always regenerate outputs even when already up-to-date",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "bulk":
        parser = build_bulk_parser()
        args = parser.parse_args(argv[1:])
        return run_bulk(
            root=args.root,
            pattern=args.pattern,
            fps_internal=float(args.fps_internal),
            follow_links=args.follow_links,
            skip_up_to_date=args.skip_up_to_date,
        )

    parser = build_single_parser()
    args = parser.parse_args(argv)
    try:
        inp = Path(args.input)
        out = Path(args.output) if args.output else inp.with_suffix(".lmf.json")
        extractor.extract_lmf(inp, out, fps_internal=float(args.fps_internal))
        print(out)
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
