"""CLI wrapper for the uniformizer service."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.features.uniformizer import service


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Uniformizer v1.10 (no split)")
    sub = parser.add_subparsers(dest="command", required=True)

    json2json = sub.add_parser("json2json", help="Retarget/convert un fichier JSON rotroot")
    json2json.add_argument("input")
    json2json.add_argument("output")
    json2json.add_argument("--target-skel", choices=["smpl22"], help="Squelette cible prédéfini")
    json2json.add_argument("--target-map", help="Mapping JSON personnalisé")
    json2json.add_argument("--resample", type=float, default=0.0)
    json2json.add_argument("--prompts", action="store_true")
    json2json.set_defaults(handler=service.cmd_json2json)

    npz2json = sub.add_parser("npz2json", help="Convertit un NPZ (positions/quats) vers JSON rotroot")
    npz2json.add_argument("input")
    npz2json.add_argument("output")
    npz2json.add_argument("--target-skel", choices=["smpl22"], help="Squelette cible prédéfini")
    npz2json.add_argument("--target-map", help="Mapping JSON personnalisé")
    npz2json.add_argument("--resample", type=float, default=0.0)
    npz2json.add_argument("--prompts", action="store_true")
    npz2json.set_defaults(handler=service.cmd_npz2json)

    dir2json = sub.add_parser("dir2json", help="Conversion récursive d'un dossier .npz -> .json")
    dir2json.add_argument("input_dir")
    dir2json.add_argument("output_dir")
    dir2json.add_argument("--target-skel", choices=["smpl22"], help="Squelette cible prédéfini")
    dir2json.add_argument("--target-map", help="Mapping JSON personnalisé")
    dir2json.add_argument("--resample", type=float, default=0.0)
    dir2json.add_argument("--flat", action="store_true")
    dir2json.add_argument("--prompts", action="store_true")
    dir2json.set_defaults(handler=service.cmd_dir2json)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except service.UniformizerError as exc:  # type: ignore[attr-defined]
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - unexpected branch
        print(f"[UNEXPECTED] {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
