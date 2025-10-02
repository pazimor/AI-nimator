"""Bulk conversion utilities for LMF extraction."""

from __future__ import annotations

import fnmatch
import os
import sys
from pathlib import Path
from typing import Callable

from src.features.lmf import extractor


def _find_candidates(root: Path, pattern: str, follow_links: bool) -> list[Path]:
    pattern_lower = pattern.lower()
    matches: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_links):
        for name in filenames:
            if fnmatch.fnmatch(name.lower(), pattern_lower):
                matches.append(Path(dirpath) / name)
    matches.sort()
    return matches


def _default_err_logger(message: str) -> None:
    print(message, file=sys.stderr)


def run_bulk(
    root: Path | str = ".",
    pattern: str = "animation.json",
    fps_internal: float = 15.0,
    *,
    follow_links: bool = False,
    skip_up_to_date: bool = True,
    extractor_label: str = "python -m src.cli.lmf_extractor_cli",
    log: Callable[[str], None] = print,
    err_log: Callable[[str], None] = _default_err_logger,
) -> int:
    """Bulk convert animation clips under ``root`` using the LMF extractor.

    Returns 0 on success, 1 if some conversions failed.
    """

    root_path = Path(root).resolve()

    log(f"[INFO] Extractor       : {extractor_label}")
    log(f"[INFO] Root            : {root_path}")
    log(f"[INFO] FPS internal    : {fps_internal}")
    log(f"[INFO] Pattern         : {pattern}")
    log(f"[INFO] Follow links    : {int(follow_links)}")
    log(f"[INFO] Skip up-to-date : {int(skip_up_to_date)}")

    candidates = _find_candidates(root_path, pattern, follow_links)
    count = len(candidates)
    log(f"[CHECK] Fichiers trouvés : {count}")
    if count == 0:
        log("[HINT ] Rien trouvé.")
        log(f"        - Essaie un motif plus large : \"*{pattern}*\"")
        log(
            "        - Suivre les liens : FOLLOW_LINKS=1 python -m src.cli.lmf_extractor_cli bulk \"{root}\"".format(
                root=root_path
            )
        )
        log(
            "        - Vérif manuelle : find \"{root}\" -type f -iname \"{pattern}\" | head -n 20".format(
                root=root_path, pattern=pattern
            )
        )
        return 0

    log("[CHECK] Exemples :")
    for example in candidates[:5]:
        log(f" - {example}")

    log("[RUN  ] Lancement des conversions…")
    errors = 0
    for source in candidates:
        out = source.parent / "animation.lmf.json"
        if skip_up_to_date and out.exists() and out.stat().st_mtime >= source.stat().st_mtime:
            log(f"[SKIP] {source} (déjà à jour)")
            continue

        log(f"[DO   ] {source} -> {out}")
        try:
            extractor.extract_lmf(source, out, fps_internal=float(fps_internal))
        except Exception as exc:  # pragma: no cover - reported and continue
            errors += 1
            err_log(f"[ERROR] Échec sur: {source} -- {exc}")

    log("[DONE ] Terminé.")
    return 1 if errors else 0


__all__ = ["run_bulk"]
