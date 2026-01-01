#!/usr/bin/env python3
"""CLI tools for inspecting checkpoints and model information."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _formatSize(numBytes: int) -> str:
    """Format bytes to human readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if numBytes < 1024:
            return f"{numBytes:.2f} {unit}"
        numBytes /= 1024
    return f"{numBytes:.2f} TB"


def _countParameters(stateDict: Dict[str, Any]) -> Dict[str, int]:
    """Count parameters in a state dict."""
    total = 0
    trainable = 0
    for key, value in stateDict.items():
        if isinstance(value, torch.Tensor):
            total += value.numel()
    return {"total": total, "size_mb": total * 4 / (1024 * 1024)}  # Assuming float32


def inspectCheckpoint(checkpointPath: str, verbose: bool = False) -> None:
    """
    Inspect a PyTorch checkpoint and display its contents.

    Parameters
    ----------
    checkpointPath : str
        Path to the checkpoint file.
    verbose : bool
        If True, show detailed layer information.
    """
    path = Path(checkpointPath)
    if not path.exists():
        print(f"❌ Fichier non trouvé: {checkpointPath}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"📁 Checkpoint: {path.name}")
    print(f"{'='*60}")

    # File info
    fileSize = path.stat().st_size
    print(f"📦 Taille fichier: {_formatSize(fileSize)}")

    # Load checkpoint
    try:
        ckpt = torch.load(checkpointPath, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"❌ Erreur de chargement: {e}")
        sys.exit(1)

    # Display metadata
    print(f"\n{'─'*40}")
    print("📊 Métadonnées:")
    print(f"{'─'*40}")

    metaKeys = ["epoch", "loss", "train_loss", "val_loss", "best_val_loss", "global_step"]
    for key in metaKeys:
        if key in ckpt:
            value = ckpt[key]
            if isinstance(value, float):
                print(f"  • {key}: {value:.6f}")
            else:
                print(f"  • {key}: {value}")

    # Display all keys
    print(f"\n{'─'*40}")
    print("🔑 Clés disponibles:")
    print(f"{'─'*40}")
    for key in ckpt.keys():
        if key == "model_state_dict":
            paramInfo = _countParameters(ckpt[key])
            print(f"  • {key}: {paramInfo['total']:,} params ({paramInfo['size_mb']:.1f} MB)")
        elif key == "optimizer_state_dict":
            print(f"  • {key}: (optimizer state)")
        elif key == "scheduler_state_dict":
            print(f"  • {key}: (scheduler state)")
        else:
            print(f"  • {key}")

    # Model architecture details
    if "model_state_dict" in ckpt:
        stateDict = ckpt["model_state_dict"]
        print(f"\n{'─'*40}")
        print("🏗️  Architecture:")
        print(f"{'─'*40}")

        # Group by module
        modules: Dict[str, int] = {}
        for key in stateDict.keys():
            parts = key.split(".")
            if len(parts) >= 1:
                moduleName = parts[0]
                if moduleName not in modules:
                    modules[moduleName] = 0
                if isinstance(stateDict[key], torch.Tensor):
                    modules[moduleName] += stateDict[key].numel()

        for moduleName, paramCount in sorted(modules.items(), key=lambda x: -x[1]):
            sizeMb = paramCount * 4 / (1024 * 1024)
            print(f"  • {moduleName}: {paramCount:,} params ({sizeMb:.1f} MB)")

        if verbose:
            print(f"\n{'─'*40}")
            print("📋 Détail des couches:")
            print(f"{'─'*40}")
            for key, value in stateDict.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {list(value.shape)}")

    print()


def compareCheckpoints(path1: str, path2: str) -> None:
    """
    Compare two checkpoints.

    Parameters
    ----------
    path1 : str
        Path to first checkpoint.
    path2 : str
        Path to second checkpoint.
    """
    print(f"\n{'='*60}")
    print("🔄 Comparaison de checkpoints")
    print(f"{'='*60}")

    ckpt1 = torch.load(path1, map_location="cpu", weights_only=False)
    ckpt2 = torch.load(path2, map_location="cpu", weights_only=False)

    print(f"\n📁 {Path(path1).name} vs {Path(path2).name}")
    print(f"{'─'*40}")

    # Compare metadata
    for key in ["epoch", "loss", "train_loss", "val_loss"]:
        v1 = ckpt1.get(key, "N/A")
        v2 = ckpt2.get(key, "N/A")
        if isinstance(v1, float) and isinstance(v2, float):
            diff = v2 - v1
            arrow = "↓" if diff < 0 else "↑" if diff > 0 else "="
            print(f"  {key}: {v1:.4f} → {v2:.4f} ({arrow} {abs(diff):.4f})")
        else:
            print(f"  {key}: {v1} → {v2}")

    print()


def listCheckpoints(directory: str) -> None:
    """
    List all checkpoints in a directory.

    Parameters
    ----------
    directory : str
        Path to directory containing checkpoints.
    """
    dirPath = Path(directory)
    if not dirPath.exists():
        print(f"❌ Dossier non trouvé: {directory}")
        sys.exit(1)

    checkpoints = list(dirPath.glob("*.pt")) + list(dirPath.glob("*.pth"))
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    print(f"\n{'='*60}")
    print(f"📂 Checkpoints dans: {directory}")
    print(f"{'='*60}")

    if not checkpoints:
        print("  (aucun checkpoint trouvé)")
        return

    for ckptPath in checkpoints:
        fileSize = _formatSize(ckptPath.stat().st_size)
        try:
            ckpt = torch.load(ckptPath, map_location="cpu", weights_only=False)
            epoch = ckpt.get("epoch", "?")
            loss = ckpt.get("loss", ckpt.get("val_loss", ckpt.get("train_loss", "?")))
            if isinstance(loss, float):
                lossStr = f"{loss:.4f}"
            else:
                lossStr = str(loss)
            print(f"  📄 {ckptPath.name:30} | {fileSize:>10} | epoch={epoch:>4} | loss={lossStr}")
        except Exception as e:
            print(f"  📄 {ckptPath.name:30} | {fileSize:>10} | ❌ {e}")

    print()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="🔧 Outils d'inspection des checkpoints AI-nimator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Inspecter un checkpoint
  poetry run python -m src.cli.tools inspect output/checkpoints/best_model.pt

  # Inspecter avec détails des couches
  poetry run python -m src.cli.tools inspect output/checkpoints/best_model.pt --verbose

  # Lister tous les checkpoints d'un dossier
  poetry run python -m src.cli.tools list output/checkpoints/

  # Comparer deux checkpoints
  poetry run python -m src.cli.tools compare ckpt1.pt ckpt2.pt
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")

    # Inspect command
    inspectParser = subparsers.add_parser("inspect", help="Inspecter un checkpoint")
    inspectParser.add_argument("checkpoint", help="Chemin vers le checkpoint")
    inspectParser.add_argument(
        "-v", "--verbose", action="store_true", help="Afficher les détails des couches"
    )

    # List command
    listParser = subparsers.add_parser("list", help="Lister les checkpoints d'un dossier")
    listParser.add_argument("directory", help="Chemin vers le dossier")

    # Compare command
    compareParser = subparsers.add_parser("compare", help="Comparer deux checkpoints")
    compareParser.add_argument("checkpoint1", help="Premier checkpoint")
    compareParser.add_argument("checkpoint2", help="Deuxième checkpoint")

    args = parser.parse_args()

    if args.command == "inspect":
        inspectCheckpoint(args.checkpoint, args.verbose)
    elif args.command == "list":
        listCheckpoints(args.directory)
    elif args.command == "compare":
        compareCheckpoints(args.checkpoint1, args.checkpoint2)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
