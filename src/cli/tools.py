#!/usr/bin/env python3
"""CLI tools for inspecting checkpoints and model information."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.shared.constants.rotation import ROTATION_CHANNELS_ROT6D


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


def _summarizeModuleParams(module: torch.nn.Module) -> Dict[str, int]:
    """Summarize parameter counts and size for a module."""
    total = 0
    trainable = 0
    totalBytes = 0
    trainableBytes = 0
    for param in module.parameters():
        num = param.numel()
        total += num
        totalBytes += num * param.element_size()
        if param.requires_grad:
            trainable += num
            trainableBytes += num * param.element_size()
    return {
        "total": total,
        "trainable": trainable,
        "total_bytes": totalBytes,
        "trainable_bytes": trainableBytes,
    }


def _printParamSummary(label: str, module: torch.nn.Module) -> None:
    """Print a readable parameter summary for a module."""
    summary = _summarizeModuleParams(module)
    totalSize = _formatSize(summary["total_bytes"])
    trainableSize = _formatSize(summary["trainable_bytes"])
    print(
        "Params %s: total=%s (%s) trainable=%s (%s)"
        % (
            label,
            f"{summary['total']:,}",
            totalSize,
            f"{summary['trainable']:,}",
            trainableSize,
        )
    )


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


def _resolveDevice(device: str) -> torch.device:
    """Resolve device string to torch.device."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def shapeCheck(
    networkConfigPath: str,
    networkProfile: str,
    batchSize: int,
    frames: int,
    device: str,
    motionChannels: Optional[int],
    full: bool,
    modelName: str,
    maxLength: int,
) -> None:
    """
    Run a shape-only validation pass for the generation network.

    Parameters
    ----------
    networkConfigPath : str
        Path to network.yaml.
    networkProfile : str
        Profile name in network.yaml.
    batchSize : int
        Batch size for dummy inputs.
    frames : int
        Number of frames for dummy motion.
    device : str
        Torch device string (cpu, cuda, mps, auto).
    motionChannels : Optional[int]
        Channels per bone (None uses network config).
    full : bool
        If True, include CLIP text encoder and quaternion conversion.
    modelName : str
        Hugging Face model name for CLIP text encoder.
    maxLength : int
        Max token length for dummy prompts.
    """
    from src.shared.config_loader import loadNetworkConfig

    resolvedDevice = _resolveDevice(device)
    networkConfig = loadNetworkConfig(Path(networkConfigPath), profile=networkProfile)

    embedDim = networkConfig.embedDim
    numHeads = networkConfig.generation.numHeads
    numLayers = networkConfig.generation.numLayers
    numSpatialLayers = networkConfig.generation.numSpatialLayers
    numBones = networkConfig.generation.numBones
    rotationRepr = networkConfig.generation.rotationRepr
    spatiotemporalMode = networkConfig.generation.spatiotemporalMode
    diffusionSteps = networkConfig.generation.diffusionSteps
    if motionChannels is None:
        motionChannels = networkConfig.generation.motionChannels
    elif motionChannels != networkConfig.generation.motionChannels:
        print(
            "⚠️  motion-channels override ignored; using network config."
        )
        motionChannels = networkConfig.generation.motionChannels

    if embedDim % numHeads != 0:
        print(
            "❌ Invalid config: embedDim must be divisible by numHeads "
            f"(embedDim={embedDim}, numHeads={numHeads})."
        )
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ Shape check: generation network")
    print("=" * 60)
    print(f"network.yaml: {networkConfigPath} (profile={networkProfile})")
    print(
        "D=%d H=%d L=%d S=%d K=%d C=%d R=%s M=%s T=%d device=%s"
        % (
            embedDim,
            numHeads,
            numLayers,
            numSpatialLayers,
            numBones,
            motionChannels,
            rotationRepr,
            spatiotemporalMode,
            frames,
            resolvedDevice,
        )
    )

    tags = [None] * batchSize
    timesteps = torch.randint(
        0,
        diffusionSteps,
        (batchSize,),
        device=resolvedDevice,
        dtype=torch.long,
    )
    noisyMotion = torch.randn(
        batchSize,
        frames,
        numBones,
        motionChannels,
        device=resolvedDevice,
    )

    with torch.no_grad():
        if full:
            from src.shared.model.generation.motion_generator import MotionGenerator

            model = MotionGenerator(
                embedDim=embedDim,
                numHeads=numHeads,
                numLayers=numLayers,
                numSpatialLayers=numSpatialLayers,
                motionChannels=motionChannels,
                rotationRepr=rotationRepr,
                spatiotemporalMode=spatiotemporalMode,
                numBones=numBones,
                diffusionSteps=diffusionSteps,
                modelName=modelName,
                clipCheckpoint=None,
            ).to(resolvedDevice)
            model.eval()

            _printParamSummary("MotionGenerator", model)
            _printParamSummary("MotionDenoiser", model.denoiser)
            _printParamSummary("CLIP", model.clip)

            prompts = ["shape check"] * batchSize
            encoded = model.clip.tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=maxLength,
                return_tensors="pt",
            )
            inputIds = encoded["input_ids"].to(resolvedDevice)
            attentionMask = encoded["attention_mask"].to(resolvedDevice)

            outputs = model(
                textInputIds=inputIds,
                textAttentionMask=attentionMask,
                tags=tags,
                noisyMotion=noisyMotion,
                timesteps=timesteps,
                targetNoise=None,
            )
            predictedNoise = outputs["predicted_noise"]

            if predictedNoise.shape != noisyMotion.shape:
                print(
                    "❌ Shape mismatch: predicted_noise %s vs noisyMotion %s"
                    % (tuple(predictedNoise.shape), tuple(noisyMotion.shape))
                )
                sys.exit(1)

            motionFlat = predictedNoise.view(
                batchSize,
                frames,
                numBones * motionChannels,
            )
            smoothed = model.smoothing(motionFlat)
            motion = smoothed.view(batchSize, frames, numBones, motionChannels)
            motionQuat = model._rotationToQuaternion(motion)
            motionQuat = model.velocityReg(motionQuat)

            expectedQuat = (batchSize, frames, numBones, 4)
            if motionQuat.shape != expectedQuat:
                print(
                    "❌ Shape mismatch: quaternion %s vs expected %s"
                    % (tuple(motionQuat.shape), expectedQuat)
                )
                sys.exit(1)
            print("✅ Full check OK (denoiser + CLIP + post-processing).")
        else:
            from src.shared.model.generation.denoiser import MotionDenoiser
            from src.shared.model.layers.correction import (
                Renormalization,
                Smoothing,
                VelocityRegularization,
            )

            denoiser = MotionDenoiser(
                embedDim=embedDim,
                numHeads=numHeads,
                numLayers=numLayers,
                numSpatialLayers=numSpatialLayers,
                spatiotemporalMode=spatiotemporalMode,
                numBones=numBones,
                motionChannels=motionChannels,
            ).to(resolvedDevice)
            denoiser.eval()

            _printParamSummary("MotionDenoiser", denoiser)

            textEmbedding = torch.randn(batchSize, embedDim, device=resolvedDevice)
            predictedNoise = denoiser(
                noisyMotion=noisyMotion,
                textEmbedding=textEmbedding,
                tags=tags,
                timesteps=timesteps,
            )

            if predictedNoise.shape != noisyMotion.shape:
                print(
                    "❌ Shape mismatch: predicted_noise %s vs noisyMotion %s"
                    % (tuple(predictedNoise.shape), tuple(noisyMotion.shape))
                )
                sys.exit(1)

            motionFlat = predictedNoise.view(
                batchSize,
                frames,
                numBones * motionChannels,
            )
            smoothing = Smoothing(channels=numBones * motionChannels).to(resolvedDevice)
            smoothed = smoothing(motionFlat)
            motion = smoothed.view(batchSize, frames, numBones, motionChannels)
            if motionChannels == ROTATION_CHANNELS_ROT6D:
                renorm = Renormalization().to(resolvedDevice)
                velreg = VelocityRegularization().to(resolvedDevice)
                rotMat = renorm(motion)
                rotMat = velreg(rotMat)

                expectedRot = (batchSize, frames, numBones, 3, 3)
                if rotMat.shape != expectedRot:
                    print(
                        "❌ Shape mismatch: rotation matrix %s vs expected %s"
                        % (tuple(rotMat.shape), expectedRot)
                    )
                    sys.exit(1)
                print("✅ Fast check OK (denoiser + post-processing).")
            else:
                print(
                    "⚠️  Fast check OK (denoiser + smoothing). "
                    "Renormalization skipped for non-6D rotations."
                )

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

  # Shape check (denoiser + post-processing)
  poetry run python -m src.cli.tools shape-check --network-profile default

  # Shape check with CLIP text encoder and quaternion conversion
  poetry run python -m src.cli.tools shape-check --full --device cpu
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

    # Shape check command
    shapeParser = subparsers.add_parser(
        "shape-check",
        help="Valider les formes tensors du réseau de génération",
    )
    shapeParser.add_argument(
        "--network-config",
        default="src/configs/network.yaml",
        help="Chemin vers network.yaml",
    )
    shapeParser.add_argument(
        "--network-profile",
        default="default",
        help="Profil dans network.yaml (default, spark, lightweight)",
    )
    shapeParser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    shapeParser.add_argument("--frames", type=int, default=16, help="Nombre de frames")
    shapeParser.add_argument(
        "--device",
        default="auto",
        help="Device torch (auto, cpu, cuda, mps)",
    )
    shapeParser.add_argument(
        "--motion-channels",
        type=int,
        default=None,
        help="Channels par bone (None = config)",
    )
    shapeParser.add_argument(
        "--full",
        action="store_true",
        help="Inclure CLIP + conversion quaternion",
    )
    shapeParser.add_argument(
        "--model-name",
        default="xlm-roberta-base",
        help="Nom du modele textuel (si --full)",
    )
    shapeParser.add_argument(
        "--max-length",
        type=int,
        default=64,
        help="Longueur max des prompts (si --full)",
    )

    args = parser.parse_args()

    if args.command == "inspect":
        inspectCheckpoint(args.checkpoint, args.verbose)
    elif args.command == "list":
        listCheckpoints(args.directory)
    elif args.command == "compare":
        compareCheckpoints(args.checkpoint1, args.checkpoint2)
    elif args.command == "shape-check":
        shapeCheck(
            networkConfigPath=args.network_config,
            networkProfile=args.network_profile,
            batchSize=args.batch_size,
            frames=args.frames,
            device=args.device,
            motionChannels=args.motion_channels,
            full=args.full,
            modelName=args.model_name,
            maxLength=args.max_length,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
