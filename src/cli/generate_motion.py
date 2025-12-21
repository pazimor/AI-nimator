"""CLI entry point for motion generation inference."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from src.shared.model.generation.motion_generator import MotionGenerator
from src.shared.types import VALID_TAGS, validateTag

LOGGER = logging.getLogger("generation.inference_cli")


def buildArgumentParser() -> argparse.ArgumentParser:
    """
    Create the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """
    parser = argparse.ArgumentParser(
        description="Generate motion from text prompt using trained diffusion model.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained generation model checkpoint.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text description of the motion to generate.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help=f"Optional motion category tag. Valid values: {', '.join(VALID_TAGS)}",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=60,
        help="Number of frames to generate (default: 60).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save the generated motion JSON file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for inference (default: auto).",
    )
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=50,
        help="Number of DDIM sampling steps (default: 50).",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=64,
        help="Embedding dimension (must match training, default: 64).",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads (must match training, default: 4).",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of denoiser layers (must match training, default: 6).",
    )
    return parser


def main() -> None:
    """CLI entry point for motion generation."""
    logging.basicConfig(level=logging.INFO)
    parser = buildArgumentParser()
    args = parser.parse_args()

    try:
        # Validate tag if provided
        if args.tag is not None:
            validateTag(args.tag)

        # Validate checkpoint exists
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

        # Resolve device
        device = _resolveDevice(args.device)
        LOGGER.info("Using device: %s", device)

        # Load model
        LOGGER.info("Loading model from %s", args.checkpoint)
        model = MotionGenerator(
            embedDim=args.embed_dim,
            numHeads=args.num_heads,
            numLayers=args.num_layers,
        )
        _loadCheckpoint(args.checkpoint, model)
        model = model.to(device)
        model.eval()

        # Generate motion
        tagDisplay = args.tag if args.tag else "(none)"
        LOGGER.info(
            "Generating %d frames for prompt: '%s' with tag: %s",
            args.frames,
            args.prompt,
            tagDisplay,
        )
        motionQuat = model.generate(
            prompt=args.prompt,
            tag=args.tag,
            numFrames=args.frames,
            ddimSteps=args.ddim_steps,
            device=device,
        )

        # Save output
        _saveMotion(motionQuat, args.output)
        LOGGER.info("Motion saved to %s", args.output)

        parser.exit(0, f"Generated {args.frames} frames successfully.\n")

    except Exception as error:  # noqa: BLE001
        LOGGER.exception("Generation failed")
        parser.exit(1, f"{error}\n")


def _loadCheckpoint(checkpointPath: Path, model: MotionGenerator) -> None:
    """
    Load model weights from checkpoint.

    Parameters
    ----------
    checkpointPath : Path
        Path to checkpoint file.
    model : MotionGenerator
        Model to load weights into.
    """
    checkpoint = torch.load(checkpointPath, weights_only=False, map_location="cpu")

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    elif "denoiser_state_dict" in checkpoint:
        model.denoiser.load_state_dict(checkpoint["denoiser_state_dict"])
    else:
        raise ValueError("Checkpoint does not contain model weights")


def _saveMotion(motion: torch.Tensor, outputPath: Path) -> None:
    """
    Save generated motion to JSON file.

    Parameters
    ----------
    motion : torch.Tensor
        Generated motion shaped (1, frames, bones, 4).
    outputPath : Path
        Path to save JSON file.
    """
    # Remove batch dimension and convert to list
    motionData = motion.squeeze(0).cpu().tolist()

    output = {
        "format": "quaternion",
        "shape": {
            "frames": len(motionData),
            "bones": len(motionData[0]) if motionData else 0,
            "channels": 4,
        },
        "data": motionData,
    }

    outputPath.parent.mkdir(parents=True, exist_ok=True)
    with open(outputPath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


def _resolveDevice(choice: str) -> torch.device:
    """
    Resolve the torch.device based on CLI arguments.

    Parameters
    ----------
    choice : str
        Requested backend.

    Returns
    -------
    torch.device
        Device satisfying the request.
    """
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend not available.")
        return torch.device("cuda")
    if choice == "mps":
        hasMps = hasattr(torch.backends, "mps")
        available = torch.backends.mps.is_available() if hasMps else False
        if not available:
            raise RuntimeError("Apple MPS backend not available.")
        return torch.device("mps")
    return torch.device(choice)


if __name__ == "__main__":
    main()
