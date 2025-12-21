"""CLI tool to convert JSON animation files to Collada format."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch

from src.features.dataset_builder.animation_rebuilder import AnimationRebuilder
from src.features.dataset_builder.config_loader import loadBuilderConfig
from src.shared.quaternion import Rotation
from src.shared.types import AnimationSample

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("json_to_collada")


def buildArgumentParser() -> argparse.ArgumentParser:
    """
    Create the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """
    parser = argparse.ArgumentParser(
        description="Convert a JSON animation file to Collada (.dae) format.",
    )
    parser.add_argument(
        "--input_file",
        "-i",
        type=Path,
        help="Path to the input JSON animation file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help=(
            "Path for the output .dae file. Defaults to input filename "
            "with .dae extension."
        ),
    )
    parser.add_argument(
        "--zero_root_translation",
        action="store_true",
        help="Force pelvis translation to zero for all frames.",
    )
    parser.add_argument(
        "--anchor_root_translation",
        action="store_true",
        help="Recenter pelvis translation by subtracting the first frame.",
    )
    return parser


def load_json_animation(file_path: Path) -> Dict[str, Any]:
    """Load the JSON animation file."""
    with open(file_path, "r") as f:
        return json.load(f)


def convert_json_to_sample(
    data: Dict[str, Any],
    file_path: Path,
) -> AnimationSample:
    """
    Convert JSON data to AnimationSample.

    Expected keys:
    - meta.fps
    - bones[].frames[].rotation (6 floats as 6D rotation)
    """
    meta = data.get("meta", {})
    fps = int(meta.get("fps", 30))
    
    bones_data = data.get("bones", [])
    if not bones_data:
        raise ValueError("No bone data found in JSON.")

    # Determine number of frames
    num_frames = 0
    if bones_data:
        num_frames = len(bones_data[0].get("frames", []))
    
    # Rebuild SMPL24-ordered axis-angle buffer from SMPL22 JSON.
    
    from src.shared.constants.skeletons import SMPL24_BONE_ORDER
    
    smpl24_map = {name: i for i, name in enumerate(SMPL24_BONE_ORDER)}
    num_smpl24_bones = len(SMPL24_BONE_ORDER)
    
    # Initialize array with identity rotations (axis angle 0,0,0)
    # Shape: (frames, num_bones, 3)
    axis_angles_full = np.zeros(
        (num_frames, num_smpl24_bones, 3),
        dtype=np.float32,
    )
    
    seen_bones = set()
    for bone in bones_data:
        bone_name = bone.get("name")
        if bone_name not in smpl24_map:
            LOGGER.warning(
                "Bone %s not found in SMPL24 skeleton, skipping.",
                bone_name,
            )
            continue
        seen_bones.add(bone_name)
        target_idx = smpl24_map[bone_name]
        frames = bone.get("frames", [])
        if len(frames) != num_frames:
            LOGGER.warning(
                "Bone %s has %s frames, expected %s; clamping or padding.",
                bone_name,
                len(frames),
                num_frames,
            )
            if len(frames) < num_frames:
                if frames:
                    frames = frames + [frames[-1]] * (num_frames - len(frames))
                else:
                    frames = []
            else:
                frames = frames[:num_frames]
        
        # Extract rotations (assuming 6D from JSON)
        # JSON 6D: [r1, r2, r3, r4, r5, r6]
        rotations_6d = []
        for f in frames:
            rot = f.get("rotation")
            if rot and len(rot) == 6:
                rotations_6d.append(rot)
            else:
                rotations_6d.append([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        
        if not rotations_6d:
            continue
            
        rotations_6d_np = np.array(rotations_6d, dtype=np.float32)
        
        # Convert 6D to Axis-Angle
        # We can use our Rotation class
        rot_tensor = torch.from_numpy(rotations_6d_np)
        rot_obj = Rotation(rot_tensor, kind="rot6d")
        axis_angle = rot_obj.axis_angle.numpy()

        # Diagnostics: warn if consecutive frames are identical
        if axis_angle.shape[0] > 1:
            identical = np.all(
                np.isclose(axis_angle[1:], axis_angle[:-1]),
                axis=1,
            )
            if identical.any():
                idxs = np.nonzero(identical)[0][:5].tolist()
                LOGGER.warning(
                    "Bone %s has %s consecutive identical frames (first: %s).",
                    bone_name,
                    int(identical.sum()),
                    idxs,
                )
        magnitudes = np.linalg.norm(axis_angle, axis=1)
        LOGGER.debug(
            "Bone %s axis-angle norm min=%.4f max=%.4f mean=%.4f",
            bone_name,
            float(np.min(magnitudes)),
            float(np.max(magnitudes)),
            float(np.mean(magnitudes)),
        )
        
        axis_angles_full[:, target_idx, :] = axis_angle

    # Flatten to (frames, bones*3) as expected by AnimationSample.
    axis_angles_flat = axis_angles_full.reshape(num_frames, -1)
    
    extras = data.get("extras", {})
    missing = [name for name in SMPL24_BONE_ORDER if name not in seen_bones]
    if missing:
        LOGGER.warning(
            "Missing bones in JSON, filled with identity: %s",
            ", ".join(missing),
        )
    LOGGER.info(
        "Loaded animation: fps=%s frames=%s bones=%s (SMPL24 slots=%s)",
        fps,
        num_frames,
        len(bones_data),
        num_smpl24_bones,
    )
    
    return AnimationSample(
        relativePath=file_path,
        resolvedPath=file_path.resolve(),
        axisAngles=axis_angles_flat,
        fps=fps,
        extras=extras,
    )


def main() -> None:
    """CLI entry-point."""
    parser = buildArgumentParser()
    args = parser.parse_args()

    if args.zero_root_translation and args.anchor_root_translation:
        parser.error(
            "Use either --zero_root_translation or "
            "--anchor_root_translation, not both."
        )
    
    input_path = args.input_file
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")
        
    output_path = args.output
    if output_path is None:
        output_path = input_path.with_suffix(".dae")
        
    LOGGER.info(f"Loading {input_path}...")
    try:
        data = load_json_animation(input_path)
        sample = convert_json_to_sample(data, input_path)
        
        # Load real config so AnimationRebuilder resolves paths correctly.
        config_path = Path("src/configs/dataset.yaml")
        if config_path.exists():
            config = loadBuilderConfig(config_path)
        else:
            # For now assuming it exists as per user context.
            raise FileNotFoundError("Could not find src/configs/dataset.yaml")
        
        rebuilder = AnimationRebuilder(config)
        
        LOGGER.info(f"Exporting to {output_path}...")
        rebuilder.exportCollada(
            sample,
            output_path,
            zeroRootTranslation=args.zero_root_translation,
            anchorRootTranslation=args.anchor_root_translation,
        )
        LOGGER.info("Done.")
        
    except Exception as e:
        LOGGER.error(f"Conversion failed: {e}")
        exit(1)


#TODO: tester avec un fichier pour valider la coherance de l'animation
if __name__ == "__main__":
    main()
