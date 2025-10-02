"""Reusable utilities to load animation assets from disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from src.shared.constants import DEFAULT_FPS, SMPL24_BONES
from src.shared.quaternion import QuaternionConverter


class AnimationIOError(RuntimeError):
    """Raised when animation payloads cannot be decoded."""


def load_animation_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str], float]:
    """Read an animation NPZ file and normalize its content."""

    archive = _open_npz(path)
    keys = list(archive.keys())
    source = str(path)
    if _is_amass_pose(keys):
        return _load_amass_pose(archive, source)
    native = _load_native_npz(archive, keys, source)
    if native is not None:
        return native
    if _is_amass_shape(keys):
        raise AnimationIOError(
            f"{path}: shape archive detected (missing pose data)"
        )
    raise AnimationIOError(f"{path}: unsupported NPZ structure: {keys}")


def load_animation_json(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str], float]:
    """Load a JSON animation file that follows the rotroot convention."""

    if not path.exists():
        raise AnimationIOError(f"JSON file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - unexpected disk failure
        raise AnimationIOError(f"{path}: invalid JSON payload ({exc})") from exc
    bones = payload.get("bones")
    frames = payload.get("frames")
    if not isinstance(bones, list) or not isinstance(frames, list):
        raise AnimationIOError(f"{path}: payload requires 'bones' and 'frames'")
    fps = float(payload.get("meta", {}).get("fps", DEFAULT_FPS))
    fmt = payload.get("format", "rotroot")
    position = np.zeros((len(frames), len(bones), 3), dtype=np.float32)
    rotation = np.concatenate(
        [np.zeros((len(frames), len(bones), 3), dtype=np.float32), np.ones((len(frames), len(bones), 1), dtype=np.float32)],
        axis=-1,
    )
    index_of = {name: idx for idx, name in enumerate(bones)}

    if fmt != "rotroot":
        raise AnimationIOError(f"{path}: unsupported animation format '{fmt}'")
    for frame_index, frame_payload in enumerate(frames):
        if not isinstance(frame_payload, dict):
            raise AnimationIOError(f"{path}: frame {frame_index} must be an object")
        root_position = frame_payload.get("root_pos", [0.0, 0.0, 0.0])
        _assign_root_position(position, frame_index, root_position, path)
        for bone_name, quat in frame_payload.items():
            if bone_name == "root_pos":
                continue
            bone_index = index_of.get(bone_name)
            if bone_index is None:
                continue
            _assign_quaternion(rotation, frame_index, bone_index, quat, path, bone_name)
    return position, rotation, bones, fps


def _open_npz(path: Path):
    if not path.exists():
        raise AnimationIOError(f"NPZ file not found: {path}")
    try:
        return np.load(path, allow_pickle=True)
    except Exception as exc:  # pragma: no cover - disk failures
        raise AnimationIOError(f"{path}: unable to load NPZ ({exc})") from exc


def _load_native_npz(archive, keys: Iterable[str], source: str):
    position_key = _best_key(archive, ["positions", "joints", "xyz", "joints_pos"])
    rotation_key = _best_key(archive, ["rotations", "quats", "quat", "joint_quats"])
    name_key = _best_key(archive, ["bone_names", "bones", "names", "joint_names"])
    fps_key = _best_key(archive, ["fps", "rate", "frame_rate", "mocap_framerate"])
    if position_key is None and rotation_key is None:
        return None
    if position_key is not None:
        position = archive[position_key]
        _assert_shape(position, 3, source, position_key)
    else:
        position = None
    if rotation_key is not None:
        rotation = archive[rotation_key]
        _assert_shape(rotation, 4, source, rotation_key)
    else:
        rotation = None
    frame_count, joint_count = _infer_dims(position, rotation, source)
    bones = _load_bone_names(archive, name_key, joint_count, source)
    fps = float(archive[fps_key]) if fps_key is not None else DEFAULT_FPS
    position = _coalesce_position(position, frame_count, joint_count)
    rotation = _coalesce_rotation(rotation, frame_count, joint_count)
    return position, rotation, bones, fps


def _load_amass_pose(archive, source: str) -> Tuple[np.ndarray, np.ndarray, List[str], float]:
    poses = archive["poses"]
    if poses.ndim != 2 or poses.shape[1] % 3 != 0:
        raise AnimationIOError(
            f"{source}: expected poses shaped (T, 3*J), got {poses.shape}"
        )
    frame_count, axis_count = poses.shape
    joint_count = axis_count // 3
    axis_angles = poses.reshape(frame_count, joint_count, 3).astype(np.float32)
    rotation = axis_angle_to_quaternion(axis_angles)
    position = np.zeros((frame_count, joint_count, 3), dtype=np.float32)
    if "trans" in archive and archive["trans"].shape == (frame_count, 3):
        position[:, 0, :] = archive["trans"].astype(np.float32)
    bones = _guess_bones(joint_count)
    fps = float(archive.get("mocap_framerate", DEFAULT_FPS))
    return position, rotation, bones, fps


def axis_angle_to_quaternion(axis_angles: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(axis_angles, axis=-1, keepdims=True)
    half = 0.5 * theta
    sin_over_theta = np.where(theta > 1e-5, np.sin(half) / (theta + 1e-9), 0.5)
    xyz = axis_angles * sin_over_theta
    w = np.cos(half)
    quaternion = np.concatenate([xyz, w], axis=-1)
    return normalize_quaternion(quaternion)


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    norm = np.where(np.isfinite(norm), norm, 0.0)
    norm = np.maximum(norm, 1e-8)
    return quaternion / norm


def _best_key(dataset, candidates: Iterable[str]):
    entries = set(dataset.keys())
    for key in candidates:
        if key in entries:
            return key
    lowered = {key.lower(): key for key in dataset.keys()}
    for key in candidates:
        if key in lowered:
            return lowered[key]
    return None


def _is_amass_pose(keys: Iterable[str]) -> bool:
    keys_set = set(keys)
    return "poses" in keys_set and ("trans" in keys_set or "mocap_framerate" in keys_set)


def _is_amass_shape(keys: Iterable[str]) -> bool:
    keys_set = set(keys)
    return "betas" in keys_set and "poses" not in keys_set and "trans" not in keys_set


def _assert_shape(array: np.ndarray, trailing: int, source: str, key: str) -> None:
    if array.ndim != 3 or array.shape[-1] != trailing:
        raise AnimationIOError(
            f"{source}: key '{key}' must be shaped (T, J, {trailing}), got {array.shape}"
        )


def _infer_dims(position: np.ndarray | None, rotation: np.ndarray | None, source: str):
    if position is None and rotation is None:
        raise AnimationIOError(f"{source}: empty NPZ payload")
    frame_count = position.shape[0] if position is not None else rotation.shape[0]
    joint_count = position.shape[1] if position is not None else rotation.shape[1]
    return frame_count, joint_count


def _load_bone_names(archive, key: str | None, joint_count: int, source: str) -> List[str]:
    if key is None:
        return [f"joint_{index}" for index in range(joint_count)]
    try:
        raw = archive[key]
        names = [str(entry) for entry in raw.tolist()]
        if len(names) != joint_count:
            raise AnimationIOError(
                f"{source}: bone names count mismatch ({len(names)} != {joint_count})"
            )
        return names
    except Exception as exc:
        raise AnimationIOError(
            f"{source}: unable to decode '{key}' as bone names"
        ) from exc


def _coalesce_position(position: np.ndarray | None, frames: int, joints: int) -> np.ndarray:
    if position is not None:
        return position.astype(np.float32)
    return np.zeros((frames, joints, 3), dtype=np.float32)


def _coalesce_rotation(rotation: np.ndarray | None, frames: int, joints: int) -> np.ndarray:
    if rotation is not None:
        return normalize_quaternion(rotation.astype(np.float32))
    identity = np.zeros((frames, joints, 4), dtype=np.float32)
    identity[..., 3] = 1.0
    return identity


def _guess_bones(joint_count: int) -> List[str]:
    if joint_count == 24:
        return list(SMPL24_BONES)
    return [f"joint_{index}" for index in range(joint_count)]


def _assign_root_position(position: np.ndarray, frame_index: int, root: Iterable[float], path: Path) -> None:
    if len(root) != 3:
        raise AnimationIOError(f"{path}: frame {frame_index} root_pos must contain 3 floats")
    position[frame_index, 0, :] = np.asarray(root, dtype=np.float32)


def _assign_quaternion(
    rotation: np.ndarray,
    frame_index: int,
    joint_index: int,
    quaternion_value,
    path: Path,
    bone_name: str,
) -> None:
    quaternion = np.asarray(quaternion_value, dtype=np.float32)
    if quaternion.shape != (4,):
        raise AnimationIOError(
            f"{path}: frame {frame_index} bone '{bone_name}' expects 4 values"
        )
    rotation[frame_index, joint_index, :] = normalize_quaternion(quaternion)


def loadAnimationFile(jsonPath: Path) -> Dict[str, List[str]]:
    """Load rotations for training from an animation JSON file.

    Parameters
    ----------
    jsonPath : Path
        Path to an ``animation.json`` generated by the uniformizer pipeline.

    Returns
    -------
    Dict[str, List[str]]
        Mapping of bone names to per-frame quaternions. Each quaternion is
        serialised as ``"x|y|z|w"`` to remain compatible with legacy loaders.

    Raises
    ------
    ValueError
        If the payload cannot be interpreted as a uniformizer export.
    """

    with open(jsonPath, "r", encoding="utf-8") as fileHandle:
        loaded = json.load(fileHandle)

    if isinstance(loaded, dict):
        if "bones" in loaded and "frames" in loaded:
            return _parse_uniformizer_rotations(loaded, jsonPath)
        if "rotations" in loaded and isinstance(loaded["rotations"], dict):
            return _normalise_quaternion_mapping(loaded["rotations"], jsonPath)
        return _normalise_quaternion_mapping(loaded, jsonPath)
    raise ValueError(f"Format inattendu pour {jsonPath}")


def _parse_uniformizer_rotations(
    payload: Dict[str, Any],
    path: Path,
) -> Dict[str, List[str]]:
    """Convert a uniformizer rotroot payload to a quaternion mapping.

    Parameters
    ----------
    payload : Dict[str, Any]
        JSON structure containing ``bones`` and ``frames`` sections.
    path : Path
        Path of the source file, used to enrich error messages.

    Returns
    -------
    Dict[str, List[str]]
        Mapping of bone names to pipe-separated quaternion strings.

    Raises
    ------
    ValueError
        If the payload misses expected sections or contains malformed frames.
    """

    boneNames = payload.get("bones")
    frames = payload.get("frames")
    if not isinstance(boneNames, list) or not boneNames:
        raise ValueError(f"{path}: champ 'bones' manquant ou invalide")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{path}: champ 'frames' manquant ou invalide")
    rotations: Dict[str, List[str]] = {name: [] for name in boneNames}
    for frameIndex, framePayload in enumerate(frames):
        if not isinstance(framePayload, dict):
            raise ValueError(f"{path}: frame {frameIndex} doit être un objet JSON")
        for boneName in boneNames:
            quaternion = framePayload.get(boneName)
            if quaternion is None:
                raise ValueError(
                    f"{path}: frame {frameIndex} missing quaternion for '{boneName}'"
                )
            rotations[boneName].append(
                _format_quaternion_components(quaternion, path, frameIndex, boneName)
            )
    return rotations


def _normalise_quaternion_mapping(
    rotations: Dict[str, Sequence[Any]],
    path: Path,
) -> Dict[str, List[str]]:
    """Ensure a generic mapping contains serialised quaternions.

    Parameters
    ----------
    rotations : Dict[str, Sequence[Any]]
        Mapping of bone names to raw quaternion payloads.
    path : Path
        Path of the source file, used to annotate errors.

    Returns
    -------
    Dict[str, List[str]]
        Cleaned mapping with quaternions encoded as pipe-separated strings.

    Raises
    ------
    ValueError
        If rotations are not sequences or contain inconsistent frame counts.
    """

    normalised: Dict[str, List[str]] = {}
    expectedFrameCount: int | None = None
    for boneName, sequence in rotations.items():
        if isinstance(sequence, str):
            raise ValueError(
                f"{path}: rotations for '{boneName}' must be a sequence of quaternions"
            )
        if not isinstance(sequence, Sequence):
            raise ValueError(f"{path}: rotations for '{boneName}' must be a list")
        formatted = [
            _format_quaternion_components(entry, path, index, boneName)
            for index, entry in enumerate(sequence)
        ]
        normalised[boneName] = formatted
        frameCount = len(formatted)
        if expectedFrameCount is None:
            expectedFrameCount = frameCount
        elif frameCount != expectedFrameCount:
            raise ValueError(
                f"{path}: inconsistent frame count for '{boneName}' ({frameCount} != "
                f"{expectedFrameCount})"
            )
    if expectedFrameCount is None:
        raise ValueError(f"{path}: aucune rotation détectée")
    return normalised


def _format_quaternion_components(
    quaternion: Any,
    path: Path,
    frameIndex: int,
    boneName: str,
) -> str:
    """Serialise a quaternion value into the legacy pipe-separated format.

    Parameters
    ----------
    quaternion : Any
        Raw quaternion representation (string, iterable or numpy array).
    path : Path
        Path of the source JSON file.
    frameIndex : int
        Index of the frame where the quaternion originates.
    boneName : str
        Name of the bone associated with the quaternion.

    Returns
    -------
    str
        Quaternion components encoded as ``"x|y|z|w"``.

    Raises
    ------
    ValueError
        If the quaternion cannot be decoded into four numeric components.
    """

    if isinstance(quaternion, str):
        components = [float(value) for value in quaternion.split("|")]
    elif isinstance(quaternion, np.ndarray):
        components = [float(value) for value in quaternion.tolist()]
    elif isinstance(quaternion, Sequence):
        components = [float(value) for value in quaternion]
    else:
        raise ValueError(
            f"{path}: frame {frameIndex} bone '{boneName}' quaternion type non supporté"
        )
    if len(components) != 4:
        raise ValueError(
            f"{path}: frame {frameIndex} bone '{boneName}' requires 4 components"
        )
    return "|".join(f"{component:.10g}" for component in components)


def loadPromptsFile(jsonPath: Path) -> List[Dict[str, str]]:
    """Load prompt descriptors from prompt JSON files."""
    with open(jsonPath, "r", encoding="utf-8") as fileHandle:
        rawContent = json.load(fileHandle)
    if isinstance(rawContent, list):
        return rawContent
    if isinstance(rawContent, dict) and all(
        isinstance(value, dict) for value in rawContent.values()
    ):
        return list(rawContent.values())
    if isinstance(rawContent, dict):
        return [rawContent]
    raise ValueError(f"Format prompt non supporté pour {jsonPath}")


def extractBoneNames(rotations: Dict[str, List[str]]) -> List[str]:
    """Retrieve the ordered bone list from a rotation dictionary."""

    return list(rotations.keys())


def convertRotationsToTensor(
    rotations: Dict[str, List[str]],
) -> Tuple[Tensor, List[str]]:
    """Convert pipe-separated quaternions to a torch tensor."""

    bones = extractBoneNames(rotations)
    frameCount = len(next(iter(rotations.values())))
    boneCount = len(bones)
    quaternionArray = np.zeros((frameCount, boneCount, 4), dtype=np.float32)
    for boneIndex, boneName in enumerate(bones):
        sequence = rotations[boneName]
        if len(sequence) != frameCount:
            raise ValueError(f"Nombre de frames incohérent pour {boneName}")
        for frameIndex, quaternionText in enumerate(sequence):
            components = [float(component) for component in quaternionText.split("|")]
            if len(components) != 4:
                raise ValueError(
                    "Chaque quaternion doit contenir exactement quatre valeurs."
                )
            quaternionArray[frameIndex, boneIndex, :] = components
    tensor = torch.from_numpy(quaternionArray)
    return tensor, bones


def rotationsTo6d(rotations: Dict[str, List[str]]) -> Tensor:
    """Utility to convert JSON rotations directly to 6D representation."""

    quaternionTensor, _ = convertRotationsToTensor(rotations)
    return QuaternionConverter.rotation6dFromQuaternion(quaternionTensor)
