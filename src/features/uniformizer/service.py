#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uniformizer.py — AMASS support + retarget + validation + resample
Version 1.10 — 2025-08-09

Fixes
- All quaternions becoming identity when source is SMPL-H (52 joints).
  We now provide an explicit **index-based retarget** for SMPL-H -> SMPL-22.
  The retarget mapping accepts **names OR integer indices**.

What's included (same as v1.9 + fix):
- JSON format "rotroot" (default): frames with "root_pos" + per-bone quats as arrays.
- --resample <fps>: SLERP quats, linear root.
- --flat & --prompts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np

from src.shared.animation_io import (
    AnimationIOError,
    load_animation_json,
    load_animation_npz,
)
from src.shared.constants import (
    DEFAULT_FPS,
    EPSILON,
    SMPL22_BONES,
    SMPL24_BONES,
    SMPL22_NAME_ALIASES,
    SMPLH_CORE_INDEX_MAP,
)
from src.shared.naming import safe_slug, short_hash
from src.shared.types import UniformizerDirectoryJob, UniformizerJob

# ------------------------------
# Errors & helpers
# ------------------------------

class UniformizerError(Exception): pass
def _fail(msg: str) -> None: raise UniformizerError(msg)

# ------------------------------
# Quaternions
# ------------------------------

def qnormalize(q: np.ndarray, eps: float = EPSILON) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.where(np.isfinite(n), n, 0.0)
    n = np.maximum(n, eps)
    return q / n

def qsafe(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    bad = ~np.isfinite(n) | (n < 1e-6)
    q = q.copy()
    if np.any(bad):
        q[bad.squeeze(-1)] = np.array([0.0,0.0,0.0,1.0], dtype=q.dtype)
    return qnormalize(q)

def qmul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1,y1,z1,w1 = np.split(q1, 4, axis=-1)
    x2,y2,z2,w2 = np.split(q2, 4, axis=-1)
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return qsafe(np.concatenate([x,y,z,w], axis=-1))

def aa_to_quat(aa: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    half = 0.5 * theta
    s = np.where(theta > EPSILON, np.sin(half) / (theta + EPSILON), 0.5 + 0 * theta)
    xyz = aa * s
    w = np.cos(half)
    q = np.concatenate([xyz, w], axis=-1)
    return qsafe(q)

def qslerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    q0 = q0.astype(np.float32); q1 = q1.astype(np.float32)
    dot = np.sum(q0*q1, axis=-1, keepdims=True)
    negate = (dot < 0.0).astype(np.float32)
    q1 = q1 * (1.0 - 2.0*negate)
    dot = np.sum(q0*q1, axis=-1, keepdims=True)
    dot = np.clip(dot, -1.0, 1.0)
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)
    close = (sin_omega < 1e-6).astype(np.float32)
    out = ((1.0 - a) * q0 + a * q1)
    s0 = np.sin((1.0 - a) * omega) / (sin_omega + 1e-9)
    s1 = np.sin(a * omega) / (sin_omega + 1e-9)
    out = close * out + (1.0 - close) * (s0 * q0 + s1 * q1)
    return qsafe(out)

# ------------------------------
# Save JSON (rotroot)
# ------------------------------

def save_json_rotroot(path: Path, pos, rot, bones, fps, source=""):
    T,J,_ = pos.shape
    meta = {
        "fps": float(fps), "frames": int(T), "joints": int(J),
        "source": source, "format": "rotroot"
    }
    frames = []
    for t in range(T):
        d = {"root_pos": pos[t,0].astype(float).tolist()}
        for j, name in enumerate(bones):
            d[name] = rot[t, j].astype(float).tolist()
        frames.append(d)
    payload = {"meta": meta, "bones": bones, "frames": frames}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

def save_npz(path: Path, pos, rot, bones, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, positions=pos, rotations=qsafe(rot), bone_names=np.array(bones, dtype=object), fps=float(fps))

# ------------------------------
# Retarget
# ------------------------------

def load_target_map(path: Optional[Path], target_skel: Optional[str]):
    if path:
        try: payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e: _fail(f"Mapping JSON invalide '{path}': {e}")
        target_bones = payload.get("target_bones"); map_dict = payload.get("map", {})
        offset_quat = payload.get("offset_quat", {}); ax = payload.get("axis_correction", {})
        pre_q = ax.get("pre_q"); post_q = ax.get("post_q")
        if not target_bones or not isinstance(target_bones, list): _fail(f"{path}: 'target_bones' liste requise.")
        return target_bones, map_dict, offset_quat, pre_q, post_q

    if target_skel and target_skel.lower() in ("smpl22","smpl-22","smpl_22","smlp-22","smlp22"):
        target_bones = SMPL22_BONES[:]
        aliases = {k: v[:] for k, v in SMPL22_NAME_ALIASES.items()}
        return target_bones, aliases, {}, None, None

    _fail("Aucun target_skel ou mapping fourni. Utilise --target-skel smpl22 ou --target-map mapping.json.")
    return None


def augment_map_with_indices(map_by_name: Dict[str, List[str]], src_joint_count: int) -> Dict[str, List[Union[str,int]]]:
    """
    If the source looks like SMPL-H (J==52), add index mapping for a robust retarget.
    """
    m = {k: (v[:] if isinstance(v, list) else [v]) for k,v in map_by_name.items()}
    if src_joint_count == 52:
        for k, i in SMPLH_CORE_INDEX_MAP.items():
            m.setdefault(k, []).append(i)
    return m

def retarget(pos, rot, src_names, target_bones, map_dict, offset_quat=None, axis_pre_q=None, axis_post_q=None):
    T = rot.shape[0]; Jt = len(target_bones)
    pos_out = np.zeros((T, Jt, 3), dtype=np.float32)
    rot_out = np.concatenate([np.zeros((T,Jt,3),np.float32), np.ones((T,Jt,1),np.float32)], axis=-1)

    pre = None if axis_pre_q is None else np.asarray(axis_pre_q, dtype=np.float32).reshape(1,1,4).repeat(T, axis=0)
    post = None if axis_post_q is None else np.asarray(axis_post_q, dtype=np.float32).reshape(1,1,4).repeat(T, axis=0)

    name2idx = {n.lower(): i for i,n in enumerate(src_names)}
    def find_idx(cands):
        # cands can contain strings (names) or ints (indices)
        if not isinstance(cands, (list, tuple)):
            cands = [cands]
        for c in cands:
            if isinstance(c, int):
                if 0 <= c < len(src_names): return c
                continue
            i = name2idx.get(str(c).lower())
            if i is not None: return i
        return None

    root_src_idx = find_idx(map_dict.get("pelvis", ["pelvis","hips","root","hip"]))
    if root_src_idx is not None: pos_out[:, 0, :] = pos[:, root_src_idx, :]

    for j,tb in enumerate(target_bones):
        idx = find_idx(map_dict.get(tb, [tb]))
        if idx is None: continue
        pos_out[:, j] = pos[:, idx]
        q = rot[:, idx:idx+1, :]
        if pre is not None: q = qmul(pre, q)
        if post is not None: q = qmul(q, post)
        if offset_quat and tb in offset_quat:
            off = np.asarray(offset_quat[tb], dtype=np.float32).reshape(1,1,4).repeat(T, axis=0)
            q = qmul(off, q)
        rot_out[:, j] = q.squeeze(1)

    return pos_out, qsafe(rot_out), target_bones

# ------------------------------
# Resample
# ------------------------------

def resample_clip(pos: np.ndarray, rot: np.ndarray, fps_in: float, fps_out: float) -> Tuple[np.ndarray, np.ndarray]:
    if abs(fps_in - fps_out) < 1e-6:
        return pos, rot
    T_in = pos.shape[0]
    if T_in == 0:
        return pos, rot
    duration = (T_in - 1) / max(fps_in, 1e-6)
    T_out = int(round(duration * fps_out)) + 1
    t_in = np.arange(T_in, dtype=np.float32) / float(fps_in)
    t_out = np.arange(T_out, dtype=np.float32) / float(fps_out)
    src = np.clip(t_out * fps_in, 0, T_in - 1 - 1e-6)
    i0 = np.floor(src).astype(np.int32)
    i1 = np.clip(i0 + 1, 0, T_in - 1)
    a = (src - i0).astype(np.float32)[..., None]

    p0 = pos[i0, 0, :]
    p1 = pos[i1, 0, :]
    root_out = (1.0 - a) * p0 + a * p1

    T,J,_ = rot.shape
    rot_out = np.zeros((T_out, J, 4), dtype=np.float32)
    for j in range(J):
        q0 = rot[i0, j, :]
        q1 = rot[i1, j, :]
        qo = np.zeros_like(q0)
        for t in range(T_out):
            qo[t] = qslerp(q0[t:t+1], q1[t:t+1], float(a[t]))
        rot_out[:, j, :] = qo

    pos_out = np.zeros((T_out, J, 3), dtype=np.float32)
    pos_out[:, 0, :] = root_out
    return pos_out, rot_out

# ------------------------------
# Split & utils
# ------------------------------

def write_prompts_json(folder: Path, meta: dict, overwrite: bool=False):
    path = folder / "prompts.json"
    if path.exists() and not overwrite: return
    tpl = {
        "meta": {
            "source": meta.get("source",""),
            "fps": meta.get("fps", DEFAULT_FPS),
            "frames": meta.get("frames", 0),
            "joints": meta.get("joints", 0)
        },
        "simple": "",
        "detailed": "",
        "expert": ""
    }
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tpl, ensure_ascii=False, indent=2), encoding="utf-8")


def convert_npz_to_json(job: UniformizerJob) -> Tuple[int, float]:
    try:
        pos, rot, names, fps = load_animation_npz(job.input_path)
    except AnimationIOError as exc:
        _fail(str(exc))
    bones = names
    if job.target_skeleton or job.target_map:
        tb, m, off, pre, post = load_target_map(job.target_map, job.target_skeleton)
        m = augment_map_with_indices(m, src_joint_count=rot.shape[1])
        pos, rot, bones = retarget(pos, rot, names, tb, m, off, pre, post)
    if job.resample_fps and job.resample_fps > 0:
        pos, rot = resample_clip(pos, rot, fps, float(job.resample_fps))
        fps = float(job.resample_fps)
    save_json_rotroot(job.output_path, pos, rot, bones, fps, source=str(job.input_path))
    if job.include_prompts:
        meta = {"source": str(job.input_path), "fps": fps, "frames": int(pos.shape[0]), "joints": len(bones)}
        write_prompts_json(job.output_path.parent, meta, overwrite=True)
    return pos.shape[0], float(fps)


def convert_json_to_npz(job: UniformizerJob) -> Tuple[int, float]:
    try:
        pos, rot, bones, fps = load_animation_json(job.input_path)
    except AnimationIOError as exc:
        _fail(str(exc))
    save_npz(job.output_path, pos, rot, bones, fps)
    return pos.shape[0], float(fps)


def convert_directory(job: UniformizerDirectoryJob) -> Tuple[int, int]:
    src = job.input_dir
    if not src.exists():
        _fail(f"Dossier introuvable: {src}")

    outdir = job.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    tb = m = off = pre = post = None
    if job.target_skeleton or job.target_map:
        tb, m, off, pre, post = load_target_map(job.target_map, job.target_skeleton)

    count = 0
    skipped = 0
    for p in src.rglob("*.npz"):
        try:
            pos, rot, names, fps = load_animation_npz(p)
            bones = names
            if tb is not None:
                m2 = augment_map_with_indices(m, src_joint_count=rot.shape[1])
                pos, rot, bones = retarget(pos, rot, names, tb, m2, off, pre, post)
            if job.resample_fps and job.resample_fps > 0:
                pos, rot = resample_clip(pos, rot, fps, float(job.resample_fps))
                fps = float(job.resample_fps)

            if job.flatten_output:
                rel = str(p.relative_to(src))
                h8 = short_hash(rel)
                stem = safe_slug(p.stem)
                folder = outdir / f"{stem}_{h8}"
                folder.mkdir(parents=True, exist_ok=True)
                dst = folder / "animation.json"
                save_json_rotroot(dst, pos, rot, bones, fps, source=str(p))
                if job.include_prompts:
                    meta = {"source": str(p), "fps": fps, "frames": int(pos.shape[0]), "joints": len(bones)}
                    write_prompts_json(folder, meta, overwrite=True)
            else:
                rel = p.relative_to(src).with_suffix("")
                sub = outdir / rel
                sub.mkdir(parents=True, exist_ok=True)
                dst = sub / f"{p.stem}.json"
                save_json_rotroot(dst, pos, rot, bones, fps, source=str(p))
                if job.include_prompts:
                    meta = {"source": str(p), "fps": fps, "frames": int(pos.shape[0]), "joints": len(bones)}
                    write_prompts_json(sub, meta, overwrite=True)
            count += 1
        except UniformizerError as exc:
            print(f"[WARN] skip {p}: {exc}", file=sys.stderr)
            skipped += 1
        except AnimationIOError as exc:
            print(f"[WARN] {p}: {exc}", file=sys.stderr)
            skipped += 1
        except Exception as exc:  # pragma: no cover
            print(f"[EXC] {p}: {exc}", file=sys.stderr)
            skipped += 1
    return count, skipped

# ------------------------------
# Commands
# ------------------------------

def cmd_npz2json(args):
    job = UniformizerJob(
        input_path=Path(args.npz),
        output_path=Path(args.json),
        target_skeleton=args.target_skel,
        target_map=Path(args.target_map) if getattr(args, "target_map", None) else None,
        resample_fps=getattr(args, "resample", None),
        include_prompts=getattr(args, "prompts", False),
    )
    frames, fps = convert_npz_to_json(job)
    print(f"[OK] {args.npz} -> {args.json}  ({frames}f @ {fps}fps)")
    return 0


def cmd_json2npz(args):
    job = UniformizerJob(
        input_path=Path(args.json),
        output_path=Path(args.npz),
    )
    frames, fps = convert_json_to_npz(job)
    print(f"[OK] {args.json} -> {args.npz}  ({frames}f @ {fps}fps)")
    return 0


def cmd_dir2json(args):
    job = UniformizerDirectoryJob(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        target_skeleton=getattr(args, "target_skel", None),
        target_map=Path(args.target_map) if getattr(args, "target_map", None) else None,
        resample_fps=getattr(args, "resample", None),
        flatten_output=getattr(args, "flat", False),
        include_prompts=getattr(args, "prompts", False),
    )
    processed, skipped = convert_directory(job)
    print(f"Terminé. Fichiers traités: {processed}, ignorés: {skipped}")
    return 0
