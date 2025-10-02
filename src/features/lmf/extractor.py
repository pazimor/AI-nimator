#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LMF Extractor (v1)
- Entrée : animation.json (meta,bones,frames), quaternions [x,y,z,w], root_pos.
- Sortie : animation.lmf.json (schema LMF ci-dessus).

Fonctions clefs:
- Downsample -> fps_internal (par défaut 15 fps)
- Angles Euler sur sous-ensemble d'articulations (épaule/coude/hanche/genou/cheville + tronc/tête)
- Énergie (∑ vitesses angulaires^2 + vitesse racine pondérée)
- Segmentation par seuils avec hystérésis (lissage Gaussien si SciPy dispo, sinon moyenne glissante)
- Détection : steps (cycles cuisse), spins (yaw cumulé), sauts (proxy), sway (tronc latéral), rythme (FFT dominante)
- Snapshots de posture : 3 keyframes/segment (début, milieu, fin)

Dépendances : numpy (obligatoire), scipy.ndimage (optionnel). Le script fonctionne sans SciPy.
"""
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.shared.constants import (
    DEFAULT_FPS,
    LMF_JOINTS_SUBSET,
    LMF_POSE_EXPORT,
)
from src.shared.types import Quaternion

# ------------------------- utils --------------------------------------------

def quat_to_euler_xyz(q: List[float]) -> Tuple[float, float, float]:
    quat = Quaternion.from_iterable(q).normalized()
    return quat.to_euler_xyz()


def unwrap_deg(d: np.ndarray) -> np.ndarray:
    # wrap to [-180,180]
    d = (d + 180.0) % 360.0 - 180.0
    return d


def deriv(x: np.ndarray, dt: float) -> np.ndarray:
    if len(x) < 2:
        return np.zeros_like(x)
    dx = np.diff(x, axis=0, prepend=x[0:1]) / max(dt, 1e-6)
    return dx


# --------------------- core extraction --------------------------------------


def load_anim(path: Path | str) -> Dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def to_euler_deg(frames: List[Dict[str, Any]], joints: List[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for j in joints:
        eulers = []
        for fr in frames:
            if j not in fr:
                eulers.append([0.0, 0.0, 0.0])
            else:
                e = quat_to_euler_xyz(fr[j])
                eulers.append([math.degrees(e[0]), math.degrees(e[1]), math.degrees(e[2])])
        out[j] = np.array(eulers, dtype=np.float32)
    return out


def downsample_indices(n_src: int, fps_src: float, fps_tgt: float) -> np.ndarray:
    if fps_tgt >= fps_src:
        return np.arange(n_src)
    step = max(int(round(fps_src / fps_tgt)), 1)
    return np.arange(0, n_src, step)


def compute_energy(eul_deg: Dict[str, np.ndarray], root_pos: np.ndarray, fps: float) -> np.ndarray:
    dt = 1.0 / fps
    # angular energy
    ang_v_sq = None
    for arr in eul_deg.values():
        d = deriv(arr, dt)  # deg/s for (roll,pitch,yaw)
        l2 = np.linalg.norm(d, axis=1)
        if ang_v_sq is None:
            ang_v_sq = l2 ** 2
        else:
            ang_v_sq += l2 ** 2
    # linear root term (scaled up to matter)
    v = deriv(root_pos, dt)
    v_xy = np.linalg.norm(v[:, [0, 2]], axis=1)
    energy = ang_v_sq + (v_xy * 200.0) ** 2
    return energy


def segment_from_energy(E: np.ndarray, fps: float) -> List[Tuple[int, int]]:
    # simple hysteresis thresholding on smoothed energy
    if len(E) == 0:
        return [(0, 0)]
    try:
        from scipy.ndimage import gaussian_filter1d  # type: ignore
        Es = gaussian_filter1d(E, sigma=max(1.0, 0.5 * fps))
    except Exception:
        # fallback: moving average
        k = max(int(0.5 * fps), 1)
        kernel = np.ones(k) / k
        Es = np.convolve(E, kernel, mode="same")
    thr_on = max(float(np.percentile(Es, 60)), float(np.mean(Es) * 1.2))
    thr_off = thr_on * 0.6
    segs: List[Tuple[int, int]] = []
    in_seg = False
    s = 0
    for i in range(len(Es)):
        if not in_seg and Es[i] > thr_on:
            in_seg = True
            s = i
        elif in_seg and Es[i] < thr_off:
            in_seg = False
            segs.append((s, i))
    if in_seg:
        segs.append((s, len(Es) - 1))
    if not segs:
        segs = [(0, len(Es) - 1)]
    return segs


def detect_steps(thigh_pitch_L: np.ndarray, thigh_pitch_R: np.ndarray, fps: float) -> Tuple[int, int]:
    # very simple cycle count on derivative; robust enough for gait-like cycles
    dt = 1.0 / fps
    dL = deriv(thigh_pitch_L, dt)
    dR = deriv(thigh_pitch_R, dt)

    def count_zero_cross(d: np.ndarray) -> int:
        if len(d) < 3:
            return 0
        mag = np.abs(d)
        if np.all(mag == 0):
            return 0
        th = np.percentile(mag, 75)
        mask = mag > th
        signs = np.sign(d) * mask
        crossings = np.sum((signs[1:] > 0) & (signs[:-1] <= 0))
        return int(crossings)

    return count_zero_cross(dL), count_zero_cross(dR)


def detect_spins(yaw_deg: np.ndarray) -> Tuple[float, float]:
    dy = unwrap_deg(np.diff(yaw_deg, prepend=yaw_deg[:1]))
    total = float(np.sum(dy))
    net = float(yaw_deg[-1] - yaw_deg[0]) if len(yaw_deg) else 0.0
    return total, net


def detect_jump(root_y: np.ndarray, fps: float) -> Tuple[int, float]:
    # proxy: look for brief upward then downward spikes in root_y velocity
    if len(root_y) < 3:
        return 0, 0.0
    vy = deriv(root_y, 1.0 / fps)
    up = vy > np.percentile(vy, 90)
    down = vy < np.percentile(vy, 10)
    count = int(max(0, min(np.sum(up), np.sum(down)) // 2))
    return count, 0.0


def dominant_freq(signal: np.ndarray, fps: float):
    if len(signal) < 8:
        return None
    s = signal - np.mean(signal)
    if np.allclose(s, 0.0):
        return None
    n = int(2 ** np.ceil(np.log2(len(s))))
    S = np.fft.rfft(s, n=n)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    idx = int(np.argmax(np.abs(S[1:])) + 1)
    return float(freqs[idx]) if idx < len(freqs) else None


def snapshots(eulers: Dict[str, np.ndarray], t: np.ndarray, k: int = 3) -> List[int]:
    # pick k frames: start/mid/end by default
    if len(t) == 0:
        return []
    idxs = [0, len(t) // 2, len(t) - 1]
    idxs = sorted(set(int(i) for i in idxs))[:k]
    return idxs


def quantize_deg(x: float, step: int = 1) -> int:
    return int(round(x / step) * step)


def build_pose(fr_idx: int, eul: Dict[str, np.ndarray]) -> Dict[str, Any]:
    # assemble compact posture fields from POSE_EXPORT
    data: Dict[str, Any] = {}
    # spine/head
    data["posture"] = {
        "spine_flex": quantize_deg(eul[LMF_POSE_EXPORT["spine_flex"][0]][fr_idx, LMF_POSE_EXPORT["spine_flex"][1]]),
        "spine_side": quantize_deg(eul[LMF_POSE_EXPORT["spine_side"][0]][fr_idx, LMF_POSE_EXPORT["spine_side"][1]]),
        "head_pitch": quantize_deg(eul[LMF_POSE_EXPORT["head_pitch"][0]][fr_idx, LMF_POSE_EXPORT["head_pitch"][1]]),
        "head_yaw": quantize_deg(eul[LMF_POSE_EXPORT["head_yaw"][0]][fr_idx, LMF_POSE_EXPORT["head_yaw"][1]]),
    }
    data["arms"] = {
        "L": {
            "shoulder_flex": quantize_deg(eul[LMF_POSE_EXPORT["L_shoulder_flex"][0]][fr_idx, LMF_POSE_EXPORT["L_shoulder_flex"][1]]),
            "abduction":     quantize_deg(eul[LMF_POSE_EXPORT["L_abduction"][0]][fr_idx, LMF_POSE_EXPORT["L_abduction"][1]]),
            "elbow_flex":    quantize_deg(eul[LMF_POSE_EXPORT["L_elbow_flex"][0]][fr_idx, LMF_POSE_EXPORT["L_elbow_flex"][1]]),
        },
        "R": {
            "shoulder_flex": quantize_deg(eul[LMF_POSE_EXPORT["R_shoulder_flex"][0]][fr_idx, LMF_POSE_EXPORT["R_shoulder_flex"][1]]),
            "abduction":     quantize_deg(eul[LMF_POSE_EXPORT["R_abduction"][0]][fr_idx, LMF_POSE_EXPORT["R_abduction"][1]]),
            "elbow_flex":    quantize_deg(eul[LMF_POSE_EXPORT["R_elbow_flex"][0]][fr_idx, LMF_POSE_EXPORT["R_elbow_flex"][1]]),
        },
    }
    data["legs"] = {
        "L": {
            "hip_flex":   quantize_deg(eul[LMF_POSE_EXPORT["L_hip_flex"][0]][fr_idx, LMF_POSE_EXPORT["L_hip_flex"][1]]),
            "knee_flex":  quantize_deg(eul[LMF_POSE_EXPORT["L_knee_flex"][0]][fr_idx, LMF_POSE_EXPORT["L_knee_flex"][1]]),
            "ankle_flex": quantize_deg(eul[LMF_POSE_EXPORT["L_ankle_flex"][0]][fr_idx, LMF_POSE_EXPORT["L_ankle_flex"][1]]),
        },
        "R": {
            "hip_flex":   quantize_deg(eul[LMF_POSE_EXPORT["R_hip_flex"][0]][fr_idx, LMF_POSE_EXPORT["R_hip_flex"][1]]),
            "knee_flex":  quantize_deg(eul[LMF_POSE_EXPORT["R_knee_flex"][0]][fr_idx, LMF_POSE_EXPORT["R_knee_flex"][1]]),
            "ankle_flex": quantize_deg(eul[LMF_POSE_EXPORT["R_ankle_flex"][0]][fr_idx, LMF_POSE_EXPORT["R_ankle_flex"][1]]),
        },
    }
    # hand proximity hints (coarse, proxy via elbow/shoulder flex)
    LnearTorso = data["arms"]["L"]["shoulder_flex"] > 20 and data["arms"]["L"]["elbow_flex"] > 45
    RnearTorso = data["arms"]["R"]["shoulder_flex"] > 20 and data["arms"]["R"]["elbow_flex"] > 45
    data["hands"] = {
        "L": {"near_head": False, "near_torso": bool(LnearTorso)},
        "R": {"near_head": False, "near_torso": bool(RnearTorso)},
    }
    return data


def extract_lmf(anim_path: Path | str, out_path: Path | str, fps_internal: float = 15.0) -> Dict[str, Any]:
    anim_path = Path(anim_path)
    out_path = Path(out_path)
    anim = load_anim(str(anim_path))
    frames = anim.get("frames", [])
    if not frames:
        out = {"meta": {"version": "lmf-1", "frames": 0}, "segments": []}
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return out

    fps_src = float(anim.get("meta", {}).get("fps", 60.0))
    idx = downsample_indices(len(frames), fps_src, fps_internal)
    frames_ds = [frames[int(i)] for i in idx]
    t = np.arange(len(frames_ds)) / fps_internal

    # arrays
    root = np.array([fr.get("root_pos", [0.0, 0.0, 0.0]) for fr in frames_ds], dtype=np.float32)
    eul = to_euler_deg(frames_ds, LMF_JOINTS_SUBSET + ["pelvis", "head", "spine3", "thigh_l", "thigh_r"])  # ensure keys

    # energy & segmentation
    E = compute_energy({k: eul[k] for k in LMF_JOINTS_SUBSET}, root, fps_internal)
    seg_idx = segment_from_energy(E, fps_internal)

    # global features
    disp = root[-1] - root[0]
    v = deriv(root, 1.0 / fps_internal)
    v_xy = np.linalg.norm(v[:, [0, 2]], axis=1)
    yaw = eul.get("pelvis", eul.get("spine3")).copy()[:, 2]
    yaw_total, yaw_net = detect_spins(yaw)

    # rhythm
    dom_hz = dominant_freq(E, fps_internal)

    # body usage proxies (upper vs lower)
    upper_mov = np.linalg.norm(deriv(eul["upperarm_l"], 1.0 / fps_internal), axis=1).mean() \
              + np.linalg.norm(deriv(eul["upperarm_r"], 1.0 / fps_internal), axis=1).mean()
    lower_mov = np.linalg.norm(deriv(eul["thigh_l"], 1.0 / fps_internal), axis=1).mean() \
              + np.linalg.norm(deriv(eul["thigh_r"], 1.0 / fps_internal), axis=1).mean()
    denom = max(upper_mov + lower_mov, 1e-6)

    out: Dict[str, Any] = {
        "meta": {
            "version": "lmf-1",
            "source": anim.get("meta", {}).get("source", os.path.basename(anim_path)),
            "skeleton": "SMPL-22",
            "fps_source": fps_src,
            "fps_internal": fps_internal,
            "duration_s": float(len(frames_ds) / fps_internal),
            "frames": int(len(frames_ds)),
        },
        "global": {
            "root": {
                "displacement": [float(disp[0]), float(disp[1]), float(disp[2])],
                "path_len": float(np.sum(np.linalg.norm(np.diff(root[:, [0, 2]], axis=0, prepend=root[0:1, [0, 2]]), axis=1))),
                "avg_speed": float(v_xy.mean()),
                "max_speed": float(v_xy.max()),
                "yaw_total_deg": float(yaw_total),
                "yaw_net_deg": float(yaw_net),
            },
            "energy": {"mean": float(E.mean()), "p95": float(np.percentile(E, 95))},
            "rhythm": {"cadence_hz": None, "dominant_hz": dom_hz, "stability": 0.0},
            "body_use": {
                "upper": float(upper_mov / denom),
                "lower": float(lower_mov / denom),
                "left":  0.5,
                "right": 0.5,
            },
        },
        "segments": [],
        "events": [],
        "llm_hints": {"verbs": [], "style": [], "intensity": "unknown", "short_caption": ""},
    }

    # per-segment details
    n_frames = len(t)
    if n_frames == 0:
        seg_idx = []

    for s0, s1 in seg_idx:
        if s0 >= n_frames:
            continue
        s1 = max(s0, min(s1, n_frames - 1))
        t0 = float(t[s0])
        t1 = float(t[s1])
        seg: Dict[str, Any] = {
            "t0": t0, "t1": t1,
            "labels": [],
            "motion": {"turn_deg": 0.0, "steps_L": 0, "steps_R": 0, "jump": {"count": 0, "max_hang_s": 0.0},
                        "sway_amp": 0.0, "sway_hz": None},
            "poses": [],
            "contacts": {"cadence_hz": None, "L": [], "R": []},
            "orientation_polyline": [],
        }
        # compute subarrays
        yaw_seg = yaw[s0:s1+1]
        yaw_total_seg, yaw_net_seg = detect_spins(yaw_seg)
        seg["motion"]["turn_deg"] = float(yaw_net_seg)

        # steps
        stepsL, stepsR = detect_steps(eul["thigh_l"][s0:s1+1, 1], eul["thigh_r"][s0:s1+1, 1], fps_internal)
        seg["motion"]["steps_L"] = stepsL
        seg["motion"]["steps_R"] = stepsR

        # sway (use spine3 roll amplitude)
        spine_roll = eul["spine3"][s0:s1+1, 0]
        if len(spine_roll) > 1:
            seg["motion"]["sway_amp"] = float(np.max(spine_roll) - np.min(spine_roll))
            seg["motion"]["sway_hz"] = dominant_freq(spine_roll, fps_internal)

        # snapshots
        for i in snapshots(eul, t[s0:s1+1], k=3):
            fr_idx = s0 + i
            pose = build_pose(fr_idx, eul)
            pose["t"] = float(t[fr_idx])
            seg["poses"].append(pose)

        # simple labels based on heuristics
        energy_seg = float(np.mean(E[s0:s1+1]))
        moving = energy_seg > max(float(np.mean(E)) * 0.75, float(np.percentile(E, 30)))
        if moving:
            seg["labels"].append({"name": "moving", "score": 0.7})
        else:
            seg["labels"].append({"name": "idle", "score": 0.7})
        if abs(seg["motion"]["turn_deg"]) > 45:
            seg["labels"].append({"name": "turn", "score": 0.6})
        if stepsL + stepsR >= 2:
            seg["labels"].append({"name": "walk_like", "score": 0.6})

        # tiny orientation polyline (start/mid/end)
        XY = root[s0:s1+1][:, [0, 2]]
        pts = [XY[0].tolist(), XY[len(XY)//2].tolist(), XY[-1].tolist()]
        seg["orientation_polyline"] = [
            [float(pts[0][0]), float(pts[0][1]), float(yaw_seg[0])],
            [float(pts[1][0]), float(pts[1][1]), float(yaw_seg[len(yaw_seg)//2])],
            [float(pts[2][0]), float(pts[2][1]), float(yaw_seg[-1])],
        ]

        out["segments"].append(seg)

    # crude caption seeds
    if len(out["segments"]) == 1 and out["segments"][0]["labels"][0]["name"] == "idle":
        out["llm_hints"]["short_caption"] = "Personnage immobile (posture debout/assise), légères variations locales."
        out["llm_hints"]["intensity"] = "low"
    else:
        out["llm_hints"]["short_caption"] = "Séquence avec mouvements articulaires, voir poses et steps."
        out["llm_hints"]["intensity"] = "medium"

    # write
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


__all__ = ["extract_lmf"]
