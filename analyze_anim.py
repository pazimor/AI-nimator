"""Quick animation diagnostic script."""
import json
import numpy as np

with open(r"C:\Users\eddy.hummel\Desktop\output\my_animation_raw.json", "r") as f:
    data = json.load(f)

meta = data["meta"]
print("META:", json.dumps(meta, indent=2))
print()

bones = data["bones"]
for i, bone in enumerate(bones):
    frames = bone["frames"]
    rotations = np.array([f["rotation"] for f in frames])
    name = bone["name"]
    length = bone["length"]
    print(
        f"Bone {i:2d} ({name:15s}): length={length:.4f}, "
        f"frames={rotations.shape}, range=[{rotations.min():.4f}, {rotations.max():.4f}], "
        f"std={rotations.std():.6f}"
    )

print()
print("--- Temporal analysis ---")
for i in range(len(bones)):
    frames = bones[i]["frames"]
    rotations = np.array([f["rotation"] for f in frames])
    if rotations.shape[0] > 1:
        vel = np.diff(rotations, axis=0)
        name = bones[i]["name"]
        print(
            f"Bone {i:2d} ({name:15s}): mean_vel={np.abs(vel).mean():.6f}, "
            f"max_vel={np.abs(vel).max():.6f}"
        )

print()
extras = data["extras"]
print("Prompt:", extras.get("prompt"))
print("Checkpoint:", extras.get("checkpoint"))
print("DDIM steps:", extras.get("ddimSteps"))

if "trans" in extras:
    t = np.array(extras["trans"])
    print(f"Translation: shape={t.shape}, range=[{t.min():.4f}, {t.max():.4f}], std={t.std():.6f}")
    if t.shape[0] > 1:
        tvel = np.diff(t, axis=0)
        print(f"Trans velocity: mean={np.abs(tvel).mean():.6f}, max={np.abs(tvel).max():.6f}")

print("\n--- Frozen Frame Detection ---")
total_bones = len(bones)
frozen_bones = 0
for i, bone in enumerate(bones):
    rotations = np.array([f["rotation"] for f in bone["frames"]])
    if rotations.shape[0] > 1:
        vel = np.diff(rotations, axis=0)
        if np.abs(vel).max() < 0.001:
            frozen_bones += 1
            print(f"  FROZEN: Bone {i} ({bone['name']})")
print(f"\n{frozen_bones}/{total_bones} bones are effectively frozen")

print("\n--- Frame Comparison (first vs middle vs last) ---")
mid = len(bones[0]["frames"]) // 2
for i, bone in enumerate(bones):
    frames = bone["frames"]
    r0 = np.array(frames[0]["rotation"])
    rm = np.array(frames[mid]["rotation"])
    rl = np.array(frames[-1]["rotation"])
    d_start_mid = np.sqrt(np.sum((r0 - rm) ** 2))
    d_mid_end = np.sqrt(np.sum((rm - rl) ** 2))
    d_start_end = np.sqrt(np.sum((r0 - rl) ** 2))
    print(
        f"Bone {i:2d} ({bone['name']:15s}): "
        f"start-mid={d_start_mid:.4f}, mid-end={d_mid_end:.4f}, start-end={d_start_end:.4f}"
    )

print("\n--- Rotation Format ---")
r0 = np.array(bones[0]["frames"][0]["rotation"])
print(f"Rotation channels: {len(r0)} (values: {r0})")
if len(r0) == 6:
    print("Format: 6D rotation (PROBLEM - should be quaternion for animation!)")
elif len(r0) == 4:
    norm = np.sqrt(np.sum(r0 ** 2))
    print(f"Format: quaternion, norm={norm:.4f}")
