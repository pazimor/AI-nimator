// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"

class UPoseableMeshComponent;

/**
 * Applies FRigBinder's per-frame retarget output onto a live
 * `UPoseableMeshComponent` (`apps/spec/rig_binding.md` §2.2 "Application
 * au squelette").
 *
 * Design decision (v1, documented per the task's request for an explicit
 * choice): **UPoseableMeshComponent**, not a custom `FAnimNode`.
 * Rationale:
 * - `UPoseableMeshComponent` exposes `SetBoneRotationByName` /
 *   `SetBoneLocationByName` directly from game/C++ code, with no
 *   AnimBlueprint graph to author, compile, or keep in sync with the
 *   RigMap — the whole B3-bis pipeline (Runtime -> PostProcess ->
 *   RigBinder) is already pure C++, so driving bone transforms directly
 *   is the lowest-friction, most robust integration point, and the one
 *   most amenable to unit-testing the math in isolation (this file is
 *   the ONLY piece that actually touches the component; everything
 *   upstream is engine-independent).
 * - A custom `FAnimNode` would require an AnimBlueprint asset per
 *   character (or a native anim instance class), UE's anim graph
 *   threading/eval model (which changed materially across 5.x), and
 *   compiling against `AnimGraphRuntime`/`Engine` anim internals that
 *   are harder to verify without a running editor — higher risk for a
 *   first validation pass under "Unreal non exécutable ici".
 * - Trade-off acknowledged: `UPoseableMeshComponent` bypasses the
 *   AnimBlueprint pipeline entirely (no blending with other
 *   Montages/state machines through the normal anim graph). A project
 *   that needs to blend AI-nimator's output with hand-authored
 *   AnimBlueprint content should upgrade to a custom `FAnimNode` later,
 *   OR use `UPoseableMeshComponent`'s pose as an input pose passed into
 *   an anim graph via `FPoseSnapshot`/`SetSnapshotPose`-style plumbing
 *   (out of scope here — flagged as an open question in the session
 *   report, not invented here).
 *
 * `SetBoneRotationByName` on `UPoseableMeshComponent` takes rotations in
 * `EBoneSpaces::LocalSpace` when asked (see `EBoneSpaces`), matching
 * exactly what `FRigBinder::RetargetFrame` computes.
 */
namespace AInimatorPoseableMeshApplier
{
	/**
	 * Writes one frame's retargeted local bone rotations onto
	 * PoseableMesh via `SetBoneRotationByName(..., EBoneSpaces::
	 * LocalSpace)`, one call per (BoneNames[i], BoneLocalRotations[i])
	 * pair. Silently skips any BoneNames[i] not found on the mesh's
	 * skeleton (logged once per unique missing name per call to avoid
	 * spamming — see the .cpp) rather than asserting, since a RigMap
	 * authored against a slightly different skeleton variant should
	 * degrade gracefully, not crash the game.
	 *
	 * Parameters
	 * ----------
	 * PoseableMesh : target component, already attached to a
	 *     SkeletalMesh whose skeleton the RigMap was authored against.
	 * BoneNames / BoneLocalRotations : parallel arrays, same length,
	 *     as produced by FRigBinder::RetargetFrame.
	 */
	void ApplyFrame(
		UPoseableMeshComponent* PoseableMesh,
		const TArray<FName>& BoneNames,
		const TArray<FQuat>& BoneLocalRotations);
}
