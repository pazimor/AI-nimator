// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"
#include "AInimatorSmpl22Skeleton.h"

/**
 * Minimal forward-kinematics solver for the SMPL-22 skeleton: converts
 * a rotation6d bone frame into root-local joint world-orientation
 * rotations + positions. Ports the strict necessary subset of
 * `ainimator/geometry/components/ops.py::rot6dToJointXYZ` /
 * `sixdToRotationMatrix` to C++ (Goal B vérité #1: the plugin reads the
 * model's output, it never redefines geometry independently — this is
 * a straight port, mirroring
 * `apps/unity-sentis/.../PostProcess/SmplForwardKinematics.cs` exactly).
 *
 * Joint positions returned by ComputeJointPositions are in the
 * **root-local frame**: the pelvis (bone 0) sits at the origin, and
 * every other joint's position is relative to it. Its rotation6d
 * already encodes the skeleton's actual world-facing orientation
 * (`ainimator.geometry.root_local`: pelvis yaw is extracted directly
 * from rot6d[0], never re-derived from the engine's integrated yaw) —
 * so callers add the engine's integrated world-space root position
 * (never re-apply the integrated yaw as an extra rotation; see
 * RootLocalToWorld).
 */
namespace AInimatorForwardKinematics
{
	/**
	 * Convert one bone's 6D rotation representation to a 3x3 rotation
	 * matrix (stored in an FMatrix, translation/last row/col unused)
	 * via Gram-Schmidt orthogonalization — matches sixdToRotationMatrix
	 * exactly: columns are (b1, b2, b3) with b1 = normalize(a1),
	 * b2 = normalize(a2 - (b1.a2)b1), b3 = b1 x b2.
	 *
	 * Parameters
	 * ----------
	 * SixD : 6 values [a1.x, a1.y, a1.z, a2.x, a2.y, a2.z].
	 */
	inline void SixDToRotationMatrix(const float SixD[6], FMatrix& OutRotation)
	{
		const FVector A1(SixD[0], SixD[1], SixD[2]);
		const FVector A2(SixD[3], SixD[4], SixD[5]);

		const FVector B1 = A1.GetSafeNormal();
		const float Dot = FVector::DotProduct(B1, A2);
		const FVector B2 = (A2 - Dot * B1).GetSafeNormal();
		const FVector B3 = FVector::CrossProduct(B1, B2);

		OutRotation = FMatrix::Identity;
		// Columns, matching torch.stack([b1, b2, b3], dim=-1).
		OutRotation.SetColumn(0, B1);
		OutRotation.SetColumn(1, B2);
		OutRotation.SetColumn(2, B3);
	}

	/**
	 * Project every bone's accumulated 6D rotation back onto the
	 * manifold of valid representations, in place: the same
	 * Gram-Schmidt as SixDToRotationMatrix, keeping (b1, b2) as the new
	 * 6D value. Normative rollout step (apps/spec/inference_contract.md
	 * §3.6, parity with the Python reference orthonormalizeRot6d and
	 * the Unity runtime): applied to the accumulated bone frame BEFORE
	 * it enters the state window — without it, long rollouts drift
	 * off-manifold and the pose degenerates. Idempotent on valid 6D.
	 */
	inline void OrthonormalizeFrame(TArray<float>& BoneFrame)
	{
		for (int32 Offset = 0; Offset + 6 <= BoneFrame.Num(); Offset += 6)
		{
			const FVector A1(BoneFrame[Offset], BoneFrame[Offset + 1], BoneFrame[Offset + 2]);
			const FVector A2(BoneFrame[Offset + 3], BoneFrame[Offset + 4], BoneFrame[Offset + 5]);

			const FVector B1 = A1.GetSafeNormal();
			const float Dot = FVector::DotProduct(B1, A2);
			const FVector B2 = (A2 - Dot * B1).GetSafeNormal();

			BoneFrame[Offset] = static_cast<float>(B1.X);
			BoneFrame[Offset + 1] = static_cast<float>(B1.Y);
			BoneFrame[Offset + 2] = static_cast<float>(B1.Z);
			BoneFrame[Offset + 3] = static_cast<float>(B2.X);
			BoneFrame[Offset + 4] = static_cast<float>(B2.Y);
			BoneFrame[Offset + 5] = static_cast<float>(B2.Z);
		}
	}

	/**
	 * Compute root-local global rotations and positions for every bone
	 * in BoneFrame (row-major, NumBones * 6), following
	 * AInimatorSmpl22Skeleton::ParentIndices / BoneOffsets. Both output
	 * arrays must already be sized NumBones (caller-owned, no per-frame
	 * allocation on the hot path).
	 */
	inline void ComputeJointPositions(
		const TArray<float>& BoneFrame,
		int32 NumBones,
		TArray<FMatrix>& OutGlobalRotations,
		TArray<FVector>& OutGlobalPositions)
	{
		check(BoneFrame.Num() == NumBones * AInimatorSmpl22Skeleton::RotationChannelsPerBone);
		check(OutGlobalRotations.Num() == NumBones);
		check(OutGlobalPositions.Num() == NumBones);

		for (int32 Bone = 0; Bone < NumBones; ++Bone)
		{
			float SixD[6];
			const int32 Base = Bone * AInimatorSmpl22Skeleton::RotationChannelsPerBone;
			for (int32 Channel = 0; Channel < 6; ++Channel)
			{
				SixD[Channel] = BoneFrame[Base + Channel];
			}
			FMatrix LocalRotation;
			SixDToRotationMatrix(SixD, LocalRotation);

			const int32 Parent = AInimatorSmpl22Skeleton::ParentIndices[Bone];
			if (Parent < 0)
			{
				OutGlobalRotations[Bone] = LocalRotation;
				OutGlobalPositions[Bone] = AInimatorSmpl22Skeleton::BoneOffsets[Bone];
				continue;
			}

			const FMatrix& ParentRotation = OutGlobalRotations[Parent];
			const FVector& ParentPosition = OutGlobalPositions[Parent];
			OutGlobalRotations[Bone] = ParentRotation * LocalRotation;
			const FVector ChildOffset =
				ParentRotation.TransformVector(AInimatorSmpl22Skeleton::BoneOffsets[Bone]);
			OutGlobalPositions[Bone] = ParentPosition + ChildOffset;
		}
	}

	/**
	 * Transform a root-local joint position (from
	 * ComputeJointPositions) into world space, given the engine's
	 * integrated root world position. No extra yaw rotation is applied
	 * here: the pelvis (bone 0)'s own rotation6d already carries the
	 * skeleton's world-facing orientation (`ainimator.geometry.root_local`),
	 * so the root-local FK positions are already expressed with the
	 * correct world orientation baked in — only a translation by the
	 * integrated root position remains, matching
	 * `controller_sequences.py::deriveFootContacts`'s
	 * `worldFoot = footXyz + rootTranslation`.
	 */
	inline FVector RootLocalToWorld(const FVector& RootLocalPosition, const FVector& RootWorldPosition)
	{
		return RootLocalPosition + RootWorldPosition;
	}
}
