// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"

/**
 * Root-local motion helpers, mirroring
 * ainimator/geometry/root_local.py::pelvisYawFromRot6d exactly so the
 * yaw extracted from a rotation6d pelvis frame agrees with the Python
 * reference (needed to seed CurrentWorldYaw from a seed window, and
 * for any diagnostics comparing against the reference rollout).
 */
namespace AInimatorRootLocalMath
{
	/** Index of the pelvis bone in the SMPL-22 skeleton (bone 0). */
	constexpr int32 PelvisBoneIndex = 0;

	/**
	 * Extracts the yaw angle (radians) of a pelvis 6D rotation.
	 *
	 * The 6D representation stores the first two columns of the
	 * rotation matrix (Zhou et al. 2019); column 0 is the local
	 * forward/X axis in world space. Yaw is the signed angle around
	 * the Y (up) axis, from atan2(forwardZ, forwardX) — this matches
	 * pelvisYawFromRot6d bit-for-bit in intent (channel order: the
	 * first three of the six values are column 0 = (x, y, z)).
	 *
	 * Parameters
	 * ----------
	 * PelvisRot6d : the 6 rotation channels for the pelvis bone,
	 *     [col0.x, col0.y, col0.z, col1.x, col1.y, col1.z].
	 */
	inline float PelvisYawFromRot6d(const float PelvisRot6d[6])
	{
		const float ForwardX = PelvisRot6d[0];
		const float ForwardZ = PelvisRot6d[2];
		return FMath::Atan2(ForwardZ, ForwardX);
	}
}
