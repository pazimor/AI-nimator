// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"

/**
 * Analytic two-bone IK solver (hip -> knee -> ankle), used by
 * FFootLockIK to hold a locked foot at its captured world position
 * (`apps/spec/footlock_blending.md` §3).
 *
 * Standard law-of-cosines two-bone solve: given a root (hip), a target
 * effector position, an upper/lower bone length and a pole vector (the
 * knee direction in the controller's own pose, never a fixed pole —
 * spec §3.2), returns the corrected knee (mid) and effector (ankle)
 * positions. This does not rotate actual bone transforms — B4's
 * consumer (FFootLockIK) only needs corrected joint **positions** to
 * feed back into a position-space pose correction; wiring this onto an
 * actual Skeleton/AnimGraph node (e.g. a two-bone IK AnimNode) is an
 * engine-integration step left to the host project.
 *
 * Mirrors `apps/unity-sentis/.../PostProcess/TwoBoneIkSolver.cs`
 * step-for-step (Goal B vérité #7 parity).
 */
namespace AInimatorTwoBoneIkSolver
{
	struct FResult
	{
		FVector MidPosition = FVector::ZeroVector;
		FVector EffectorPosition = FVector::ZeroVector;
		bool bTargetReachable = false;
	};

	inline FVector RotateAroundAxis(const FVector& Vector, const FVector& Axis, float AngleRadians)
	{
		const FQuat Rotation(Axis, AngleRadians);
		return Rotation.RotateVector(Vector);
	}

	/**
	 * Solve a two-bone chain so the effector reaches TargetPosition as
	 * closely as possible.
	 *
	 * Parameters
	 * ----------
	 * RootPosition : hip (chain root) world position — fixed, not
	 *     moved by this solve.
	 * MidPosition : current knee world position (used only to derive
	 *     the pole direction).
	 * EffectorPosition : current ankle world position (used only to
	 *     derive the pole direction).
	 * TargetPosition : desired world position for the effector (the
	 *     locked foot position).
	 * UpperLength : hip -> knee bone length (meters).
	 * LowerLength : knee -> ankle bone length (meters).
	 *
	 * The pole vector is derived from the *current controller pose's*
	 * knee direction (never a fixed world-space pole), preserving the
	 * natural bend orientation per spec §3.2. When the target is
	 * unreachable (farther than UpperLength + LowerLength), the chain
	 * is fully extended toward the target along the root->target
	 * direction and FResult::bTargetReachable is false — callers
	 * (FFootLockIK) use this to decide whether to still apply the
	 * correction or fall back (the 0.3m clamp in spec §3.3 already
	 * prevents this case from mattering in practice for foot-lock, but
	 * the solver itself stays correct/stable for a target at any
	 * distance).
	 */
	inline FResult Solve(
		const FVector& RootPosition,
		const FVector& MidPosition,
		const FVector& EffectorPosition,
		const FVector& TargetPosition,
		float UpperLength,
		float LowerLength)
	{
		const FVector ToTarget = TargetPosition - RootPosition;
		const float TargetDistance = ToTarget.Size();
		const float MaxReach = UpperLength + LowerLength;
		const float MinReach = FMath::Abs(UpperLength - LowerLength);

		const bool bReachable =
			TargetDistance <= MaxReach && TargetDistance >= MinReach;
		const float ClampedDistance =
			FMath::Clamp(TargetDistance, MinReach + 1e-6f, MaxReach - 1e-6f);

		if (TargetDistance < 1e-6f)
		{
			// Degenerate: target coincides with the root. Fall back to
			// the current pose unchanged rather than dividing by zero.
			FResult Degenerate;
			Degenerate.MidPosition = MidPosition;
			Degenerate.EffectorPosition = EffectorPosition;
			Degenerate.bTargetReachable = false;
			return Degenerate;
		}

		const FVector Forward = ToTarget / TargetDistance;

		// Pole direction: current knee offset from the root->effector
		// axis, projected out of Forward and normalized. Preserves the
		// pose's natural bend plane (spec §3.2) instead of a fixed
		// pole vector.
		const FVector MidOffset = MidPosition - RootPosition;
		FVector PoleRaw = MidOffset - FVector::DotProduct(MidOffset, Forward) * Forward;
		const float PoleLengthSq = PoleRaw.SizeSquared();
		FVector Pole = PoleLengthSq > 1e-10f
			? PoleRaw / FMath::Sqrt(PoleLengthSq)
			: FVector::CrossProduct(Forward, FVector::UpVector).GetSafeNormal();
		if (Pole.SizeSquared() < 1e-10f)
		{
			// Forward was parallel to UpVector: pick any perpendicular.
			Pole = FVector::CrossProduct(Forward, FVector::ForwardVector).GetSafeNormal();
		}

		// Law of cosines: angle at the root between "upper bone" and
		// "root -> target" axis.
		float CosRootAngle =
			(UpperLength * UpperLength + ClampedDistance * ClampedDistance - LowerLength * LowerLength)
			/ (2.0f * UpperLength * ClampedDistance);
		CosRootAngle = FMath::Clamp(CosRootAngle, -1.0f, 1.0f);
		const float RootAngle = FMath::Acos(CosRootAngle);

		// Rotate Forward by RootAngle around the plane normal
		// (Forward x Pole) to get the direction from root to mid.
		const FVector PlaneNormal =
			FVector::CrossProduct(Forward, Pole).GetSafeNormal();
		const FVector MidDirection = RotateAroundAxis(Forward, PlaneNormal, RootAngle);
		const FVector NewMidPosition = RootPosition + MidDirection * UpperLength;

		// Effector sits LowerLength further along the mid->target
		// direction constrained by the remaining triangle side (see
		// the C# solver's comment for the closed-form derivation).
		const FVector MidToTargetDir = TargetPosition - NewMidPosition;
		const float MidToTargetLen = MidToTargetDir.Size();
		const FVector NewEffectorPosition = MidToTargetLen > 1e-6f
			? NewMidPosition + MidToTargetDir / MidToTargetLen * LowerLength
			: NewMidPosition + Forward * LowerLength;

		FResult Result;
		Result.MidPosition = NewMidPosition;
		Result.EffectorPosition = NewEffectorPosition;
		Result.bTargetReachable = bReachable;
		return Result;
	}
}
