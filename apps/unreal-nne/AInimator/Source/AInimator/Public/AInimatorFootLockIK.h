// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"
#include "AInimatorFootContactDetector.h"

/**
 * Foot-lock IK post-processor (`apps/spec/footlock_blending.md` §3):
 * per locked foot, captures its world position at the frame contact
 * begins, then holds it there via analytic two-bone IK, clamping and
 * fading the correction per spec. Consumes (but never mutates) the
 * controller's raw pose/state — this is purely a downstream, cosmetic
 * correction (§1: "la fenêtre autorégressive reçoit TOUJOURS l'état
 * brut, jamais la pose corrigée").
 *
 * Per locked foot, every frame:
 * 1. Capture the world position at contact entry (`LockedPosition`).
 * 2. Two-bone IK (hip -> knee -> ankle) toward `LockedPosition`, pole
 *    vector = current pose's knee direction (never fixed).
 * 3. Clamp: if `|LockedPosition - PosePosition| > MaxCorrectionMeters`
 *    (spec default 0.3m), release the lock — the controller "decided"
 *    a large displacement, do not stretch the leg.
 * 4. Release fade: on exiting contact, lerp the correction to zero over
 *    `ReleaseFadeSeconds` (spec default 0.1s) — never a snap.
 *
 * This class outputs corrected **joint positions** only (ankle/knee
 * per leg); applying them onto an actual Skeleton/AnimGraph two-bone
 * IK node (or an FK-reconstruction-then-reapply pipeline) is an
 * engine-integration step for the host project, mirroring
 * `apps/unity-sentis/`'s design (no direct Animator/rig binding
 * assumed at this layer either).
 */
class AINIMATOR_API FFootLockIK
{
public:
	/** Correction released if the controller's own pose displaces the
	 *  foot further than this from the locked position (meters). */
	static constexpr float DefaultMaxCorrectionMeters = 0.3f;

	/** Duration (seconds) over which the correction fades to zero
	 *  after a lock releases — never an instantaneous snap. */
	static constexpr float DefaultReleaseFadeSeconds = 0.1f;

	struct FLegChain
	{
		int32 HipBone = INDEX_NONE;
		int32 KneeBone = INDEX_NONE;
		int32 AnkleBone = INDEX_NONE;
		float UpperLength = 0.0f;
		float LowerLength = 0.0f;
	};

	FFootLockIK(
		float InMaxCorrectionMeters = DefaultMaxCorrectionMeters,
		float InReleaseFadeSeconds = DefaultReleaseFadeSeconds);

	/**
	 * Runs one frame of foot-lock correction for both legs.
	 *
	 * Parameters
	 * ----------
	 * ContactDetector : already Update()-d this frame (caller drives
	 *     it — this class only consumes IsLocked()).
	 * LeftChain / RightChain : bone indices + bone lengths for each
	 *     leg (from AInimatorSmpl22Skeleton).
	 * WorldHipPosition / WorldKneePosition / WorldAnklePosition :
	 *     current (uncorrected) world joint positions from FK + the
	 *     engine's integrated root position, indexed
	 *     [Left, Right] (2 entries each).
	 * DeltaSeconds : real time elapsed since the previous call, used
	 *     to drive the release fade.
	 * OutCorrectedKneePosition / OutCorrectedAnklePosition : resized to
	 *     2 and filled with the corrected world positions (equal to
	 *     the input pose positions where no correction applies).
	 */
	void Solve(
		const FFootContactDetector& ContactDetector,
		const FLegChain& LeftChain,
		const FLegChain& RightChain,
		const FVector WorldHipPosition[2],
		const FVector WorldKneePosition[2],
		const FVector WorldAnklePosition[2],
		float DeltaSeconds,
		FVector OutCorrectedKneePosition[2],
		FVector OutCorrectedAnklePosition[2]);

	/** Resets captured locks and fade state (call on a hard state
	 *  reset, e.g. re-seeding the pawn). */
	void Reset();

private:
	struct FFootState
	{
		bool bWasLocked = false;
		FVector LockedPosition = FVector::ZeroVector;
		/** 1 at full correction, fading to 0 over ReleaseFadeSeconds
		 *  after a lock releases. */
		float FadeWeight = 0.0f;
	};

	void SolveOneFoot(
		int32 FootSlot,
		bool bIsLocked,
		const FLegChain& Chain,
		const FVector& WorldHip,
		const FVector& WorldKnee,
		const FVector& WorldAnkle,
		float DeltaSeconds,
		FVector& OutCorrectedKnee,
		FVector& OutCorrectedAnkle);

	float MaxCorrectionMeters;
	float ReleaseFadeSeconds;

	FFootState FeetState[2];
};
