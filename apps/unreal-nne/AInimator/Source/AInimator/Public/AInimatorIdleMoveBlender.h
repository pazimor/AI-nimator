// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"

/**
 * Idle <-> move pose-space cross-fade state machine
 * (`apps/spec/footlock_blending.md` §4).
 *
 * Rules (normative, spec §4):
 * - Entering `move`: `‖(vx, vz)‖ > MoveSpeedThreshold` (0.005 m/frame)
 *   on the **raw** requested control (never the normalized one).
 * - Entering `idle`: below that threshold continuously for
 *   `IdleDelaySeconds` (0.25s).
 * - Transition: `CrossFadeSeconds` (0.2s) cross-fade between the
 *   controller pose and the engine's idle pose (its own idle asset,
 *   or the controller's `idle` preset pose as a fallback).
 * - The blend is purely pose-space, downstream: the controller keeps
 *   running (and feeding its own autoregressive window) throughout the
 *   fade — this class never touches FStateBuffer.
 *
 * This class only tracks *state* (are we idle or moving, and the
 * current cross-fade weight) — applying `MoveWeight`/`IdleWeight` to
 * two actual poses (skeletal mesh blend, or a simple lerp of joint
 * positions) is the caller's job, since that depends on how the host
 * project represents a pose.
 *
 * Mirrors the same rule set the Unity plugin implements from the same
 * shared spec (Goal B vérité #7 parity) — no Unity C# equivalent
 * exists yet at time of writing; this is not a port, but it must stay
 * numerically identical to whatever Unity implements against
 * `footlock_blending.md` §4.
 */
class AINIMATOR_API FIdleMoveBlender
{
public:
	static constexpr float DefaultMoveSpeedThreshold = 0.005f;
	static constexpr float DefaultIdleDelaySeconds = 0.25f;
	static constexpr float DefaultCrossFadeSeconds = 0.2f;

	explicit FIdleMoveBlender(
		float InMoveSpeedThreshold = DefaultMoveSpeedThreshold,
		float InIdleDelaySeconds = DefaultIdleDelaySeconds,
		float InCrossFadeSeconds = DefaultCrossFadeSeconds);

	/**
	 * Advances the state machine by DeltaSeconds given this frame's
	 * raw (unnormalized) requested control (Vx, Vz).
	 *
	 * Returns the resulting move-pose blend weight in [0, 1]: 1 = pure
	 * controller/move pose, 0 = pure idle pose, in between = cross-fade
	 * in progress. `GetIdleWeight()` is always `1 - GetMoveWeight()`.
	 */
	float Tick(float RawVx, float RawVz, float DeltaSeconds);

	float GetMoveWeight() const { return MoveWeight; }
	float GetIdleWeight() const { return 1.0f - MoveWeight; }

	/** True once the raw control has been below threshold continuously
	 *  for at least IdleDelaySeconds (i.e. the target state is idle,
	 *  independent of whether the cross-fade has finished). */
	bool IsTargetIdle() const { return bTargetIsIdle; }

	void Reset();

private:
	float MoveSpeedThreshold;
	float IdleDelaySeconds;
	float CrossFadeSeconds;

	float MoveWeight = 1.0f;
	bool bTargetIsIdle = false;
	float SecondsBelowThreshold = 0.0f;
};
