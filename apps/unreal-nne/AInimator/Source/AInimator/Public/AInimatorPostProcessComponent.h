// Copyright AI-nimator.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "AInimatorFootContactDetector.h"
#include "AInimatorFootLockIK.h"
#include "AInimatorIdleMoveBlender.h"
#include "AInimatorFootSlidingMetric.h"
#include "AInimatorPostProcessComponent.generated.h"

class UAInimatorControllerRuntime;

/**
 * B4 post-processing component: foot-lock IK + idle/move blending,
 * downstream of `UAInimatorControllerRuntime::Tick()`
 * (`apps/spec/footlock_blending.md`).
 *
 * Kept as a **separate** component rather than folded into
 * `UAInimatorControllerRuntime` deliberately, mirroring the B2 design
 * note that the runtime "returns a plausible unlocked pose" and
 * post-processing is the caller's responsibility: this keeps
 * `UAInimatorControllerRuntime` a pure inference+integration engine
 * (no pose-space/IK concerns), and lets a game opt out of B4 entirely
 * (e.g. to A/B test foot-sliding) by simply not adding this component.
 *
 * Hard rule enforced by construction: this component reads
 * `Runtime->GetLatestBoneFrame()` / `GetWorldRootPosition()` /
 * `GetWorldYaw()` **after** `Runtime->Tick()` has already pushed the
 * raw state into `FStateBuffer` — it never calls back into the
 * runtime to mutate state, so the autoregressive window always sees
 * the controller's raw output, never the IK-corrected pose (spec §1).
 *
 * Call order each frame (game code, not automated by this component to
 * keep tick-group ordering explicit and inspectable):
 * ```
 * Runtime->Tick();
 * PostProcess->TickPostProcess(DeltaSeconds, RawControlVx, RawControlVz);
 * // read PostProcess->GetCorrectedAnklePosition(...) / GetMoveWeight()
 * ```
 */
UCLASS(ClassGroup = (AInimator), meta = (BlueprintSpawnableComponent))
class AINIMATOR_API UAInimatorPostProcessComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UAInimatorPostProcessComponent();

	/** The runtime this component post-processes. Must already have a
	 *  bundle loaded before TickPostProcess is called. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|PostProcess")
	TObjectPtr<UAInimatorControllerRuntime> Runtime;

	/** Master on/off switch for foot-lock IK (spec §3). When false,
	 *  GetCorrectedAnklePosition/GetCorrectedKneePosition simply return
	 *  the uncorrected FK pose every frame — useful for the mandatory
	 *  before/after foot-sliding comparison (spec §5). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|PostProcess|FootLock")
	bool bEnableFootLockIK = true;

	/** Master on/off switch for idle<->move cross-fade (spec §4). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|PostProcess|Blending")
	bool bEnableIdleMoveBlend = true;

	/** spec §2 default 0.05m. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|PostProcess|FootContact")
	float ContactHeightThreshold = FFootContactDetector::DefaultHeightThreshold;

	/** spec §2 default 0.01 m/frame. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|PostProcess|FootContact")
	float ContactSpeedThreshold = FFootContactDetector::DefaultSpeedThreshold;

	/** spec §2 default 2 frames. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|PostProcess|FootContact")
	int32 ContactExitFrames = FFootContactDetector::DefaultExitFrames;

	/** spec §2 default 1.5x (i.e. 0.075m release height with the
	 *  default 0.05m threshold). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|PostProcess|FootContact")
	float ContactReleaseHeightMultiplier = FFootContactDetector::DefaultReleaseHeightMultiplier;

	/** spec §3.3 default 0.3m. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|PostProcess|FootLock")
	float MaxCorrectionMeters = FFootLockIK::DefaultMaxCorrectionMeters;

	/** spec §3.4 default 0.1s. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|PostProcess|FootLock")
	float ReleaseFadeSeconds = FFootLockIK::DefaultReleaseFadeSeconds;

	/** spec §4 default 0.005 m/frame, evaluated on the raw control. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|PostProcess|Blending")
	float MoveSpeedThreshold = FIdleMoveBlender::DefaultMoveSpeedThreshold;

	/** spec §4 default 0.25s. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|PostProcess|Blending")
	float IdleDelaySeconds = FIdleMoveBlender::DefaultIdleDelaySeconds;

	/** spec §4 default 0.2s. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|PostProcess|Blending")
	float CrossFadeSeconds = FIdleMoveBlender::DefaultCrossFadeSeconds;

	/** Enables FFootSlidingMetric accumulation every TickPostProcess
	 *  call — off by default (diagnostic only, spec §5), never
	 *  required for the runtime loop itself. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AInimator|PostProcess|Debug")
	bool bRecordFootSlidingMetric = false;

	/**
	 * Runs one frame of B4 post-processing. Must be called AFTER
	 * `Runtime->Tick()` for this frame (see class comment call order).
	 *
	 * Parameters
	 * ----------
	 * DeltaSeconds : real elapsed time, drives the release fade and
	 *     idle/move cross-fade timers (both are wall-clock, not
	 *     frame-count based, per spec §3.4/§4).
	 * RawControlVx / RawControlVz : this frame's RAW (unnormalized)
	 *     requested control — the same values passed to
	 *     `Runtime->SetControl()`/`SetPreset()`, needed because spec §4
	 *     requires the raw, not normalized, control for the idle/move
	 *     threshold.
	 */
	UFUNCTION(BlueprintCallable, Category = "AInimator|PostProcess")
	void TickPostProcess(float DeltaSeconds, float RawControlVx, float RawControlVz);

	/** Corrected world ankle position for FootSlot (0=left, 1=right)
	 *  from the most recent TickPostProcess call. Equals the
	 *  uncorrected FK pose when bEnableFootLockIK is false or the foot
	 *  is not locked. */
	UFUNCTION(BlueprintPure, Category = "AInimator|PostProcess")
	FVector GetCorrectedAnklePosition(int32 FootSlot) const;

	/** Corrected world knee position for FootSlot, see
	 *  GetCorrectedAnklePosition. */
	UFUNCTION(BlueprintPure, Category = "AInimator|PostProcess")
	FVector GetCorrectedKneePosition(int32 FootSlot) const;

	/** Whether FootSlot is currently considered locked (in contact),
	 *  from the most recent TickPostProcess call. */
	UFUNCTION(BlueprintPure, Category = "AInimator|PostProcess")
	bool IsFootLocked(int32 FootSlot) const;

	/** Current move-pose blend weight in [0,1] (1 = full controller
	 *  pose, 0 = full idle pose) — see FIdleMoveBlender. */
	UFUNCTION(BlueprintPure, Category = "AInimator|PostProcess")
	float GetMoveBlendWeight() const { return LastMoveWeight; }

	/** Mean planar foot displacement per contact frame recorded so far
	 *  (meters/frame) when bRecordFootSlidingMetric is true; 0 with no
	 *  samples. See FFootSlidingMetric / spec §5. */
	UFUNCTION(BlueprintPure, Category = "AInimator|PostProcess|Debug")
	float GetAverageFootSlidingMeters() const;

	UFUNCTION(BlueprintCallable, Category = "AInimator|PostProcess|Debug")
	void ResetFootSlidingMetric();

	/** Resets all internal post-process state (locks, fades, blend
	 *  weight) — call alongside any hard reset of the Runtime (e.g.
	 *  re-seeding a pawn). Does NOT reset the foot-sliding metric
	 *  (that is an explicit, separate opt-in via
	 *  ResetFootSlidingMetric, since a metric recording usually spans
	 *  multiple resets/re-seeds of the same evaluation run). */
	UFUNCTION(BlueprintCallable, Category = "AInimator|PostProcess")
	void ResetPostProcessState();

protected:
	virtual void BeginPlay() override;

private:
	TUniquePtr<FFootContactDetector> ContactDetector;
	TUniquePtr<FFootLockIK> FootLockIK;
	TUniquePtr<FIdleMoveBlender> IdleMoveBlender;
	FFootSlidingMetric SlidingMetric;

	FVector CorrectedKneePosition[2] = {FVector::ZeroVector, FVector::ZeroVector};
	FVector CorrectedAnklePosition[2] = {FVector::ZeroVector, FVector::ZeroVector};
	float LastMoveWeight = 1.0f;

	void InitializeSubsystems();
};
