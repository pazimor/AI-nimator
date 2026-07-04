// Copyright AI-nimator.

#include "AInimatorPostProcessComponent.h"
#include "AInimatorControllerRuntime.h"
#include "AInimatorForwardKinematics.h"
#include "AInimatorSmpl22Skeleton.h"
#include "AInimatorLog.h"

UAInimatorPostProcessComponent::UAInimatorPostProcessComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
	InitializeSubsystems();
}

void UAInimatorPostProcessComponent::BeginPlay()
{
	Super::BeginPlay();
	InitializeSubsystems();
}

void UAInimatorPostProcessComponent::InitializeSubsystems()
{
	ContactDetector = MakeUnique<FFootContactDetector>(
		/*FootCount=*/2,
		ContactHeightThreshold,
		ContactSpeedThreshold,
		ContactExitFrames,
		ContactReleaseHeightMultiplier);
	FootLockIK = MakeUnique<FFootLockIK>(MaxCorrectionMeters, ReleaseFadeSeconds);
	IdleMoveBlender = MakeUnique<FIdleMoveBlender>(
		MoveSpeedThreshold, IdleDelaySeconds, CrossFadeSeconds);
}

void UAInimatorPostProcessComponent::TickPostProcess(
	float DeltaSeconds,
	float RawControlVx,
	float RawControlVz)
{
	if (!Runtime || !Runtime->IsLoaded())
	{
		UE_LOG(LogAInimator, Warning,
			TEXT("AInimator: TickPostProcess called with no loaded Runtime; ")
			TEXT("skipping this frame's post-processing."));
		return;
	}

	const FAInimatorManifest& Manifest = Runtime->GetManifest();
	const int32 NumBones = Manifest.NumBones;
	check(NumBones == AInimatorSmpl22Skeleton::NumBones);

	// Forward-kinematics the controller's raw current pose (root-local,
	// per AInimatorForwardKinematics's contract) then translate by the
	// engine's integrated world root position — never re-applying an
	// extra yaw rotation, since the pelvis rotation6d already carries
	// world-facing orientation (spec design note, footlock_blending.md
	// §1 + AInimatorForwardKinematics.h comment).
	TArray<FMatrix> GlobalRotations;
	TArray<FVector> RootLocalPositions;
	GlobalRotations.SetNum(NumBones);
	RootLocalPositions.SetNum(NumBones);
	AInimatorForwardKinematics::ComputeJointPositions(
		Runtime->GetLatestBoneFrame(), NumBones, GlobalRotations, RootLocalPositions);

	const FVector WorldRoot = Runtime->GetWorldRootPosition();

	using namespace AInimatorSmpl22Skeleton;
	const FVector WorldHip[2] = {
		AInimatorForwardKinematics::RootLocalToWorld(RootLocalPositions[LeftHip], WorldRoot),
		AInimatorForwardKinematics::RootLocalToWorld(RootLocalPositions[RightHip], WorldRoot),
	};
	const FVector WorldKnee[2] = {
		AInimatorForwardKinematics::RootLocalToWorld(RootLocalPositions[LeftKnee], WorldRoot),
		AInimatorForwardKinematics::RootLocalToWorld(RootLocalPositions[RightKnee], WorldRoot),
	};
	const FVector WorldAnkle[2] = {
		AInimatorForwardKinematics::RootLocalToWorld(RootLocalPositions[LeftAnkle], WorldRoot),
		AInimatorForwardKinematics::RootLocalToWorld(RootLocalPositions[RightAnkle], WorldRoot),
	};
	// Foot-contact joints (10/11) per spec §2 — used for the contact
	// criterion, distinct from the IK effector (ankle, 7/8) per spec
	// §3's chain definition (hip->knee->ankle).
	const FVector WorldFoot[2] = {
		AInimatorForwardKinematics::RootLocalToWorld(RootLocalPositions[LeftFoot], WorldRoot),
		AInimatorForwardKinematics::RootLocalToWorld(RootLocalPositions[RightFoot], WorldRoot),
	};

	bool bIsLocked[2];
	for (int32 FootSlot = 0; FootSlot < 2; ++FootSlot)
	{
		bIsLocked[FootSlot] = ContactDetector->Update(FootSlot, WorldFoot[FootSlot]);
	}

	if (bEnableFootLockIK)
	{
		FFootLockIK::FLegChain LeftChain;
		LeftChain.HipBone = LeftHip;
		LeftChain.KneeBone = LeftKnee;
		LeftChain.AnkleBone = LeftAnkle;
		LeftChain.UpperLength = LeftThighLength();
		LeftChain.LowerLength = LeftShinLength();

		FFootLockIK::FLegChain RightChain;
		RightChain.HipBone = RightHip;
		RightChain.KneeBone = RightKnee;
		RightChain.AnkleBone = RightAnkle;
		RightChain.UpperLength = RightThighLength();
		RightChain.LowerLength = RightShinLength();

		FootLockIK->Solve(
			*ContactDetector,
			LeftChain,
			RightChain,
			WorldHip,
			WorldKnee,
			WorldAnkle,
			DeltaSeconds,
			CorrectedKneePosition,
			CorrectedAnklePosition);
	}
	else
	{
		for (int32 FootSlot = 0; FootSlot < 2; ++FootSlot)
		{
			CorrectedKneePosition[FootSlot] = WorldKnee[FootSlot];
			CorrectedAnklePosition[FootSlot] = WorldAnkle[FootSlot];
		}
	}

	if (bEnableIdleMoveBlend)
	{
		LastMoveWeight = IdleMoveBlender->Tick(RawControlVx, RawControlVz, DeltaSeconds);
	}
	else
	{
		LastMoveWeight = 1.0f;
	}

	if (bRecordFootSlidingMetric)
	{
		for (int32 FootSlot = 0; FootSlot < 2; ++FootSlot)
		{
			SlidingMetric.AccumulateSample(FootSlot, WorldFoot[FootSlot], bIsLocked[FootSlot]);
		}
	}
}

FVector UAInimatorPostProcessComponent::GetCorrectedAnklePosition(int32 FootSlot) const
{
	check(FootSlot >= 0 && FootSlot < 2);
	return CorrectedAnklePosition[FootSlot];
}

FVector UAInimatorPostProcessComponent::GetCorrectedKneePosition(int32 FootSlot) const
{
	check(FootSlot >= 0 && FootSlot < 2);
	return CorrectedKneePosition[FootSlot];
}

bool UAInimatorPostProcessComponent::IsFootLocked(int32 FootSlot) const
{
	check(FootSlot >= 0 && FootSlot < 2);
	return ContactDetector.IsValid() && ContactDetector->IsLocked(FootSlot);
}

float UAInimatorPostProcessComponent::GetAverageFootSlidingMeters() const
{
	return SlidingMetric.ComputeAverageSlidingMeters();
}

void UAInimatorPostProcessComponent::ResetFootSlidingMetric()
{
	SlidingMetric.Reset();
}

void UAInimatorPostProcessComponent::ResetPostProcessState()
{
	if (ContactDetector.IsValid())
	{
		ContactDetector->Reset();
	}
	if (FootLockIK.IsValid())
	{
		FootLockIK->Reset();
	}
	if (IdleMoveBlender.IsValid())
	{
		IdleMoveBlender->Reset();
	}
	LastMoveWeight = 1.0f;
}
