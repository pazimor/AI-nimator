// Copyright AI-nimator.

#include "Misc/AutomationTest.h"
#include "AInimatorForwardKinematics.h"
#include "AInimatorSmpl22Skeleton.h"

#if WITH_DEV_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorForwardKinematicsIdentityPoseMatchesOffsetsTest,
	"AInimator.ForwardKinematics.IdentityPoseYieldsCumulativeOffsets",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorForwardKinematicsIdentityPoseMatchesOffsetsTest::RunTest(const FString& Parameters)
{
	using namespace AInimatorSmpl22Skeleton;

	// Identity rotation6d per bone: [1,0,0, 0,1,0] (column0=X, column1=Y).
	TArray<float> BoneFrame;
	BoneFrame.Reserve(NumBones * RotationChannelsPerBone);
	for (int32 Bone = 0; Bone < NumBones; ++Bone)
	{
		BoneFrame.Append({1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f});
	}

	TArray<FMatrix> GlobalRotations;
	TArray<FVector> GlobalPositions;
	GlobalRotations.SetNum(NumBones);
	GlobalPositions.SetNum(NumBones);
	AInimatorForwardKinematics::ComputeJointPositions(
		BoneFrame, NumBones, GlobalRotations, GlobalPositions);

	// With identity rotations everywhere, global position is simply the
	// cumulative sum of bone-local offsets along the parent chain.
	TestEqual(TEXT("Pelvis sits at its own offset (origin)"),
		GlobalPositions[Pelvis], BoneOffsets[Pelvis]);

	const FVector ExpectedLeftKnee = BoneOffsets[LeftHip] + BoneOffsets[LeftKnee];
	TestTrue(TEXT("Left knee position is hip+knee offsets summed"),
		(GlobalPositions[LeftKnee] - ExpectedLeftKnee).Size() < 1e-4f);

	const FVector ExpectedLeftAnkle =
		BoneOffsets[LeftHip] + BoneOffsets[LeftKnee] + BoneOffsets[LeftAnkle];
	TestTrue(TEXT("Left ankle position sums the full hip->knee->ankle chain"),
		(GlobalPositions[LeftAnkle] - ExpectedLeftAnkle).Size() < 1e-4f);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorForwardKinematicsSixDNormalizesNonOrthonormalInputTest,
	"AInimator.ForwardKinematics.SixDToRotationMatrixOrthonormalizesInput",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorForwardKinematicsSixDNormalizesNonOrthonormalInputTest::RunTest(const FString& Parameters)
{
	// Deliberately non-unit, non-orthogonal input -- Gram-Schmidt must
	// still produce an orthonormal basis (columns unit length,
	// mutually perpendicular).
	const float SixD[6] = {2.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f};
	FMatrix Rotation;
	AInimatorForwardKinematics::SixDToRotationMatrix(SixD, Rotation);

	const FVector Col0 = Rotation.GetColumn(0);
	const FVector Col1 = Rotation.GetColumn(1);
	const FVector Col2 = Rotation.GetColumn(2);

	TestTrue(TEXT("Column 0 is unit length"), FMath::IsNearlyEqual(Col0.Size(), 1.0f, 1e-4f));
	TestTrue(TEXT("Column 1 is unit length"), FMath::IsNearlyEqual(Col1.Size(), 1.0f, 1e-4f));
	TestTrue(TEXT("Column 2 is unit length"), FMath::IsNearlyEqual(Col2.Size(), 1.0f, 1e-4f));
	TestTrue(TEXT("Columns 0/1 are orthogonal"),
		FMath::IsNearlyZero(FVector::DotProduct(Col0, Col1), 1e-4f));
	TestTrue(TEXT("Columns 0/2 are orthogonal"),
		FMath::IsNearlyZero(FVector::DotProduct(Col0, Col2), 1e-4f));
	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
