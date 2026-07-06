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

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorForwardKinematicsOrthonormalizeFrameTest,
	"AInimator.ForwardKinematics.OrthonormalizeFrameProjectsAndIsIdempotent",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorForwardKinematicsOrthonormalizeFrameTest::RunTest(const FString& Parameters)
{
	// Two bones: one already-valid identity 6D (must be a numeric no-op,
	// inference_contract.md section 3.6), one drifted off-manifold (must come
	// back to an orthonormal pair).
	TArray<float> Frame = {
		1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
		1.7f, 0.2f, -0.4f, 0.9f, 1.3f, 0.1f,
	};
	AInimatorForwardKinematics::OrthonormalizeFrame(Frame);

	TestTrue(TEXT("Valid identity 6D is unchanged"),
		FMath::IsNearlyEqual(Frame[0], 1.0f, 1e-5f) &&
		FMath::IsNearlyZero(Frame[1], 1e-5f) &&
		FMath::IsNearlyEqual(Frame[4], 1.0f, 1e-5f));

	const FVector B1(Frame[6], Frame[7], Frame[8]);
	const FVector B2(Frame[9], Frame[10], Frame[11]);
	TestTrue(TEXT("Drifted b1 is unit length"), FMath::IsNearlyEqual(B1.Size(), 1.0f, 1e-4f));
	TestTrue(TEXT("Drifted b2 is unit length"), FMath::IsNearlyEqual(B2.Size(), 1.0f, 1e-4f));
	TestTrue(TEXT("Drifted b1/b2 are orthogonal"),
		FMath::IsNearlyZero(FVector::DotProduct(B1, B2), 1e-4f));

	TArray<float> Reprojected = Frame;
	AInimatorForwardKinematics::OrthonormalizeFrame(Reprojected);
	for (int32 Index = 0; Index < Frame.Num(); ++Index)
	{
		TestTrue(TEXT("Projection is idempotent"),
			FMath::IsNearlyEqual(Frame[Index], Reprojected[Index], 1e-5f));
	}
	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
