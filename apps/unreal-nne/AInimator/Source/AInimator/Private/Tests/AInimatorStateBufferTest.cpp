// Copyright AI-nimator.

#include "Misc/AutomationTest.h"
#include "AInimatorStateBuffer.h"

#if WITH_DEV_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorStateBufferSeedTest,
	"AInimator.StateBuffer.SeedFillsEverySlot",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorStateBufferSeedTest::RunTest(const FString& Parameters)
{
	FStateBuffer Buffer(/*ContextFrames=*/3, /*BoneFrameWidth=*/2,
		/*GlobalFrameWidth=*/1);
	Buffer.SeedWithFrame({1.f, 2.f}, {9.f});

	const TArray<float>& BoneWindow = Buffer.GetBoneWindow();
	TestEqual(TEXT("BoneWindow length"), BoneWindow.Num(), 6);
	for (int32 Frame = 0; Frame < 3; ++Frame)
	{
		TestEqual(TEXT("Seeded bone channel 0"), BoneWindow[Frame * 2 + 0], 1.f);
		TestEqual(TEXT("Seeded bone channel 1"), BoneWindow[Frame * 2 + 1], 2.f);
	}

	const TArray<float>& GlobalWindow = Buffer.GetGlobalWindow();
	TestEqual(TEXT("GlobalWindow length"), GlobalWindow.Num(), 3);
	for (float Value : GlobalWindow)
	{
		TestEqual(TEXT("Seeded global channel"), Value, 9.f);
	}
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorStateBufferPushOrderTest,
	"AInimator.StateBuffer.PushEvictsOldestKeepsChronologicalOrder",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorStateBufferPushOrderTest::RunTest(const FString& Parameters)
{
	FStateBuffer Buffer(/*ContextFrames=*/3, /*BoneFrameWidth=*/1,
		/*GlobalFrameWidth=*/1);
	Buffer.SeedWithFrame({0.f}, {0.f});

	Buffer.PushFrame({1.f}, {10.f});
	Buffer.PushFrame({2.f}, {20.f});
	Buffer.PushFrame({3.f}, {30.f});

	const TArray<float>& BoneWindow = Buffer.GetBoneWindow();
	// After 3 pushes into a window of 3 (seeded with 0), the oldest
	// seed frame has been fully evicted: window is [1, 2, 3].
	TestEqual(TEXT("Oldest bone frame"), BoneWindow[0], 1.f);
	TestEqual(TEXT("Middle bone frame"), BoneWindow[1], 2.f);
	TestEqual(TEXT("Newest bone frame"), BoneWindow[2], 3.f);

	const TArray<float>& GlobalWindow = Buffer.GetGlobalWindow();
	TestEqual(TEXT("Oldest global frame"), GlobalWindow[0], 10.f);
	TestEqual(TEXT("Newest global frame"), GlobalWindow[2], 30.f);

	TArray<float> LastBoneFrame;
	Buffer.GetLastBoneFrame(LastBoneFrame);
	TestEqual(TEXT("GetLastBoneFrame length"), LastBoneFrame.Num(), 1);
	TestEqual(TEXT("GetLastBoneFrame value"), LastBoneFrame[0], 3.f);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorStateBufferSingleFrameWindowTest,
	"AInimator.StateBuffer.ContextFramesOneDegeneratesCorrectly",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorStateBufferSingleFrameWindowTest::RunTest(const FString& Parameters)
{
	FStateBuffer Buffer(/*ContextFrames=*/1, /*BoneFrameWidth=*/2,
		/*GlobalFrameWidth=*/1);
	Buffer.SeedWithFrame({0.f, 0.f}, {0.f});
	Buffer.PushFrame({5.f, 6.f}, {7.f});

	const TArray<float>& BoneWindow = Buffer.GetBoneWindow();
	TestEqual(TEXT("Window length stays BoneFrameWidth"), BoneWindow.Num(), 2);
	TestEqual(TEXT("Only frame value 0"), BoneWindow[0], 5.f);
	TestEqual(TEXT("Only frame value 1"), BoneWindow[1], 6.f);
	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
