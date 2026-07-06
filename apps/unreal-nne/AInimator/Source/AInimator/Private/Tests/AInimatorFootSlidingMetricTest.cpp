// Copyright AI-nimator.

#include "Misc/AutomationTest.h"
#include "AInimatorFootSlidingMetric.h"

#if WITH_DEV_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorFootSlidingMetricComputesAverageOverContactFramesTest,
	"AInimator.FootSlidingMetric.AveragesPlanarDisplacementOverContactFramesOnly",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorFootSlidingMetricComputesAverageOverContactFramesTest::RunTest(const FString& Parameters)
{
	FFootSlidingMetric Metric;

	// Frame 0: first sample, no previous -> no displacement counted yet.
	Metric.AccumulateSample(0, FVector(0.0f, 0.0f, 0.0f), /*bIsInContact=*/true);
	TestEqual(TEXT("No samples counted after the first frame"), Metric.GetSampleCount(), 0);

	// Frame 1: still in contact, moved 0.02m planar (X) -- counts.
	Metric.AccumulateSample(0, FVector(0.02f, 0.0f, 0.0f), true);
	TestEqual(TEXT("One contact-to-contact pair counted"), Metric.GetSampleCount(), 1);
	TestTrue(TEXT("Average matches the single displacement"),
		FMath::IsNearlyEqual(Metric.ComputeAverageSlidingMeters(), 0.02f, 1e-4f));

	// Frame 2: not in contact -- must not contribute even though it
	// moved a lot (airborne motion is not "sliding").
	Metric.AccumulateSample(0, FVector(1.0f, 0.0f, 0.0f), false);
	TestEqual(TEXT("Airborne frame does not add a sample"), Metric.GetSampleCount(), 1);

	// Frame 3: back in contact, but previous frame was airborne -> the
	// transition itself is not counted (only contact-to-contact pairs).
	Metric.AccumulateSample(0, FVector(1.01f, 0.0f, 0.0f), true);
	TestEqual(TEXT("Contact-after-airborne transition frame not counted"),
		Metric.GetSampleCount(), 1);

	// Frame 4: another contact frame -- now counts against frame 3.
	Metric.AccumulateSample(0, FVector(1.03f, 0.0f, 0.0f), true);
	TestEqual(TEXT("Second contact-to-contact pair now counted"), Metric.GetSampleCount(), 2);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorFootSlidingMetricNoDataReturnsZeroTest,
	"AInimator.FootSlidingMetric.NoSamplesReturnsZeroNotNaN",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorFootSlidingMetricNoDataReturnsZeroTest::RunTest(const FString& Parameters)
{
	FFootSlidingMetric Metric;
	TestEqual(TEXT("No samples -> average is 0"), Metric.ComputeAverageSlidingMeters(), 0.0f);
	TestEqual(TEXT("No samples -> sample count is 0"), Metric.GetSampleCount(), 0);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FAInimatorFootSlidingMetricTracksFeetIndependentlyTest,
	"AInimator.FootSlidingMetric.TracksMultipleFeetIndependently",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilterMask)

bool FAInimatorFootSlidingMetricTracksFeetIndependentlyTest::RunTest(const FString& Parameters)
{
	FFootSlidingMetric Metric;

	Metric.AccumulateSample(0, FVector(0.0f, 0.0f, 0.0f), true);
	Metric.AccumulateSample(1, FVector(10.0f, 0.0f, 0.0f), true);

	Metric.AccumulateSample(0, FVector(0.01f, 0.0f, 0.0f), true);
	Metric.AccumulateSample(1, FVector(10.05f, 0.0f, 0.0f), true);

	TestEqual(TEXT("Both feet contribute one pair each"), Metric.GetSampleCount(), 2);
	const float Expected = (0.01f + 0.05f) / 2.0f;
	TestTrue(TEXT("Average blends both feet's contributions"),
		FMath::IsNearlyEqual(Metric.ComputeAverageSlidingMeters(), Expected, 1e-4f));
	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
