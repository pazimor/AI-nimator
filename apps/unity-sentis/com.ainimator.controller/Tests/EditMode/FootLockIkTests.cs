using NUnit.Framework;
using UnityEngine;
using AInimator.Controller.PostProcess;

namespace AInimator.Controller.Tests
{
    /// <summary>
    /// Tests for <see cref="FootLockIk"/>: capture-on-entry, clamp release,
    /// and the 0.1s fade-out on release (spec §3).
    /// </summary>
    public class FootLockIkTests
    {
        private const float Upper = 0.4f;
        private const float Lower = 0.42f;

        private static FootLockIk NewIk(float clamp = FootLockIk.DefaultClampDistance, float fade = FootLockIk.DefaultReleaseFadeSeconds)
        {
            return new FootLockIk(Upper, Lower, clamp, fade);
        }

        [Test]
        public void Resolve_HoldsCapturedPosition_WhileLockedAndPoseDrifts()
        {
            var ik = NewIk();
            var hip = new Vector3(0f, 1f, 0f);
            var kneeAtRest = new Vector3(0f, 1f - Upper, 0.02f);
            var ankleAtRest = new Vector3(0f, 1f - Upper - Lower, 0.02f);

            // Frame 1: enters lock, captures ankleAtRest.
            var (knee1, ankle1) = ik.Resolve(true, hip, kneeAtRest, ankleAtRest, 1f / 30f);
            Assert.That(Vector3.Distance(ankle1, ankleAtRest), Is.LessThan(1e-3f));

            // Frame 2: pose "wants" to drift the ankle slightly (small, within clamp);
            // the IK should still hold close to the original capture.
            var driftedAnkle = ankleAtRest + new Vector3(0.02f, 0f, 0f);
            var driftedKnee = kneeAtRest + new Vector3(0.02f, 0f, 0f);
            var (_, ankle2) = ik.Resolve(true, hip, driftedKnee, driftedAnkle, 1f / 30f);

            Assert.That(Vector3.Distance(ankle2, ankleAtRest), Is.LessThan(1e-2f));
        }

        [Test]
        public void Resolve_BeyondClamp_ReleasesLockOutright()
        {
            var ik = NewIk(clamp: 0.3f);
            var hip = new Vector3(0f, 1f, 0f);
            var knee = new Vector3(0f, 1f - Upper, 0.02f);
            var ankle = new Vector3(0f, 1f - Upper - Lower, 0.02f);

            ik.Resolve(true, hip, knee, ankle, 1f / 30f); // capture

            // Pose moves the ankle by more than the 0.3m clamp.
            var farAnkle = ankle + new Vector3(0.5f, 0f, 0f);
            var farKnee = knee + new Vector3(0.5f, 0f, 0f);
            var (kneeOut, ankleOut) = ik.Resolve(true, hip, farKnee, farAnkle, 1f / 30f);

            // Released: output equals the raw (uncorrected) pose, not the capture.
            Assert.That(ankleOut, Is.EqualTo(farAnkle));
            Assert.That(kneeOut, Is.EqualTo(farKnee));
        }

        [Test]
        public void Resolve_OnRelease_FadesCorrectionToZeroOverConfiguredDuration()
        {
            const float fadeSeconds = 0.1f;
            var ik = NewIk(fade: fadeSeconds);
            var hip = new Vector3(0f, 1f, 0f);
            var knee = new Vector3(0f, 1f - Upper, 0.02f);
            var ankle = new Vector3(0f, 1f - Upper - Lower, 0.02f);

            ik.Resolve(true, hip, knee, ankle, 1f / 30f); // capture at `ankle`

            // Release: raw pose has moved on (small enough not to matter),
            // released immediately.
            var releasedAnkle = ankle + new Vector3(0.05f, 0f, 0f);
            var releasedKnee = knee + new Vector3(0.05f, 0f, 0f);

            var (_, midFade) = ik.Resolve(false, hip, releasedKnee, releasedAnkle, fadeSeconds / 2f);
            // Midway through the fade the corrected ankle should sit strictly
            // between the raw released position and the locked position.
            var distToRaw = Vector3.Distance(midFade, releasedAnkle);
            var distToLocked = Vector3.Distance(midFade, ankle);
            Assert.That(distToRaw, Is.GreaterThan(0f));
            Assert.That(distToLocked, Is.GreaterThan(0f));

            // After the full fade duration, correction must be fully gone.
            var (_, endFade) = ik.Resolve(false, hip, releasedAnkle, releasedAnkle, fadeSeconds);
            Assert.That(Vector3.Distance(endFade, releasedAnkle), Is.LessThan(1e-3f));
        }

        [Test]
        public void Resolve_NeverLocked_PassesRawPoseThrough()
        {
            var ik = NewIk();
            var hip = new Vector3(0f, 1f, 0f);
            var knee = new Vector3(0f, 1f - Upper, 0.1f);
            var ankle = new Vector3(0f, 1f - Upper - Lower, 0.1f);

            var (kneeOut, ankleOut) = ik.Resolve(false, hip, knee, ankle, 1f / 30f);

            Assert.That(kneeOut, Is.EqualTo(knee));
            Assert.That(ankleOut, Is.EqualTo(ankle));
        }
    }
}
