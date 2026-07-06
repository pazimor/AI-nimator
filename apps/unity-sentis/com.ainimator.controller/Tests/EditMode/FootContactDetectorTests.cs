using NUnit.Framework;
using UnityEngine;
using AInimator.Controller.PostProcess;

namespace AInimator.Controller.Tests
{
    /// <summary>
    /// Tests for <see cref="FootContactDetector"/> against synthetic
    /// planted/lifted foot trajectories, per
    /// <c>apps/spec/footlock_blending.md</c> §2 (height &lt; 0.05m AND
    /// planar speed &lt; 0.01 m/frame to enter; 2-frame or 1.5x-height
    /// hysteresis to exit).
    /// </summary>
    public class FootContactDetectorTests
    {
        [Test]
        public void PlantedStationaryFoot_EntersAndStaysInContact()
        {
            var detector = new FootContactDetector(footCount: 1);
            var plantedPos = new Vector3(0f, 0.02f, 0f);

            var first = detector.Update(0, plantedPos);
            var second = detector.Update(0, plantedPos);

            Assert.That(first, Is.True);
            Assert.That(second, Is.True);
        }

        [Test]
        public void LiftedFoot_NeverEntersContact()
        {
            var detector = new FootContactDetector(footCount: 1);
            var liftedPos = new Vector3(0f, 0.3f, 0f);

            var result = detector.Update(0, liftedPos);

            Assert.That(result, Is.False);
        }

        [Test]
        public void FastMovingLowFoot_DoesNotEnterContact_SpeedCriterion()
        {
            var detector = new FootContactDetector(footCount: 1);

            detector.Update(0, new Vector3(0f, 0.02f, 0f));
            // Planar speed 0.05 m/frame > 0.01 threshold, height still low.
            var result = detector.Update(0, new Vector3(0.05f, 0.02f, 0f));

            Assert.That(result, Is.False);
        }

        [Test]
        public void ExitRequiresTwoConsecutiveViolatingFrames()
        {
            var detector = new FootContactDetector(footCount: 1, exitFrames: 2);
            var plantedPos = new Vector3(0f, 0.02f, 0f);

            detector.Update(0, plantedPos);
            Assert.That(detector.IsLocked(0), Is.True);

            // One violating frame (height above threshold but below release multiplier): stays locked.
            var stillLocked = detector.Update(0, new Vector3(0f, 0.06f, 0f));
            Assert.That(stillLocked, Is.True, "single violating frame must not release the lock");

            // Second consecutive violating frame: releases.
            var released = detector.Update(0, new Vector3(0f, 0.06f, 0f));
            Assert.That(released, Is.False);
        }

        [Test]
        public void ExitFrameCounterResetsOnNonViolatingFrame()
        {
            var detector = new FootContactDetector(footCount: 1, exitFrames: 2);
            var plantedPos = new Vector3(0f, 0.02f, 0f);

            detector.Update(0, plantedPos);
            detector.Update(0, new Vector3(0f, 0.06f, 0f)); // 1 violation
            detector.Update(0, plantedPos);                  // resets counter (raw contact true again)
            var stillLocked = detector.Update(0, new Vector3(0f, 0.06f, 0f)); // 1 violation again, not 2

            Assert.That(stillLocked, Is.True);
        }

        [Test]
        public void HeightAboveReleaseThreshold_ReleasesImmediately()
        {
            var detector = new FootContactDetector(footCount: 1);
            var plantedPos = new Vector3(0f, 0.02f, 0f);

            detector.Update(0, plantedPos);
            Assert.That(detector.IsLocked(0), Is.True);

            // 0.075m = 1.5x the 0.05m threshold -> immediate release, no hysteresis delay.
            var released = detector.Update(0, new Vector3(0f, 0.08f, 0f));

            Assert.That(released, Is.False);
        }

        [Test]
        public void Reset_ClearsLockedStateAndHistory()
        {
            var detector = new FootContactDetector(footCount: 1);
            detector.Update(0, new Vector3(0f, 0.02f, 0f));
            Assert.That(detector.IsLocked(0), Is.True);

            detector.Reset();

            Assert.That(detector.IsLocked(0), Is.False);
        }
    }
}
