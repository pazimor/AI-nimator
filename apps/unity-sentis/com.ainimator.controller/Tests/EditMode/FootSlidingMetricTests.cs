using NUnit.Framework;
using UnityEngine;
using AInimator.Controller.PostProcess;

namespace AInimator.Controller.Tests
{
    /// <summary>
    /// Tests for <see cref="FootSlidingMetric"/> — the before/after
    /// foot-sliding measurement utility (spec §5).
    /// </summary>
    public class FootSlidingMetricTests
    {
        [Test]
        public void StationaryContactFrames_ZeroSliding()
        {
            var metric = new FootSlidingMetric();
            var pos = new Vector3(1f, 0.02f, 2f);

            metric.Accumulate(pos, isInContact: true);
            metric.Accumulate(pos, isInContact: true);
            metric.Accumulate(pos, isInContact: true);

            Assert.That(metric.MeanPlanarDisplacementPerContactFrame, Is.EqualTo(0.0).Within(1e-9));
        }

        [Test]
        public void SlidingContactFrames_AccumulatesPlanarDisplacement()
        {
            var metric = new FootSlidingMetric();

            metric.Accumulate(new Vector3(0f, 0.02f, 0f), isInContact: true);
            metric.Accumulate(new Vector3(0.01f, 0.02f, 0f), isInContact: true);
            metric.Accumulate(new Vector3(0.02f, 0.02f, 0f), isInContact: true);

            // Two consecutive-contact transitions of 0.01m each.
            Assert.That(metric.ContactFrameCount, Is.EqualTo(2));
            Assert.That(metric.MeanPlanarDisplacementPerContactFrame, Is.EqualTo(0.01).Within(1e-6));
        }

        [Test]
        public void NonContactFrames_AreNotCounted()
        {
            var metric = new FootSlidingMetric();

            metric.Accumulate(new Vector3(0f, 0.3f, 0f), isInContact: false);
            metric.Accumulate(new Vector3(1f, 0.3f, 0f), isInContact: false);

            Assert.That(metric.ContactFrameCount, Is.EqualTo(0));
            Assert.That(metric.MeanPlanarDisplacementPerContactFrame, Is.EqualTo(0.0));
        }

        [Test]
        public void TransitionIntoContact_DoesNotCountTheEntryFrame()
        {
            var metric = new FootSlidingMetric();

            metric.Accumulate(new Vector3(0f, 0.3f, 0f), isInContact: false);
            // First frame of contact: no "previous contact frame" to diff against.
            metric.Accumulate(new Vector3(0f, 0.02f, 0f), isInContact: true);

            Assert.That(metric.ContactFrameCount, Is.EqualTo(0));
        }

        [Test]
        public void Reset_ClearsAccumulatedState()
        {
            var metric = new FootSlidingMetric();
            metric.Accumulate(new Vector3(0f, 0.02f, 0f), isInContact: true);
            metric.Accumulate(new Vector3(0.01f, 0.02f, 0f), isInContact: true);

            metric.Reset();

            Assert.That(metric.ContactFrameCount, Is.EqualTo(0));
            Assert.That(metric.TotalPlanarDisplacement, Is.EqualTo(0.0));
        }
    }
}
