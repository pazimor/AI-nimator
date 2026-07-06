using NUnit.Framework;
using AInimator.Controller.Rig;

namespace AInimator.Controller.Tests
{
    /// <summary>
    /// Tests for <see cref="RigScale"/>: the rig-scale ratio and its
    /// application to a root-local motion delta (<c>apps/spec/rig_binding.md</c> §2.3).
    /// </summary>
    public class RigScaleTests
    {
        [Test]
        public void FromPelvisHeight_MatchingSmplHeight_ReturnsOne()
        {
            var scale = RigScale.FromPelvisHeight(RigScale.SmplPelvisRestHeight);

            Assert.That(scale, Is.EqualTo(1f).Within(1e-5f));
        }

        [Test]
        public void FromPelvisHeight_DoubleSmplHeight_ReturnsTwo()
        {
            var scale = RigScale.FromPelvisHeight(RigScale.SmplPelvisRestHeight * 2f);

            Assert.That(scale, Is.EqualTo(2f).Within(1e-5f));
        }

        [Test]
        public void FromPelvisHeight_NonPositiveHeight_Throws()
        {
            Assert.Throws<System.ArgumentOutOfRangeException>(() => RigScale.FromPelvisHeight(0f));
            Assert.Throws<System.ArgumentOutOfRangeException>(() => RigScale.FromPelvisHeight(-1f));
        }

        [Test]
        public void ScaleGlobalDelta_ScalesTranslationChannelsOnly()
        {
            float[] rawDelta = { 0.1f, 0.2f, 0.3f, 0.5f }; // (dFwd, dLat, dHeight, dYaw)
            var destination = new float[4];

            RigScale.ScaleGlobalDelta(rawDelta, 2f, destination);

            Assert.That(destination[0], Is.EqualTo(0.2f).Within(1e-6f));
            Assert.That(destination[1], Is.EqualTo(0.4f).Within(1e-6f));
            Assert.That(destination[2], Is.EqualTo(0.6f).Within(1e-6f));
            // Δyaw (channel 3) is dimensionless -- never scaled.
            Assert.That(destination[3], Is.EqualTo(0.5f).Within(1e-6f));
        }

        [Test]
        public void ScaleGlobalDelta_ThreeChannelDelta_ScalesAllOfThem()
        {
            float[] rawDelta = { 1f, 2f, 3f };
            var destination = new float[3];

            RigScale.ScaleGlobalDelta(rawDelta, 0.5f, destination);

            Assert.That(destination[0], Is.EqualTo(0.5f).Within(1e-6f));
            Assert.That(destination[1], Is.EqualTo(1.0f).Within(1e-6f));
            Assert.That(destination[2], Is.EqualTo(1.5f).Within(1e-6f));
        }

        [Test]
        public void ScaleGlobalDelta_ThrowsOnLengthMismatch()
        {
            float[] rawDelta = { 1f, 2f, 3f, 4f };
            var destination = new float[3];

            Assert.Throws<System.ArgumentException>(() => RigScale.ScaleGlobalDelta(rawDelta, 1f, destination));
        }
    }
}
