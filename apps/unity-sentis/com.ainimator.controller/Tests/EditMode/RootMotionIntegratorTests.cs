using AInimator.Controller.Runtime;
using NUnit.Framework;

namespace AInimator.Controller.Tests
{
    /// <summary>
    /// Verifies the C# integration matches
    /// <c>ainimator.model.controller_rollout._integrateOneStep</c> exactly
    /// (same yaw-rotation formula, Y-up right-handed).
    /// </summary>
    public class RootMotionIntegratorTests
    {
        [Test]
        public void Integrate_AtZeroYaw_ForwardMapsToPlusZ()
        {
            var integrator = RootMotionIntegrator.AtOrigin();
            integrator.Integrate(new float[] { 1f, 0f, 0f, 0f }); // dFwd=1

            Assert.That(integrator.PositionX, Is.EqualTo(0f).Within(1e-6f));
            Assert.That(integrator.PositionZ, Is.EqualTo(1f).Within(1e-6f));
            Assert.That(integrator.PositionY, Is.EqualTo(0f).Within(1e-6f));
        }

        [Test]
        public void Integrate_AtZeroYaw_LateralMapsToPlusX()
        {
            var integrator = RootMotionIntegrator.AtOrigin();
            integrator.Integrate(new float[] { 0f, 1f, 0f, 0f }); // dLat=1

            Assert.That(integrator.PositionX, Is.EqualTo(1f).Within(1e-6f));
            Assert.That(integrator.PositionZ, Is.EqualTo(0f).Within(1e-6f));
        }

        [Test]
        public void Integrate_AccumulatesYaw()
        {
            var integrator = RootMotionIntegrator.AtOrigin();
            integrator.Integrate(new float[] { 0f, 0f, 0f, 0.1f });
            integrator.Integrate(new float[] { 0f, 0f, 0f, 0.2f });

            Assert.That(integrator.YawRadians, Is.EqualTo(0.3f).Within(1e-6f));
        }

        [Test]
        public void Integrate_AfterQuarterTurn_ForwardMapsToPlusX()
        {
            var integrator = RootMotionIntegrator.AtOrigin();
            integrator.Integrate(new float[] { 0f, 0f, 0f, System.MathF.PI / 2f });
            integrator.Integrate(new float[] { 1f, 0f, 0f, 0f });

            Assert.That(integrator.PositionX, Is.EqualTo(1f).Within(1e-4f));
            Assert.That(integrator.PositionZ, Is.EqualTo(0f).Within(1e-4f));
        }

        [Test]
        public void Integrate_HeightDeltaMapsDirectlyToPositionY()
        {
            var integrator = RootMotionIntegrator.AtOrigin();
            integrator.Integrate(new float[] { 0f, 0f, 0.05f, 0f });

            Assert.That(integrator.PositionY, Is.EqualTo(0.05f).Within(1e-6f));
        }
    }
}
