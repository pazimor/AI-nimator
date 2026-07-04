using NUnit.Framework;
using UnityEngine;
using AInimator.Controller.PostProcess;

namespace AInimator.Controller.Tests
{
    /// <summary>
    /// Tests for the analytic <see cref="TwoBoneIkSolver"/> against
    /// reachable and unreachable targets.
    /// </summary>
    public class TwoBoneIkSolverTests
    {
        [Test]
        public void Solve_ReachableTarget_EffectorLandsOnTarget()
        {
            const float upper = 0.4f;
            const float lower = 0.42f;
            var root = new Vector3(0f, 1f, 0f);
            var mid = new Vector3(0f, 1f - upper, 0.05f);
            var effector = new Vector3(0f, 1f - upper - lower, 0.05f);
            // Well within reach (upper+lower = 0.82m).
            var target = new Vector3(0.1f, 0.5f, 0.1f);

            var result = TwoBoneIkSolver.Solve(root, mid, effector, target, upper, lower);

            Assert.That(result.TargetReachable, Is.True);
            Assert.That(Vector3.Distance(result.EffectorPosition, target), Is.LessThan(1e-3f));

            // Bone lengths must be preserved.
            Assert.That(Vector3.Distance(root, result.MidPosition), Is.EqualTo(upper).Within(1e-3f));
            Assert.That(Vector3.Distance(result.MidPosition, result.EffectorPosition), Is.EqualTo(lower).Within(1e-3f));
        }

        [Test]
        public void Solve_UnreachableTarget_ReportsUnreachableAndFullyExtends()
        {
            const float upper = 0.4f;
            const float lower = 0.42f;
            var root = Vector3.zero;
            var mid = new Vector3(0f, -upper, 0f);
            var effector = new Vector3(0f, -upper - lower, 0f);
            // Far beyond max reach (0.82m).
            var target = new Vector3(0f, -5f, 0f);

            var result = TwoBoneIkSolver.Solve(root, mid, effector, target, upper, lower);

            Assert.That(result.TargetReachable, Is.False);
            // Fully extended: effector distance from root approx upper+lower.
            var reach = Vector3.Distance(root, result.EffectorPosition);
            Assert.That(reach, Is.EqualTo(upper + lower).Within(1e-2f));
        }

        [Test]
        public void Solve_PreservesPoleDirection_KneeStaysOnSameSideAsInputPose()
        {
            const float upper = 0.4f;
            const float lower = 0.42f;
            var root = new Vector3(0f, 1f, 0f);
            // Knee bent forward (+Z).
            var mid = new Vector3(0f, 1f - 0.3f, 0.3f);
            var effector = new Vector3(0f, 1f - 0.6f, 0.3f);
            var target = new Vector3(0f, 0.7f, 0.05f);

            var result = TwoBoneIkSolver.Solve(root, mid, effector, target, upper, lower);

            // The solved knee should still be biased forward (+Z), not
            // flipped to the back, preserving the pose's natural bend.
            Assert.That(result.MidPosition.z, Is.GreaterThan(0f));
        }

        [Test]
        public void Solve_TargetAtRoot_FallsBackWithoutThrowing()
        {
            const float upper = 0.4f;
            const float lower = 0.42f;
            var root = new Vector3(1f, 1f, 1f);
            var mid = new Vector3(1f, 0.6f, 1f);
            var effector = new Vector3(1f, 0.2f, 1f);

            Assert.DoesNotThrow(() =>
            {
                var result = TwoBoneIkSolver.Solve(root, mid, effector, root, upper, lower);
                Assert.That(result.TargetReachable, Is.False);
            });
        }
    }
}
