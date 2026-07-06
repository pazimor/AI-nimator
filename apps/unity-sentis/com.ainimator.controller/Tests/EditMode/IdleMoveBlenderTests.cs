using NUnit.Framework;
using UnityEngine;
using AInimator.Controller.PostProcess;

namespace AInimator.Controller.Tests
{
    /// <summary>
    /// Tests for <see cref="IdleMoveBlender"/>: move-speed threshold entry,
    /// idle-entry delay, and cross-fade timing (spec §4).
    /// </summary>
    public class IdleMoveBlenderTests
    {
        [Test]
        public void MoveInput_ImmediatelyLeavesIdle_NoDelay()
        {
            var blender = new IdleMoveBlender();
            // Force idle first.
            for (var i = 0; i < 30; i++)
            {
                blender.Update(0f, 0f, 1f / 30f);
            }

            Assert.That(blender.IsIdle, Is.True);

            blender.Update(0f, 0.033f, 1f / 30f);

            Assert.That(blender.IsIdle, Is.False);
        }

        [Test]
        public void BelowThresholdControl_EntersIdleOnlyAfterEntrySeconds()
        {
            var blender = new IdleMoveBlender(idleEntrySeconds: 0.25f);
            var dt = 1f / 30f;
            var frames = Mathf.CeilToInt(0.25f / dt);

            for (var i = 0; i < frames - 1; i++)
            {
                blender.Update(0f, 0f, dt);
                Assert.That(blender.IsIdle, Is.False, $"must not enter idle before 0.25s elapsed (frame {i})");
            }

            blender.Update(0f, 0f, dt);
            Assert.That(blender.IsIdle, Is.True);
        }

        [Test]
        public void CrossFade_ReachesFullWeightAfterConfiguredDuration()
        {
            var blender = new IdleMoveBlender(idleEntrySeconds: 0f, crossFadeSeconds: 0.2f);
            blender.Update(0f, 0f, 0f); // triggers idle immediately (0s entry delay)

            var dt = 0.05f;
            blender.Update(0f, 0f, dt);
            Assert.That(blender.IdleWeight, Is.GreaterThan(0f).And.LessThan(1f));

            for (var i = 0; i < 10; i++)
            {
                blender.Update(0f, 0f, dt);
            }

            Assert.That(blender.IdleWeight, Is.EqualTo(1f).Within(1e-4f));
        }

        [Test]
        public void MoveSpeedThreshold_ExactlyAtThreshold_IsNotConsideredMoving()
        {
            var blender = new IdleMoveBlender(moveSpeedThreshold: 0.005f, idleEntrySeconds: 0f);
            // Speed exactly at threshold: `> threshold` is false, so this counts as idle-seeking.
            blender.Update(0.005f, 0f, 1f);

            Assert.That(blender.IsIdle, Is.True);
        }
    }
}
