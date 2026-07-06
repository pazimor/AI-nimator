using NUnit.Framework;
using UnityEngine;
using AInimator.Controller.Rig;

namespace AInimator.Controller.Tests
{
    /// <summary>
    /// Tests for <see cref="RigRetargeter"/>: the canonical retarget formula
    /// (<c>apps/spec/rig_binding.md</c> §2.2) on a synthetic 3-bone mini-rig
    /// (pelvis -> spine1 -> neck, matching the start of the real SMPL-22
    /// hierarchy) with known, hand-computed rest-pose offsets, so expected
    /// local rotations can be verified numerically without a live Unity rig.
    /// </summary>
    public class RigRetargeterTests
    {
        private const float Tolerance = 1e-4f;

        [Test]
        public void ComputeWorldTarget_ComposesSmplWorldWithRigRest()
        {
            var smplWorld = Quaternion.Euler(0f, 30f, 0f);
            var rigRest = Quaternion.Euler(0f, 0f, 90f); // e.g. a rig authored with a different rest-pose basis.

            var worldTarget = RigRetargeter.ComputeWorldTarget(smplWorld, rigRest);

            var expected = smplWorld * rigRest;
            Assert.That(Quaternion.Angle(worldTarget, expected), Is.LessThan(Tolerance));
        }

        [Test]
        public void ComputeLocalTarget_WithIdentityParent_EqualsWorldTarget()
        {
            var worldTarget = Quaternion.Euler(10f, 20f, 30f);

            var local = RigRetargeter.ComputeLocalTarget(worldTarget, Quaternion.identity);

            Assert.That(Quaternion.Angle(local, worldTarget), Is.LessThan(Tolerance));
        }

        [Test]
        public void ComputeLocalTarget_SubtractsParentWorldRotation()
        {
            // Parent bone target is a 45-degree yaw; child bone's own world
            // target is a 45-degree yaw plus a 20-degree additional pitch.
            // The expected local (child-relative-to-parent) rotation is
            // exactly the 20-degree pitch.
            var parentWorldTarget = Quaternion.Euler(0f, 45f, 0f);
            var childPitch = Quaternion.Euler(20f, 0f, 0f);
            var childWorldTarget = parentWorldTarget * childPitch;

            var local = RigRetargeter.ComputeLocalTarget(childWorldTarget, parentWorldTarget);

            Assert.That(Quaternion.Angle(local, childPitch), Is.LessThan(Tolerance));
        }

        [Test]
        public void RetargetFrame_TwoBoneChain_ProducesExpectedLocalRotations()
        {
            // Synthetic mini-rig: pelvis (root) -> spine1 -> neck, mirroring
            // Smpl22Skeleton's first three indices (0=pelvis, 3=spine1 is
            // the real hierarchy; here we use indices 0 and 3 explicitly and
            // leave the rest of the 22-slot arrays "unmapped" so only these
            // two bones matter to the assertions).
            const int numBones = Smpl22Skeleton.NumBones;
            var smplWorldRotations = new Matrix4x4[numBones];
            var rigRestWorldRotations = new Quaternion[numBones];
            var isMapped = new bool[numBones];
            var nearestMappedParent = new int[numBones];

            for (var i = 0; i < numBones; i++)
            {
                smplWorldRotations[i] = Matrix4x4.identity;
                rigRestWorldRotations[i] = Quaternion.identity;
            }

            const int pelvis = Smpl22Skeleton.Pelvis; // 0, parent -1
            const int spine1 = Smpl22Skeleton.Spine1; // 3, parent pelvis

            // Pelvis: SMPL predicts a 90-degree yaw this frame; the rig's
            // rest pose has no offset (identity) -- so pelvis local target
            // == the 90-degree yaw itself.
            var pelvisSmplWorld = Quaternion.Euler(0f, 90f, 0f);
            smplWorldRotations[pelvis] = Matrix4x4.Rotate(pelvisSmplWorld);
            isMapped[pelvis] = true;
            nearestMappedParent[pelvis] = -1;

            // Spine1: SMPL predicts a further 90-degree yaw *on top of* the
            // pelvis's world rotation (Matrix4x4 world rotations from FK are
            // already composed with the parent -- mirrors
            // SmplForwardKinematics.ComputeJointPositions's globalRotations
            // semantics), and the rig has a rest-pose offset of a 10-degree
            // pitch (a rig authored with a slight forward lean at spine1).
            var spine1SmplWorld = pelvisSmplWorld * Quaternion.Euler(0f, 90f, 0f);
            smplWorldRotations[spine1] = Matrix4x4.Rotate(spine1SmplWorld);
            var spine1RigRest = Quaternion.Euler(10f, 0f, 0f);
            rigRestWorldRotations[spine1] = spine1RigRest;
            isMapped[spine1] = true;
            nearestMappedParent[spine1] = RigRetargeter.FindNearestMappedParent(spine1, isMapped);

            var worldTargetsScratch = new Quaternion[numBones];
            var localTargets = new Quaternion[numBones];

            RigRetargeter.RetargetFrame(
                smplWorldRotations, rigRestWorldRotations, isMapped, nearestMappedParent,
                worldTargetsScratch, localTargets);

            // Pelvis: identity rig rest -> local target equals the SMPL world yaw directly.
            Assert.That(Quaternion.Angle(localTargets[pelvis], pelvisSmplWorld), Is.LessThan(Tolerance));

            // Spine1: worldTarget = spine1SmplWorld * spine1RigRest;
            // localTarget = pelvisWorldTarget^-1 * spine1WorldTarget.
            var pelvisWorldTarget = RigRetargeter.ComputeWorldTarget(pelvisSmplWorld, Quaternion.identity);
            var spine1WorldTarget = RigRetargeter.ComputeWorldTarget(spine1SmplWorld, spine1RigRest);
            var expectedSpine1Local = Quaternion.Inverse(pelvisWorldTarget) * spine1WorldTarget;

            Assert.That(Quaternion.Angle(localTargets[spine1], expectedSpine1Local), Is.LessThan(Tolerance));
        }

        [Test]
        public void FindNearestMappedParent_SkipsUnmappedAncestors()
        {
            // leftKnee's real parent is leftHip; if leftHip is unmapped but
            // pelvis (leftHip's own parent) is mapped, the nearest mapped
            // parent of leftKnee must be pelvis, not leftHip (spec §2.1:
            // an unmapped bone is transparent to the hierarchy).
            const int numBones = Smpl22Skeleton.NumBones;
            var isMapped = new bool[numBones];
            isMapped[Smpl22Skeleton.Pelvis] = true;
            isMapped[Smpl22Skeleton.LeftHip] = false;
            isMapped[Smpl22Skeleton.LeftKnee] = true;

            var parent = RigRetargeter.FindNearestMappedParent(Smpl22Skeleton.LeftKnee, isMapped);

            Assert.That(parent, Is.EqualTo(Smpl22Skeleton.Pelvis));
        }

        [Test]
        public void FindNearestMappedParent_RootWithNoMappedAncestor_ReturnsMinusOne()
        {
            const int numBones = Smpl22Skeleton.NumBones;
            var isMapped = new bool[numBones];
            isMapped[Smpl22Skeleton.LeftWrist] = true;
            // Every ancestor of leftWrist left unmapped.

            var parent = RigRetargeter.FindNearestMappedParent(Smpl22Skeleton.LeftWrist, isMapped);

            Assert.That(parent, Is.EqualTo(-1));
        }

        [Test]
        public void RetargetFrame_ThrowsOnMismatchedSpanLengths()
        {
            var smplWorldRotations = new Matrix4x4[3];
            var rigRestWorldRotations = new Quaternion[2]; // mismatched length
            var isMapped = new bool[3];
            var nearestMappedParent = new int[3];
            var worldTargetsScratch = new Quaternion[3];
            var localTargets = new Quaternion[3];

            Assert.Throws<System.ArgumentException>(() =>
                RigRetargeter.RetargetFrame(
                    smplWorldRotations, rigRestWorldRotations, isMapped, nearestMappedParent,
                    worldTargetsScratch, localTargets));
        }
    }
}
