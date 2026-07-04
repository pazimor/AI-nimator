using NUnit.Framework;
using UnityEngine;
using AInimator.Controller.PostProcess;

namespace AInimator.Controller.Tests
{
    /// <summary>
    /// Tests for <see cref="SmplForwardKinematics"/>: identity T-pose sanity
    /// checks (root at origin, known bone offsets reproduced), matching the
    /// intent of <c>ops.py::rot6dToJointXYZ</c>.
    /// </summary>
    public class SmplForwardKinematicsTests
    {
        private const int NumBones = Smpl22Skeleton.NumBones;

        // Identity 6D rotation: a1=(1,0,0), a2=(0,1,0) -> identity matrix.
        private static float[] IdentityBoneFrame()
        {
            var frame = new float[NumBones * Smpl22Skeleton.RotationChannelsPerBone];
            for (var bone = 0; bone < NumBones; bone++)
            {
                var offset = bone * Smpl22Skeleton.RotationChannelsPerBone;
                frame[offset + 0] = 1f;
                frame[offset + 1] = 0f;
                frame[offset + 2] = 0f;
                frame[offset + 3] = 0f;
                frame[offset + 4] = 1f;
                frame[offset + 5] = 0f;
            }

            return frame;
        }

        [Test]
        public void IdentityPose_PelvisAtOrigin()
        {
            var frame = IdentityBoneFrame();
            var rotations = new Matrix4x4[NumBones];
            var positions = new Vector3[NumBones];

            SmplForwardKinematics.ComputeJointPositions(frame, NumBones, rotations, positions);

            Assert.That(positions[Smpl22Skeleton.Pelvis], Is.EqualTo(Vector3.zero));
        }

        [Test]
        public void IdentityPose_LeftKneePosition_IsSumOfHipAndKneeOffsets()
        {
            var frame = IdentityBoneFrame();
            var rotations = new Matrix4x4[NumBones];
            var positions = new Vector3[NumBones];

            SmplForwardKinematics.ComputeJointPositions(frame, NumBones, rotations, positions);

            var expected = Smpl22Skeleton.BoneOffsets[Smpl22Skeleton.LeftHip]
                           + Smpl22Skeleton.BoneOffsets[Smpl22Skeleton.LeftKnee];

            Assert.That(Vector3.Distance(positions[Smpl22Skeleton.LeftKnee], expected), Is.LessThan(1e-5f));
        }

        [Test]
        public void IdentityPose_LeftAnklePosition_ChainsThroughHipKneeAnkle()
        {
            var frame = IdentityBoneFrame();
            var rotations = new Matrix4x4[NumBones];
            var positions = new Vector3[NumBones];

            SmplForwardKinematics.ComputeJointPositions(frame, NumBones, rotations, positions);

            var expected = Smpl22Skeleton.BoneOffsets[Smpl22Skeleton.LeftHip]
                           + Smpl22Skeleton.BoneOffsets[Smpl22Skeleton.LeftKnee]
                           + Smpl22Skeleton.BoneOffsets[Smpl22Skeleton.LeftAnkle];

            Assert.That(Vector3.Distance(positions[Smpl22Skeleton.LeftAnkle], expected), Is.LessThan(1e-5f));
        }

        [Test]
        public void RootLocalToWorld_TranslatesByRootWorldPositionOnly()
        {
            var rootLocal = new Vector3(0.1f, 0.2f, 0.3f);
            var rootWorld = new Vector3(5f, 0f, -2f);

            var world = SmplForwardKinematics.RootLocalToWorld(rootLocal, rootWorld);

            Assert.That(world, Is.EqualTo(rootLocal + rootWorld));
        }

        [Test]
        public void SixDToRotationMatrix_IdentityInput_ProducesIdentityMatrix()
        {
            float[] sixd = { 1f, 0f, 0f, 0f, 1f, 0f };

            SmplForwardKinematics.SixDToRotationMatrix(sixd, out var rotation);

            Assert.That(rotation.isIdentity, Is.True);
        }

        [Test]
        public void ComputeJointPositions_ThrowsOnLengthMismatch()
        {
            var badFrame = new float[10];
            var rotations = new Matrix4x4[NumBones];
            var positions = new Vector3[NumBones];

            Assert.Throws<System.ArgumentException>(() =>
                SmplForwardKinematics.ComputeJointPositions(badFrame, NumBones, rotations, positions));
        }
    }
}
