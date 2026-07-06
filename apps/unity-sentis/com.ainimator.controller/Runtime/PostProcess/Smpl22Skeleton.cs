using UnityEngine;

namespace AInimator.Controller.PostProcess
{
    /// <summary>
    /// SMPL-22 kinematic constants (bone order, parent hierarchy, T-pose
    /// bone-local offsets) ported to C# for engine-side forward kinematics.
    /// </summary>
    /// <remarks>
    /// Source of truth (never edit these values independently — regenerate
    /// this file from the Python side if the skeleton constants ever
    /// change): <c>src/ainimator/core/constants/skeletons.py</c>
    /// (<c>SMPL22_BONE_ORDER</c>, <c>SMPL22_HIERARCHY</c>,
    /// <c>SMPL22_DEFAULT_OFFSETS</c>), consumed by
    /// <c>src/ainimator/geometry/components/ops.py::smpl22KinematicParams</c>
    /// / <c>rot6dToJointXYZ</c>. Offsets are approximate T-pose bone-local
    /// translations for the mean SMPL shape, in meters, <c>(x, y, z)</c>
    /// Y-up. The engine only needs this for foot-lock IK's leg chains
    /// (pelvis..ankle, indices 0-8 plus the two foot joints 10/11) — B4
    /// scope, `apps/spec/footlock_blending.md` §2/§3.
    /// </remarks>
    public static class Smpl22Skeleton
    {
        public const int NumBones = 22;
        public const int RotationChannelsPerBone = 6;

        public const int Pelvis = 0;
        public const int LeftHip = 1;
        public const int RightHip = 2;
        public const int Spine1 = 3;
        public const int LeftKnee = 4;
        public const int RightKnee = 5;
        public const int Spine2 = 6;
        public const int LeftAnkle = 7;
        public const int RightAnkle = 8;
        public const int Spine3 = 9;
        public const int LeftFoot = 10;
        public const int RightFoot = 11;
        public const int Neck = 12;
        public const int LeftCollar = 13;
        public const int RightCollar = 14;
        public const int Head = 15;
        public const int LeftShoulder = 16;
        public const int RightShoulder = 17;
        public const int LeftElbow = 18;
        public const int RightElbow = 19;
        public const int LeftWrist = 20;
        public const int RightWrist = 21;

        /// <summary>Foot-contact joints (footlock_blending.md §2): left, right.</summary>
        public static readonly int[] FootJointIndices = { LeftFoot, RightFoot };

        /// <summary>Left-leg two-bone IK chain (footlock_blending.md §3): hip -> knee -> ankle.</summary>
        public static readonly int[] LeftLegChain = { LeftHip, LeftKnee, LeftAnkle };

        /// <summary>Right-leg two-bone IK chain (footlock_blending.md §3): hip -> knee -> ankle.</summary>
        public static readonly int[] RightLegChain = { RightHip, RightKnee, RightAnkle };

        /// <summary>Parent index per bone, -1 for the root (pelvis).</summary>
        public static readonly int[] ParentIndices =
        {
            /* pelvis        */ -1,
            /* leftHip       */ Pelvis,
            /* rightHip      */ Pelvis,
            /* spine1        */ Pelvis,
            /* leftKnee      */ LeftHip,
            /* rightKnee     */ RightHip,
            /* spine2        */ Spine1,
            /* leftAnkle     */ LeftKnee,
            /* rightAnkle    */ RightKnee,
            /* spine3        */ Spine2,
            /* leftFoot      */ LeftAnkle,
            /* rightFoot     */ RightAnkle,
            /* neck          */ Spine3,
            /* leftCollar    */ Spine3,
            /* rightCollar   */ Spine3,
            /* head          */ Neck,
            /* leftShoulder  */ LeftCollar,
            /* rightShoulder */ RightCollar,
            /* leftElbow     */ LeftShoulder,
            /* rightElbow    */ RightShoulder,
            /* leftWrist     */ LeftElbow,
            /* rightWrist    */ RightElbow,
        };

        /// <summary>
        /// Bone-local T-pose offsets (meters, Y-up), row-major
        /// <c>(NumBones, 3)</c> matching <c>SMPL22_DEFAULT_OFFSETS</c>
        /// exactly (same order, same values).
        /// </summary>
        public static readonly Vector3[] BoneOffsets =
        {
            /* pelvis        */ new(0.0f, 0.0f, 0.0f),
            /* leftHip       */ new(0.07f, -0.04f, 0.0f),
            /* rightHip      */ new(-0.07f, -0.04f, 0.0f),
            /* spine1        */ new(0.0f, 0.1f, 0.02f),
            /* leftKnee      */ new(0.0f, -0.40f, 0.0f),
            /* rightKnee     */ new(0.0f, -0.40f, 0.0f),
            /* spine2        */ new(0.0f, 0.15f, -0.02f),
            /* leftAnkle     */ new(0.0f, -0.42f, 0.0f),
            /* rightAnkle    */ new(0.0f, -0.42f, 0.0f),
            /* spine3        */ new(0.0f, 0.15f, 0.0f),
            /* leftFoot      */ new(0.0f, -0.06f, 0.12f),
            /* rightFoot     */ new(0.0f, -0.06f, 0.12f),
            /* neck          */ new(0.0f, 0.12f, 0.0f),
            /* leftCollar    */ new(0.06f, 0.08f, -0.02f),
            /* rightCollar   */ new(-0.06f, 0.08f, -0.02f),
            /* head          */ new(0.0f, 0.12f, 0.04f),
            /* leftShoulder  */ new(0.12f, 0.0f, 0.0f),
            /* rightShoulder */ new(-0.12f, 0.0f, 0.0f),
            /* leftElbow     */ new(0.26f, 0.0f, 0.0f),
            /* rightElbow    */ new(-0.26f, 0.0f, 0.0f),
            /* leftWrist     */ new(0.24f, 0.0f, 0.0f),
            /* rightWrist    */ new(-0.24f, 0.0f, 0.0f),
        };

        /// <summary>Length of the left thigh (hip -> knee), meters, from <see cref="BoneOffsets"/>.</summary>
        public static float LeftThighLength => BoneOffsets[LeftKnee].magnitude;

        /// <summary>Length of the left shin (knee -> ankle), meters, from <see cref="BoneOffsets"/>.</summary>
        public static float LeftShinLength => BoneOffsets[LeftAnkle].magnitude;

        /// <summary>Length of the right thigh (hip -> knee), meters, from <see cref="BoneOffsets"/>.</summary>
        public static float RightThighLength => BoneOffsets[RightKnee].magnitude;

        /// <summary>Length of the right shin (knee -> ankle), meters, from <see cref="BoneOffsets"/>.</summary>
        public static float RightShinLength => BoneOffsets[RightAnkle].magnitude;
    }
}
