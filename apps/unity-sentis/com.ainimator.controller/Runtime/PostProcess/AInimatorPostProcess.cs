using System;
using AInimator.Controller.Bundle;
using UnityEngine;

namespace AInimator.Controller.PostProcess
{
    /// <summary>
    /// Wires <see cref="FootContactDetector"/> + <see cref="FootLockIk"/>
    /// (both legs) + <see cref="IdleMoveBlender"/> onto one
    /// <see cref="AInimatorController"/>'s per-frame output, exposing on/off
    /// toggles + the spec's default thresholds in the Inspector
    /// (<c>apps/spec/footlock_blending.md</c>, Goal B phase B4).
    /// </summary>
    /// <remarks>
    /// Pipeline order (spec §1, verbatim): the controller ticks and pushes
    /// its **raw** state into the autoregressive window first
    /// (<see cref="AInimatorController.Tick"/> already does this
    /// internally); only *after* that does this component derive foot
    /// contacts from the raw pose, run the foot-lock IK correction, and
    /// compute the idle/move blend weight — all strictly aval,
    /// non-contaminating. This component does not itself apply the
    /// corrected pose to a skeleton/rig (engine-specific); it exposes the
    /// corrected joint positions + blend weight for a renderer to consume.
    /// </remarks>
    public sealed class AInimatorPostProcess
    {
        [Serializable]
        public sealed class Options
        {
            [Tooltip("Enable foot-contact re-derivation + foot-lock IK.")]
            public bool enableFootLockIk = true;

            [Tooltip("Enable idle<->move pose-space cross-fade.")]
            public bool enableIdleMoveBlend = true;

            [Tooltip("World height below which a foot may be in contact (meters).")]
            public float contactHeightThreshold = FootContactDetector.DefaultHeightThreshold;

            [Tooltip("Planar speed below which a foot may be in contact (meters/frame).")]
            public float contactSpeedThreshold = FootContactDetector.DefaultSpeedThreshold;

            [Tooltip("Consecutive violating frames required to release a foot lock.")]
            public int contactExitFrames = FootContactDetector.DefaultExitFrames;

            [Tooltip("1.5x contactHeightThreshold by convention -- immediate release above this height.")]
            public float releaseHeightMultiplier = FootContactDetector.DefaultReleaseHeightMultiplier;

            [Tooltip("Foot-lock IK correction clamp (meters) -- beyond this, the lock releases outright.")]
            public float ikClampDistance = FootLockIk.DefaultClampDistance;

            [Tooltip("Foot-lock IK release fade duration (seconds).")]
            public float ikReleaseFadeSeconds = FootLockIk.DefaultReleaseFadeSeconds;

            [Tooltip("Raw planar control speed (m/frame) above which the character is considered moving.")]
            public float idleMoveSpeedThreshold = IdleMoveBlender.DefaultMoveSpeedThreshold;

            [Tooltip("Seconds of below-threshold control required before entering idle.")]
            public float idleEntrySeconds = IdleMoveBlender.DefaultIdleEntrySeconds;

            [Tooltip("Idle<->move cross-fade duration (seconds).")]
            public float idleCrossFadeSeconds = IdleMoveBlender.DefaultCrossFadeSeconds;
        }

        /// <summary>Per-leg corrected joint world positions for the current frame.</summary>
        public readonly struct LegResult
        {
            public readonly bool IsLocked;
            public readonly Vector3 HipWorldPosition;
            public readonly Vector3 KneeWorldPosition;
            public readonly Vector3 AnkleWorldPosition;

            public LegResult(bool isLocked, Vector3 hip, Vector3 knee, Vector3 ankle)
            {
                IsLocked = isLocked;
                HipWorldPosition = hip;
                KneeWorldPosition = knee;
                AnkleWorldPosition = ankle;
            }
        }

        private const int LeftLegSlot = 0;
        private const int RightLegSlot = 1;

        private readonly Options _options;
        private readonly int _numBones;

        private readonly FootContactDetector _contactDetector;
        private readonly FootLockIk _leftLegIk;
        private readonly FootLockIk _rightLegIk;
        private readonly IdleMoveBlender _idleMoveBlender;

        private readonly Matrix4x4[] _globalRotationsScratch;
        private readonly Vector3[] _globalPositionsScratch;

        /// <summary>The idle blend weight computed on the most recent <see cref="Process"/> call.</summary>
        public float IdleWeight { get; private set; }

        public AInimatorPostProcess(Manifest manifest, Options options = null)
        {
            if (manifest == null)
            {
                throw new ArgumentNullException(nameof(manifest));
            }

            _options = options ?? new Options();
            _numBones = manifest.num_bones;

            _contactDetector = new FootContactDetector(
                Smpl22Skeleton.FootJointIndices.Length,
                _options.contactHeightThreshold,
                _options.contactSpeedThreshold,
                _options.contactExitFrames,
                _options.releaseHeightMultiplier);

            _leftLegIk = new FootLockIk(
                Smpl22Skeleton.LeftThighLength, Smpl22Skeleton.LeftShinLength,
                _options.ikClampDistance, _options.ikReleaseFadeSeconds);
            _rightLegIk = new FootLockIk(
                Smpl22Skeleton.RightThighLength, Smpl22Skeleton.RightShinLength,
                _options.ikClampDistance, _options.ikReleaseFadeSeconds);

            _idleMoveBlender = new IdleMoveBlender(
                _options.idleMoveSpeedThreshold, _options.idleEntrySeconds, _options.idleCrossFadeSeconds);

            _globalRotationsScratch = new Matrix4x4[_numBones];
            _globalPositionsScratch = new Vector3[_numBones];
        }

        /// <summary>
        /// Post-process one frame. Must be called with the controller's
        /// **raw, already-pushed** bone frame (spec §1 — never before the
        /// autoregressive window has received it).
        /// </summary>
        /// <param name="rawBoneFrame">Raw bone rotation6d frame, row-major (numBones*6).</param>
        /// <param name="rootWorldPosition">Engine-integrated world root position for this frame.</param>
        /// <param name="rawControlVx">Raw (unnormalized) requested control vx this frame.</param>
        /// <param name="rawControlVz">Raw (unnormalized) requested control vz this frame.</param>
        /// <param name="deltaTimeSeconds">Frame time (seconds).</param>
        /// <returns>Corrected left/right leg joint positions (world space) and the idle blend weight.</returns>
        public (LegResult left, LegResult right) Process(
            ReadOnlySpan<float> rawBoneFrame,
            Vector3 rootWorldPosition,
            float rawControlVx,
            float rawControlVz,
            float deltaTimeSeconds)
        {
            SmplForwardKinematics.ComputeJointPositions(
                rawBoneFrame, _numBones, _globalRotationsScratch, _globalPositionsScratch);

            var leftHip = ToWorld(Smpl22Skeleton.LeftHip, rootWorldPosition);
            var leftKnee = ToWorld(Smpl22Skeleton.LeftKnee, rootWorldPosition);
            var leftAnkle = ToWorld(Smpl22Skeleton.LeftAnkle, rootWorldPosition);
            var leftFoot = ToWorld(Smpl22Skeleton.LeftFoot, rootWorldPosition);

            var rightHip = ToWorld(Smpl22Skeleton.RightHip, rootWorldPosition);
            var rightKnee = ToWorld(Smpl22Skeleton.RightKnee, rootWorldPosition);
            var rightAnkle = ToWorld(Smpl22Skeleton.RightAnkle, rootWorldPosition);
            var rightFoot = ToWorld(Smpl22Skeleton.RightFoot, rootWorldPosition);

            bool leftLocked = false, rightLocked = false;
            var leftKneeOut = leftKnee;
            var leftAnkleOut = leftAnkle;
            var rightKneeOut = rightKnee;
            var rightAnkleOut = rightAnkle;

            if (_options.enableFootLockIk)
            {
                // Contact is derived from the *foot* joint (10/11, spec §2)
                // even though the IK effector is the ankle (7/8, spec §3) —
                // the foot joint is the fixed offset child used purely for
                // ground-contact sensing.
                leftLocked = _contactDetector.Update(LeftLegSlot, leftFoot);
                rightLocked = _contactDetector.Update(RightLegSlot, rightFoot);

                (leftKneeOut, leftAnkleOut) = _leftLegIk.Resolve(
                    leftLocked, leftHip, leftKnee, leftAnkle, deltaTimeSeconds);
                (rightKneeOut, rightAnkleOut) = _rightLegIk.Resolve(
                    rightLocked, rightHip, rightKnee, rightAnkle, deltaTimeSeconds);
            }

            IdleWeight = _options.enableIdleMoveBlend
                ? _idleMoveBlender.Update(rawControlVx, rawControlVz, deltaTimeSeconds)
                : 0f;

            var left = new LegResult(leftLocked, leftHip, leftKneeOut, leftAnkleOut);
            var right = new LegResult(rightLocked, rightHip, rightKneeOut, rightAnkleOut);
            return (left, right);
        }

        private Vector3 ToWorld(int boneIndex, Vector3 rootWorldPosition)
        {
            return SmplForwardKinematics.RootLocalToWorld(_globalPositionsScratch[boneIndex], rootWorldPosition);
        }

        /// <summary>Reset all sub-systems to their unlocked/no-history state (e.g. respawn/teleport).</summary>
        public void Reset()
        {
            _contactDetector.Reset();
            _leftLegIk.Reset();
            _rightLegIk.Reset();
            _idleMoveBlender.Reset();
        }
    }
}
