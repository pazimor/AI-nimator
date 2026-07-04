using System;
using AInimator.Controller.PostProcess;
using UnityEngine;

namespace AInimator.Controller.Rig
{
    /// <summary>
    /// Applies the canonical SMPL-22 -&gt; rig retargeting
    /// (<c>apps/spec/rig_binding.md</c> §2) to a real character rig every
    /// frame, using a <see cref="Rig.RigMap"/> asset and the FK-derived
    /// world rotations from <see cref="SmplForwardKinematics"/>.
    /// </summary>
    /// <remarks>
    /// Pipeline position (spec §1): this is the last stage before the pose
    /// reaches the rendered skeleton, applied after
    /// <c>AInimatorPostProcess</c> (B4 foot-lock/idle-blend) runs for the
    /// frame. It consumes the controller's rotation6d bone frame directly
    /// (B4's corrections are joint-<b>position</b>-space, not fed back into
    /// rotation6d — see <c>AInimatorCharacter</c>'s remarks) and never
    /// writes back into the controller's autoregressive <c>StateBuffer</c>
    /// (that window only ever sees the raw controller output, per B4 spec
    /// §1 — this class does not touch it at all).
    /// <para/>
    /// Calibration happens once, in <see cref="Calibrate"/> (called from
    /// <c>Awake</c>/<c>Start</c> by default): captures each mapped bone's
    /// world rotation while the rig sits in its own rest pose, plus the
    /// pelvis rest height for <see cref="Rig.RigScale"/>. Bone
    /// <b>translations</b> are never written by <see cref="Apply"/> — only
    /// <see cref="Transform.localRotation"/> — preserving the rig's
    /// proportions (spec §2.2 last line).
    /// </remarks>
    public sealed class RigBinder : MonoBehaviour
    {
        [Tooltip("Bone-name -> Transform map for this rig (spec §2.1). Required.")]
        [SerializeField] private RigMap rigMap;

        [Tooltip("If true, capture the rig's rest-pose calibration in Awake(). Uncheck to call Calibrate() manually (e.g. after procedurally attaching a rig).")]
        [SerializeField] private bool calibrateOnAwake = true;

        [Tooltip("Root Transform to receive root motion (position + yaw), scaled by rigScale (spec §2.3). Defaults to this GameObject's transform.")]
        [SerializeField] private Transform actorRoot;

        private Quaternion[] _rigRestWorldRotations;
        private bool[] _isMapped;
        private int[] _nearestMappedParent;
        private Quaternion[] _worldTargetsScratch;
        private Quaternion[] _localTargetsScratch;
        private Matrix4x4[] _smplGlobalRotationsScratch;
        private Vector3[] _smplGlobalPositionsScratch;

        /// <summary>True once <see cref="Calibrate"/> has run successfully.</summary>
        public bool IsCalibrated { get; private set; }

        /// <summary><c>rigScale</c> (spec §2.3): rig pelvis rest height / SMPL pelvis rest height. Valid only after <see cref="Calibrate"/>.</summary>
        public float RigScaleFactor { get; private set; } = 1f;

        /// <summary>The active <see cref="RigMap"/>.</summary>
        public RigMap Map
        {
            get => rigMap;
            set => rigMap = value;
        }

        private void Awake()
        {
            if (actorRoot == null)
            {
                actorRoot = transform;
            }

            if (calibrateOnAwake && rigMap != null)
            {
                Calibrate();
            }
        }

        /// <summary>
        /// Capture the rig's current pose as its rest pose (spec §2.2 step 1
        /// requires the rig to already be posed in its rest pose when this
        /// is called): for every mapped bone, record its world rotation and
        /// its nearest-mapped-ancestor index; also compute
        /// <see cref="RigScaleFactor"/> from the pelvis bone's world height.
        /// </summary>
        public void Calibrate()
        {
            if (rigMap == null)
            {
                throw new InvalidOperationException("RigBinder.Calibrate: no RigMap assigned.");
            }

            rigMap.EnsureEntriesSized();

            var numBones = Smpl22Skeleton.NumBones;
            _rigRestWorldRotations = new Quaternion[numBones];
            _isMapped = new bool[numBones];
            _nearestMappedParent = new int[numBones];
            _worldTargetsScratch = new Quaternion[numBones];
            _localTargetsScratch = new Quaternion[numBones];
            _smplGlobalRotationsScratch = new Matrix4x4[numBones];
            _smplGlobalPositionsScratch = new Vector3[numBones];

            for (var bone = 0; bone < numBones; bone++)
            {
                var rigBone = rigMap.GetRigBone(bone);
                _isMapped[bone] = rigBone != null;
                _rigRestWorldRotations[bone] = rigBone != null ? rigBone.rotation : Quaternion.identity;
            }

            for (var bone = 0; bone < numBones; bone++)
            {
                _nearestMappedParent[bone] = _isMapped[bone]
                    ? RigRetargeter.FindNearestMappedParent(bone, _isMapped)
                    : -1;
            }

            var pelvisBone = rigMap.GetRigBone(Smpl22Skeleton.Pelvis);
            RigScaleFactor = pelvisBone != null
                ? RigScale.FromPelvisHeight(Mathf.Max(pelvisBone.position.y, 1e-4f))
                : 1f;

            IsCalibrated = true;
        }

        /// <summary>
        /// Apply one frame's corrected SMPL-22 bone rotations to the rig
        /// (spec §2.2/§2.3). Must be called after <see cref="Calibrate"/>
        /// and after the frame's foot-lock/idle-blend post-processing.
        /// </summary>
        /// <param name="boneFrame">Row-major rotation6d, length <c>numBones * 6</c> (the controller's per-frame bone output).</param>
        /// <param name="rawGlobalDelta">This frame's root-local motion delta <c>(Δforward, Δlateral, Δheight, Δyaw)</c>, pre-scale.</param>
        public void Apply(ReadOnlySpan<float> boneFrame, ReadOnlySpan<float> rawGlobalDelta)
        {
            if (!IsCalibrated)
            {
                throw new InvalidOperationException("RigBinder.Apply called before Calibrate().");
            }

            var numBones = Smpl22Skeleton.NumBones;
            SmplForwardKinematics.ComputeJointPositions(
                correctedBoneFrame, numBones, _smplGlobalRotationsScratch, _smplGlobalPositionsScratch);

            RigRetargeter.RetargetFrame(
                _smplGlobalRotationsScratch,
                _rigRestWorldRotations,
                _isMapped,
                _nearestMappedParent,
                _worldTargetsScratch,
                _localTargetsScratch);

            // Rotation only -- bone translations are never modified
            // (spec §2.2 last line: rig proportions preserved).
            ApplyLocalRotations(numBones);
            ApplyRootMotion(rawGlobalDelta);
        }

        private void ApplyLocalRotations(int numBones)
        {
            for (var bone = 0; bone < numBones; bone++)
            {
                if (!_isMapped[bone])
                {
                    continue;
                }

                var rigBone = rigMap.GetRigBone(bone);
                rigBone.localRotation = _localTargetsScratch[bone];
            }
        }

        private void ApplyRootMotion(ReadOnlySpan<float> rawGlobalDelta)
        {
            if (rawGlobalDelta.Length == 0 || actorRoot == null)
            {
                return;
            }

            Span<float> scaled = stackalloc float[rawGlobalDelta.Length];
            RigScale.ScaleGlobalDelta(rawGlobalDelta, RigScaleFactor, scaled);

            var dFwd = scaled[0];
            var dLat = scaled.Length > 1 ? scaled[1] : 0f;
            var dHeight = scaled.Length > 2 ? scaled[2] : 0f;
            var dYaw = scaled.Length > 3 ? scaled[3] : 0f;

            var yaw = actorRoot.eulerAngles.y * Mathf.Deg2Rad;
            var cos = Mathf.Cos(yaw);
            var sin = Mathf.Sin(yaw);
            var worldDx = cos * dFwd - sin * dLat;
            var worldDz = sin * dFwd + cos * dLat;

            actorRoot.position += new Vector3(worldDx, dHeight, worldDz);
            actorRoot.Rotate(0f, dYaw * Mathf.Rad2Deg, 0f, Space.World);
        }
    }
}
