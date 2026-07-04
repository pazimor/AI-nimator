using System;
using AInimator.Controller.PostProcess;
using UnityEngine;

namespace AInimator.Controller.Rig
{
    /// <summary>
    /// Pure retargeting math (<c>apps/spec/rig_binding.md</c> §2.2, canonical
    /// formula, byte-for-byte identical intent to the Unreal implementation):
    /// given the FK-derived SMPL-22 world rotations for one frame and a
    /// rig's calibrated rest-pose world rotations, compute the rig's
    /// per-bone local rotation to apply.
    /// </summary>
    /// <remarks>
    /// Free of any <see cref="UnityEngine.Component"/>/<see cref="Transform"/>
    /// mutation so it is unit-testable (EditMode, no live scene) against a
    /// synthetic mini-rig fixture. <see cref="Rig.RigBinder"/> is the
    /// MonoBehaviour that owns calibration state and actually assigns the
    /// results to real <see cref="Transform"/>s.
    /// <para/>
    /// Formula (spec §2.2, quoted verbatim):
    /// <code>
    /// worldRot_target(bone)  = R_smpl_world(bone) * worldRot_rig_rest(bone)
    /// localRot_target(bone)  = worldRot_target(parent)^-1 * worldRot_target(bone)
    /// </code>
    /// where <c>parent</c> is the nearest **mapped** ancestor in the rig (an
    /// SMPL bone with no rig equivalent is skipped when walking up the
    /// hierarchy — spec §2.1). Bone translations are never touched here;
    /// callers must not write <see cref="Transform.localPosition"/> from
    /// this class's output.
    /// </remarks>
    public static class RigRetargeter
    {
        /// <summary>
        /// Compute the world-space target rotation for one bone:
        /// <c>R_smpl_world(bone) * worldRot_rig_rest(bone)</c>.
        /// </summary>
        public static Quaternion ComputeWorldTarget(Quaternion smplWorldRotation, Quaternion rigRestWorldRotation)
        {
            return smplWorldRotation * rigRestWorldRotation;
        }

        /// <summary>
        /// Compute the local rotation to apply to a mapped rig bone, given
        /// its own world target rotation and its nearest-mapped-ancestor's
        /// world target rotation (identity if the bone is the root of the
        /// mapped chain, i.e. no mapped ancestor exists).
        /// </summary>
        public static Quaternion ComputeLocalTarget(Quaternion boneWorldTarget, Quaternion parentWorldTarget)
        {
            return Quaternion.Inverse(parentWorldTarget) * boneWorldTarget;
        }

        /// <summary>
        /// Convert a <see cref="SmplForwardKinematics"/> global rotation
        /// (<see cref="Matrix4x4"/>, root-local frame) into a
        /// <see cref="Quaternion"/> for use with <see cref="ComputeWorldTarget"/>.
        /// </summary>
        public static Quaternion ToQuaternion(Matrix4x4 rotationMatrix)
        {
            return rotationMatrix.rotation;
        }

        /// <summary>
        /// Retarget one full frame: for every SMPL bone mapped in
        /// <paramref name="parentMapped"/> (nearest mapped ancestor index,
        /// or -1 for none), compute its local target rotation into
        /// <paramref name="localTargets"/>. Unmapped bones
        /// (<paramref name="isMapped"/> false) are left untouched in the
        /// output (caller must not apply them — spec §2.1: unmapped = ignored).
        /// </summary>
        /// <param name="smplWorldRotations">Per-bone SMPL world rotations (root-local frame), length numBones.</param>
        /// <param name="rigRestWorldRotations">Per-bone calibrated rig rest world rotations, length numBones (only meaningful where <paramref name="isMapped"/> is true).</param>
        /// <param name="isMapped">Per-bone: true if this SMPL bone has a rig <see cref="Transform"/> assigned.</param>
        /// <param name="nearestMappedParent">Per-bone: index of the nearest mapped ancestor, or -1 if none (this bone is the mapped root).</param>
        /// <param name="localTargets">Output: per-bone local rotation to apply (only written where <paramref name="isMapped"/> is true).</param>
        public static void RetargetFrame(
            ReadOnlySpan<Matrix4x4> smplWorldRotations,
            ReadOnlySpan<Quaternion> rigRestWorldRotations,
            ReadOnlySpan<bool> isMapped,
            ReadOnlySpan<int> nearestMappedParent,
            Span<Quaternion> worldTargetsScratch,
            Span<Quaternion> localTargets)
        {
            var numBones = smplWorldRotations.Length;
            if (rigRestWorldRotations.Length != numBones || isMapped.Length != numBones ||
                nearestMappedParent.Length != numBones || worldTargetsScratch.Length != numBones ||
                localTargets.Length != numBones)
            {
                throw new ArgumentException("All spans passed to RetargetFrame must have the same length (numBones).");
            }

            for (var bone = 0; bone < numBones; bone++)
            {
                if (!isMapped[bone])
                {
                    continue;
                }

                var smplWorld = ToQuaternion(smplWorldRotations[bone]);
                worldTargetsScratch[bone] = ComputeWorldTarget(smplWorld, rigRestWorldRotations[bone]);
            }

            for (var bone = 0; bone < numBones; bone++)
            {
                if (!isMapped[bone])
                {
                    continue;
                }

                var parent = nearestMappedParent[bone];
                var parentWorldTarget = parent >= 0 ? worldTargetsScratch[parent] : Quaternion.identity;
                localTargets[bone] = ComputeLocalTarget(worldTargetsScratch[bone], parentWorldTarget);
            }
        }

        /// <summary>
        /// Walk <see cref="Smpl22Skeleton.ParentIndices"/> upward from
        /// <paramref name="boneIndex"/> until a mapped ancestor is found (or
        /// the root is reached), skipping unmapped bones (spec §2.1: an
        /// unmapped SMPL bone is transparent to the hierarchy for retargeting
        /// purposes — its rotation is simply not composed anywhere).
        /// </summary>
        public static int FindNearestMappedParent(int boneIndex, ReadOnlySpan<bool> isMapped)
        {
            var parent = Smpl22Skeleton.ParentIndices[boneIndex];
            while (parent >= 0)
            {
                if (isMapped[parent])
                {
                    return parent;
                }

                parent = Smpl22Skeleton.ParentIndices[parent];
            }

            return -1;
        }
    }
}
