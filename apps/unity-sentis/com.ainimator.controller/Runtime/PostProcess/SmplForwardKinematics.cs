using System;
using UnityEngine;

namespace AInimator.Controller.PostProcess
{
    /// <summary>
    /// Minimal forward-kinematics solver for the SMPL-22 skeleton: converts
    /// a rotation6d bone frame into root-local joint world-orientation
    /// rotations + positions. Ports the strict necessary subset of
    /// <c>src/ainimator/geometry/components/ops.py::rot6dToJointXYZ</c> /
    /// <c>sixdToRotationMatrix</c> to C# (Goal B verite #1: the plugin
    /// reads the model's output, it never redefines geometry independently
    /// — this is a straight port, not a reinterpretation).
    /// </summary>
    /// <remarks>
    /// Joint positions returned by <see cref="ComputeJointPositions"/> are
    /// in the **root-local frame**: the pelvis (bone 0) sits at the origin,
    /// and every other joint's position is relative to it. Its rotation6d
    /// already encodes the skeleton's actual world-facing orientation
    /// (<c>ainimator.geometry.root_local</c>: pelvis yaw is extracted
    /// directly from <c>rot6d[0]</c>, never re-derived from the engine's
    /// integrated yaw) — so callers add the engine's integrated
    /// world-space root position (never re-apply the integrated yaw as an
    /// extra rotation; see <see cref="RootLocalToWorld"/>).
    /// </remarks>
    public static class SmplForwardKinematics
    {
        /// <summary>
        /// Convert one bone's 6D rotation representation to a 3x3 rotation
        /// matrix via Gram-Schmidt orthogonalization (matches
        /// <c>sixdToRotationMatrix</c> exactly: columns are
        /// <c>(b1, b2, b3)</c> with <c>b1 = normalize(a1)</c>,
        /// <c>b2 = normalize(a2 - (b1.a2)b1)</c>, <c>b3 = b1 x b2</c>).
        /// </summary>
        public static void SixDToRotationMatrix(ReadOnlySpan<float> sixd, out Matrix4x4 rotation)
        {
            var a1 = new Vector3(sixd[0], sixd[1], sixd[2]);
            var a2 = new Vector3(sixd[3], sixd[4], sixd[5]);

            var b1 = a1.normalized;
            var dot = Vector3.Dot(b1, a2);
            var b2 = (a2 - dot * b1).normalized;
            var b3 = Vector3.Cross(b1, b2);

            rotation = Matrix4x4.identity;
            // Columns, matching torch.stack([b1, b2, b3], dim=-1).
            rotation.SetColumn(0, new Vector4(b1.x, b1.y, b1.z, 0f));
            rotation.SetColumn(1, new Vector4(b2.x, b2.y, b2.z, 0f));
            rotation.SetColumn(2, new Vector4(b3.x, b3.y, b3.z, 0f));
            rotation.SetColumn(3, new Vector4(0f, 0f, 0f, 1f));
        }

        /// <summary>
        /// Compute root-local global rotations and positions for every bone
        /// in <paramref name="boneFrame"/> (row-major, <c>numBones * 6</c>),
        /// following <see cref="Smpl22Skeleton.ParentIndices"/> /
        /// <see cref="Smpl22Skeleton.BoneOffsets"/>. Both output arrays must
        /// already be sized <c>numBones</c> (caller-owned, no per-frame
        /// allocation on the hot path).
        /// </summary>
        public static void ComputeJointPositions(
            ReadOnlySpan<float> boneFrame,
            int numBones,
            Span<Matrix4x4> globalRotations,
            Span<Vector3> globalPositions)
        {
            if (boneFrame.Length != numBones * Smpl22Skeleton.RotationChannelsPerBone)
            {
                throw new ArgumentException(
                    $"boneFrame length must be {numBones * Smpl22Skeleton.RotationChannelsPerBone}; " +
                    $"got {boneFrame.Length}.");
            }

            if (globalRotations.Length != numBones || globalPositions.Length != numBones)
            {
                throw new ArgumentException("globalRotations/globalPositions must be sized numBones.");
            }

            for (var bone = 0; bone < numBones; bone++)
            {
                var slice = boneFrame.Slice(bone * Smpl22Skeleton.RotationChannelsPerBone, Smpl22Skeleton.RotationChannelsPerBone);
                SixDToRotationMatrix(slice, out var localRotation);

                var parent = Smpl22Skeleton.ParentIndices[bone];
                if (parent < 0)
                {
                    globalRotations[bone] = localRotation;
                    globalPositions[bone] = Smpl22Skeleton.BoneOffsets[bone];
                    continue;
                }

                var parentRotation = globalRotations[parent];
                var parentPosition = globalPositions[parent];
                globalRotations[bone] = parentRotation * localRotation;
                var childOffset = parentRotation.MultiplyVector(Smpl22Skeleton.BoneOffsets[bone]);
                globalPositions[bone] = parentPosition + childOffset;
            }
        }

        /// <summary>
        /// Transform a root-local joint position (from
        /// <see cref="ComputeJointPositions"/>) into world space, given the
        /// engine's integrated root world position. No extra yaw rotation
        /// is applied here: the pelvis (bone 0)'s own rotation6d already
        /// carries the skeleton's world-facing orientation
        /// (<c>ainimator.geometry.root_local</c>), so the root-local
        /// FK positions are already expressed with the correct world
        /// orientation baked in — only a translation by the integrated
        /// root position remains, matching
        /// <c>controller_sequences.py::deriveFootContacts</c>'s
        /// <c>worldFoot = footXyz + rootTranslation</c>.
        /// </summary>
        public static Vector3 RootLocalToWorld(Vector3 rootLocalPosition, Vector3 rootWorldPosition)
        {
            return rootLocalPosition + rootWorldPosition;
        }
    }
}
