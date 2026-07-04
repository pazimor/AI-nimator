using System;
using AInimator.Controller.PostProcess;
using UnityEngine;

namespace AInimator.Controller.Rig
{
    /// <summary>
    /// Per-rig retargeting asset: for each of the 22 canonical SMPL-22 bones
    /// (<see cref="Smpl22Skeleton"/> order), the <see cref="Transform"/> of
    /// the equivalent bone on an arbitrary character rig — or unassigned
    /// when the rig has no equivalent bone.
    /// </summary>
    /// <remarks>
    /// This is the canonical (parity) retargeting path of
    /// <c>apps/spec/rig_binding.md</c> §2.1: an explicit bone map, one asset
    /// per rig, calibrated once at bind time by <see cref="Rig.RigBinder"/>.
    /// An SMPL bone left unassigned is simply ignored (spec §2.1 v1 rule) —
    /// it contributes no rotation to the rig and is never composed with a
    /// child's rotation.
    /// <para/>
    /// This asset stores <b>only</b> the bone-name mapping; the world-space
    /// rest-pose calibration (<c>O_self</c> / <c>worldRot_rig_rest</c>) is
    /// captured at runtime by <see cref="Rig.RigBinder"/> from the actual
    /// bound rig instance, never baked into this asset (the same
    /// <see cref="RigMap"/> can be reused across multiple rig instances that
    /// share bone names, e.g. prefab variants, as long as their rest poses
    /// match).
    /// </remarks>
    [CreateAssetMenu(
        fileName = "NewRigMap",
        menuName = "AInimator/Rig Map",
        order = 101)]
    public sealed class RigMap : ScriptableObject
    {
        [Serializable]
        public sealed class BoneEntry
        {
            [Tooltip("SMPL-22 bone name (read-only, informational — index is the source of truth).")]
            public string smplBoneName = "";

            [Tooltip("Rig Transform mapped to this SMPL bone, or null/unassigned when the rig has no equivalent.")]
            public Transform rigBone;
        }

        [Tooltip("22 entries, indexed exactly per Smpl22Skeleton bone order. Unassigned entries are ignored (spec §2.1).")]
        [SerializeField] private BoneEntry[] entries = CreateDefaultEntries();

        /// <summary>The 22 bone-map entries, indexed per <see cref="Smpl22Skeleton"/> bone order.</summary>
        public BoneEntry[] Entries => entries;

        /// <summary>Rig <see cref="Transform"/> mapped to SMPL bone <paramref name="smplBoneIndex"/>, or null if unmapped.</summary>
        public Transform GetRigBone(int smplBoneIndex)
        {
            if (smplBoneIndex < 0 || smplBoneIndex >= Smpl22Skeleton.NumBones)
            {
                throw new ArgumentOutOfRangeException(nameof(smplBoneIndex));
            }

            EnsureEntriesSized();
            return entries[smplBoneIndex].rigBone;
        }

        /// <summary>True when at least the pelvis (root) is mapped — the minimum for the rig to be drivable at all.</summary>
        public bool HasMinimalMapping()
        {
            EnsureEntriesSized();
            return entries[Smpl22Skeleton.Pelvis].rigBone != null;
        }

        /// <summary>(Re)build the 22-entry array with the canonical SMPL-22 bone names, preserving any existing assignments by name.</summary>
        public void EnsureEntriesSized()
        {
            if (entries != null && entries.Length == Smpl22Skeleton.NumBones)
            {
                return;
            }

            var rebuilt = CreateDefaultEntries();
            if (entries != null)
            {
                // Preserve assignments made before a schema/size change by name.
                foreach (var old in entries)
                {
                    if (old == null || string.IsNullOrEmpty(old.smplBoneName))
                    {
                        continue;
                    }

                    for (var i = 0; i < rebuilt.Length; i++)
                    {
                        if (rebuilt[i].smplBoneName == old.smplBoneName)
                        {
                            rebuilt[i].rigBone = old.rigBone;
                            break;
                        }
                    }
                }
            }

            entries = rebuilt;
        }

        private static BoneEntry[] CreateDefaultEntries()
        {
            var names = Smpl22BoneNames;
            var result = new BoneEntry[names.Length];
            for (var i = 0; i < names.Length; i++)
            {
                result[i] = new BoneEntry { smplBoneName = names[i], rigBone = null };
            }

            return result;
        }

        /// <summary>Human-readable SMPL-22 bone names, indexed per <see cref="Smpl22Skeleton"/> bone order (informational/UI only).</summary>
        public static readonly string[] Smpl22BoneNames =
        {
            "pelvis", "leftHip", "rightHip", "spine1", "leftKnee", "rightKnee",
            "spine2", "leftAnkle", "rightAnkle", "spine3", "leftFoot", "rightFoot",
            "neck", "leftCollar", "rightCollar", "head", "leftShoulder", "rightShoulder",
            "leftElbow", "rightElbow", "leftWrist", "rightWrist",
        };

        private void OnValidate()
        {
            EnsureEntriesSized();
        }

        private void Reset()
        {
            entries = CreateDefaultEntries();
        }
    }
}
