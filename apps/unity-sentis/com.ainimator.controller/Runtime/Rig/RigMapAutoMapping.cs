using System.Collections.Generic;
using UnityEngine;

namespace AInimator.Controller.Rig
{
    /// <summary>
    /// Best-effort auto-mapping of the 22 SMPL bones onto a rig hierarchy by
    /// common humanoid bone-name conventions (Mixamo, Unity Humanoid-exported
    /// FBX, UE Mannequin-style names) — the pure name-matching logic behind
    /// the Editor's "Auto-Map From Root" button (<c>RigMapAutoMapper</c>),
    /// kept in the Runtime assembly so a runtime component can also bind an
    /// arbitrary character instance at Play time (e.g. the RigDemo sample's
    /// custom-rig path).
    /// </summary>
    /// <remarks>
    /// Heuristic, per <c>apps/spec/rig_binding.md</c> §2.1: an unmatched
    /// SMPL bone is left unmapped (a valid, documented state) rather than
    /// guessed wrong.
    /// </remarks>
    public static class RigMapAutoMapping
    {
        // Each SMPL-22 bone -> a list of case-insensitive substrings tried,
        // in order, against every descendant Transform's name.
        private static readonly Dictionary<string, string[]> Aliases = new()
        {
            ["pelvis"] = new[] { "pelvis", "hips", "root" },
            ["leftHip"] = new[] { "leftupleg", "left_upleg", "l_upleg", "upleg_l", "thigh_l", "leftthigh", "hip_l", "l_thigh" },
            ["rightHip"] = new[] { "rightupleg", "right_upleg", "r_upleg", "upleg_r", "thigh_r", "rightthigh", "hip_r", "r_thigh" },
            ["spine1"] = new[] { "spine1", "spine_01", "spine" },
            ["leftKnee"] = new[] { "leftleg", "left_leg", "l_leg", "leg_l", "calf_l", "shin_l" },
            ["rightKnee"] = new[] { "rightleg", "right_leg", "r_leg", "leg_r", "calf_r", "shin_r" },
            ["spine2"] = new[] { "spine2", "spine_02" },
            ["leftAnkle"] = new[] { "leftfoot", "left_foot", "l_foot", "foot_l", "ankle_l" },
            ["rightAnkle"] = new[] { "rightfoot", "right_foot", "r_foot", "foot_r", "ankle_r" },
            ["spine3"] = new[] { "spine3", "spine_03", "chest" },
            ["leftFoot"] = new[] { "lefttoe", "left_toe", "l_toe", "toe_l", "ball_l" },
            ["rightFoot"] = new[] { "righttoe", "right_toe", "r_toe", "toe_r", "ball_r" },
            ["neck"] = new[] { "neck" },
            ["leftCollar"] = new[] { "leftshoulder", "left_shoulder", "l_shoulder", "shoulder_l", "clavicle_l" },
            ["rightCollar"] = new[] { "rightshoulder", "right_shoulder", "r_shoulder", "shoulder_r", "clavicle_r" },
            ["head"] = new[] { "head" },
            ["leftShoulder"] = new[] { "leftarm", "left_arm", "l_arm", "arm_l", "upperarm_l" },
            ["rightShoulder"] = new[] { "rightarm", "right_arm", "r_arm", "arm_r", "upperarm_r" },
            ["leftElbow"] = new[] { "leftforearm", "left_forearm", "l_forearm", "forearm_l", "lowerarm_l" },
            ["rightElbow"] = new[] { "rightforearm", "right_forearm", "r_forearm", "forearm_r", "lowerarm_r" },
            ["leftWrist"] = new[] { "lefthand", "left_hand", "l_hand", "hand_l" },
            ["rightWrist"] = new[] { "righthand", "right_hand", "r_hand", "hand_r" },
        };

        /// <summary>
        /// Exact SMPL-22 -> Mixamo bone names (suffix after the
        /// <c>mixamorig[N]:</c> prefix), used in preference to
        /// <see cref="Aliases"/> when the rig is detected as Mixamo. The
        /// generic substring pass cannot get the Mixamo spine chain right:
        /// SMPL <c>spine1/spine2/spine3</c> correspond to Mixamo
        /// <c>Spine/Spine1/Spine2</c> (off by one), so substring "spine1"
        /// would grab the wrong bone.
        /// </summary>
        private static readonly Dictionary<string, string> MixamoNames = new()
        {
            ["pelvis"] = "Hips",
            ["leftHip"] = "LeftUpLeg",
            ["rightHip"] = "RightUpLeg",
            ["spine1"] = "Spine",
            ["leftKnee"] = "LeftLeg",
            ["rightKnee"] = "RightLeg",
            ["spine2"] = "Spine1",
            ["leftAnkle"] = "LeftFoot",
            ["rightAnkle"] = "RightFoot",
            ["spine3"] = "Spine2",
            ["leftFoot"] = "LeftToeBase",
            ["rightFoot"] = "RightToeBase",
            ["neck"] = "Neck",
            ["leftCollar"] = "LeftShoulder",
            ["rightCollar"] = "RightShoulder",
            ["head"] = "Head",
            ["leftShoulder"] = "LeftArm",
            ["rightShoulder"] = "RightArm",
            ["leftElbow"] = "LeftForeArm",
            ["rightElbow"] = "RightForeArm",
            ["leftWrist"] = "LeftHand",
            ["rightWrist"] = "RightHand",
        };

        /// <summary>
        /// Populate every unmapped row of <paramref name="map"/> by
        /// searching <paramref name="rigRoot"/>'s descendants for a name
        /// matching one of the aliases for that SMPL-22 bone — with an exact
        /// Mixamo-name pass first when any bone carries the
        /// <c>mixamorig</c> prefix. Already-set rows are left untouched.
        /// Returns the number of rows mapped (including previously-set ones).
        /// </summary>
        public static int AutoMap(RigMap map, Transform rigRoot)
        {
            map.EnsureEntriesSized();
            var allTransforms = rigRoot.GetComponentsInChildren<Transform>(true);

            if (IsMixamoRig(allTransforms))
            {
                MapMixamoExact(map, allTransforms);
            }

            var mapped = 0;
            for (var i = 0; i < map.Entries.Length; i++)
            {
                var entry = map.Entries[i];
                if (entry.rigBone != null)
                {
                    mapped++;
                    continue;
                }

                if (!Aliases.TryGetValue(entry.smplBoneName, out var aliases))
                {
                    continue;
                }

                var match = FindBestMatch(allTransforms, aliases);
                if (match != null)
                {
                    entry.rigBone = match;
                    mapped++;
                }
            }

            return mapped;
        }

        private static bool IsMixamoRig(Transform[] candidates)
        {
            foreach (var candidate in candidates)
            {
                if (candidate.name.StartsWith("mixamorig", System.StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }

            return false;
        }

        private static void MapMixamoExact(RigMap map, Transform[] candidates)
        {
            foreach (var entry in map.Entries)
            {
                if (entry.rigBone != null || !MixamoNames.TryGetValue(entry.smplBoneName, out var mixamoName))
                {
                    continue;
                }

                foreach (var candidate in candidates)
                {
                    // "mixamorig:Hips" / "mixamorig1:Hips" / bare "Hips".
                    if (candidate.name == mixamoName || candidate.name.EndsWith(":" + mixamoName))
                    {
                        entry.rigBone = candidate;
                        break;
                    }
                }
            }
        }

        private static Transform FindBestMatch(Transform[] candidates, string[] aliases)
        {
            foreach (var alias in aliases)
            {
                foreach (var candidate in candidates)
                {
                    var name = candidate.name.ToLowerInvariant().Replace(" ", "").Replace("-", "_");
                    if (name.Contains(alias))
                    {
                        return candidate;
                    }
                }
            }

            return null;
        }
    }
}
