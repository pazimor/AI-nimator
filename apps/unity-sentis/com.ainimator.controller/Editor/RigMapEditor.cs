using System.Collections.Generic;
using AInimator.Controller.Rig;
using UnityEditor;
using UnityEngine;

namespace AInimator.Controller.Editor
{
    /// <summary>
    /// Custom Inspector for <see cref="RigMap"/>: a 22-row table (one per
    /// SMPL-22 bone, canonical order) with an object-field drag target per
    /// row, plus an "Auto-Map From Selected Root" button that best-effort
    /// matches common humanoid bone names (<c>apps/spec/rig_binding.md</c> §2.1).
    /// </summary>
    [CustomEditor(typeof(RigMap))]
    public sealed class RigMapEditor : UnityEditor.Editor
    {
        private SerializedProperty _entriesProp;
        private Transform _autoMapRoot;

        private void OnEnable()
        {
            ((RigMap)target).EnsureEntriesSized();
            _entriesProp = serializedObject.FindProperty("entries");
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            EditorGUILayout.HelpBox(
                "One row per SMPL-22 bone (canonical order). Leave a row " +
                "empty when the rig has no equivalent bone -- it is simply " +
                "ignored at retarget time (rig_binding.md §2.1).",
                MessageType.Info);

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Auto-map", EditorStyles.boldLabel);
            using (new EditorGUILayout.HorizontalScope())
            {
                _autoMapRoot = (Transform)EditorGUILayout.ObjectField(
                    "Rig Root", _autoMapRoot, typeof(Transform), true);

                using (new EditorGUI.DisabledScope(_autoMapRoot == null))
                {
                    if (GUILayout.Button("Auto-Map From Root", GUILayout.Width(160)))
                    {
                        var mapped = RigMapAutoMapper.AutoMap((RigMap)target, _autoMapRoot);
                        EditorUtility.DisplayDialog(
                            "AInimator", $"Auto-mapped {mapped}/{RigMap.Smpl22BoneNames.Length} bone(s).", "OK");
                        serializedObject.Update();
                    }
                }
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Bone map (22 entries)", EditorStyles.boldLabel);
            DrawEntries();

            serializedObject.ApplyModifiedProperties();
        }

        private void DrawEntries()
        {
            for (var i = 0; i < _entriesProp.arraySize; i++)
            {
                var element = _entriesProp.GetArrayElementAtIndex(i);
                var nameProp = element.FindPropertyRelative("smplBoneName");
                var boneProp = element.FindPropertyRelative("rigBone");

                using (new EditorGUILayout.HorizontalScope())
                {
                    EditorGUILayout.LabelField(
                        $"{i:D2}  {nameProp.stringValue}", GUILayout.Width(160));
                    EditorGUILayout.PropertyField(boneProp, GUIContent.none);
                }
            }
        }
    }

    /// <summary>
    /// Best-effort auto-mapping of SMPL-22 bone names onto a rig's
    /// hierarchy by common humanoid bone-name conventions (Mixamo, Unity
    /// Humanoid-exported FBX, UE Mannequin-style names). Editor-only,
    /// heuristic — always leaves unmatched rows empty rather than guessing
    /// wrong (rig_binding.md §2.1: unmapped is a valid, documented state).
    /// </summary>
    public static class RigMapAutoMapper
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
        /// Populate every unmapped row of <paramref name="map"/> by
        /// searching <paramref name="rigRoot"/>'s descendants for a name
        /// matching one of the aliases for that SMPL-22 bone. Already-set
        /// rows are left untouched. Returns the number of rows mapped
        /// (including previously-set ones).
        /// </summary>
        public static int AutoMap(RigMap map, Transform rigRoot)
        {
            map.EnsureEntriesSized();
            var allTransforms = rigRoot.GetComponentsInChildren<Transform>(true);

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

            EditorUtility.SetDirty(map);
            return mapped;
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
