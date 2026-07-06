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
    /// Editor facade over <see cref="RigMapAutoMapping"/> (the Runtime
    /// name-matching logic, shared with the RigDemo sample's Play-time
    /// custom-rig path): same behavior, plus asset dirty-marking so the
    /// Inspector button's result persists.
    /// </summary>
    public static class RigMapAutoMapper
    {
        /// <summary>See <see cref="RigMapAutoMapping.AutoMap"/>; also marks <paramref name="map"/> dirty.</summary>
        public static int AutoMap(RigMap map, Transform rigRoot)
        {
            var mapped = RigMapAutoMapping.AutoMap(map, rigRoot);
            EditorUtility.SetDirty(map);
            return mapped;
        }
    }
}
