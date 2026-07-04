using System.IO;
using AInimator.Controller.Authoring;
using AInimator.Controller.Presets;
using UnityEditor;
using UnityEngine;

namespace AInimator.Controller.Editor
{
    /// <summary>
    /// Custom Inspector for <see cref="AInimatorActionBinder"/>: an editable
    /// key→preset binding list (each row is an in-project
    /// <see cref="ControlPreset"/> picker, no code required), plus two
    /// authoring shortcuts (ROADMAP_PLUGINS.md §4 B3):
    /// "Create New Preset" (creates a <see cref="ControlPreset"/> asset and
    /// wires it into a new binding row) and "Import Presets From Bundle..."
    /// (delegates to the existing <see cref="ControlPresetImporter"/>).
    /// </summary>
    [CustomEditor(typeof(AInimatorActionBinder))]
    public sealed class AInimatorActionBinderEditor : UnityEditor.Editor
    {
        private const string DefaultPresetFolder = "Assets/AInimatorPresets";

        private SerializedProperty _bindingsProp;
        private SerializedProperty _idlePresetProp;
        private SerializedProperty _bundleDirectoryOverrideProp;
        private SerializedProperty _seedBoneFrameOverrideProp;
        private SerializedProperty _autoInitializeProp;

        private void OnEnable()
        {
            _bindingsProp = serializedObject.FindProperty("bindings");
            _idlePresetProp = serializedObject.FindProperty("idlePreset");
            _bundleDirectoryOverrideProp = serializedObject.FindProperty("bundleDirectoryOverride");
            _seedBoneFrameOverrideProp = serializedObject.FindProperty("seedBoneFrameOverride");
            _autoInitializeProp = serializedObject.FindProperty("autoInitializeFromDefaultBundle");
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            EditorGUILayout.LabelField("Idle / fallback", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(_idlePresetProp, new GUIContent("Idle Preset"));

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Key -> Preset bindings", EditorStyles.boldLabel);
            DrawBindingsList();

            EditorGUILayout.Space();
            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button("+ Add Binding"))
                {
                    _bindingsProp.InsertArrayElementAtIndex(_bindingsProp.arraySize);
                }

                if (GUILayout.Button("Create New Preset..."))
                {
                    CreateNewPresetAndBind();
                }

                if (GUILayout.Button("Import Presets From Bundle..."))
                {
                    var bundleDirectory = EditorUtility.OpenFolderPanel(
                        "Select controller bundle directory", "", "");
                    if (!string.IsNullOrEmpty(bundleDirectory))
                    {
                        var count = ControlPresetImporter.ImportFromBundle(bundleDirectory, DefaultPresetFolder);
                        EditorUtility.DisplayDialog(
                            "AInimator", $"Imported {count} preset(s) into '{DefaultPresetFolder}'.", "OK");
                    }
                }
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Bundle loading", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(_autoInitializeProp, new GUIContent("Auto-Initialize From Default Bundle"));
            EditorGUILayout.PropertyField(_bundleDirectoryOverrideProp, new GUIContent("Bundle Directory Override"));
            EditorGUILayout.PropertyField(_seedBoneFrameOverrideProp, new GUIContent("Seed Bone Frame Override"), true);

            serializedObject.ApplyModifiedProperties();
        }

        private void DrawBindingsList()
        {
            for (var i = 0; i < _bindingsProp.arraySize; i++)
            {
                var element = _bindingsProp.GetArrayElementAtIndex(i);
                var keyProp = element.FindPropertyRelative("key");
                var presetProp = element.FindPropertyRelative("preset");
                var labelProp = element.FindPropertyRelative("label");

                using (new EditorGUILayout.HorizontalScope(EditorStyles.helpBox))
                {
                    EditorGUILayout.PropertyField(keyProp, GUIContent.none, GUILayout.Width(110));
                    EditorGUILayout.PropertyField(presetProp, GUIContent.none, GUILayout.MinWidth(140));
                    EditorGUILayout.PropertyField(labelProp, GUIContent.none, GUILayout.Width(90));

                    if (GUILayout.Button("x", GUILayout.Width(22)))
                    {
                        _bindingsProp.DeleteArrayElementAtIndex(i);
                        break;
                    }
                }
            }
        }

        /// <summary>
        /// Create a new <see cref="ControlPreset"/> asset (blank, snake_case
        /// name prompted via the folder+name save panel) under
        /// <see cref="DefaultPresetFolder"/> and append a binding row wired
        /// to it — the "field, no code" authoring path for a brand-new
        /// action.
        /// </summary>
        private void CreateNewPresetAndBind()
        {
            EnsureFolder(DefaultPresetFolder);

            var path = EditorUtility.SaveFilePanelInProject(
                "Create Control Preset", "NewControlPreset", "asset",
                "Name the new ControlPreset asset.", DefaultPresetFolder);
            if (string.IsNullOrEmpty(path))
            {
                return;
            }

            var preset = CreateInstance<ControlPreset>();
            AssetDatabase.CreateAsset(preset, path);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();

            var index = _bindingsProp.arraySize;
            _bindingsProp.InsertArrayElementAtIndex(index);
            var element = _bindingsProp.GetArrayElementAtIndex(index);
            element.FindPropertyRelative("preset").objectReferenceValue = preset;
            element.FindPropertyRelative("key").enumValueIndex = (int)KeyCode.None;
            element.FindPropertyRelative("label").stringValue = Path.GetFileNameWithoutExtension(path);

            serializedObject.ApplyModifiedProperties();
            EditorGUIUtility.PingObject(preset);
        }

        private static void EnsureFolder(string folder)
        {
            if (AssetDatabase.IsValidFolder(folder))
            {
                return;
            }

            var parent = Path.GetDirectoryName(folder)?.Replace('\\', '/');
            var leaf = Path.GetFileName(folder);
            if (!string.IsNullOrEmpty(parent) && !AssetDatabase.IsValidFolder(parent))
            {
                EnsureFolder(parent);
            }

            AssetDatabase.CreateFolder(string.IsNullOrEmpty(parent) ? "Assets" : parent, leaf);
        }
    }
}
