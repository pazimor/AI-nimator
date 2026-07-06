using AInimator.Controller.Presets;
using UnityEditor;
using UnityEngine;

namespace AInimator.Controller.Editor
{
    /// <summary>
    /// Default <see cref="ControlPreset"/> Inspector plus one authoring
    /// shortcut: "Load Prompt Embedding JSON..." loads the output of
    /// <c>python -m ainimator.cli.encode_prompt</c>
    /// (<c>apps/spec/rig_binding.md</c> §4.1) directly into this preset's
    /// <c>Prompt</c>/<c>PromptEmb</c> fields — the no-Python-in-the-loop
    /// authoring path once a prompt has been encoded once.
    /// </summary>
    [CustomEditor(typeof(ControlPreset))]
    public sealed class ControlPresetEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            EditorGUILayout.Space();
            if (GUILayout.Button("Load Prompt Embedding JSON... (encode_prompt output)"))
            {
                var path = EditorUtility.OpenFilePanel("Select encode_prompt output JSON", "", "json");
                if (!string.IsNullOrEmpty(path))
                {
                    PromptEmbeddingImporter.LoadInto((ControlPreset)target, path);
                    AssetDatabase.SaveAssets();
                }
            }
        }
    }
}
