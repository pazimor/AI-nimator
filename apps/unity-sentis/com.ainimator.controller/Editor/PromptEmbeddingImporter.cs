using System.IO;
using AInimator.Controller.Bundle;
using AInimator.Controller.Presets;
using UnityEditor;
using UnityEngine;

namespace AInimator.Controller.Editor
{
    /// <summary>
    /// Editor-only loader for the JSON produced by the repo-side CLI
    /// <c>python -m ainimator.cli.encode_prompt --prompt "..." --encoder-artifact
    /// ... --output x.json</c> (<c>apps/spec/rig_binding.md</c> §4.1): a
    /// <c>{"prompt": str, "prompt_emb": float[]}</c> file, distinct from the
    /// full <c>presets/*.json</c> bundle schema (no <c>control</c> section).
    /// Parsing itself lives in <see cref="PromptEmbeddingData"/> (Runtime)
    /// so it stays testable without an Editor assembly.
    /// </summary>
    public static class PromptEmbeddingImporter
    {
        /// <summary>
        /// Load an <c>encode_prompt</c> JSON file into <paramref name="preset"/>'s
        /// <see cref="ControlPreset.Prompt"/>/<see cref="ControlPreset.PromptEmb"/>
        /// fields via reflection-free serialized-property writes (keeps
        /// <see cref="ControlPreset"/>'s fields private/Inspector-only).
        /// </summary>
        public static void LoadInto(ControlPreset preset, string jsonPath)
        {
            var data = PromptEmbeddingData.Parse(File.ReadAllText(jsonPath));

            var so = new SerializedObject(preset);
            so.FindProperty("prompt").stringValue = data.Prompt;

            var embProp = so.FindProperty("promptEmb");
            embProp.arraySize = data.PromptEmb.Length;
            for (var i = 0; i < data.PromptEmb.Length; i++)
            {
                embProp.GetArrayElementAtIndex(i).floatValue = data.PromptEmb[i];
            }

            so.ApplyModifiedProperties();
            EditorUtility.SetDirty(preset);
        }

        [MenuItem("CONTEXT/ControlPreset/Load Prompt Embedding JSON...")]
        private static void LoadPromptEmbeddingJsonMenu(MenuCommand command)
        {
            var preset = (ControlPreset)command.context;
            var path = EditorUtility.OpenFilePanel(
                "Select encode_prompt output JSON", "", "json");
            if (string.IsNullOrEmpty(path))
            {
                return;
            }

            LoadInto(preset, path);
            AssetDatabase.SaveAssets();
        }
    }
}
