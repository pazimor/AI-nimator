using System.IO;
using AInimator.Controller.Bundle;
using AInimator.Controller.Presets;
using UnityEditor;
using UnityEngine;

namespace AInimator.Controller.Editor
{
    /// <summary>
    /// Editor-only utilities to hydrate <see cref="ControlPreset"/> assets
    /// from a bundle's <c>presets/*.json</c> files. Editor-only because it
    /// depends on <c>UnityEditor</c> (asset creation) — never referenced
    /// from Runtime code.
    /// </summary>
    public static class ControlPresetImporter
    {
        /// <summary>
        /// Import every <c>presets/*.json</c> file under
        /// <paramref name="bundleDirectory"/> as a <see cref="ControlPreset"/>
        /// asset under <paramref name="destinationFolder"/> (created if needed).
        /// </summary>
        [MenuItem("AInimator/Import Presets From Bundle...")]
        public static void ImportFromBundleMenu()
        {
            var bundleDirectory = EditorUtility.OpenFolderPanel(
                "Select controller bundle directory", "", "");
            if (string.IsNullOrEmpty(bundleDirectory))
            {
                return;
            }

            const string destinationFolder = "Assets/AInimatorPresets";
            var count = ImportFromBundle(bundleDirectory, destinationFolder);
            EditorUtility.DisplayDialog(
                "AInimator", $"Imported {count} preset(s) into '{destinationFolder}'.", "OK");
        }

        /// <summary>
        /// Non-interactive entry point (also callable from tests/tools):
        /// reads every JSON file in <c>&lt;bundleDirectory&gt;/presets/</c>
        /// and writes/updates a <see cref="ControlPreset"/> asset per file.
        /// </summary>
        /// <returns>Number of presets imported.</returns>
        public static int ImportFromBundle(string bundleDirectory, string destinationFolder)
        {
            var presetsDir = Path.Combine(bundleDirectory, BundleLoader.PresetsSubdirectory);
            if (!Directory.Exists(presetsDir))
            {
                throw new BundleLoadException($"No '{BundleLoader.PresetsSubdirectory}' directory found under '{bundleDirectory}'.");
            }

            EnsureFolder(destinationFolder);

            var imported = 0;
            foreach (var file in Directory.GetFiles(presetsDir, "*.json"))
            {
                var data = ControlPresetData.Parse(File.ReadAllText(file));
                var assetPath = $"{destinationFolder}/{data.name}.asset";

                var preset = AssetDatabase.LoadAssetAtPath<ControlPreset>(assetPath);
                if (preset == null)
                {
                    preset = ScriptableObject.CreateInstance<ControlPreset>();
                    AssetDatabase.CreateAsset(preset, assetPath);
                }

                preset.HydrateFrom(data);
                EditorUtility.SetDirty(preset);
                imported++;
            }

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            return imported;
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
