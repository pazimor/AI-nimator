using System.IO;
using AInimator.Controller.Bundle;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace AInimator.Controller.Editor
{
    /// <summary>
    /// Editor-side conversion of the bundle's raw ONNX models to serialized
    /// Sentis models (<c>.sentis</c>) — the only format Sentis 2.x can load
    /// at runtime (<see cref="BundleLoader.SentisFileName"/> remarks). Runs
    /// automatically after every domain reload for the default
    /// StreamingAssets bundle, and on demand from the
    /// <c>AInimator/Convert Bundle ONNX to Sentis</c> menu.
    /// </summary>
    /// <remarks>
    /// The package's ONNX converter is internal, so this goes through the
    /// asset pipeline instead: the <c>.onnx</c> is copied to a temporary
    /// folder under <c>Assets/</c> (StreamingAssets content is never
    /// imported), imported into a <see cref="ModelAsset"/>, serialized next
    /// to the source file with <see cref="ModelWriter.Save(string, ModelAsset)"/>,
    /// then the temp asset is deleted. A <c>.sentis</c> newer than its
    /// <c>.onnx</c> is considered up to date and skipped, so the automatic
    /// pass is a cheap timestamp check in the common case.
    /// </remarks>
    public static class BundleSentisConverter
    {
        private const string TempImportFolder = "Assets/AInimatorSentisImportTemp";
        private const string TextEncoderOnnxFileName = "text_encoder.onnx";
        private const string TextEncoderSentisFileName = "text_encoder.sentis";

        [InitializeOnLoadMethod]
        private static void AutoConvertAfterDomainReload()
        {
            // The asset pipeline is not safe to drive during the reload
            // itself; run once the editor is idle.
            EditorApplication.delayCall += () => ConvertBundleIfNeeded(
                BundlePaths.DefaultBundleDirectory, logWhenNothingToDo: false);
        }

        [MenuItem("AInimator/Convert Bundle ONNX to Sentis (StreamingAssets)")]
        public static void ConvertDefaultBundle()
        {
            ConvertBundleIfNeeded(BundlePaths.DefaultBundleDirectory, logWhenNothingToDo: true);
        }

        /// <summary>
        /// Convert <c>controller.onnx</c> and (when present)
        /// <c>text_encoder.onnx</c> in <paramref name="bundleDirectory"/> to
        /// their <c>.sentis</c> siblings, skipping up-to-date outputs.
        /// </summary>
        public static void ConvertBundleIfNeeded(string bundleDirectory, bool logWhenNothingToDo)
        {
            if (!Directory.Exists(bundleDirectory))
            {
                if (logWhenNothingToDo)
                {
                    Debug.LogWarning($"AInimator: no bundle directory at '{bundleDirectory}' — nothing to convert.");
                }

                return;
            }

            var converted = 0;
            converted += ConvertOne(bundleDirectory, BundleLoader.OnnxFileName, BundleLoader.SentisFileName) ? 1 : 0;
            converted += ConvertOne(bundleDirectory, TextEncoderOnnxFileName, TextEncoderSentisFileName) ? 1 : 0;

            if (converted > 0)
            {
                AssetDatabase.Refresh();
            }
            else if (logWhenNothingToDo)
            {
                Debug.Log($"AInimator: bundle at '{bundleDirectory}' already has up-to-date .sentis models.");
            }
        }

        private static bool ConvertOne(string bundleDirectory, string onnxFileName, string sentisFileName)
        {
            var onnxPath = Path.Combine(bundleDirectory, onnxFileName);
            if (!File.Exists(onnxPath))
            {
                return false;
            }

            var sentisPath = Path.Combine(bundleDirectory, sentisFileName);
            if (File.Exists(sentisPath) &&
                File.GetLastWriteTimeUtc(sentisPath) >= File.GetLastWriteTimeUtc(onnxPath))
            {
                return false;
            }

            var tempAssetPath = $"{TempImportFolder}/{onnxFileName}";
            try
            {
                Directory.CreateDirectory(TempImportFolder);
                File.Copy(onnxPath, tempAssetPath, overwrite: true);
                AssetDatabase.ImportAsset(tempAssetPath, ImportAssetOptions.ForceSynchronousImport);

                var modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(tempAssetPath);
                if (modelAsset == null)
                {
                    Debug.LogError(
                        $"AInimator: importing '{onnxFileName}' did not produce a ModelAsset — " +
                        "the ONNX may be unsupported by this Inference Engine version. " +
                        $"'{sentisFileName}' was NOT written.");
                    return false;
                }

                ModelWriter.Save(sentisPath, modelAsset);
                Debug.Log($"AInimator: converted '{onnxFileName}' -> '{sentisPath}' (Sentis-serialized model).");
                return true;
            }
            finally
            {
                AssetDatabase.DeleteAsset(tempAssetPath);
                AssetDatabase.DeleteAsset(TempImportFolder);
            }
        }
    }
}
