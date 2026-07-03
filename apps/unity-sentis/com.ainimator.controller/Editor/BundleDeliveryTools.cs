using System;
using System.IO;
using AInimator.Controller.Bundle;
using UnityEditor;
using UnityEngine;

namespace AInimator.Controller.Editor
{
    /// <summary>
    /// Batch-mode entry points invoked by the <c>apps/build/</c> orchestrator
    /// (ROADMAP_PLUGINS.md §3.4) after it has copied a fresh bundle into
    /// <c>StreamingAssets/AInimatorBundle/</c>. Never invoked interactively
    /// in normal editor use — only via
    /// <c>Unity -batchmode -quit -executeMethod ...</c>.
    /// </summary>
    /// <remarks>
    /// This plugin never calls Python or the export CLI itself (Goal B
    /// verite #9): it only validates that a bundle is present where the
    /// runtime expects it, then triggers a standard player build. The
    /// orchestrator is responsible for the copy step (export → validate →
    /// deliver) before invoking this method.
    /// </remarks>
    public static class BundleDeliveryTools
    {
        /// <summary>
        /// Verify a bundle is present under StreamingAssets, then build the
        /// active build target's player. Intended invocation:
        /// <c>Unity -batchmode -quit -projectPath &lt;proj&gt;
        /// -executeMethod AInimator.Controller.Editor.BundleDeliveryTools.PackPlayer</c>.
        /// </summary>
        public static void PackPlayer()
        {
            var bundlePath = BundlePaths.DefaultBundleDirectory;
            if (!Directory.Exists(bundlePath) || !File.Exists(Path.Combine(bundlePath, "manifest.json")))
            {
                throw new InvalidOperationException(
                    $"No controller bundle found at '{bundlePath}'. The apps/build/ orchestrator must " +
                    "copy a fresh bundle (export -> validate -> deliver) before invoking PackPlayer.");
            }

            var scenes = Array.ConvertAll(
                EditorBuildSettings.scenes, s => s.path);

            var report = BuildPipeline.BuildPlayer(
                scenes,
                Path.Combine("Build", "AInimatorDemo"),
                EditorUserBuildSettings.activeBuildTarget,
                BuildOptions.None);

            if (report.summary.result != UnityEditor.Build.Reporting.BuildResult.Succeeded)
            {
                throw new InvalidOperationException($"Build failed: {report.summary.result}.");
            }
        }
    }
}
