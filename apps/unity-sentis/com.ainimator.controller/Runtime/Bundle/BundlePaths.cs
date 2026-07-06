using System.IO;
using UnityEngine;

namespace AInimator.Controller.Bundle
{
    /// <summary>
    /// Resolves the default on-disk location of the controller bundle at
    /// runtime. The bundle is a build artifact (never a versioned asset,
    /// Goal B verite #9): it is delivered under
    /// <c>StreamingAssets/AInimatorBundle/</c> by the <c>apps/build/</c>
    /// orchestrator before the engine build, and read back here with
    /// <see cref="Application.streamingAssetsPath"/> so the same code path
    /// works in the editor and in a built player.
    /// </summary>
    public static class BundlePaths
    {
        public const string DefaultStreamingAssetsSubfolder = "AInimatorBundle";

        /// <summary>
        /// Absolute path to the default bundle directory for this platform.
        /// </summary>
        /// <remarks>
        /// On most platforms this is a plain filesystem path readable with
        /// <see cref="File"/>/<see cref="Directory"/> APIs (Editor, Windows,
        /// macOS, Linux). Android/WebGL bundle StreamingAssets inside a
        /// compressed archive/URL that <see cref="File.ReadAllBytes"/> cannot
        /// read directly — those platforms need a small
        /// <c>UnityWebRequest</c>-based loader swapped in for
        /// <see cref="BundleLoader.Load"/> before shipping there. Not needed
        /// for the B1 desktop demo target.
        /// </remarks>
        public static string DefaultBundleDirectory =>
            Path.Combine(Application.streamingAssetsPath, DefaultStreamingAssetsSubfolder);
    }
}
