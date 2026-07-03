using System.Collections.Generic;

namespace AInimator.Controller.Bundle
{
    /// <summary>
    /// A fully-loaded, validated controller bundle: manifest, normalization
    /// stats, the raw ONNX bytes, and any presets found alongside it.
    /// </summary>
    /// <remarks>
    /// Construct via <see cref="BundleLoader"/> — never directly — so every
    /// instance in memory has already passed manifest validation
    /// (Goal B verite #4, fail-fast).
    /// </remarks>
    public sealed class ControllerBundle
    {
        public Manifest Manifest { get; }
        public NormStats NormStats { get; }
        public byte[] OnnxModelBytes { get; }
        public IReadOnlyDictionary<string, ControlPresetData> Presets { get; }

        internal ControllerBundle(
            Manifest manifest,
            NormStats normStats,
            byte[] onnxModelBytes,
            IReadOnlyDictionary<string, ControlPresetData> presets)
        {
            Manifest = manifest;
            NormStats = normStats;
            OnnxModelBytes = onnxModelBytes;
            Presets = presets;
        }
    }
}
