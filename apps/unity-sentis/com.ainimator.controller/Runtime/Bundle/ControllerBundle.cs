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

        /// <summary>Raw <c>text_encoder.onnx</c> bytes, or <c>null</c> (bundle without B7 encoder).</summary>
        public byte[] TextEncoderModelBytes { get; }

        /// <summary>Verbatim <c>tokenizer/vocab.json</c> content, or <c>null</c>.</summary>
        public string TokenizerVocabJson { get; }

        /// <summary>Verbatim <c>tokenizer/merges.txt</c> content, or <c>null</c>.</summary>
        public string TokenizerMergesText { get; }

        /// <summary>True when the bundle ships the in-engine text encoder (B7 / A7.1+).</summary>
        public bool HasTextEncoder => TextEncoderModelBytes != null;

        internal ControllerBundle(
            Manifest manifest,
            NormStats normStats,
            byte[] onnxModelBytes,
            IReadOnlyDictionary<string, ControlPresetData> presets,
            byte[] textEncoderModelBytes = null,
            string tokenizerVocabJson = null,
            string tokenizerMergesText = null)
        {
            Manifest = manifest;
            NormStats = normStats;
            OnnxModelBytes = onnxModelBytes;
            Presets = presets;
            TextEncoderModelBytes = textEncoderModelBytes;
            TokenizerVocabJson = tokenizerVocabJson;
            TokenizerMergesText = tokenizerMergesText;
        }
    }
}
