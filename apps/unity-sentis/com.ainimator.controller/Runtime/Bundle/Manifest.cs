using System;
using System.Collections.Generic;

namespace AInimator.Controller.Bundle
{
    /// <summary>
    /// Plain data mirror of <c>manifest.json</c> (apps/spec/manifest.schema.json).
    /// </summary>
    /// <remarks>
    /// This is a serialization of the frozen I/O contract defined in
    /// <c>doc/ROADMAP_DETERMINIST.md §2.2</c>; it does not redefine it. Field
    /// names intentionally mirror the JSON keys (snake_case) so
    /// <c>JsonUtility</c> can deserialize the manifest without a custom
    /// converter.
    /// </remarks>
    [Serializable]
    public sealed class Manifest
    {
        public string bundle_version = "";
        public int state_channels;
        public int num_bones;
        public int rotation_channels_per_bone;
        public int root_local_motion_channels;
        public int control_channels;
        public List<string> control_layout = new();
        public int phase_channels;
        public int prompt_emb_channels;
        public int context_frames;
        public string output_layout = "";
        public string coord_system = "";
        public string normalization_note = "";
        public List<string> reserved_input_groups = new();
        public TextEncoderSection text_encoder = new();

        /// <summary>True when the manifest declares an aim (2D) channel pair.</summary>
        public bool HasAim => control_channels == 4;

        /// <summary>True when the bundle expects a <c>promptEmb</c> ONNX input.</summary>
        public bool HasPrompt => prompt_emb_channels > 0;

        /// <summary>True when the bundle expects a <c>(cos, sin)</c> phase ONNX input.</summary>
        public bool HasPhase => phase_channels == 2;

        /// <summary>
        /// True when the bundle ships an in-engine text encoder (B7 /
        /// A7.1+, <c>apps/spec/text_encoding.md</c> §1). The section is
        /// optional in the schema; JsonUtility leaves the default (empty
        /// <c>file</c>) when the key is absent.
        /// </summary>
        public bool HasTextEncoder => !string.IsNullOrEmpty(text_encoder?.file);

        /// <summary>
        /// Optional <c>text_encoder</c> manifest section (B7): the pooled
        /// encoder ONNX file plus its paired CLIP BPE tokenizer assets.
        /// </summary>
        [Serializable]
        public sealed class TextEncoderSection
        {
            public string file = "";
            public TokenizerSection tokenizer = new();
            public string pooling = "";
            public int embedding_channels;
        }

        /// <summary>Tokenizer sub-section of <see cref="TextEncoderSection"/>.</summary>
        [Serializable]
        public sealed class TokenizerSection
        {
            public string type = "";
            public string vocab = "";
            public string merges = "";
            public int max_length;
            public int bos_id;
            public int eos_id;
            public int pad_id;
        }
    }
}
