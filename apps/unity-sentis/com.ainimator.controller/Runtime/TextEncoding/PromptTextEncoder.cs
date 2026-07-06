using System;
using AInimator.Controller.Bundle;
using Unity.InferenceEngine;

namespace AInimator.Controller.TextEncoding
{
    /// <summary>
    /// Owns the Sentis <see cref="Worker"/> for the bundle's
    /// <c>text_encoder.onnx</c> (Goal B phase B7,
    /// <c>apps/spec/text_encoding.md</c> §1/§3) and runs one encode pass
    /// per prompt change: <c>input_ids + attention_mask → prompt_emb</c>.
    /// </summary>
    /// <remarks>
    /// The masked-mean pooling is baked INTO the ONNX graph — this class
    /// never pools. Encoding happens at action scale (a prompt change),
    /// never per frame, so a synchronous readback is acceptable here
    /// (unlike the controller hot path).
    /// <para/>
    /// Sentis API caveat: same as <see cref="Runtime.ControllerRuntime"/> —
    /// written against the documented Sentis 1.x surface
    /// (<c>ModelLoader.Load</c>, <c>Worker.SetInput/Schedule/PeekOutput</c>);
    /// **must be smoke-tested against the installed package before
    /// shipping** (no Unity editor available in this session). The ONNX
    /// <c>input_ids</c> input is int64; Sentis loads it as its int tensor
    /// type, so a <c>Tensor&lt;int&gt;</c> upload is expected to match.
    /// </remarks>
    public sealed class PromptTextEncoder : IDisposable
    {
        private const string InputIdsName = "input_ids";
        private const string AttentionMaskName = "attention_mask";
        private const string PromptEmbOutputName = "prompt_emb";

        private readonly ClipBpeTokenizer _tokenizer;
        private readonly int _embeddingChannels;
        private readonly Model _model;
        private readonly Worker _worker;

        // Reused per encode — a prompt change allocates nothing.
        private readonly int[] _inputIdsScratch;
        private readonly float[] _attentionMaskScratch;
        private Tensor<int> _inputIdsTensor;
        private Tensor<float> _attentionMaskTensor;

        public ClipBpeTokenizer Tokenizer => _tokenizer;
        public int EmbeddingChannels => _embeddingChannels;

        /// <param name="bundle">
        /// A bundle already validated by <see cref="BundleLoader"/> and
        /// carrying a text encoder (<see cref="ControllerBundle.HasTextEncoder"/>).
        /// </param>
        /// <param name="backendType">
        /// Sentis backend. CPU is a sensible default here — the encode is
        /// off the per-frame hot path and a synchronous readback follows.
        /// </param>
        public PromptTextEncoder(ControllerBundle bundle, BackendType backendType = BackendType.CPU)
        {
            if (bundle == null)
            {
                throw new ArgumentNullException(nameof(bundle));
            }

            if (!bundle.HasTextEncoder)
            {
                throw new InvalidOperationException(
                    "This bundle ships no text_encoder (manifest has no text_encoder section, " +
                    "bundle predates A7.1 or was exported without --encoder-artifact) — " +
                    "in-engine prompt encoding is unavailable (text_encoding.md §3).");
            }

            var section = bundle.Manifest.text_encoder;
            _tokenizer = new ClipBpeTokenizer(
                bundle.TokenizerVocabJson,
                bundle.TokenizerMergesText,
                section.tokenizer.max_length,
                section.tokenizer.bos_id,
                section.tokenizer.eos_id,
                section.tokenizer.pad_id);
            _embeddingChannels = section.embedding_channels;

            // Sentis 2.x has no ModelLoader.Load(byte[]) overload — wrap the
            // bytes in a stream (must be a serialized .sentis model, not raw
            // .onnx — Sentis 2.x cannot parse .onnx at runtime).
            using (var modelStream = new System.IO.MemoryStream(bundle.TextEncoderModelBytes))
            {
                _model = ModelLoader.Load(modelStream);
            }

            _worker = new Worker(_model, backendType);

            _inputIdsScratch = new int[_tokenizer.MaxLength];
            _attentionMaskScratch = new float[_tokenizer.MaxLength];
            _inputIdsTensor = new Tensor<int>(new TensorShape(1, _tokenizer.MaxLength));
            _attentionMaskTensor = new Tensor<float>(new TensorShape(1, _tokenizer.MaxLength));
        }

        /// <summary>
        /// Tokenize <paramref name="text"/> and run one encoder pass,
        /// writing the pooled embedding into <paramref name="destination"/>
        /// (length <see cref="EmbeddingChannels"/>).
        /// </summary>
        public void Encode(string text, Span<float> destination)
        {
            if (destination.Length != _embeddingChannels)
            {
                throw new ArgumentException(
                    $"destination length must be {_embeddingChannels}; got {destination.Length}.");
            }

            _tokenizer.Encode(text, _inputIdsScratch, _attentionMaskScratch);
            _inputIdsTensor.Upload(_inputIdsScratch);
            _attentionMaskTensor.Upload(_attentionMaskScratch);

            _worker.SetInput(InputIdsName, _inputIdsTensor);
            _worker.SetInput(AttentionMaskName, _attentionMaskTensor);
            _worker.Schedule();

            if (_worker.PeekOutput(PromptEmbOutputName) is not Tensor<float> output)
            {
                throw new InvalidOperationException(
                    "Sentis worker produced no 'prompt_emb' output — check text_encoder.onnx " +
                    "output names against the contract (text_encoding.md §1).");
            }

            // Sentis 2.x: DownloadToArray() takes no argument and returns a
            // fresh float[] (blocking GPU readback included).
            output.DownloadToArray().AsSpan().CopyTo(destination);
        }

        public void Dispose()
        {
            _inputIdsTensor?.Dispose();
            _attentionMaskTensor?.Dispose();
            _worker?.Dispose();
        }
    }
}
