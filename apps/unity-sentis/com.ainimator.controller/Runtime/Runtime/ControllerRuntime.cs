using System;
using AInimator.Controller.Bundle;
using AInimator.Controller.Normalization;
using Unity.Sentis;

namespace AInimator.Controller.Runtime
{
    /// <summary>
    /// Owns the Sentis <see cref="Worker"/>/<see cref="Model"/> for a loaded
    /// controller bundle and runs exactly one forward pass per
    /// <see cref="Tick"/>. Contains no gameplay logic: callers own the
    /// <see cref="StateBuffer"/>, the active control vector, and applying
    /// the returned pose to a skeleton.
    /// </summary>
    /// <remarks>
    /// Sentis API notes (verify against the installed package — Sentis'
    /// API churned significantly from Barracuda and across 1.x minors;
    /// this was written against the documented Sentis 1.x surface:
    /// <c>ModelLoader.Load</c>, <c>Worker</c> (constructed with a
    /// <see cref="BackendType"/>), <c>Worker.SetInput</c>/<c>Schedule</c>/
    /// <c>PeekOutput</c>, and <c>Tensor&lt;float&gt;</c> for typed tensors).
    /// **Must be smoke-tested against the actual installed Sentis package
    /// before shipping** — this class cannot be compiled in this session
    /// (no Unity editor available).
    /// <para/>
    /// One forward = one frame (Goal B verite #2): the autoregressive loop,
    /// the Δstate integration, and the state window all live here/in
    /// <see cref="StateBuffer"/>/<see cref="RootMotionIntegrator"/> — never
    /// inside the ONNX graph.
    /// <para/>
    /// Input/output tensor names match <c>controller.onnx</c> exactly
    /// (apps/spec/inference_contract.md §2): <c>bone_window</c>,
    /// <c>control</c>, <c>global_window</c>, optional <c>prompt_emb</c>,
    /// optional <c>phase</c>; outputs <c>bone_delta</c>, <c>global_delta</c>.
    /// </remarks>
    public sealed class ControllerRuntime : IDisposable
    {
        private const string BoneWindowInputName = "bone_window";
        private const string ControlInputName = "control";
        private const string GlobalWindowInputName = "global_window";
        private const string PromptEmbInputName = "prompt_emb";
        private const string PhaseInputName = "phase";
        private const string BoneDeltaOutputName = "bone_delta";
        private const string GlobalDeltaOutputName = "global_delta";

        private readonly Manifest _manifest;
        private readonly Normalizer _normalizer;
        private readonly Model _model;
        private readonly Worker _worker;

        // Pre-allocated per-frame tensors — reused every Tick, never
        // reallocated (Goal B performance rule: zero per-frame GC on the
        // hot path).
        private Tensor<float> _boneWindowTensor;
        private Tensor<float> _controlTensor;
        private Tensor<float> _globalWindowTensor;
        private Tensor<float> _promptEmbTensor;
        private Tensor<float> _phaseTensor;

        // Scratch buffers for normalization math (also reused every frame,
        // including the prompt/phase upload buffers — no per-frame heap
        // allocation on the hot path).
        private readonly float[] _normBoneWindowScratch;
        private readonly float[] _normGlobalWindowScratch;
        private readonly float[] _normControlScratch;
        private readonly float[] _promptEmbScratch;
        private readonly float[] _phaseScratch;
        private readonly float[] _rawBoneDeltaScratch;
        private readonly float[] _rawGlobalDeltaScratch;

        public Manifest Manifest => _manifest;
        public Normalizer Normalizer => _normalizer;

        /// <param name="bundle">A bundle already validated by <see cref="BundleLoader"/>.</param>
        /// <param name="backendType">
        /// Sentis backend. Prefer <see cref="BackendType.GPUCompute"/> when
        /// available; fall back to <see cref="BackendType.CPU"/>.
        /// </param>
        public ControllerRuntime(ControllerBundle bundle, BackendType backendType = BackendType.GPUCompute)
        {
            if (bundle == null)
            {
                throw new ArgumentNullException(nameof(bundle));
            }

            _manifest = bundle.Manifest;
            _normalizer = new Normalizer(bundle.NormStats, bundle.Manifest);

            _model = ModelLoader.Load(bundle.OnnxModelBytes);
            _worker = new Worker(_model, backendType);

            var boneWindowLen = _manifest.context_frames * _normalizer.BoneFrameLength;
            var globalWindowLen = _manifest.context_frames * _normalizer.GlobalFrameLength;

            _normBoneWindowScratch = new float[boneWindowLen];
            _normGlobalWindowScratch = new float[globalWindowLen];
            _normControlScratch = new float[_manifest.control_channels];
            _promptEmbScratch = _manifest.HasPrompt ? new float[_manifest.prompt_emb_channels] : Array.Empty<float>();
            _phaseScratch = _manifest.HasPhase ? new float[2] : Array.Empty<float>();
            _rawBoneDeltaScratch = new float[_normalizer.BoneFrameLength];
            _rawGlobalDeltaScratch = new float[_normalizer.GlobalFrameLength];

            _boneWindowTensor = new Tensor<float>(
                new TensorShape(1, _manifest.context_frames, _manifest.num_bones, _manifest.rotation_channels_per_bone));
            _controlTensor = new Tensor<float>(new TensorShape(1, _manifest.control_channels));
            _globalWindowTensor = new Tensor<float>(
                new TensorShape(1, _manifest.context_frames, _manifest.root_local_motion_channels));

            if (_manifest.HasPrompt)
            {
                _promptEmbTensor = new Tensor<float>(new TensorShape(1, _manifest.prompt_emb_channels));
            }

            if (_manifest.HasPhase)
            {
                _phaseTensor = new Tensor<float>(new TensorShape(1, 2));
            }
        }

        /// <summary>
        /// Run one forward pass: normalize <paramref name="stateBuffer"/> +
        /// <paramref name="rawControl"/> (+ optional prompt/phase), execute
        /// the graph once, and write the denormalized bone/global deltas
        /// into <paramref name="rawBoneDelta"/> / <paramref name="rawGlobalDelta"/>.
        /// </summary>
        /// <param name="stateBuffer">The rolling context-frames window (already seeded).</param>
        /// <param name="rawControl">
        /// Raw control vector, length <c>manifest.control_channels</c>
        /// (<c>vx, vz[, aim_x, aim_z]</c>).
        /// </param>
        /// <param name="rawPromptEmb">
        /// Raw prompt embedding, length <c>manifest.prompt_emb_channels</c>, or
        /// <c>null</c> to use the bundle's learned <c>prompt.null_emb</c>
        /// (never zeros — inference_contract.md §4). Ignored when the
        /// manifest declares no prompt channel.
        /// </param>
        /// <param name="rawPhase">
        /// Raw <c>(cos, sin)</c> phase, or <c>null</c> when the manifest
        /// declares no phase channel. Required (throws) when the manifest
        /// declares <c>phase_channels == 2</c> and this is <c>null</c>.
        /// </param>
        /// <param name="rawBoneDelta">
        /// Destination for the denormalized bone delta, length
        /// <c>numBones * rotationChannelsPerBone</c>.
        /// </param>
        /// <param name="rawGlobalDelta">
        /// Destination for the denormalized global delta
        /// <c>(Δforward, Δlateral, Δheight, Δyaw)</c>, length
        /// <c>root_local_motion_channels</c>.
        /// </param>
        public void Tick(
            StateBuffer stateBuffer,
            ReadOnlySpan<float> rawControl,
            ReadOnlySpan<float> rawPromptEmb,
            ReadOnlySpan<float> rawPhase,
            Span<float> rawBoneDelta,
            Span<float> rawGlobalDelta)
        {
            if (!stateBuffer.IsFull)
            {
                throw new InvalidOperationException(
                    "StateBuffer must be seeded (SeedUniform or context_frames pushes) before Tick.");
            }

            EncodeInputs(stateBuffer, rawControl, rawPromptEmb, rawPhase);
            RunForward();
            DecodeOutputs(stateBuffer, rawBoneDelta, rawGlobalDelta);
        }

        private void EncodeInputs(
            StateBuffer stateBuffer,
            ReadOnlySpan<float> rawControl,
            ReadOnlySpan<float> rawPromptEmb,
            ReadOnlySpan<float> rawPhase)
        {
            // --- bone_window: normalize each frame of the window in place.
            Span<float> rawBoneWindow = stackalloc float[stateBuffer.ContextFrames * stateBuffer.BoneFrameLength];
            stateBuffer.CopyBoneWindowOrdered(rawBoneWindow);
            for (var frame = 0; frame < stateBuffer.ContextFrames; frame++)
            {
                var slice = rawBoneWindow.Slice(frame * stateBuffer.BoneFrameLength, stateBuffer.BoneFrameLength);
                var dest = _normBoneWindowScratch.AsSpan(frame * stateBuffer.BoneFrameLength, stateBuffer.BoneFrameLength);
                _normalizer.EncodeBoneFrame(slice, dest);
            }

            _boneWindowTensor.Upload(_normBoneWindowScratch);

            // --- global_window: normalize each frame of the window in place.
            Span<float> rawGlobalWindow = stackalloc float[stateBuffer.ContextFrames * stateBuffer.GlobalFrameLength];
            stateBuffer.CopyGlobalWindowOrdered(rawGlobalWindow);
            for (var frame = 0; frame < stateBuffer.ContextFrames; frame++)
            {
                var slice = rawGlobalWindow.Slice(frame * stateBuffer.GlobalFrameLength, stateBuffer.GlobalFrameLength);
                var dest = _normGlobalWindowScratch.AsSpan(frame * stateBuffer.GlobalFrameLength, stateBuffer.GlobalFrameLength);
                _normalizer.EncodeGlobalFrame(slice, dest);
            }

            _globalWindowTensor.Upload(_normGlobalWindowScratch);

            // --- control: vx,vz z-normed; aim_x,aim_z passthrough.
            _normalizer.EncodeControl(rawControl, _normControlScratch);
            _controlTensor.Upload(_normControlScratch);

            // --- prompt_emb: required whenever the manifest declares it;
            // a caller passing an empty span means "no active prompt" and
            // MUST be resolved to the learned null_emb by the caller
            // (ControllerRuntime does not silently substitute zeros).
            if (_manifest.HasPrompt)
            {
                if (rawPromptEmb.Length != _manifest.prompt_emb_channels)
                {
                    throw new ArgumentException(
                        $"rawPromptEmb length must be {_manifest.prompt_emb_channels} " +
                        "(pass norm_stats.json's prompt.null_emb when no prompt is active — " +
                        "never zeros, inference_contract.md §4); got " +
                        $"{rawPromptEmb.Length}.");
                }

                rawPromptEmb.CopyTo(_promptEmbScratch);
                _promptEmbTensor.Upload(_promptEmbScratch);
            }

            // --- phase: required whenever the manifest declares it.
            if (_manifest.HasPhase)
            {
                if (rawPhase.Length != 2)
                {
                    throw new ArgumentException(
                        $"rawPhase length must be 2 (cos, sin); got {rawPhase.Length}.");
                }

                rawPhase.CopyTo(_phaseScratch);
                _phaseTensor.Upload(_phaseScratch);
            }
        }

        private void RunForward()
        {
            _worker.SetInput(BoneWindowInputName, _boneWindowTensor);
            _worker.SetInput(ControlInputName, _controlTensor);
            _worker.SetInput(GlobalWindowInputName, _globalWindowTensor);
            if (_manifest.HasPrompt)
            {
                _worker.SetInput(PromptEmbInputName, _promptEmbTensor);
            }

            if (_manifest.HasPhase)
            {
                _worker.SetInput(PhaseInputName, _phaseTensor);
            }

            _worker.Schedule();
        }

        private void DecodeOutputs(StateBuffer stateBuffer, Span<float> rawBoneDelta, Span<float> rawGlobalDelta)
        {
            // NOTE (verify against installed Sentis): `PeekOutput` returns a
            // tensor that is still owned/backed by the worker/backend — it
            // must NOT be disposed here, and it may still be a GPU-resident
            // tensor requiring a readback before its CPU data is valid. The
            // exact readback call churned across Sentis versions
            // (`MakeReadable()`, `ReadbackAndClone()`, `CompleteOperationsAndDownload`,
            // etc.) — confirm the correct one-shot synchronous accessor for
            // the installed package version and update this method
            // accordingly. This code assumes a `Tensor<float>` result
            // exposing a synchronous `DownloadToArray`-style accessor after
            // `Schedule()` has completed; if the installed API instead
            // returns awaitable/async readback only, `Tick` must be adapted
            // (e.g. a synchronous blocking readback for parity/determinism —
            // Goal B requires per-frame determinism, so async readback must
            // not desynchronize the StateBuffer window).
            var boneDeltaOut = _worker.PeekOutput(BoneDeltaOutputName) as Tensor<float>;
            var globalDeltaOut = _worker.PeekOutput(GlobalDeltaOutputName) as Tensor<float>;

            if (boneDeltaOut == null || globalDeltaOut == null)
            {
                throw new InvalidOperationException(
                    "Sentis worker produced no output tensors for bone_delta/global_delta — " +
                    "check the loaded controller.onnx output names against the contract.");
            }

            boneDeltaOut.DownloadToArray(_rawBoneDeltaScratch);
            globalDeltaOut.DownloadToArray(_rawGlobalDeltaScratch);

            _normalizer.DecodeBoneDelta(_rawBoneDeltaScratch, rawBoneDelta);
            _normalizer.DecodeGlobalDelta(_rawGlobalDeltaScratch, rawGlobalDelta);
        }

        public void Dispose()
        {
            _boneWindowTensor?.Dispose();
            _controlTensor?.Dispose();
            _globalWindowTensor?.Dispose();
            _promptEmbTensor?.Dispose();
            _phaseTensor?.Dispose();
            _worker?.Dispose();
        }
    }
}
