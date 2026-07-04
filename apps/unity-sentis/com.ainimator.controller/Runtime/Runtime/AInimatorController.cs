using System;
using AInimator.Controller.Bundle;
using AInimator.Controller.Presets;
using AInimator.Controller.Prompting;
using Unity.Sentis;
using UnityEngine;

namespace AInimator.Controller.Runtime
{
    /// <summary>
    /// Engine-facing façade: owns one <see cref="ControllerRuntime"/> +
    /// <see cref="StateBuffer"/> + <see cref="RootMotionIntegrator"/> for a
    /// single character and exposes a single <see cref="Tick"/> entry point
    /// per frame (Goal B §3.3 pseudo-loop, C# side).
    /// </summary>
    /// <remarks>
    /// This MonoBehaviour is engine glue only: it does not itself implement
    /// any normalization/integration math (that lives in
    /// <c>Normalization.Normalizer</c> and <c>RootMotionIntegrator</c>) so
    /// those stay unit-testable without a running Unity instance.
    /// <para/>
    /// Seeding: at construction time the state window is seeded uniformly
    /// with a caller-supplied rest-pose frame (typically the
    /// model's bind pose in rotation6d, with a zero root-local motion
    /// frame). A production integration may instead seed from a real
    /// animation clip snippet for a cleaner warm start — this is left to
    /// the caller since it is content-dependent.
    /// <para/>
    /// Prompt-at-runtime (<c>apps/spec/rig_binding.md</c> §3):
    /// <see cref="SetPrompt"/>/<see cref="SetPromptEmbedding"/>/
    /// <see cref="ClearPrompt"/> change the <b>active</b> prompt embedding;
    /// the actual value fed to the graph each frame is cross-faded in
    /// embedding space over <see cref="PromptCrossFadeSeconds"/> by an owned
    /// <see cref="PromptCrossFader"/> — see that class for the
    /// heuristic/fallback caveat. <see cref="ClearPrompt"/> fades toward the
    /// bundle's learned <c>prompt.null_emb</c>, never zeros.
    /// </remarks>
    public sealed class AInimatorController : IDisposable
    {
        private readonly ControllerRuntime _runtime;
        private readonly StateBuffer _stateBuffer;
        private RootMotionIntegrator _rootMotion;

        private readonly float[] _rawBoneDeltaScratch;
        private readonly float[] _rawGlobalDeltaScratch;
        private readonly float[] _rawBoneFrameScratch;

        private readonly PromptCrossFader _promptCrossFader;
        private readonly float[] _promptEmbScratch;
        private readonly float[] _activePromptOverride;
        private bool _hasActivePromptOverride;

        public Manifest Manifest => _runtime.Manifest;
        public ControllerBundle Bundle { get; }

        /// <summary>Current world-space root position (Y-up, meters).</summary>
        public Vector3 RootPosition => new(_rootMotion.PositionX, _rootMotion.PositionY, _rootMotion.PositionZ);

        /// <summary>Current world-space yaw, radians around +Y.</summary>
        public float RootYawRadians => _rootMotion.YawRadians;

        /// <summary>
        /// This frame's raw (un-scaled) root-local motion delta
        /// <c>(Δforward, Δlateral, Δheight, Δyaw)</c>, as produced by the
        /// most recent <see cref="Tick"/> — the same values already folded
        /// into <see cref="RootPosition"/>/<see cref="RootYawRadians"/> at
        /// <c>rigScale == 1</c>. A <see cref="Rig.RigBinder"/> reads this to
        /// apply its own <c>rigScale</c>-corrected root motion instead
        /// (<c>apps/spec/rig_binding.md</c> §2.3) rather than re-deriving it
        /// from the (unscaled) integrated total.
        /// </summary>
        public ReadOnlySpan<float> LastRawGlobalDelta => _rawGlobalDeltaScratch;

        /// <param name="bundle">A bundle already validated by <see cref="BundleLoader"/>.</param>
        /// <param name="seedBoneFrame">
        /// Rest-pose bone frame, row-major <c>(numBones * rotationChannelsPerBone)</c>,
        /// used to uniformly seed the context window.
        /// </param>
        /// <param name="backendType">Sentis backend (GPUCompute preferred, CPU fallback).</param>
        public AInimatorController(
            ControllerBundle bundle,
            ReadOnlySpan<float> seedBoneFrame,
            BackendType backendType = BackendType.GPUCompute)
        {
            Bundle = bundle ?? throw new ArgumentNullException(nameof(bundle));
            _runtime = new ControllerRuntime(bundle, backendType);

            var manifest = bundle.Manifest;
            _stateBuffer = new StateBuffer(
                manifest.context_frames, _runtime.Normalizer.BoneFrameLength, _runtime.Normalizer.GlobalFrameLength);

            Span<float> zeroGlobalFrame = stackalloc float[_runtime.Normalizer.GlobalFrameLength];
            _stateBuffer.SeedUniform(seedBoneFrame, zeroGlobalFrame);

            _rootMotion = RootMotionIntegrator.AtOrigin();

            _rawBoneDeltaScratch = new float[_runtime.Normalizer.BoneFrameLength];
            _rawGlobalDeltaScratch = new float[_runtime.Normalizer.GlobalFrameLength];
            _rawBoneFrameScratch = new float[_runtime.Normalizer.BoneFrameLength];

            if (manifest.HasPrompt)
            {
                _promptCrossFader = new PromptCrossFader(manifest.prompt_emb_channels);
                _promptEmbScratch = new float[manifest.prompt_emb_channels];
                _activePromptOverride = new float[manifest.prompt_emb_channels];
                // Start with no active prompt: snap to the learned null_emb
                // (inference_contract.md §4 — never zeros).
                _promptCrossFader.SnapTo(bundle.NormStats.PromptNullEmb);
            }
        }

        /// <summary>Cross-fade duration (seconds) applied by <see cref="SetPrompt"/>/<see cref="SetPromptEmbedding"/>/<see cref="ClearPrompt"/> (default 0.3s, spec §3).</summary>
        public float PromptCrossFadeSeconds
        {
            get => _promptCrossFader?.CrossFadeSeconds ?? PromptCrossFader.DefaultCrossFadeSeconds;
            set
            {
                if (_promptCrossFader != null)
                {
                    _promptCrossFader.CrossFadeSeconds = value;
                }
            }
        }

        /// <summary>True while a prompt cross-fade is in progress.</summary>
        public bool IsPromptFading => _promptCrossFader?.IsFading ?? false;

        /// <summary>
        /// Begin cross-fading toward <paramref name="preset"/>'s precomputed
        /// <c>prompt_emb</c> (rig_binding.md §3). No-op (throws) if the
        /// manifest declares no prompt channel, or if the preset carries no
        /// embedding.
        /// </summary>
        public void SetPrompt(ControlPreset preset)
        {
            if (preset == null)
            {
                throw new ArgumentNullException(nameof(preset));
            }

            if (preset.PromptEmb is not { Length: > 0 } presetEmb)
            {
                throw new ArgumentException(
                    $"ControlPreset '{preset.PresetName}' carries no prompt_emb — cannot SetPrompt from it.",
                    nameof(preset));
            }

            SetPromptEmbedding(presetEmb);
        }

        /// <summary>
        /// Begin cross-fading toward a raw prompt embedding supplied by the
        /// game (rig_binding.md §3), length <c>manifest.prompt_emb_channels</c>.
        /// </summary>
        public void SetPromptEmbedding(ReadOnlySpan<float> embedding)
        {
            RequirePromptSupport();
            if (embedding.Length != Manifest.prompt_emb_channels)
            {
                throw new ArgumentException(
                    $"embedding length must be {Manifest.prompt_emb_channels}; got {embedding.Length}.");
            }

            embedding.CopyTo(_activePromptOverride);
            _hasActivePromptOverride = true;
            Span<float> current = stackalloc float[_promptEmbScratch.Length];
            _promptCrossFader.BeginFadeTo(_activePromptOverride, current);
        }

        /// <summary>
        /// Begin cross-fading back to the bundle's learned null prompt
        /// embedding (never zeros — inference_contract.md §4).
        /// </summary>
        public void ClearPrompt()
        {
            RequirePromptSupport();
            _hasActivePromptOverride = false;
            Span<float> current = stackalloc float[_promptEmbScratch.Length];
            _promptCrossFader.BeginFadeTo(Bundle.NormStats.PromptNullEmb, current);
        }

        private void RequirePromptSupport()
        {
            if (!Manifest.HasPrompt)
            {
                throw new InvalidOperationException(
                    "This bundle's manifest declares prompt_emb_channels == 0 -- prompt-at-runtime is unavailable.");
            }
        }

        /// <summary>
        /// Advance one frame: run the controller forward pass, integrate
        /// the root motion delta, and push the new state into the window.
        /// </summary>
        /// <param name="preset">Active control preset for this frame.</param>
        /// <param name="continuousAim">
        /// Optional continuous aim override (e.g. mouse/stick), applied
        /// instead of the preset's static aim when the manifest declares
        /// <c>control_channels == 4</c>. Pass <c>(0, 0)</c> to defer to the preset.
        /// </param>
        /// <param name="deltaTimeSeconds">
        /// Frame time (seconds), used only to advance the prompt cross-fade
        /// clock (<see cref="PromptCrossFadeSeconds"/>). Defaults to
        /// <see cref="Time.deltaTime"/> when called from Unity's main thread;
        /// pass an explicit value from tests/fixed-step loops.
        /// </param>
        /// <returns>The new bone-rotation6d frame, row-major (numBones * rotationChannelsPerBone).</returns>
        public ReadOnlySpan<float> Tick(ControlPreset preset, Vector2 continuousAim = default, float? deltaTimeSeconds = null)
        {
            if (preset == null)
            {
                throw new ArgumentNullException(nameof(preset));
            }

            Span<float> rawControl = stackalloc float[Manifest.control_channels];
            preset.WriteRawControl(rawControl, Manifest.HasAim);
            if (Manifest.HasAim && continuousAim != default)
            {
                rawControl[2] = continuousAim.x;
                rawControl[3] = continuousAim.y;
            }

            _promptCrossFader?.Tick(deltaTimeSeconds ?? Time.deltaTime);
            var promptEmb = ResolvePromptEmb(preset);

            _runtime.Tick(
                _stateBuffer,
                rawControl,
                promptEmb,
                ReadOnlySpan<float>.Empty, // phase not modeled by presets yet (manifest.phase_channels==0 in B1 reference bundle)
                _rawBoneDeltaScratch,
                _rawGlobalDeltaScratch);

            _stateBuffer.CopyLatestBoneFrame(_rawBoneFrameScratch);
            for (var i = 0; i < _rawBoneFrameScratch.Length; i++)
            {
                _rawBoneFrameScratch[i] += _rawBoneDeltaScratch[i];
            }

            _rootMotion.Integrate(_rawGlobalDeltaScratch);
            _stateBuffer.Push(_rawBoneFrameScratch, _rawGlobalDeltaScratch);

            return _rawBoneFrameScratch;
        }

        /// <summary>
        /// Resolve this frame's prompt embedding. When the caller has never
        /// invoked <see cref="SetPrompt"/>/<see cref="SetPromptEmbedding"/>/
        /// <see cref="ClearPrompt"/>, this transparently falls back to the
        /// legacy B1 behaviour of reading the active preset's static
        /// <c>prompt_emb</c> (or the null embedding) directly, un-faded —
        /// preserving the capsule/B1 code path exactly. Once any prompt API
        /// call has been made, the <see cref="PromptCrossFader"/> becomes
        /// the sole source of truth for this frame's embedding (rig_binding.md §3).
        /// </summary>
        private ReadOnlySpan<float> ResolvePromptEmb(ControlPreset preset)
        {
            if (!Manifest.HasPrompt)
            {
                return ReadOnlySpan<float>.Empty;
            }

            if (!_hasActivePromptOverride && !_promptCrossFader.IsFading)
            {
                // Legacy B1 path: no explicit prompt API call yet this
                // session -- read the preset directly, un-faded.
                if (preset.PromptEmb is { Length: > 0 } presetEmb)
                {
                    return presetEmb;
                }

                return Bundle.NormStats.PromptNullEmb;
            }

            _promptCrossFader.Evaluate(_promptEmbScratch);
            return _promptEmbScratch;
        }

        public void Dispose()
        {
            _runtime.Dispose();
        }
    }
}
