using System;
using AInimator.Controller.Bundle;
using AInimator.Controller.Presets;
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
    /// </remarks>
    public sealed class AInimatorController : IDisposable
    {
        private readonly ControllerRuntime _runtime;
        private readonly StateBuffer _stateBuffer;
        private RootMotionIntegrator _rootMotion;

        private readonly float[] _rawBoneDeltaScratch;
        private readonly float[] _rawGlobalDeltaScratch;
        private readonly float[] _rawBoneFrameScratch;

        public Manifest Manifest => _runtime.Manifest;
        public ControllerBundle Bundle { get; }

        /// <summary>Current world-space root position (Y-up, meters).</summary>
        public Vector3 RootPosition => new(_rootMotion.PositionX, _rootMotion.PositionY, _rootMotion.PositionZ);

        /// <summary>Current world-space yaw, radians around +Y.</summary>
        public float RootYawRadians => _rootMotion.YawRadians;

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
        /// <returns>The new bone-rotation6d frame, row-major (numBones * rotationChannelsPerBone).</returns>
        public ReadOnlySpan<float> Tick(ControlPreset preset, Vector2 continuousAim = default)
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

        private ReadOnlySpan<float> ResolvePromptEmb(ControlPreset preset)
        {
            if (!Manifest.HasPrompt)
            {
                return ReadOnlySpan<float>.Empty;
            }

            if (preset.PromptEmb is { Length: > 0 } presetEmb)
            {
                return presetEmb;
            }

            // No active prompt: feed the learned null embedding, never zeros
            // (inference_contract.md §4).
            return Bundle.NormStats.PromptNullEmb;
        }

        public void Dispose()
        {
            _runtime.Dispose();
        }
    }
}
