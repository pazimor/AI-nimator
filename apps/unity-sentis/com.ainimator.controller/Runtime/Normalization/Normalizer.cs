using System;
using AInimator.Controller.Bundle;

namespace AInimator.Controller.Normalization
{
    /// <summary>
    /// Pure z-normalization math driven by the bundle's <see cref="NormStats"/>.
    /// </summary>
    /// <remarks>
    /// Mirrors <c>ainimator.model.motion_normalizer.MotionNormalizer</c> and
    /// the engine-parity test
    /// <c>test/ainimator/export/test_bundle.py::test_bundle_engine_parity_with_normalization</c>:
    /// <c>(x - mean) / std</c> for encode, <c>x * std + mean</c> for decode.
    /// No engine dependency — pure arrays in, pure arrays out — so this
    /// class is unit-testable without Unity running (EditMode tests).
    /// </remarks>
    public sealed class Normalizer
    {
        // Same floor as the Python side (EPSILON_STD in motion_normalizer.py)
        // so std==0 channels (e.g. locked bones) never divide by zero.
        private const float EpsilonStd = 1e-5f;

        private readonly NormStats _stats;
        private readonly int _numBones;
        private readonly int _rotationChannels;
        private readonly int _globalChannels;
        private readonly bool _hasAim;

        public Normalizer(NormStats stats, Manifest manifest)
        {
            _stats = stats ?? throw new ArgumentNullException(nameof(stats));
            _numBones = manifest.num_bones;
            _rotationChannels = manifest.rotation_channels_per_bone;
            _globalChannels = manifest.root_local_motion_channels;
            _hasAim = manifest.HasAim;
        }

        /// <summary>
        /// Z-normalize one frame of bone rotation6d state, shape
        /// <c>(numBones * rotationChannels)</c> row-major, in place into
        /// <paramref name="destination"/>.
        /// </summary>
        public void EncodeBoneFrame(ReadOnlySpan<float> rawFrame, Span<float> destination)
        {
            EncodeFlat(rawFrame, _stats.BoneMean, _stats.BoneStd, destination);
        }

        /// <summary>
        /// Z-normalize one frame of root-local global motion, shape
        /// <c>(globalChannels)</c>.
        /// </summary>
        public void EncodeGlobalFrame(ReadOnlySpan<float> rawFrame, Span<float> destination)
        {
            EncodeFlat(rawFrame, _stats.GlobalMean, _stats.GlobalStd, destination);
        }

        /// <summary>
        /// Z-normalize a raw control vector <c>(vx, vz[, aim_x, aim_z])</c>.
        /// <c>vx, vz</c> use <c>control.mean/std</c>; <c>aim_x, aim_z</c>
        /// (when present) pass through unchanged — they are unit-norm by
        /// construction, never z-normalized (inference_contract.md §3.2).
        /// </summary>
        public void EncodeControl(ReadOnlySpan<float> rawControl, Span<float> destination)
        {
            var expectedLen = _hasAim ? 4 : 2;
            if (rawControl.Length != expectedLen || destination.Length != expectedLen)
            {
                throw new ArgumentException(
                    $"EncodeControl expects length {expectedLen} (aim={_hasAim}); " +
                    $"got raw={rawControl.Length}, dest={destination.Length}.");
            }

            // No epsilon clamp: mirrors the raw `(control - mean) / std` used
            // by the training/rollout code path (controller_training_v2.py,
            // controller_generalization_v2.py) — control stats are never
            // routed through MotionNormalizer's clamped std.
            destination[0] = (rawControl[0] - _stats.ControlMean[0]) / _stats.ControlStd[0];
            destination[1] = (rawControl[1] - _stats.ControlMean[1]) / _stats.ControlStd[1];
            if (_hasAim)
            {
                destination[2] = rawControl[2];
                destination[3] = rawControl[3];
            }
        }

        /// <summary>
        /// Denormalize one frame of predicted bone delta, shape
        /// <c>(numBones * rotationChannels)</c>: <c>x * std + mean</c>.
        /// </summary>
        public void DecodeBoneDelta(ReadOnlySpan<float> normDelta, Span<float> destination)
        {
            DecodeFlat(normDelta, _stats.DeltaBoneMean, _stats.DeltaBoneStd, destination);
        }

        /// <summary>
        /// Denormalize one frame of predicted global delta
        /// <c>(Δforward, Δlateral, Δheight, Δyaw)</c>, shape <c>(globalChannels)</c>.
        /// </summary>
        public void DecodeGlobalDelta(ReadOnlySpan<float> normDelta, Span<float> destination)
        {
            DecodeFlat(normDelta, _stats.DeltaGlobalMean, _stats.DeltaGlobalStd, destination);
        }

        /// <summary>Number of scalar channels in one bone-state frame.</summary>
        public int BoneFrameLength => _numBones * _rotationChannels;

        /// <summary>Number of scalar channels in one global-state frame.</summary>
        public int GlobalFrameLength => _globalChannels;

        private static void EncodeFlat(
            ReadOnlySpan<float> raw, ReadOnlySpan<float> mean, ReadOnlySpan<float> std, Span<float> destination)
        {
            if (raw.Length != mean.Length || destination.Length != mean.Length)
            {
                throw new ArgumentException(
                    $"Length mismatch: raw={raw.Length}, stats={mean.Length}, dest={destination.Length}.");
            }

            for (var i = 0; i < raw.Length; i++)
            {
                destination[i] = (raw[i] - mean[i]) / MathF.Max(std[i], EpsilonStd);
            }
        }

        private static void DecodeFlat(
            ReadOnlySpan<float> norm, ReadOnlySpan<float> mean, ReadOnlySpan<float> std, Span<float> destination)
        {
            if (norm.Length != mean.Length || destination.Length != mean.Length)
            {
                throw new ArgumentException(
                    $"Length mismatch: norm={norm.Length}, stats={mean.Length}, dest={destination.Length}.");
            }

            // No epsilon clamp here: mirrors MotionNormalizer.denormalizeBone,
            // which uses the raw std as-is (the stored std was already
            // floored at fit time in fitFromTensors — decode never re-clamps).
            for (var i = 0; i < norm.Length; i++)
            {
                destination[i] = norm[i] * std[i] + mean[i];
            }
        }
    }
}
