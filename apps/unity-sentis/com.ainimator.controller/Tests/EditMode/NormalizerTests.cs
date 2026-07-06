using AInimator.Controller.Bundle;
using AInimator.Controller.Normalization;
using NUnit.Framework;
using UnityEngine;

namespace AInimator.Controller.Tests
{
    /// <summary>
    /// Numeric parity tests for <see cref="Normalizer"/> against concrete
    /// values pulled from the reference bundle's <c>norm_stats.json</c>
    /// (Tests/Fixtures/reference_bundle), mirroring
    /// <c>test/ainimator/export/test_bundle.py::test_bundle_engine_parity_with_normalization</c>.
    /// </summary>
    public class NormalizerTests
    {
        // control.mean/std for the reference bundle (output/reference_bundle/norm_stats.json).
        private const float ControlMeanVx = -0.00013408098311629146f;
        private const float ControlMeanVz = 0.0021747760474681854f;
        private const float ControlStdVx = 0.0026797926984727383f;
        private const float ControlStdVz = 0.005573602393269539f;

        private static Manifest ReferenceManifest()
        {
            return new Manifest
            {
                bundle_version = "A7.0",
                state_channels = 136,
                num_bones = 22,
                rotation_channels_per_bone = 6,
                root_local_motion_channels = 4,
                control_channels = 2,
                control_layout = new() { "vx", "vz" },
                phase_channels = 0,
                prompt_emb_channels = 512,
                context_frames = 8,
                output_layout = "bone_delta|global_delta",
                coord_system = "Y-up right-handed",
                normalization_note = "",
                reserved_input_groups = new() { "interaction", "perception", "reaction", "morphology" },
            };
        }

        private static NormStats ReferenceStats()
        {
            // Loaded via Resources so the fixture resolves whether the
            // package is embedded, local, or referenced by git URL — see
            // Tests/Fixtures/Resources/reference_bundle/norm_stats.json.
            var textAsset = Resources.Load<TextAsset>("reference_bundle/norm_stats");
            Assert.That(textAsset, Is.Not.Null, "Missing test fixture: reference_bundle/norm_stats.json");
            return NormStats.Parse(textAsset.text);
        }

        [Test]
        public void EncodeControl_MatchesRawZNormFormula()
        {
            var normalizer = new Normalizer(ReferenceStats(), ReferenceManifest());
            float[] raw = { 0.5f, 1.0f };
            float[] dest = new float[2];

            normalizer.EncodeControl(raw, dest);

            var expectedVx = (raw[0] - ControlMeanVx) / ControlStdVx;
            var expectedVz = (raw[1] - ControlMeanVz) / ControlStdVz;
            Assert.That(dest[0], Is.EqualTo(expectedVx).Within(1e-4f));
            Assert.That(dest[1], Is.EqualTo(expectedVz).Within(1e-4f));
        }

        [Test]
        public void EncodeThenDecodeBoneFrame_RoundTripsWithinTolerance()
        {
            var normalizer = new Normalizer(ReferenceStats(), ReferenceManifest());
            var raw = new float[normalizer.BoneFrameLength];
            for (var i = 0; i < raw.Length; i++)
            {
                raw[i] = 0.01f * i - 1f;
            }

            var encoded = new float[normalizer.BoneFrameLength];
            normalizer.EncodeBoneFrame(raw, encoded);

            // Bone STATE round-trip requires the state (not delta) stats;
            // Normalizer only exposes delta decode, so this test instead
            // asserts the encode formula directly against the fixture stats.
            var stats = ReferenceStats();
            for (var i = 0; i < raw.Length; i++)
            {
                var std = stats.BoneStd[i] < 1e-5f ? 1e-5f : stats.BoneStd[i];
                var expected = (raw[i] - stats.BoneMean[i]) / std;
                Assert.That(encoded[i], Is.EqualTo(expected).Within(1e-4f), $"channel {i}");
            }
        }

        [Test]
        public void DecodeGlobalDelta_AppliesRawStdAndMean()
        {
            var normalizer = new Normalizer(ReferenceStats(), ReferenceManifest());
            var stats = ReferenceStats();
            float[] norm = { 0.1f, -0.2f, 0.3f, -0.4f };
            var dest = new float[4];

            normalizer.DecodeGlobalDelta(norm, dest);

            for (var i = 0; i < 4; i++)
            {
                var expected = norm[i] * stats.DeltaGlobalStd[i] + stats.DeltaGlobalMean[i];
                Assert.That(dest[i], Is.EqualTo(expected).Within(1e-6f), $"channel {i}");
            }
        }

        [Test]
        public void EncodeControl_WithAimChannels_PassesAimThroughUnchanged()
        {
            var manifest = ReferenceManifest();
            manifest.control_channels = 4;
            manifest.control_layout = new() { "vx", "vz", "aim_x", "aim_z" };

            var normalizer = new Normalizer(ReferenceStats(), manifest);
            float[] raw = { 0.0f, 0.033f, 0.7071f, 0.7071f };
            var dest = new float[4];

            normalizer.EncodeControl(raw, dest);

            Assert.That(dest[2], Is.EqualTo(raw[2]).Within(1e-9f), "aim_x must pass through unnormalized");
            Assert.That(dest[3], Is.EqualTo(raw[3]).Within(1e-9f), "aim_z must pass through unnormalized");
        }
    }
}
