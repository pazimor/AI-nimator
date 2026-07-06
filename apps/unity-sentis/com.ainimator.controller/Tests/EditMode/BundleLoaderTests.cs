using System.IO;
using AInimator.Controller.Bundle;
using NUnit.Framework;

namespace AInimator.Controller.Tests
{
    /// <summary>
    /// Manifest/bundle validation tests: a bundle missing a required file,
    /// or declaring an incompatible/invalid contract, must fail loudly
    /// (Goal B verite #4 — never a silent default).
    /// </summary>
    /// <remarks>
    /// These tests copy the fixture <c>reference_bundle/</c> (manifest +
    /// norm_stats + presets, no <c>.onnx</c> — never committed) into a temp
    /// directory per test so <see cref="BundleLoader.Load"/> can be
    /// exercised on a real filesystem path. Tests that would reach the
    /// missing-onnx failure path assert that specific, expected exception
    /// rather than trying to run inference (no Sentis model is available
    /// in these fixtures).
    /// </remarks>
    public class BundleLoaderTests
    {
        private string _tempDir;

        [SetUp]
        public void SetUp()
        {
            _tempDir = Path.Combine(Path.GetTempPath(), "ainimator_bundle_test_" + System.Guid.NewGuid());
            CopyFixtureInto(_tempDir);
        }

        [TearDown]
        public void TearDown()
        {
            if (Directory.Exists(_tempDir))
            {
                Directory.Delete(_tempDir, recursive: true);
            }
        }

        private static void CopyFixtureInto(string destination)
        {
            var source = TestFixturePaths.ReferenceBundleDirectory();
            CopyDirectory(source, destination);
        }

        private static void CopyDirectory(string source, string destination)
        {
            Directory.CreateDirectory(destination);
            foreach (var file in Directory.GetFiles(source))
            {
                File.Copy(file, Path.Combine(destination, Path.GetFileName(file)), overwrite: true);
            }

            foreach (var dir in Directory.GetDirectories(source))
            {
                CopyDirectory(dir, Path.Combine(destination, Path.GetFileName(dir)));
            }
        }

        [Test]
        public void Load_MissingDirectory_ThrowsBundleLoadException()
        {
            Assert.Throws<BundleLoadException>(
                () => BundleLoader.Load(Path.Combine(_tempDir, "does_not_exist")));
        }

        [Test]
        public void Load_MissingOnnxFile_ThrowsBundleLoadExceptionMentioningFileName()
        {
            // The fixture intentionally has no controller.onnx (never committed).
            var ex = Assert.Throws<BundleLoadException>(() => BundleLoader.Load(_tempDir));
            StringAssert.Contains("controller.onnx", ex.Message);
        }

        [Test]
        public void Load_IncompatibleBundleVersion_ThrowsBundleLoadException()
        {
            RewriteManifestField(_tempDir, "\"bundle_version\": \"A7.0\"", "\"bundle_version\": \"B1.0\"");

            var ex = Assert.Throws<BundleLoadException>(() => BundleLoader.Load(_tempDir));
            StringAssert.Contains("Incompatible bundle_version", ex.Message);
        }

        [Test]
        public void Load_WrongStateChannels_ThrowsBundleLoadException()
        {
            RewriteManifestField(_tempDir, "\"num_bones\": 22", "\"num_bones\": 21");

            var ex = Assert.Throws<BundleLoadException>(() => BundleLoader.Load(_tempDir));
            StringAssert.Contains("num_bones", ex.Message);
        }

        [Test]
        public void Load_ZeroPromptNullEmbLength_ThrowsWhenManifestDeclaresPrompt()
        {
            // Corrupt norm_stats.json's prompt.null_emb to the wrong length
            // while the manifest still declares prompt_emb_channels=512.
            var normStatsPath = Path.Combine(_tempDir, "norm_stats.json");
            var json = File.ReadAllText(normStatsPath);
            // Replace the prompt section's null_emb with a too-short array.
            var corrupted = System.Text.RegularExpressions.Regex.Replace(
                json,
                "\"null_emb\": \\[[^\\]]*\\]",
                "\"null_emb\": [0.0, 0.0]");
            File.WriteAllText(normStatsPath, corrupted);

            var ex = Assert.Throws<BundleLoadException>(() => BundleLoader.Load(_tempDir));
            StringAssert.Contains("null_emb", ex.Message);
        }

        private static void RewriteManifestField(string bundleDir, string oldText, string newText)
        {
            var manifestPath = Path.Combine(bundleDir, "manifest.json");
            var json = File.ReadAllText(manifestPath);
            Assert.That(json, Does.Contain(oldText), "Test fixture assumption changed; update the test.");
            File.WriteAllText(manifestPath, json.Replace(oldText, newText));
        }
    }
}
