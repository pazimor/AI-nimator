using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using AInimator.Controller.TextCommand;

namespace AInimator.Controller.Tests
{
    /// <summary>
    /// Parity + normalization tests for <see cref="TextToControlResolver"/>
    /// (Goal B phase B6). The parametrized cases in
    /// <see cref="ResolvesExpectedVector"/> / <see cref="UnresolvableReturnsNull"/>
    /// are duplicated verbatim from the shared reference suite
    /// <c>test/apps/test_text_to_control.py</c> — same phrases, same
    /// expected vectors (<c>text_to_control.md</c> §5).
    /// </summary>
    public class TextToControlResolverTests
    {
        private const float Walk = 0.033f;
        private const float Run = 0.1f;
        private static readonly float Diag = 1f / UnityEngine.Mathf.Sqrt(2f);

        private static Dictionary<string, object> _table;

        [OneTimeSetUp]
        public void LoadTable()
        {
            var packageInfo = UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                typeof(TextToControlResolverTests).Assembly);
            var packageRoot = packageInfo != null
                ? packageInfo.resolvedPath
                : Path.Combine("Packages", "com.ainimator.controller");

            var tablePath = Path.Combine(
                packageRoot, "Runtime", "TextCommand", "Resources", "text_to_control.json");
            var json = File.ReadAllText(tablePath);
            _table = TextToControlResolver.ParseTable(json);
        }

        // -- Shared parity cases (test/apps/test_text_to_control.py::test_resolves_expected_vector) --

        [TestCase("cours vers la gauche", -Run, 0f)]
        [TestCase("run left", -Run, 0f)]
        [TestCase("walk forward", 0f, Walk)]
        [TestCase("avance", 0f, Walk)]
        [TestCase("recule lentement", 0f, -0.015f)]
        [TestCase("sprint", 0f, 0.15f)]
        [TestCase("marche a droite", Walk, 0f)]
        public void ResolvesExpectedVector(string text, float vx, float vz)
        {
            var result = TextToControlResolver.Resolve(text, _table);
            Assert.That(result, Is.Not.Null, text);
            Assert.That(result.Value.Vx, Is.EqualTo(vx).Within(1e-6f));
            Assert.That(result.Value.Vz, Is.EqualTo(vz).Within(1e-6f));
        }

        [Test]
        public void ResolvesExpectedVector_RunForwardAndLeft_Diagonal()
        {
            var result = TextToControlResolver.Resolve("run forward and left", _table);
            Assert.That(result, Is.Not.Null);
            Assert.That(result.Value.Vx, Is.EqualTo(-Run * Diag).Within(1e-6f));
            Assert.That(result.Value.Vz, Is.EqualTo(Run * Diag).Within(1e-6f));
        }

        [Test]
        public void ResolvesExpectedVector_CoursLentement_MaxSpeedWins()
        {
            // "cours" (0.1) and "lentement" (0.015) both present: the max
            // speed keyword wins (text_to_control.md §2 rule 3), not "lentement".
            var result = TextToControlResolver.Resolve("cours lentement", _table);
            Assert.That(result, Is.Not.Null);
            Assert.That(result.Value.Vx, Is.EqualTo(0f).Within(1e-6f));
            Assert.That(result.Value.Vz, Is.EqualTo(Run).Within(1e-6f));
        }

        // -- test/apps/test_text_to_control.py::test_stop_always_wins_and_zeroes_control --

        [Test]
        public void StopAlwaysWinsAndZeroesControl()
        {
            var result = TextToControlResolver.Resolve("cours vite et arrete a gauche", _table);
            Assert.That(result, Is.Not.Null);
            Assert.That(result.Value.Vx, Is.EqualTo(0f).Within(1e-9f));
            Assert.That(result.Value.Vz, Is.EqualTo(0f).Within(1e-9f));
            Assert.That(result.Value.AimX, Is.EqualTo(0f).Within(1e-9f));
            Assert.That(result.Value.AimZ, Is.EqualTo(1f).Within(1e-9f));
        }

        // -- test/apps/test_text_to_control.py::test_aim_follows_movement_direction --

        [Test]
        public void AimFollowsMovementDirection()
        {
            var result = TextToControlResolver.Resolve("run left", _table);
            Assert.That(result, Is.Not.Null);
            Assert.That(result.Value.AimX, Is.EqualTo(-1f).Within(1e-9f));
            Assert.That(result.Value.AimZ, Is.EqualTo(0f).Within(1e-9f));
        }

        // -- test/apps/test_text_to_control.py::test_unresolvable_returns_none --

        [TestCase("bonjour tout le monde")]
        [TestCase("")]
        [TestCase("gauche droite")]
        public void UnresolvableReturnsNull(string text)
        {
            Assert.That(TextToControlResolver.Resolve(text, _table), Is.Null);
        }

        // -- test/apps/test_text_to_control.py::test_accents_are_stripped --

        [Test]
        public void AccentsAreStripped()
        {
            var result = TextToControlResolver.Resolve("arrête-toi", _table);
            Assert.That(result, Is.Not.Null);
            Assert.That(result.Value.Vx, Is.EqualTo(0f).Within(1e-9f));
            Assert.That(result.Value.Vz, Is.EqualTo(0f).Within(1e-9f));
        }

        // -- Additional Unity-side accent coverage (not in the Python suite,
        // exercising the replacement-table approach documented on the class) --

        [TestCase("À DROITE", Walk, 0f)]
        [TestCase("derrière", 0f, -Walk)]
        [TestCase("où es-tu", null, null)] // "où" -> "ou" strips to a non-keyword token; unresolved
        public void AccentedTokensMatchUnaccentedEquivalents(string text, float? vx, float? vz)
        {
            var result = TextToControlResolver.Resolve(text, _table);
            if (vx == null)
            {
                Assert.That(result, Is.Null);
                return;
            }

            Assert.That(result, Is.Not.Null);
            Assert.That(result.Value.Vx, Is.EqualTo(vx.Value).Within(1e-6f));
            Assert.That(result.Value.Vz, Is.EqualTo(vz.Value).Within(1e-6f));
        }

        [Test]
        public void EmbeddedTableMatchesCanonicalSpecCopy()
        {
            var packageInfo = UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                typeof(TextToControlResolverTests).Assembly);
            var packageRoot = packageInfo != null
                ? packageInfo.resolvedPath
                : Path.Combine("Packages", "com.ainimator.controller");

            var embeddedPath = Path.Combine(
                packageRoot, "Runtime", "TextCommand", "Resources", "text_to_control.json");

            // Walk up from the package root (apps/unity-sentis/com.ainimator.controller)
            // to the repo root, then down to apps/spec — mirrors how the
            // fixture bundle resolves the canonical bundle in BundleLoaderTests.
            var repoRoot = Path.GetFullPath(Path.Combine(packageRoot, "..", ".."));
            var canonicalPath = Path.Combine(repoRoot, "apps", "spec", "text_to_control.json");

            if (!File.Exists(canonicalPath))
            {
                Assert.Ignore($"Canonical spec copy not found at '{canonicalPath}' (package installed outside the monorepo checkout) — skipping verbatim-copy check.");
                return;
            }

            Assert.That(File.ReadAllText(embeddedPath), Is.EqualTo(File.ReadAllText(canonicalPath)),
                "Embedded Runtime/TextCommand/Resources/text_to_control.json has drifted from apps/spec/text_to_control.json (must stay a verbatim copy).");
        }
    }
}
