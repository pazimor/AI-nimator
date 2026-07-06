using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using AInimator.Controller.Bundle;
using AInimator.Controller.TextEncoding;

namespace AInimator.Controller.Tests
{
    /// <summary>
    /// Parity tests for <see cref="ClipBpeTokenizer"/> (Goal B phase B7).
    /// The canonical cases live in the verbatim fixture copy of
    /// <c>apps/spec/text_encoding_parity.json</c> — same prompts, same ids
    /// as the Python reference (<c>clip_bpe_reference.py</c>) and the
    /// Unreal Automation suite (<c>text_encoding.md</c> §4).
    /// </summary>
    /// <remarks>
    /// The vocabulary (<c>vocab.json</c> / <c>merges.txt</c>, ~1.5 MB) is a
    /// build artifact delivered inside the bundle — never committed. The
    /// suite loads it from the delivered
    /// <c>StreamingAssets/AInimatorBundle/tokenizer/</c> and self-ignores
    /// when the delivered bundle predates A7.1 (no tokenizer directory):
    /// run <c>make plugin-unity … ENCODER_ARTIFACT=…</c> first.
    /// </remarks>
    public class ClipBpeTokenizerTests
    {
        [Serializable]
        private sealed class ParityFile
        {
            public int max_length;
            public int bos_id;
            public int eos_id;
            public int pad_id;
            public List<ParityCase> cases;
        }

        [Serializable]
        private sealed class ParityCase
        {
            public string prompt;
            public int[] input_ids;
            public int[] attention_mask;
        }

        private static ClipBpeTokenizer _tokenizer;
        private static ParityFile _parity;

        private static string PackageRoot()
        {
            var packageInfo = UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                typeof(ClipBpeTokenizerTests).Assembly);
            return packageInfo != null
                ? packageInfo.resolvedPath
                : Path.Combine("Packages", "com.ainimator.controller");
        }

        [OneTimeSetUp]
        public void LoadTokenizer()
        {
            var parityPath = Path.Combine(
                PackageRoot(), "Tests", "Fixtures", "Resources", "text_encoding_parity.json");
            _parity = UnityEngine.JsonUtility.FromJson<ParityFile>(File.ReadAllText(parityPath));
            Assert.That(_parity.cases, Is.Not.Null.And.Not.Empty, "parity fixture has no cases");

            var tokenizerDir = Path.Combine(
                PackageRoot(), "StreamingAssets", "AInimatorBundle", "tokenizer");
            var vocabPath = Path.Combine(tokenizerDir, "vocab.json");
            var mergesPath = Path.Combine(tokenizerDir, "merges.txt");
            if (!File.Exists(vocabPath) || !File.Exists(mergesPath))
            {
                Assert.Ignore(
                    "Delivered bundle ships no tokenizer/ (A7.0 bundle) — deliver an A7.1 bundle " +
                    "(make plugin-unity … ENCODER_ARTIFACT=output/clip_text_artifact) to run this suite.");
            }

            _tokenizer = new ClipBpeTokenizer(
                File.ReadAllText(vocabPath),
                File.ReadAllText(mergesPath),
                _parity.max_length,
                _parity.bos_id,
                _parity.eos_id,
                _parity.pad_id);
        }

        [Test]
        public void AllCanonicalCasesMatchReferenceIds()
        {
            var inputIds = new int[_parity.max_length];
            var attentionMask = new float[_parity.max_length];
            foreach (var parityCase in _parity.cases)
            {
                _tokenizer.Encode(parityCase.prompt, inputIds, attentionMask);
                Assert.That(inputIds, Is.EqualTo(parityCase.input_ids),
                    $"input_ids mismatch for prompt '{parityCase.prompt}'");
                for (var i = 0; i < _parity.max_length; i++)
                {
                    Assert.That((int)attentionMask[i], Is.EqualTo(parityCase.attention_mask[i]),
                        $"attention_mask[{i}] mismatch for prompt '{parityCase.prompt}'");
                }
            }
        }

        [Test]
        public void EmptyPromptEncodesBosEosOnly()
        {
            var inputIds = new int[_parity.max_length];
            var attentionMask = new float[_parity.max_length];
            _tokenizer.Encode(string.Empty, inputIds, attentionMask);
            Assert.That(inputIds[0], Is.EqualTo(_parity.bos_id));
            Assert.That(inputIds[1], Is.EqualTo(_parity.eos_id));
            Assert.That(attentionMask[0], Is.EqualTo(1f));
            Assert.That(attentionMask[1], Is.EqualTo(1f));
            Assert.That(attentionMask[2], Is.EqualTo(0f));
        }

        [Test]
        public void WrongBufferLengthThrows()
        {
            Assert.Throws<ArgumentException>(() => _tokenizer.Encode(
                "a person dances", new int[_parity.max_length - 1], new float[_parity.max_length]));
            Assert.Throws<ArgumentException>(() => _tokenizer.Encode(
                "a person dances", new int[_parity.max_length], new float[_parity.max_length + 1]));
        }

        [Test]
        public void MalformedVocabThrowsBundleLoadException()
        {
            Assert.Throws<BundleLoadException>(() => new ClipBpeTokenizer(
                "[]", "a b\n", 32, 49406, 49407, 49407));
            Assert.Throws<BundleLoadException>(() => new ClipBpeTokenizer(
                "{\"a\": 1}", "", 32, 49406, 49407, 49407));
        }
    }
}
