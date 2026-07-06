using System;
using System.Collections.Generic;
using System.Text;
using AInimator.Controller.Bundle;

namespace AInimator.Controller.TextEncoding
{
    /// <summary>
    /// Pure C# CLIP byte-level BPE tokenizer (Goal B phase B7,
    /// <c>apps/spec/text_encoding.md</c> §2): free text →
    /// fixed-length <c>input_ids</c> + <c>attention_mask</c> for the
    /// bundle's <c>text_encoder.onnx</c>.
    /// </summary>
    /// <remarks>
    /// Value-for-value mirror of <c>apps/spec/clip_bpe_reference.py</c>
    /// (itself locked to HuggingFace <c>CLIPTokenizerFast</c>); parity is
    /// verified by the EditMode suite against the shared
    /// <c>text_encoding_parity.json</c> cases. The vocabulary
    /// (<c>vocab.json</c>) and ranked merges (<c>merges.txt</c>) are the
    /// verbatim HuggingFace files shipped in the bundle's
    /// <c>tokenizer/</c> directory — never edited by hand.
    /// <para/>
    /// Normative simplification (spec §2.1): the "letter" class is
    /// <c>[a-z]</c> (after ASCII lowercasing) plus any codepoint above
    /// U+007F — identical to Unicode <c>\p{L}</c> on the FR/EN prompt
    /// domain; non-ASCII punctuation may split differently than HF but
    /// still tokenizes (byte-level BPE never fails).
    /// <para/>
    /// Pure and engine-free (no UnityEngine dependency) so it stays unit
    /// testable; construction parses ~49k vocab entries — build once per
    /// bundle, at load time, never per frame.
    /// </remarks>
    public sealed class ClipBpeTokenizer
    {
        private const string EndOfWordSuffix = "</w>";
        private const int NoMergeRank = int.MaxValue;

        // Contraction suffixes split off before word tokenization
        // (CLIP regex alternation order — checked before letter runs).
        private static readonly string[] Contractions =
            { "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" };

        private readonly Dictionary<string, int> _vocab;
        private readonly Dictionary<(string Left, string Right), int> _mergeRanks;
        private readonly string[] _byteEncoder;

        public int MaxLength { get; }
        public int BosId { get; }
        public int EosId { get; }
        public int PadId { get; }

        /// <param name="vocabJson">Verbatim <c>tokenizer/vocab.json</c> content.</param>
        /// <param name="mergesText">Verbatim <c>tokenizer/merges.txt</c> content.</param>
        /// <param name="maxLength">Fixed token budget (manifest <c>max_length</c>).</param>
        /// <param name="bosId">Begin-of-sequence id (manifest <c>bos_id</c>).</param>
        /// <param name="eosId">End-of-sequence id (manifest <c>eos_id</c>).</param>
        /// <param name="padId">Padding id (manifest <c>pad_id</c>, == eos for CLIP).</param>
        /// <exception cref="BundleLoadException">On malformed vocab/merges.</exception>
        public ClipBpeTokenizer(
            string vocabJson,
            string mergesText,
            int maxLength,
            int bosId,
            int eosId,
            int padId)
        {
            if (maxLength < 2)
            {
                throw new BundleLoadException(
                    $"tokenizer max_length must be >= 2 (bos + eos); got {maxLength}.");
            }

            MaxLength = maxLength;
            BosId = bosId;
            EosId = eosId;
            PadId = padId;
            _vocab = ParseVocab(vocabJson);
            _mergeRanks = ParseMerges(mergesText);
            _byteEncoder = BuildByteEncoder();
        }

        /// <summary>
        /// Encode <paramref name="text"/> into <paramref name="inputIds"/> /
        /// <paramref name="attentionMask"/>, both of length
        /// <see cref="MaxLength"/> (spec §2.5: truncate to max_length − 2,
        /// wrap in bos/eos, pad with the pad id, mask 1.0 on real tokens).
        /// </summary>
        public void Encode(string text, int[] inputIds, float[] attentionMask)
        {
            if (inputIds == null || inputIds.Length != MaxLength)
            {
                throw new ArgumentException($"inputIds must have length {MaxLength}.", nameof(inputIds));
            }

            if (attentionMask == null || attentionMask.Length != MaxLength)
            {
                throw new ArgumentException($"attentionMask must have length {MaxLength}.", nameof(attentionMask));
            }

            var tokens = new List<int>();
            foreach (var word in SplitWords(CleanText(text ?? string.Empty)))
            {
                foreach (var piece in ApplyBpe(word))
                {
                    if (!_vocab.TryGetValue(piece, out var id))
                    {
                        throw new BundleLoadException(
                            $"BPE piece '{piece}' missing from vocab.json — vocab/merges mismatch.");
                    }

                    tokens.Add(id);
                }
            }

            var bodyCount = Math.Min(tokens.Count, MaxLength - 2);
            inputIds[0] = BosId;
            for (var i = 0; i < bodyCount; i++)
            {
                inputIds[i + 1] = tokens[i];
            }

            inputIds[bodyCount + 1] = EosId;
            var realCount = bodyCount + 2;
            for (var i = realCount; i < MaxLength; i++)
            {
                inputIds[i] = PadId;
            }

            for (var i = 0; i < MaxLength; i++)
            {
                attentionMask[i] = i < realCount ? 1f : 0f;
            }
        }

        // ------------------------------------------------------------------
        // Text cleaning + pre-tokenization (spec §2.1 / §2.2)
        // ------------------------------------------------------------------
        private static bool IsAsciiSpace(char c) =>
            c is ' ' or '\t' or '\n' or '\r' or '\v' or '\f';

        private static bool IsLetter(char c) =>
            (c >= 'a' && c <= 'z') || c > '\x7f';

        private static bool IsDigit(char c) => c >= '0' && c <= '9';

        private static bool IsOther(char c) =>
            !IsAsciiSpace(c) && !IsLetter(c) && !IsDigit(c);

        private static string CleanText(string text)
        {
            var builder = new StringBuilder(text.Length);
            var pendingSpace = false;
            foreach (var raw in text)
            {
                var c = raw is >= 'A' and <= 'Z' ? (char)(raw + 32) : raw;
                if (IsAsciiSpace(c))
                {
                    pendingSpace = builder.Length > 0;
                    continue;
                }

                if (pendingSpace)
                {
                    builder.Append(' ');
                    pendingSpace = false;
                }

                builder.Append(c);
            }

            return builder.ToString();
        }

        private static string MatchContraction(string text, int index)
        {
            foreach (var contraction in Contractions)
            {
                if (index + contraction.Length <= text.Length &&
                    string.CompareOrdinal(text, index, contraction, 0, contraction.Length) == 0)
                {
                    return contraction;
                }
            }

            return null;
        }

        private static List<string> SplitWords(string text)
        {
            var words = new List<string>();
            var index = 0;
            while (index < text.Length)
            {
                var c = text[index];
                if (IsAsciiSpace(c))
                {
                    index++;
                    continue;
                }

                var contraction = MatchContraction(text, index);
                if (contraction != null)
                {
                    words.Add(contraction);
                    index += contraction.Length;
                }
                else if (IsLetter(c))
                {
                    index = ConsumeRun(text, index, words, IsLetter);
                }
                else if (IsDigit(c))
                {
                    words.Add(c.ToString());
                    index++;
                }
                else
                {
                    index = ConsumeRun(text, index, words, IsOther);
                }
            }

            return words;
        }

        private static int ConsumeRun(string text, int start, List<string> words, Func<char, bool> predicate)
        {
            var end = start;
            while (end < text.Length && predicate(text[end]))
            {
                // A contraction suffix terminates a run (spec §2.2 — the
                // CLIP regex alternation matches contractions first).
                if (end > start && MatchContraction(text, end) != null)
                {
                    break;
                }

                end++;
            }

            words.Add(text.Substring(start, end - start));
            return end;
        }

        // ------------------------------------------------------------------
        // Byte-level BPE (spec §2.3 / §2.4)
        // ------------------------------------------------------------------
        private List<string> ApplyBpe(string word)
        {
            var bytes = Encoding.UTF8.GetBytes(word);
            var parts = new List<string>(bytes.Length);
            for (var i = 0; i < bytes.Length; i++)
            {
                parts.Add(_byteEncoder[bytes[i]]);
            }

            if (parts.Count == 0)
            {
                return parts;
            }

            // CLIP specific: the final character carries </w> BEFORE merges.
            parts[parts.Count - 1] += EndOfWordSuffix;

            while (parts.Count > 1)
            {
                var bestRank = NoMergeRank;
                var bestIndex = -1;
                for (var i = 0; i < parts.Count - 1; i++)
                {
                    if (_mergeRanks.TryGetValue((parts[i], parts[i + 1]), out var rank) && rank < bestRank)
                    {
                        bestRank = rank;
                        bestIndex = i;
                    }
                }

                if (bestIndex < 0)
                {
                    break;
                }

                MergePair(parts, parts[bestIndex], parts[bestIndex + 1]);
            }

            return parts;
        }

        /// <summary>Merge every adjacent (left, right) occurrence, left to right.</summary>
        private static void MergePair(List<string> parts, string left, string right)
        {
            var write = 0;
            var read = 0;
            while (read < parts.Count)
            {
                if (read + 1 < parts.Count && parts[read] == left && parts[read + 1] == right)
                {
                    parts[write++] = left + right;
                    read += 2;
                }
                else
                {
                    parts[write++] = parts[read++];
                }
            }

            parts.RemoveRange(write, parts.Count - write);
        }

        // ------------------------------------------------------------------
        // Vocabulary loading
        // ------------------------------------------------------------------
        private static Dictionary<string, int> ParseVocab(string vocabJson)
        {
            if (MiniJson.Parse(vocabJson) is not Dictionary<string, object> root || root.Count == 0)
            {
                throw new BundleLoadException("tokenizer vocab.json: root is not a non-empty JSON object.");
            }

            var vocab = new Dictionary<string, int>(root.Count);
            foreach (var entry in root)
            {
                if (entry.Value is not double id)
                {
                    throw new BundleLoadException(
                        $"tokenizer vocab.json: entry '{entry.Key}' has a non-numeric id.");
                }

                vocab[entry.Key] = (int)id;
            }

            return vocab;
        }

        private static Dictionary<(string, string), int> ParseMerges(string mergesText)
        {
            var ranks = new Dictionary<(string, string), int>();
            foreach (var rawLine in mergesText.Split('\n'))
            {
                var line = rawLine.TrimEnd('\r');
                if (line.Length == 0 || line.StartsWith("#", StringComparison.Ordinal))
                {
                    continue;
                }

                var space = line.IndexOf(' ');
                if (space <= 0 || space != line.LastIndexOf(' '))
                {
                    throw new BundleLoadException(
                        $"tokenizer merges.txt: malformed merge line '{line}'.");
                }

                ranks[(line.Substring(0, space), line.Substring(space + 1))] = ranks.Count;
            }

            if (ranks.Count == 0)
            {
                throw new BundleLoadException("tokenizer merges.txt: no merge rules found.");
            }

            return ranks;
        }

        /// <summary>
        /// GPT-2/CLIP byte → printable-unicode bijection (spec §2.3):
        /// printable Latin-1 bytes map to themselves, the remaining 68
        /// bytes map to U+0100.. so every byte has a dict-safe character.
        /// </summary>
        private static string[] BuildByteEncoder()
        {
            var encoder = new string[256];
            var offset = 0;
            for (var b = 0; b < 256; b++)
            {
                var printable =
                    (b >= '!' && b <= '~') ||
                    (b >= 0xA1 && b <= 0xAC) ||
                    (b >= 0xAE && b <= 0xFF);
                encoder[b] = printable
                    ? ((char)b).ToString()
                    : ((char)(256 + offset++)).ToString();
            }

            return encoder;
        }
    }
}
