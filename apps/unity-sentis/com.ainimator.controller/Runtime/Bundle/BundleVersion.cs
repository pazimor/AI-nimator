using System;
using System.Text.RegularExpressions;

namespace AInimator.Controller.Bundle
{
    /// <summary>
    /// Parses and compares <c>bundle_version</c> strings (e.g. <c>"A7.0"</c>),
    /// matching <c>manifest.schema.json</c>'s pattern
    /// <c>^[A-Z]?[0-9]+\.[0-9]+$</c>.
    /// </summary>
    /// <remarks>
    /// Compatibility policy (Goal B verite #4 — fail-fast on a contract
    /// bump): two bundle versions are compatible with each other when they
    /// share the same optional letter prefix and the same major number.
    /// The minor number may differ (additive, non-breaking changes bump the
    /// minor only). A change to the letter prefix or the major number means
    /// the I/O layout changed and must be rejected loudly.
    /// </remarks>
    public readonly struct BundleVersion : IEquatable<BundleVersion>
    {
        private static readonly Regex Pattern = new(@"^([A-Z]?)([0-9]+)\.([0-9]+)$", RegexOptions.Compiled);

        public string Prefix { get; }
        public int Major { get; }
        public int Minor { get; }
        public string Raw { get; }

        private BundleVersion(string prefix, int major, int minor, string raw)
        {
            Prefix = prefix;
            Major = major;
            Minor = minor;
            Raw = raw;
        }

        /// <summary>
        /// Parse a version string, throwing <see cref="BundleLoadException"/>
        /// on any format that does not match the manifest schema pattern.
        /// </summary>
        public static BundleVersion Parse(string raw)
        {
            if (string.IsNullOrEmpty(raw))
            {
                throw new BundleLoadException("manifest.json: bundle_version is missing or empty.");
            }

            var match = Pattern.Match(raw);
            if (!match.Success)
            {
                throw new BundleLoadException(
                    $"manifest.json: bundle_version '{raw}' does not match the expected pattern " +
                    "'^[A-Z]?[0-9]+\\.[0-9]+$'.");
            }

            var prefix = match.Groups[1].Value;
            var major = int.Parse(match.Groups[2].Value);
            var minor = int.Parse(match.Groups[3].Value);
            return new BundleVersion(prefix, major, minor, raw);
        }

        /// <summary>
        /// True when <paramref name="other"/> is loadable by a plugin built
        /// against this version (same prefix + major; minor may differ).
        /// </summary>
        public bool IsCompatibleWith(BundleVersion other)
        {
            return Prefix == other.Prefix && Major == other.Major;
        }

        public bool Equals(BundleVersion other) => Raw == other.Raw;

        public override bool Equals(object obj) => obj is BundleVersion other && Equals(other);

        public override int GetHashCode() => Raw.GetHashCode();

        public override string ToString() => Raw;
    }
}
