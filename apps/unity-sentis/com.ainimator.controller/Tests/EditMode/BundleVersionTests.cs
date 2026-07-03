using AInimator.Controller.Bundle;
using NUnit.Framework;

namespace AInimator.Controller.Tests
{
    public class BundleVersionTests
    {
        [TestCase("A7.0", "A7.0", true)]
        [TestCase("A7.0", "A7.3", true)]
        [TestCase("A7.0", "B7.0", false)]
        [TestCase("A7.0", "A8.0", false)]
        [TestCase("7.0", "7.5", true)]
        public void IsCompatibleWith_ChecksPrefixAndMajorOnly(string a, string b, bool expected)
        {
            var versionA = BundleVersion.Parse(a);
            var versionB = BundleVersion.Parse(b);

            Assert.That(versionA.IsCompatibleWith(versionB), Is.EqualTo(expected));
        }

        [TestCase("")]
        [TestCase("not-a-version")]
        [TestCase("A7")]
        [TestCase("A.0")]
        public void Parse_InvalidFormat_ThrowsBundleLoadException(string raw)
        {
            Assert.Throws<BundleLoadException>(() => BundleVersion.Parse(raw));
        }
    }
}
