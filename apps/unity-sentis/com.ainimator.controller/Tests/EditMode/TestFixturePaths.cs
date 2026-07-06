using System.IO;
using UnityEditor;
using UnityEditor.PackageManager;

namespace AInimator.Controller.Tests
{
    /// <summary>
    /// Resolves the on-disk path of the <c>Tests/Fixtures/Resources/reference_bundle</c>
    /// directory bundled with this package, regardless of whether the
    /// package is installed embedded, local, or via git URL/tarball
    /// (each resolves to a different physical path under <c>Packages/</c> /
    /// <c>Library/PackageCache/</c>).
    /// </summary>
    internal static class TestFixturePaths
    {
        public static string ReferenceBundleDirectory()
        {
            var packageInfo = UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                typeof(TestFixturePaths).Assembly);
            var packageRoot = packageInfo != null
                ? packageInfo.resolvedPath
                : Path.Combine("Packages", "com.ainimator.controller");

            return Path.Combine(packageRoot, "Tests", "Fixtures", "Resources", "reference_bundle");
        }
    }
}
