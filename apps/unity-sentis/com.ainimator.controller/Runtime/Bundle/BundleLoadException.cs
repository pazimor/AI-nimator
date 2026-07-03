using System;

namespace AInimator.Controller.Bundle
{
    /// <summary>
    /// Raised when a controller bundle fails to load or validate. Always
    /// thrown loudly (Goal B verite #4: fail-fast on contract mismatch,
    /// never silent) — never caught and swallowed by runtime code.
    /// </summary>
    public sealed class BundleLoadException : Exception
    {
        public BundleLoadException(string message) : base(message)
        {
        }

        public BundleLoadException(string message, Exception inner) : base(message, inner)
        {
        }
    }
}
