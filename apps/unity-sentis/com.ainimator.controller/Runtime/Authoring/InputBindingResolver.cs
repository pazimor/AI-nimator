using System.Collections.Generic;
using AInimator.Controller.Presets;
using UnityEngine;

namespace AInimator.Controller.Authoring
{
    /// <summary>
    /// Pure key→preset resolution logic extracted from
    /// <see cref="AInimatorActionBinder"/> so it is unit-testable without a
    /// running <c>Input</c> subsystem (EditMode tests cannot query
    /// <see cref="Input.GetKey"/>).
    /// </summary>
    /// <remarks>
    /// Resolution order is first-match-wins over the binding list, in
    /// declaration order — this mirrors the deterministic, no-surprise rule
    /// used elsewhere in this package (Goal B verite #4: never silently
    /// pick an arbitrary winner). Bindings missing a key or a preset
    /// (<see cref="InputBinding.IsValid"/> false) are skipped.
    /// </remarks>
    public static class InputBindingResolver
    {
        /// <summary>
        /// Return the first valid binding's preset whose key is currently
        /// held, per <paramref name="isKeyHeld"/>, or <paramref name="fallback"/>
        /// (typically the "idle" preset) when none are held.
        /// </summary>
        public static ControlPreset Resolve(
            IReadOnlyList<InputBinding> bindings,
            System.Func<KeyCode, bool> isKeyHeld,
            ControlPreset fallback)
        {
            if (bindings != null)
            {
                for (var i = 0; i < bindings.Count; i++)
                {
                    var binding = bindings[i];
                    if (binding == null || !binding.IsValid)
                    {
                        continue;
                    }

                    if (isKeyHeld(binding.Key))
                    {
                        return binding.Preset;
                    }
                }
            }

            return fallback;
        }

        /// <summary>
        /// Return the preset bound to <paramref name="key"/>, or <c>null</c>
        /// if no valid binding uses that key (first match wins if duplicated).
        /// </summary>
        public static ControlPreset FindPresetForKey(IReadOnlyList<InputBinding> bindings, KeyCode key)
        {
            if (bindings == null)
            {
                return null;
            }

            for (var i = 0; i < bindings.Count; i++)
            {
                var binding = bindings[i];
                if (binding != null && binding.IsValid && binding.Key == key)
                {
                    return binding.Preset;
                }
            }

            return null;
        }
    }
}
