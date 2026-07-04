using System;
using AInimator.Controller.Presets;
using UnityEngine;

namespace AInimator.Controller.Authoring
{
    /// <summary>
    /// One authoring-time row of the "button → action" table (ROADMAP_PLUGINS.md
    /// §1.1 / §4 B3): a single <see cref="KeyCode"/> mapped to a named
    /// <see cref="ControlPreset"/> asset.
    /// </summary>
    /// <remarks>
    /// <see cref="KeyCode"/> is used (rather than Unity's newer Input System
    /// actions) to keep the runtime package free of an <c>com.unity.inputsystem</c>
    /// dependency — the same tradeoff <c>CapsuleDemoController</c> already
    /// makes in B1. A project that has the new Input System installed can
    /// still drive <see cref="AInimatorActionBinder.ActivePreset"/>/
    /// <see cref="AInimatorActionBinder.SetContinuousAim"/> directly from its
    /// own input callbacks without using this binding list at all — the
    /// binder's public API does not require <see cref="KeyCode"/> to be used.
    /// </remarks>
    [Serializable]
    public sealed class InputBinding
    {
        [Tooltip("Key that activates this binding while held.")]
        [SerializeField] private KeyCode key = KeyCode.None;

        [Tooltip("ControlPreset applied to the controller while the key is held.")]
        [SerializeField] private ControlPreset preset;

        [Tooltip("Optional human-readable label shown in the Inspector list (purely cosmetic).")]
        [SerializeField] private string label = "";

        public KeyCode Key => key;
        public ControlPreset Preset => preset;
        public string Label => label;

        public InputBinding()
        {
        }

        public InputBinding(KeyCode key, ControlPreset preset, string label = "")
        {
            this.key = key;
            this.preset = preset;
            this.label = label;
        }

        /// <summary>True once both a key and a preset are assigned.</summary>
        public bool IsValid => key != KeyCode.None && preset != null;
    }
}
