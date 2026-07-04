using System;
using System.Collections.Generic;
using AInimator.Controller.Bundle;
using AInimator.Controller.Presets;
using AInimator.Controller.Runtime;
using UnityEngine;

namespace AInimator.Controller.Authoring
{
    /// <summary>
    /// The "button → action + animation" authoring component (ROADMAP_PLUGINS.md
    /// §1.1 / §4 phase B3): a list of <see cref="InputBinding"/> rows
    /// (<see cref="KeyCode"/> → <see cref="ControlPreset"/>), resolved every
    /// frame into the active preset applied to an owned
    /// <see cref="AInimatorController"/>.
    /// </summary>
    /// <remarks>
    /// This is the "no-code" entry point for a game developer: create a
    /// <see cref="ControlPreset"/> asset, drop this component on a
    /// GameObject, add a binding row in the custom Inspector
    /// (<c>Editor/AInimatorActionBinderEditor.cs</c>), press Play. No
    /// changes to any C# file are required to add/remove a binding.
    /// <para/>
    /// <see cref="ActivePreset"/> exposes the currently resolved preset so a
    /// separate renderer/animator component can react to it without owning
    /// input logic itself. The component owns the <see cref="AInimatorController"/>
    /// lifecycle (construction from a bundle, per-frame <c>Tick</c>,
    /// disposal) so a scene only needs this one MonoBehaviour to go from
    /// "bundle on disk" to "character responds to input".
    /// </remarks>
    public sealed class AInimatorActionBinder : MonoBehaviour
    {
        [Tooltip("Key -> ControlPreset rows, evaluated top-to-bottom; first held key wins.")]
        [SerializeField] private List<InputBinding> bindings = new();

        [Tooltip("Preset applied when no bound key is held (typically 'idle').")]
        [SerializeField] private ControlPreset idlePreset;

        [Tooltip("Optional: override the bundle directory. Defaults to BundlePaths.DefaultBundleDirectory.")]
        [SerializeField] private string bundleDirectoryOverride;

        [Tooltip("Rest-pose bone frame (rotation6d, row-major, numBones*6) used to seed the state window. Leave empty for a zero-rotation placeholder seed.")]
        [SerializeField] private float[] seedBoneFrameOverride;

        [Tooltip("If true, load the bundle and construct the controller in Awake(). Set false to drive Initialize(bundle) manually (e.g. from a bootstrap script).")]
        [SerializeField] private bool autoInitializeFromDefaultBundle = true;

        private AInimatorController _controller;

        /// <summary>The bindings list, editable from a custom Inspector (Editor/).</summary>
        public List<InputBinding> Bindings => bindings;

        /// <summary>Preset applied when no bound key is held.</summary>
        public ControlPreset IdlePreset
        {
            get => idlePreset;
            set => idlePreset = value;
        }

        /// <summary>The preset resolved on the most recent <see cref="Update"/>.</summary>
        public ControlPreset ActivePreset { get; private set; }

        /// <summary>The controller this binder drives, once initialized.</summary>
        public AInimatorController Controller => _controller;

        private Vector2 _continuousAim;

        private void Awake()
        {
            if (!autoInitializeFromDefaultBundle)
            {
                return;
            }

            var bundleDirectory = string.IsNullOrEmpty(bundleDirectoryOverride)
                ? BundlePaths.DefaultBundleDirectory
                : bundleDirectoryOverride;

            var bundle = BundleLoader.Load(bundleDirectory);
            Initialize(bundle);
        }

        /// <summary>
        /// Construct the owned <see cref="AInimatorController"/> from an
        /// already-loaded bundle. Call this instead of relying on
        /// <see cref="autoInitializeFromDefaultBundle"/> when a bootstrap
        /// script owns bundle loading.
        /// </summary>
        public void Initialize(ControllerBundle bundle)
        {
            if (bundle == null)
            {
                throw new ArgumentNullException(nameof(bundle));
            }

            _controller?.Dispose();

            var seed = seedBoneFrameOverride is { Length: > 0 }
                ? seedBoneFrameOverride
                : new float[bundle.Manifest.num_bones * bundle.Manifest.rotation_channels_per_bone];

            _controller = new AInimatorController(bundle, seed);
        }

        /// <summary>
        /// Continuous aim override (e.g. mouse delta / stick), applied on
        /// top of the resolved preset's static aim when the manifest
        /// declares <c>control_channels == 4</c>. Pass <c>(0, 0)</c> to defer
        /// entirely to the preset's aim.
        /// </summary>
        public void SetContinuousAim(Vector2 aim)
        {
            _continuousAim = aim;
        }

        private void Update()
        {
            if (_controller == null)
            {
                return;
            }

            ActivePreset = ResolveActivePreset();
            if (ActivePreset == null)
            {
                return;
            }

            _controller.Tick(ActivePreset, _continuousAim);
        }

        /// <summary>
        /// Resolve the preset to apply this frame from <see cref="bindings"/>
        /// (first-held-key-wins) falling back to <see cref="idlePreset"/>.
        /// Exposed as a separate method so it is callable without a live
        /// <c>Input</c> subsystem in tests via <see cref="InputBindingResolver.Resolve"/>.
        /// </summary>
        private ControlPreset ResolveActivePreset()
        {
            return InputBindingResolver.Resolve(bindings, Input.GetKey, idlePreset);
        }

        private void OnDestroy()
        {
            _controller?.Dispose();
            _controller = null;
        }
    }
}
