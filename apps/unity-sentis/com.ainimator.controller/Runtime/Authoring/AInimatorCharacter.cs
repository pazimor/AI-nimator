using System;
using System.Collections.Generic;
using AInimator.Controller.Bundle;
using AInimator.Controller.PostProcess;
using AInimator.Controller.Presets;
using AInimator.Controller.Rig;
using AInimator.Controller.Runtime;
using AInimator.Controller.TextCommand;
using UnityEngine;

namespace AInimator.Controller.Authoring
{
    /// <summary>
    /// "Hat" component (<c>apps/spec/rig_binding.md</c>): wires one
    /// <see cref="AInimatorController"/> (B1/B2) through the B4
    /// <see cref="AInimatorPostProcess"/> stage and, when a <see cref="RigMap"/>
    /// is assigned, the B3-bis <see cref="Rig.RigBinder"/> retargeting stage
    /// — the full per-frame chain of §1: controller -&gt; post-process -&gt;
    /// rig-binder -&gt; actor transform.
    /// </summary>
    /// <remarks>
    /// This is the authoring entry point for "attach to a rigged character
    /// and drive it": drop this on the character's root GameObject alongside
    /// a <see cref="Rig.RigBinder"/> (optional — omit it to keep driving a
    /// capsule/no-rig proxy exactly like <see cref="AInimatorActionBinder"/>
    /// already does; that B1 code path is untouched and still works with no
    /// <see cref="RigMap"/> assigned at all, per spec §5 last line).
    /// <para/>
    /// <see cref="SetPrompt"/>/<see cref="SetPromptEmbedding"/>/
    /// <see cref="ClearPrompt"/> forward directly to the owned
    /// <see cref="AInimatorController"/> (spec §3) — this component adds no
    /// prompt logic of its own, only wiring.
    /// </remarks>
    public sealed class AInimatorCharacter : MonoBehaviour
    {
        [Tooltip("Key -> ControlPreset rows, evaluated top-to-bottom; first held key wins.")]
        [SerializeField] private List<InputBinding> bindings = new();

        [Tooltip("Preset applied when no bound key is held (typically 'idle').")]
        [SerializeField] private ControlPreset idlePreset;

        [Tooltip("Optional: override the bundle directory. Defaults to BundlePaths.DefaultBundleDirectory.")]
        [SerializeField] private string bundleDirectoryOverride;

        [Tooltip("Rest-pose bone frame (rotation6d, row-major, numBones*6) used to seed the state window. Leave empty for a zero-rotation placeholder seed.")]
        [SerializeField] private float[] seedBoneFrameOverride;

        [Tooltip("If true, load the bundle and construct the controller in Awake(). Set false to drive Initialize(bundle) manually.")]
        [SerializeField] private bool autoInitializeFromDefaultBundle = true;

        [Tooltip("Optional: foot-lock IK + idle/move blend post-processing (B4). Leave unassigned to skip post-processing (raw controller pose only).")]
        [SerializeField] private bool enablePostProcess = true;

        [Tooltip("Optional: retarget the corrected pose onto a rigged character (B3-bis). Leave unassigned to drive only the root transform (B1 capsule path).")]
        [SerializeField] private RigBinder rigBinder;

        private AInimatorController _controller;
        private AInimatorPostProcess _postProcess;
        private Vector2 _continuousAim;
        private ControlPreset _textCommandPreset;
        private bool _hasActiveTextCommand;

        /// <summary>
        /// The free-text command most recently accepted by
        /// <see cref="SetTextCommand"/>, or <c>null</c> once
        /// <see cref="ClearTextCommand"/> is called / a key binding takes
        /// over. Exposed for the Inspector test UX.
        /// </summary>
        public string ActiveTextCommand { get; private set; }

        /// <summary>The bindings list, editable from a custom Inspector.</summary>
        public List<InputBinding> Bindings => bindings;

        /// <summary>Preset applied when no bound key is held.</summary>
        public ControlPreset IdlePreset
        {
            get => idlePreset;
            set => idlePreset = value;
        }

        /// <summary>The preset resolved on the most recent <see cref="Update"/>.</summary>
        public ControlPreset ActivePreset { get; private set; }

        /// <summary>The controller this character drives, once initialized.</summary>
        public AInimatorController Controller => _controller;

        /// <summary>The optional rig-binder stage (null = no rig retargeting, root-transform-only).</summary>
        public RigBinder RigBinder
        {
            get => rigBinder;
            set => rigBinder = value;
        }

        /// <summary>
        /// Corrected left/right leg joint positions + idle blend weight from
        /// the most recent <see cref="AInimatorPostProcess.Process"/> call,
        /// for a renderer that drives a skinned rig's foot-lock IK directly
        /// (see remarks above <see cref="Update"/>). Default when
        /// post-processing is disabled.
        /// </summary>
        public (AInimatorPostProcess.LegResult left, AInimatorPostProcess.LegResult right) LastPostProcessResult { get; private set; }

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

        /// <summary>Construct the owned <see cref="AInimatorController"/> (+ post-process) from an already-loaded bundle.</summary>
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
            _postProcess = enablePostProcess ? new AInimatorPostProcess(bundle.Manifest) : null;

            if (rigBinder != null && !rigBinder.IsCalibrated)
            {
                rigBinder.Calibrate();
            }
        }

        /// <summary>Continuous aim override (mouse/stick), see <see cref="AInimatorActionBinder.SetContinuousAim"/>.</summary>
        public void SetContinuousAim(Vector2 aim)
        {
            _continuousAim = aim;
        }

        /// <summary>Begin cross-fading toward <paramref name="preset"/>'s precomputed prompt embedding (spec §3).</summary>
        public void SetPrompt(ControlPreset preset) => _controller?.SetPrompt(preset);

        /// <summary>Begin cross-fading toward a raw prompt embedding supplied by the game (spec §3).</summary>
        public void SetPromptEmbedding(ReadOnlySpan<float> embedding) => _controller?.SetPromptEmbedding(embedding);

        /// <summary>Begin cross-fading back to the bundle's learned null prompt embedding (spec §3).</summary>
        public void ClearPrompt() => _controller?.ClearPrompt();

        /// <summary>
        /// Resolve a free-text command (Goal B phase B6,
        /// <c>apps/spec/text_to_control.md</c>) and, if successful, drive the
        /// controller with it every frame instead of the key bindings — the
        /// SAME control-vector write path as a <see cref="ControlPreset"/>
        /// (<see cref="ControlPreset.WriteRawControl"/>), zero impact on the
        /// preset path. A held key binding still takes priority over an
        /// active text command; call <see cref="ClearTextCommand"/> to
        /// release control back to bindings/idle.
        /// </summary>
        /// <param name="text">Free-text command, French or English.</param>
        /// <returns>
        /// <c>true</c> if resolved and now active; <c>false</c> if the text
        /// could not be resolved (ambiguous or unrecognized — the previous
        /// active command/preset is left untouched, per spec §2 rule 6),
        /// logged via <see cref="Debug.LogWarning(object)"/>, never a silent
        /// fallback.
        /// </returns>
        public bool SetTextCommand(string text)
        {
            var resolved = TextToControlResolver.Resolve(text);
            if (resolved == null)
            {
                Debug.LogWarning($"AInimatorCharacter.SetTextCommand: could not resolve '{text}' (unrecognized or ambiguous) — control unchanged.");
                return false;
            }

            _textCommandPreset ??= ScriptableObject.CreateInstance<ControlPreset>();
            _textCommandPreset.SetRawControl(resolved.Value.Vx, resolved.Value.Vz, resolved.Value.AimX, resolved.Value.AimZ);
            _hasActiveTextCommand = true;
            ActiveTextCommand = text;
            return true;
        }

        /// <summary>Release the active text command; bindings/idle resolve again next frame.</summary>
        public void ClearTextCommand()
        {
            _hasActiveTextCommand = false;
            ActiveTextCommand = null;
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

            var deltaTime = Time.deltaTime;
            var rawBoneFrame = _controller.Tick(ActivePreset, _continuousAim, deltaTime);

            // Foot-lock IK / idle-move blending (B4) correct *joint
            // positions* (footlock_blending.md §1); they do not rewrite the
            // rotation6d frame, so the RigBinder retargeting stage below
            // still consumes the raw controller bone frame (its own FK pass
            // recomputes world rotations from it). The corrected leg
            // positions / idle weight remain available via
            // LastPostProcessResult for a caller that also drives a
            // skinned rig's IK effectors directly from those positions.
            var correctedBoneFrame = rawBoneFrame;
            if (_postProcess != null)
            {
                Span<float> rawControl = stackalloc float[2];
                ActivePreset.WriteRawControl(rawControl, false);
                LastPostProcessResult = _postProcess.Process(
                    rawBoneFrame, _controller.RootPosition, rawControl[0], rawControl[1], deltaTime);
            }

            if (rigBinder != null)
            {
                // RigBinder owns root-motion integration for its actor root
                // (rigScale-corrected, spec §2.3) from this frame's *raw*
                // global delta -- it does not read the controller's own
                // (unscaled) integrated RootPosition/RootYawRadians.
                rigBinder.Apply(correctedBoneFrame, _controller.LastRawGlobalDelta);
            }
            else
            {
                // No RigMap assigned: fall back to the B1 capsule path
                // exactly (spec §5 -- the capsule path remains functional
                // with no RigBinder at all).
                transform.position = _controller.RootPosition;
                transform.rotation = Quaternion.Euler(0f, _controller.RootYawRadians * Mathf.Rad2Deg, 0f);
            }
        }

        private ControlPreset ResolveActivePreset()
        {
            var boundPreset = InputBindingResolver.Resolve(bindings, Input.GetKey, null);
            if (boundPreset != null)
            {
                return boundPreset;
            }

            if (_hasActiveTextCommand)
            {
                return _textCommandPreset;
            }

            return idlePreset;
        }

        private void OnDestroy()
        {
            _controller?.Dispose();
            _controller = null;

            if (_textCommandPreset != null)
            {
                Destroy(_textCommandPreset);
                _textCommandPreset = null;
            }
        }
    }
}
