using AInimator.Controller.Authoring;
using AInimator.Controller.Bundle;
using AInimator.Controller.Presets;
using AInimator.Controller.Runtime;
using UnityEngine;

namespace AInimator.Controller.Samples.ControllerDemo
{
    /// <summary>
    /// WASD-driven demo: maps keys to <see cref="ControlPreset"/> assets
    /// (forward/backward/strafe/idle, per ROADMAP_PLUGINS.md §1.1's binding
    /// table) and applies the controller's integrated root trajectory to
    /// this <see cref="Transform"/> every frame. No IK, no skinning — B1
    /// scope only (post-processing is B4).
    /// </summary>
    public sealed class CapsuleDemoController : MonoBehaviour
    {
        [Tooltip("Forward-facing bind-pose bone frame used to seed the state window (rotation6d, row-major, numBones*6). Leave empty for the canonical identity rest-pose seed (parity with Unreal).")]
        [SerializeField] private float[] seedBoneFrameOverride;

        private AInimatorController _controller;
        private ControlPreset _idle;
        private ControlPreset _forward;
        private ControlPreset _backward;
        private ControlPreset _strafeLeft;
        private ControlPreset _strafeRight;

        /// <summary>Wire up the controller from an already-loaded bundle.</summary>
        public void Initialize(ControllerBundle bundle)
        {
            var seed = seedBoneFrameOverride is { Length: > 0 }
                ? seedBoneFrameOverride
                : AInimatorController.CreateRestPoseSeed(bundle.Manifest);

            _controller = new AInimatorController(bundle, seed);

            _idle = LoadOrFallback(bundle, "idle");
            _forward = LoadOrFallback(bundle, "forward");
            _backward = LoadOrFallback(bundle, "backward");
            _strafeLeft = LoadOrFallback(bundle, "strafe_left");
            _strafeRight = LoadOrFallback(bundle, "strafe_right");
        }

        private void Update()
        {
            if (_controller == null)
            {
                return;
            }

            var preset = SelectPresetFromInput();
            _controller.Tick(preset);

            transform.position = _controller.RootPosition;
            transform.rotation = Quaternion.Euler(0f, _controller.RootYawRadians * Mathf.Rad2Deg, 0f);
        }

        private ControlPreset SelectPresetFromInput()
        {
            if (KeyInput.GetKey(KeyCode.W) || KeyInput.GetKey(KeyCode.UpArrow))
            {
                return _forward;
            }

            if (KeyInput.GetKey(KeyCode.S) || KeyInput.GetKey(KeyCode.DownArrow))
            {
                return _backward;
            }

            if (KeyInput.GetKey(KeyCode.Q))
            {
                return _strafeLeft;
            }

            if (KeyInput.GetKey(KeyCode.E))
            {
                return _strafeRight;
            }

            return _idle;
        }

        private static ControlPreset LoadOrFallback(ControllerBundle bundle, string name)
        {
            var preset = ScriptableObject.CreateInstance<ControlPreset>();
            if (bundle.Presets.TryGetValue(name, out var data))
            {
                preset.HydrateFrom(data);
            }
            else
            {
                Debug.LogWarning(
                    $"AInimator: bundle has no preset named '{name}'; falling back to zero control.");
            }

            return preset;
        }

        private void OnDestroy()
        {
            _controller?.Dispose();
        }
    }
}
