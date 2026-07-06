using AInimator.Controller.Bundle;
using UnityEngine;

namespace AInimator.Controller.Samples.ControllerDemo
{
    /// <summary>
    /// Minimal WASD-driven capsule demo for the B1 acceptance criteria
    /// (ROADMAP_PLUGINS.md §4, phase B1): loads the bundle from
    /// <see cref="BundlePaths.DefaultBundleDirectory"/>, drives a capsule
    /// with <see cref="CapsuleDemoController"/>, and does nothing else — no
    /// foot-lock IK (that is B4), no skinning (the demo only visualizes the
    /// integrated root trajectory of a bare capsule; wiring the full 22-bone
    /// pose onto a skinned rig is a follow-up sample, not required by B1's
    /// acceptance criteria: real-time fps + trajectory parity + no NaN over
    /// 60s).
    /// </summary>
    /// <remarks>
    /// Attach this component to an empty GameObject in an otherwise empty
    /// scene (or let it self-configure via <see cref="Awake"/>, which spawns
    /// a capsule primitive, a ground plane, and a simple follow camera if
    /// none are assigned in the Inspector). This avoids shipping a
    /// hand-authored binary/YAML <c>.unity</c> scene asset that cannot be
    /// validated without a running Unity Editor in this environment — the
    /// user drops this script into any empty scene and presses Play.
    /// </remarks>
    public sealed class CapsuleDemoBootstrap : MonoBehaviour
    {
        [Tooltip("Optional: assign an existing capsule Transform. Auto-created when left empty.")]
        [SerializeField] private Transform capsule;

        [Tooltip("Optional: override the bundle directory. Defaults to BundlePaths.DefaultBundleDirectory.")]
        [SerializeField] private string bundleDirectoryOverride;

        private CapsuleDemoController _demoController;

        private void Awake()
        {
            if (capsule == null)
            {
                capsule = CreateDefaultCapsule();
            }

            CreateGroundPlane();

            var bundleDirectory = string.IsNullOrEmpty(bundleDirectoryOverride)
                ? BundlePaths.DefaultBundleDirectory
                : bundleDirectoryOverride;

            var bundle = BundleLoader.Load(bundleDirectory);
            _demoController = capsule.gameObject.AddComponent<CapsuleDemoController>();
            _demoController.Initialize(bundle);
        }

        private static Transform CreateDefaultCapsule()
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            go.name = "AInimatorDemoCapsule";
            go.transform.position = new Vector3(0f, 1f, 0f);
            return go.transform;
        }

        private static void CreateGroundPlane()
        {
            var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
            ground.name = "Ground";
            ground.transform.localScale = new Vector3(5f, 1f, 5f);
        }
    }
}
