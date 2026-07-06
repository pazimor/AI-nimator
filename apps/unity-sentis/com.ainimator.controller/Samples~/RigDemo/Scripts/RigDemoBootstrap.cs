using System.Collections.Generic;
using System.Linq;
using AInimator.Controller.Authoring;
using AInimator.Controller.PostProcess;
using AInimator.Controller.Presets;
using AInimator.Controller.Rig;
using UnityEngine;

namespace AInimator.Controller.Samples.RigDemo
{
    /// <summary>
    /// Drag-and-drop demo for B3-bis (<c>apps/spec/rig_binding.md</c>):
    /// builds a procedural SMPL-22 stick-figure mannequin (22 bone
    /// Transforms at the canonical rest pose + primitive visuals), wires it
    /// to an <see cref="AInimatorCharacter"/> through a <see cref="RigBinder"/>
    /// with a 1:1 <see cref="RigMap"/>, and adds WASD/ZQSD/arrow key
    /// bindings plus a <see cref="RigDemoHud"/> for hot prompt switching.
    /// </summary>
    /// <remarks>
    /// Everything is generated in <see cref="Awake"/> rather than authored
    /// as a binary FBX/scene asset, for the same reason as
    /// <c>CapsuleDemoBootstrap</c>: assets hand-written outside a running
    /// Unity Editor cannot be validated here. Drop the <c>RigDemo</c> prefab
    /// (or this component on an empty GameObject) into an empty scene and
    /// press Play. Requires the controller bundle under
    /// <c>Assets/StreamingAssets/AInimatorBundle/</c> (see the sample
    /// README) unless <see cref="bundleDirectoryOverride"/> is set.
    /// </remarks>
    public sealed class RigDemoBootstrap : MonoBehaviour
    {
        [Tooltip("Optional: your own rigged character instance (Mixamo/Humanoid FBX dropped in the scene, in its rest pose). Bones are auto-mapped by name at Play. Empty = a procedural SMPL-22 stick-figure mannequin is built instead.")]
        [SerializeField] private Transform customRigRoot;

        [Tooltip("Optional: absolute path to a bundle directory. Empty = StreamingAssets/AInimatorBundle.")]
        [SerializeField] private string bundleDirectoryOverride;

        private const float JointRadius = 0.05f;
        private const float HeadRadius = 0.09f;
        private const float LimbThickness = 0.045f;

        private void Awake()
        {
            CreateGround();
            EnsureDirectionalLight();
            var character = customRigRoot != null
                ? BuildCharacterFromCustomRig()
                : BuildMannequinCharacter();
            SetupFollowCamera(character.transform);
        }

        private AInimatorCharacter BuildMannequinCharacter()
        {
            var root = new GameObject("AInimatorMannequin");
            // Deferred activation: components added below must see the fully
            // wired RigMap + rest pose when their Awake() runs (RigBinder
            // calibrates on Awake, AInimatorCharacter loads the bundle).
            root.SetActive(false);

            var bones = BuildSmplBones(root.transform);
            BuildBoneVisuals(bones);

            var rigMap = ScriptableObject.CreateInstance<RigMap>();
            rigMap.EnsureEntriesSized();
            for (var i = 0; i < Smpl22Skeleton.NumBones; i++)
            {
                rigMap.Entries[i].rigBone = bones[i];
            }

            return WireDriverComponents(root, rigMap);
        }

        /// <summary>
        /// Drive the user-supplied rig instead of the mannequin: auto-map
        /// its bones by name (<see cref="RigMapAutoMapping"/> — Mixamo /
        /// Humanoid-FBX / Mannequin conventions), then reparent it under an
        /// inactive driver GameObject so the RigBinder/AInimatorCharacter
        /// Awake sequence sees the wired RigMap and the rig's rest pose,
        /// exactly like the mannequin path.
        /// </summary>
        private AInimatorCharacter BuildCharacterFromCustomRig()
        {
            var rigMap = ScriptableObject.CreateInstance<RigMap>();
            var mapped = RigMapAutoMapping.AutoMap(rigMap, customRigRoot);
            LogAutoMapResult(rigMap, mapped);

            // The controller owns the bones from here on; a live Animator
            // (FBX imports ship one) would overwrite their rotations every
            // frame right after the RigBinder writes them.
            foreach (var animator in customRigRoot.GetComponentsInChildren<Animator>(true))
            {
                animator.enabled = false;
            }

            var driver = new GameObject("AInimatorDriver");
            driver.SetActive(false);
            customRigRoot.SetParent(driver.transform, true);

            return WireDriverComponents(driver, rigMap);
        }

        private AInimatorCharacter WireDriverComponents(GameObject inactiveRoot, RigMap rigMap)
        {
            var binder = inactiveRoot.AddComponent<RigBinder>();
            binder.Map = rigMap;

            var character = inactiveRoot.AddComponent<AInimatorCharacter>();
            character.RigBinder = binder;
            character.BundleDirectoryOverride = bundleDirectoryOverride;

            inactiveRoot.SetActive(true);

            WirePresetsAndHud(character);
            return character;
        }

        private static void LogAutoMapResult(RigMap rigMap, int mapped)
        {
            Debug.Log($"RigDemoBootstrap: auto-mapped {mapped}/{Smpl22Skeleton.NumBones} SMPL bones onto the custom rig.");
            if (mapped >= Smpl22Skeleton.NumBones)
            {
                return;
            }

            for (var i = 0; i < Smpl22Skeleton.NumBones; i++)
            {
                if (rigMap.Entries[i].rigBone == null)
                {
                    Debug.LogWarning($"RigDemoBootstrap: SMPL bone '{RigMap.Smpl22BoneNames[i]}' unmapped — it will be ignored (rig_binding.md §2.1).");
                }
            }
        }

        /// <summary>
        /// Instantiate the 22 SMPL bone Transforms at the canonical rest
        /// pose (identity rotations, <see cref="Smpl22Skeleton.BoneOffsets"/>
        /// translations, pelvis at its canonical rest height so the computed
        /// rigScale is exactly 1).
        /// </summary>
        private static Transform[] BuildSmplBones(Transform root)
        {
            var bones = new Transform[Smpl22Skeleton.NumBones];
            for (var i = 0; i < Smpl22Skeleton.NumBones; i++)
            {
                var bone = new GameObject("smpl_" + RigMap.Smpl22BoneNames[i]).transform;
                var parentIndex = Smpl22Skeleton.ParentIndices[i];
                bone.SetParent(parentIndex < 0 ? root : bones[parentIndex], false);
                bone.localPosition = parentIndex < 0
                    ? new Vector3(0f, RigScale.SmplPelvisRestHeight, 0f)
                    : Smpl22Skeleton.BoneOffsets[i];
                bones[i] = bone;
            }

            return bones;
        }

        /// <summary>
        /// Stick-figure visuals: one sphere per joint (bigger for the head)
        /// and one box per bone segment, parented to the segment's PARENT
        /// bone so it follows that bone's retargeted rotation.
        /// </summary>
        private static void BuildBoneVisuals(IReadOnlyList<Transform> bones)
        {
            for (var i = 0; i < bones.Count; i++)
            {
                AddJointSphere(bones[i], i == Smpl22Skeleton.Head ? HeadRadius : JointRadius);
                var parentIndex = Smpl22Skeleton.ParentIndices[i];
                if (parentIndex >= 0)
                {
                    AddLimbBox(bones[parentIndex], Smpl22Skeleton.BoneOffsets[i]);
                }
            }
        }

        private static void AddJointSphere(Transform bone, float radius)
        {
            var sphere = CreateVisual(PrimitiveType.Sphere, bone, "Joint");
            sphere.localScale = Vector3.one * (radius * 2f);
        }

        private static void AddLimbBox(Transform parentBone, Vector3 offsetToChild)
        {
            var length = offsetToChild.magnitude;
            if (length < 1e-4f)
            {
                return;
            }

            var limb = CreateVisual(PrimitiveType.Cube, parentBone, "Limb");
            limb.localPosition = offsetToChild * 0.5f;
            limb.localRotation = Quaternion.FromToRotation(Vector3.up, offsetToChild / length);
            limb.localScale = new Vector3(LimbThickness, length, LimbThickness);
        }

        private static Transform CreateVisual(PrimitiveType type, Transform parent, string label)
        {
            var go = GameObject.CreatePrimitive(type);
            go.name = label;
            // The mannequin needs no physics; primitives ship a collider.
            Destroy(go.GetComponent<Collider>());
            go.transform.SetParent(parent, false);
            return go.transform;
        }

        /// <summary>
        /// Hydrate <see cref="ControlPreset"/> assets from the loaded
        /// bundle's presets, bind them to WASD + ZQSD + arrow keys, and add
        /// the HUD (prompt hot-swap keys + free-text fields).
        /// </summary>
        private void WirePresetsAndHud(AInimatorCharacter character)
        {
            if (character.Controller == null)
            {
                Debug.LogError(
                    "RigDemoBootstrap: bundle load failed (AInimatorCharacter has no controller). " +
                    "Copy the bundle to Assets/StreamingAssets/AInimatorBundle/ or set Bundle Directory Override " +
                    "on the RigDemo prefab — see the sample README.");
                return;
            }

            var bundlePresets = character.Controller.Bundle.Presets;
            ControlPreset Hydrate(string name)
            {
                if (!bundlePresets.TryGetValue(name, out var data))
                {
                    return null;
                }

                var preset = ScriptableObject.CreateInstance<ControlPreset>();
                preset.HydrateFrom(data);
                return preset;
            }

            character.IdlePreset = Hydrate("idle");
            BindKeys(character, Hydrate("forward"), KeyCode.W, KeyCode.Z, KeyCode.UpArrow);
            BindKeys(character, Hydrate("backward"), KeyCode.S, KeyCode.DownArrow);
            BindKeys(character, Hydrate("strafe_left"), KeyCode.A, KeyCode.Q, KeyCode.LeftArrow);
            BindKeys(character, Hydrate("strafe_right"), KeyCode.D, KeyCode.RightArrow);

            // Prompt hot-swap list (spec §3): every bundle preset shipping a
            // precomputed embedding, in stable name order for the 1..9 keys.
            var promptPresets = bundlePresets
                .Where(kv => kv.Value.promptEmb is { Length: > 0 })
                .OrderBy(kv => kv.Key)
                .Select(kv => Hydrate(kv.Key))
                .ToList();

            var hud = character.gameObject.AddComponent<RigDemoHud>();
            hud.Initialize(character, promptPresets);
        }

        private static void BindKeys(AInimatorCharacter character, ControlPreset preset, params KeyCode[] keys)
        {
            if (preset == null)
            {
                return;
            }

            foreach (var key in keys)
            {
                character.Bindings.Add(new InputBinding(key, preset, preset.PresetName));
            }
        }

        private static void CreateGround()
        {
            var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
            ground.name = "Ground";
            ground.transform.localScale = new Vector3(5f, 1f, 5f);
        }

        private static void EnsureDirectionalLight()
        {
            if (FindFirstObjectByType<Light>() != null)
            {
                return;
            }

            var light = new GameObject("RigDemoLight").AddComponent<Light>();
            light.type = LightType.Directional;
            light.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
        }

        private static void SetupFollowCamera(Transform target)
        {
            var camera = Camera.main;
            if (camera == null)
            {
                camera = new GameObject("RigDemoCamera").AddComponent<Camera>();
                camera.gameObject.tag = "MainCamera";
            }

            var follow = camera.gameObject.AddComponent<RigDemoFollowCamera>();
            follow.Target = target;
        }
    }
}
