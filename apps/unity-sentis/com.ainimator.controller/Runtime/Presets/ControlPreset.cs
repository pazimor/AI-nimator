using AInimator.Controller.Bundle;
using UnityEngine;

namespace AInimator.Controller.Presets
{
    /// <summary>
    /// Inspector-editable, versionable asset mirroring one
    /// <c>presets/*.json</c> file (apps/spec/control_preset.schema.json).
    /// </summary>
    /// <remarks>
    /// Values are RAW (un-normalized): <see cref="Vx"/>/<see cref="Vz"/> in
    /// meters/frame in the root-local ground frame; <see cref="AimX"/>/
    /// <see cref="AimZ"/> unit-norm by construction. The
    /// <see cref="Runtime.ControllerRuntime"/> applies the z-norm from
    /// <c>norm_stats.json</c> at consumption time — this asset never stores
    /// normalized values.
    /// </remarks>
    [CreateAssetMenu(
        fileName = "NewControlPreset",
        menuName = "AInimator/Control Preset",
        order = 100)]
    public sealed class ControlPreset : ScriptableObject
    {
        [Tooltip("Preset identifier (snake_case), doubles as the source JSON filename stem.")]
        [SerializeField] private string presetName = "idle";

        [Tooltip("Desired lateral velocity, meters/frame, root-local frame.")]
        [SerializeField] private float vx;

        [Tooltip("Desired forward velocity, meters/frame, root-local frame (+Z = facing).")]
        [SerializeField] private float vz;

        [Tooltip("Whether this preset supplies an aim direction (manifest must declare control_channels=4).")]
        [SerializeField] private bool hasAim;

        [Tooltip("Aim direction X component (unit-norm 2D vector with AimZ).")]
        [SerializeField] private float aimX;

        [Tooltip("Aim direction Z component (unit-norm 2D vector with AimX).")]
        [SerializeField] private float aimZ;

        [Tooltip("Optional authoring-time text prompt this preset was built for (informational).")]
        [SerializeField] private string prompt;

        [Tooltip("Optional precomputed prompt embedding; length must equal manifest.prompt_emb_channels.")]
        [SerializeField] private float[] promptEmb;

        public string PresetName => presetName;
        public float Vx => vx;
        public float Vz => vz;
        public bool HasAim => hasAim;
        public float AimX => aimX;
        public float AimZ => aimZ;
        public string Prompt => prompt;
        public float[] PromptEmb => promptEmb;

        /// <summary>
        /// Write this preset's raw control vector into
        /// <paramref name="destination"/>, ordered per <c>manifest.control_layout</c>
        /// (<c>[vx, vz]</c> or <c>[vx, vz, aim_x, aim_z]</c>).
        /// </summary>
        public void WriteRawControl(System.Span<float> destination, bool manifestHasAim)
        {
            destination[0] = vx;
            destination[1] = vz;
            if (manifestHasAim)
            {
                destination[2] = hasAim ? aimX : 0f;
                destination[3] = hasAim ? aimZ : 1f; // default: facing forward
            }
        }

        /// <summary>Hydrate this asset's fields from a parsed bundle preset file.</summary>
        public void HydrateFrom(ControlPresetData data)
        {
            presetName = data.name;
            vx = data.vx;
            vz = data.vz;
            hasAim = data.hasAim;
            aimX = data.aimX;
            aimZ = data.aimZ;
            prompt = data.prompt;
            promptEmb = data.promptEmb;
        }

        /// <summary>
        /// Overwrite only the raw control fields (<see cref="Vx"/>/<see cref="Vz"/>/
        /// <see cref="AimX"/>/<see cref="AimZ"/>), leaving <see cref="PresetName"/>/
        /// <see cref="Prompt"/>/<see cref="PromptEmb"/> untouched. Used by
        /// <c>TextCommand.TextToControlResolver</c> consumers (Goal B phase
        /// B6) to drive a transient in-memory preset from a resolved
        /// free-text command through the exact same
        /// <see cref="WriteRawControl"/> path as an authored asset — zero
        /// impact on the preset path itself.
        /// </summary>
        public void SetRawControl(float rawVx, float rawVz, float rawAimX, float rawAimZ)
        {
            vx = rawVx;
            vz = rawVz;
            hasAim = true;
            aimX = rawAimX;
            aimZ = rawAimZ;
        }
    }
}
