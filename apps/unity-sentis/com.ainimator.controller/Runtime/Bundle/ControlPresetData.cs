using System;
using System.Collections.Generic;

namespace AInimator.Controller.Bundle
{
    /// <summary>
    /// Plain data mirror of one <c>presets/*.json</c> file
    /// (apps/spec/control_preset.schema.json). Values are RAW
    /// (un-normalized) — the engine applies the z-norm from
    /// <c>norm_stats.json</c> at consumption time.
    /// </summary>
    [Serializable]
    public sealed class ControlPresetData
    {
        public string name = "";
        public float vx;
        public float vz;
        public bool hasAim;
        public float aimX;
        public float aimZ;
        public string prompt;
        public float[] promptEmb;

        /// <summary>
        /// Parse one preset JSON file's contents.
        /// </summary>
        /// <exception cref="BundleLoadException">On a malformed/missing required field.</exception>
        public static ControlPresetData Parse(string json)
        {
            var root = MiniJson.Parse(json) as Dictionary<string, object>
                       ?? throw new BundleLoadException("preset json: root is not a JSON object.");

            if (!root.TryGetValue("name", out var nameObj) || nameObj is not string name)
            {
                throw new BundleLoadException("preset json: missing required 'name' field.");
            }

            if (!root.TryGetValue("control", out var controlObj) ||
                controlObj is not Dictionary<string, object> control)
            {
                throw new BundleLoadException($"preset '{name}': missing required 'control' object.");
            }

            if (!control.TryGetValue("vx", out var vxObj) || vxObj is not double vx)
            {
                throw new BundleLoadException($"preset '{name}': missing required 'control.vx'.");
            }

            if (!control.TryGetValue("vz", out var vzObj) || vzObj is not double vz)
            {
                throw new BundleLoadException($"preset '{name}': missing required 'control.vz'.");
            }

            var hasAim = control.TryGetValue("aim_x", out var aimXObj) &&
                         control.TryGetValue("aim_z", out var aimZObj);
            var aimX = hasAim ? (float)(double)aimXObj : 0f;
            var aimZ = hasAim ? (float)(double)aimZObj : 0f;

            string prompt = null;
            if (root.TryGetValue("prompt", out var promptObj) && promptObj is string promptStr)
            {
                prompt = promptStr;
            }

            float[] promptEmb = null;
            if (root.TryGetValue("prompt_emb", out var promptEmbObj) && promptEmbObj is List<object> embList)
            {
                promptEmb = new float[embList.Count];
                for (var i = 0; i < embList.Count; i++)
                {
                    promptEmb[i] = (float)(double)embList[i];
                }
            }

            return new ControlPresetData
            {
                name = name,
                vx = (float)vx,
                vz = (float)vz,
                hasAim = hasAim,
                aimX = aimX,
                aimZ = aimZ,
                prompt = prompt,
                promptEmb = promptEmb,
            };
        }
    }
}
