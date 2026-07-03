using System;
using System.Collections.Generic;

namespace AInimator.Controller.Bundle
{
    /// <summary>
    /// Parsed contents of <c>norm_stats.json</c> (apps/spec — serialized by
    /// <c>ainimator.export.bundle</c>).
    /// </summary>
    /// <remarks>
    /// Arrays are flattened to row-major <see cref="float"/> buffers with a
    /// documented logical shape, since Unity's <c>JsonUtility</c> cannot
    /// deserialize the nested/jagged arrays the Python side emits (e.g.
    /// <c>bone_mean</c> is logically <c>(1, 1, numBones, motionChannels)</c>).
    /// The leading two singleton dims exist only because the Python
    /// <c>MotionNormalizer</c> buffers are frame-broadcastable; the engine
    /// only ever reads the trailing <c>(numBones, motionChannels)</c> or
    /// <c>(globalChannels,)</c> slice.
    /// </remarks>
    public sealed class NormStats
    {
        /// <summary>Flattened <c>(numBones, motionChannels)</c>, row-major.</summary>
        public float[] BoneMean { get; }

        /// <summary>Flattened <c>(numBones, motionChannels)</c>, row-major.</summary>
        public float[] BoneStd { get; }

        /// <summary>Flattened <c>(globalChannels,)</c>.</summary>
        public float[] GlobalMean { get; }

        /// <summary>Flattened <c>(globalChannels,)</c>.</summary>
        public float[] GlobalStd { get; }

        /// <summary>Flattened <c>(numBones, motionChannels)</c>, row-major.</summary>
        public float[] DeltaBoneMean { get; }

        /// <summary>Flattened <c>(numBones, motionChannels)</c>, row-major.</summary>
        public float[] DeltaBoneStd { get; }

        /// <summary>Flattened <c>(globalChannels,)</c>.</summary>
        public float[] DeltaGlobalMean { get; }

        /// <summary>Flattened <c>(globalChannels,)</c>.</summary>
        public float[] DeltaGlobalStd { get; }

        /// <summary>Control mean, ordered per <c>control.channels</c> (vx, vz only — never aim).</summary>
        public float[] ControlMean { get; }

        /// <summary>Control std, ordered per <c>control.channels</c> (vx, vz only — never aim).</summary>
        public float[] ControlStd { get; }

        /// <summary>Learned null prompt embedding, length <c>prompt.channels</c>. Empty if the bundle carries no text conditioning.</summary>
        public float[] PromptNullEmb { get; }

        private NormStats(
            float[] boneMean, float[] boneStd, float[] globalMean, float[] globalStd,
            float[] deltaBoneMean, float[] deltaBoneStd, float[] deltaGlobalMean, float[] deltaGlobalStd,
            float[] controlMean, float[] controlStd, float[] promptNullEmb)
        {
            BoneMean = boneMean;
            BoneStd = boneStd;
            GlobalMean = globalMean;
            GlobalStd = globalStd;
            DeltaBoneMean = deltaBoneMean;
            DeltaBoneStd = deltaBoneStd;
            DeltaGlobalMean = deltaGlobalMean;
            DeltaGlobalStd = deltaGlobalStd;
            ControlMean = controlMean;
            ControlStd = controlStd;
            PromptNullEmb = promptNullEmb;
        }

        /// <summary>
        /// Parse <c>norm_stats.json</c> raw text into flattened stat buffers.
        /// </summary>
        /// <param name="json">Raw file contents.</param>
        /// <exception cref="BundleLoadException">
        /// Thrown when a required section is missing or malformed — fail-fast,
        /// never a silent default (Goal B verite #4).
        /// </exception>
        public static NormStats Parse(string json)
        {
            var root = MiniJson.Parse(json) as Dictionary<string, object>
                       ?? throw new BundleLoadException("norm_stats.json: root is not a JSON object.");

            var state = RequireObject(root, "state", "norm_stats.json");
            var delta = RequireObject(root, "delta", "norm_stats.json");
            var control = RequireObject(root, "control", "norm_stats.json");

            var boneMean = FlattenTrailing(RequireArray(state, "bone_mean", "state"), 2);
            var boneStd = FlattenTrailing(RequireArray(state, "bone_std", "state"), 2);
            var globalMean = FlattenTrailing(RequireArray(state, "global_mean", "state"), 1);
            var globalStd = FlattenTrailing(RequireArray(state, "global_std", "state"), 1);

            var deltaBoneMean = FlattenTrailing(RequireArray(delta, "bone_mean", "delta"), 2);
            var deltaBoneStd = FlattenTrailing(RequireArray(delta, "bone_std", "delta"), 2);
            var deltaGlobalMean = FlattenTrailing(RequireArray(delta, "global_mean", "delta"), 1);
            var deltaGlobalStd = FlattenTrailing(RequireArray(delta, "global_std", "delta"), 1);

            var controlMean = FlattenTrailing(RequireArray(control, "mean", "control"), 1);
            var controlStd = FlattenTrailing(RequireArray(control, "std", "control"), 1);

            var promptNullEmb = Array.Empty<float>();
            if (root.TryGetValue("prompt", out var promptObj) && promptObj is Dictionary<string, object> prompt)
            {
                var nullEmbList = RequireArray(prompt, "null_emb", "prompt");
                promptNullEmb = ToFloatArray(nullEmbList);
            }

            return new NormStats(
                boneMean, boneStd, globalMean, globalStd,
                deltaBoneMean, deltaBoneStd, deltaGlobalMean, deltaGlobalStd,
                controlMean, controlStd, promptNullEmb);
        }

        private static Dictionary<string, object> RequireObject(
            Dictionary<string, object> parent, string key, string context)
        {
            if (!parent.TryGetValue(key, out var value) || value is not Dictionary<string, object> obj)
            {
                throw new BundleLoadException($"{context}: missing or malformed '{key}' section.");
            }

            return obj;
        }

        private static List<object> RequireArray(
            Dictionary<string, object> parent, string key, string context)
        {
            if (!parent.TryGetValue(key, out var value) || value is not List<object> list)
            {
                throw new BundleLoadException($"{context}: missing or malformed '{key}' array.");
            }

            return list;
        }

        /// <summary>
        /// Flatten a nested JSON array, keeping only the last
        /// <paramref name="trailingRank"/> dimensions (dropping the leading
        /// singleton frame/batch dims the Python normalizer always emits).
        /// </summary>
        private static float[] FlattenTrailing(List<object> nested, int trailingRank)
        {
            // Descend past leading singleton wrapper dims until we are
            // `trailingRank` levels away from scalars.
            object current = nested;
            var depth = 0;
            while (current is List<object> list)
            {
                depth++;
                if (list.Count == 0)
                {
                    break;
                }

                current = list[0];
            }

            var levelsToDrop = depth - trailingRank;
            if (levelsToDrop < 0)
            {
                levelsToDrop = 0;
            }

            object cursor = nested;
            for (var i = 0; i < levelsToDrop; i++)
            {
                cursor = ((List<object>)cursor)[0];
            }

            var flat = new List<float>();
            FlattenInto((List<object>)cursor, flat);
            return flat.ToArray();
        }

        private static void FlattenInto(List<object> node, List<float> sink)
        {
            foreach (var item in node)
            {
                switch (item)
                {
                    case List<object> child:
                        FlattenInto(child, sink);
                        break;
                    case double d:
                        sink.Add((float)d);
                        break;
                    default:
                        throw new BundleLoadException(
                            "norm_stats.json: unexpected non-numeric leaf while flattening stats.");
                }
            }
        }

        private static float[] ToFloatArray(List<object> list)
        {
            var arr = new float[list.Count];
            for (var i = 0; i < list.Count; i++)
            {
                arr[i] = (float)(double)list[i];
            }

            return arr;
        }
    }
}
