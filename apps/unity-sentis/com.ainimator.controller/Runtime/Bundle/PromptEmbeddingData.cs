using System;
using System.Collections.Generic;

namespace AInimator.Controller.Bundle
{
    /// <summary>
    /// Plain data mirror of the JSON produced by the repo-side CLI
    /// <c>python -m ainimator.cli.encode_prompt</c>
    /// (<c>apps/spec/rig_binding.md</c> §4.1): <c>{"prompt": str,
    /// "prompt_emb": float[]}</c> — distinct from the full
    /// <see cref="ControlPresetData"/> preset schema (no <c>control</c>
    /// section). Public (unlike <see cref="MiniJson"/>) so Editor code can
    /// parse this file without reaching into the internal JSON parser.
    /// </summary>
    public sealed class PromptEmbeddingData
    {
        public string Prompt { get; }
        public float[] PromptEmb { get; }

        private PromptEmbeddingData(string prompt, float[] promptEmb)
        {
            Prompt = prompt;
            PromptEmb = promptEmb;
        }

        /// <exception cref="BundleLoadException">On a malformed/missing required field.</exception>
        public static PromptEmbeddingData Parse(string json)
        {
            var root = MiniJson.Parse(json) as Dictionary<string, object>
                       ?? throw new BundleLoadException("encode_prompt json: root is not a JSON object.");

            if (!root.TryGetValue("prompt", out var promptObj) || promptObj is not string prompt)
            {
                throw new BundleLoadException("encode_prompt json: missing required 'prompt' field.");
            }

            if (!root.TryGetValue("prompt_emb", out var embObj) || embObj is not List<object> embList)
            {
                throw new BundleLoadException("encode_prompt json: missing required 'prompt_emb' array.");
            }

            var promptEmb = new float[embList.Count];
            for (var i = 0; i < embList.Count; i++)
            {
                promptEmb[i] = (float)(double)embList[i];
            }

            return new PromptEmbeddingData(prompt, promptEmb);
        }
    }
}
