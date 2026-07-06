using System;

namespace AInimator.Controller.Prompting
{
    /// <summary>
    /// Cross-fades between two prompt embeddings in embedding space
    /// (<c>apps/spec/rig_binding.md</c> §3): a linear per-channel lerp
    /// <c>embOld -&gt; embNew</c> over a configurable duration (default
    /// <see cref="DefaultCrossFadeSeconds"/> = 0.3s).
    /// </summary>
    /// <remarks>
    /// ⚠ Heuristic, not a validated behaviour: the controller was never
    /// trained on interpolated prompt embeddings, so a lerp path through
    /// embedding space may cross regions the model never saw (spec §3,
    /// explicit warning). This must be validated visually per bundle; if
    /// transitions look degraded (limb pops, unstable gait), the documented
    /// fallback is a hard prompt switch + a pose-space cross-fade instead
    /// (reusing the <c>IdleMoveBlender</c>-style mechanism from B4 §4 on the
    /// *output pose*, not the embedding). This class only implements the
    /// embedding-space path; the pose-space fallback is the caller's choice
    /// (not wired here, since it requires an idle/alternate pose source that
    /// is content-dependent).
    /// <para/>
    /// Pure math, no engine dependency: unit-testable without Unity running.
    /// Caller owns the destination buffer (no per-frame allocation).
    /// </remarks>
    public sealed class PromptCrossFader
    {
        /// <summary>Default cross-fade duration (seconds), spec §3.</summary>
        public const float DefaultCrossFadeSeconds = 0.3f;

        private readonly int _channels;
        private float[] _fromEmb;
        private float[] _toEmb;
        private float _elapsed;
        private bool _isFading;

        public float CrossFadeSeconds { get; set; }

        /// <summary>True while a cross-fade is in progress.</summary>
        public bool IsFading => _isFading;

        public PromptCrossFader(int channels, float crossFadeSeconds = DefaultCrossFadeSeconds)
        {
            if (channels <= 0)
            {
                throw new ArgumentException("channels must be > 0.", nameof(channels));
            }

            _channels = channels;
            CrossFadeSeconds = crossFadeSeconds;
            _fromEmb = new float[channels];
            _toEmb = new float[channels];
        }

        /// <summary>
        /// Snap immediately to <paramref name="embedding"/> with no fade
        /// (e.g. first-frame initialization).
        /// </summary>
        public void SnapTo(ReadOnlySpan<float> embedding)
        {
            RequireLength(embedding);
            embedding.CopyTo(_fromEmb);
            embedding.CopyTo(_toEmb);
            _isFading = false;
            _elapsed = 0f;
        }

        /// <summary>
        /// Begin a cross-fade from the current interpolated value toward
        /// <paramref name="targetEmbedding"/> (a new active prompt, or the
        /// bundle's learned null embedding when clearing — spec §3, never
        /// zeros).
        /// </summary>
        public void BeginFadeTo(ReadOnlySpan<float> targetEmbedding, Span<float> currentValueScratch)
        {
            RequireLength(targetEmbedding);
            if (currentValueScratch.Length != _channels)
            {
                throw new ArgumentException($"currentValueScratch length must be {_channels}.");
            }

            Evaluate(currentValueScratch);
            currentValueScratch.CopyTo(_fromEmb);
            targetEmbedding.CopyTo(_toEmb);
            _elapsed = 0f;
            _isFading = true;
        }

        /// <summary>Advance the fade clock by <paramref name="deltaTimeSeconds"/>.</summary>
        public void Tick(float deltaTimeSeconds)
        {
            if (!_isFading)
            {
                return;
            }

            _elapsed += deltaTimeSeconds;
            if (_elapsed >= CrossFadeSeconds)
            {
                _isFading = false;
            }
        }

        /// <summary>Write the current interpolated embedding into <paramref name="destination"/> (length <c>channels</c>).</summary>
        public void Evaluate(Span<float> destination)
        {
            if (destination.Length != _channels)
            {
                throw new ArgumentException($"destination length must be {_channels}.");
            }

            var t = CrossFadeSeconds > 0f ? Math.Clamp(_elapsed / CrossFadeSeconds, 0f, 1f) : 1f;
            for (var i = 0; i < _channels; i++)
            {
                destination[i] = _fromEmb[i] + (_toEmb[i] - _fromEmb[i]) * t;
            }
        }

        private void RequireLength(ReadOnlySpan<float> embedding)
        {
            if (embedding.Length != _channels)
            {
                throw new ArgumentException($"embedding length must be {_channels}; got {embedding.Length}.");
            }
        }
    }
}
