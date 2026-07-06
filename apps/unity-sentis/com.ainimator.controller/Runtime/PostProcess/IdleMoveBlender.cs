using UnityEngine;

namespace AInimator.Controller.PostProcess
{
    /// <summary>
    /// Idle <-> move pose-space cross-fade state machine
    /// (<c>apps/spec/footlock_blending.md</c> §4).
    /// </summary>
    /// <remarks>
    /// This is purely a **blend-weight** state machine: it decides, per
    /// frame, how much of the idle pose vs. the controller pose to show
    /// (weight in <c>[0, 1]</c>, <c>0 = fully controller, 1 = fully idle</c>).
    /// Applying that weight to two actual poses (controller state vs. idle
    /// asset/preset) is the caller's job, since it depends on the host
    /// project's pose representation (raw rotation6d, a skinned
    /// `Animator`, ...). The controller keeps ticking (and feeding its
    /// autoregressive window) throughout the fades — blending is aval/pose
    /// space only, never fed back (spec §4 last line).
    /// </remarks>
    public sealed class IdleMoveBlender
    {
        /// <summary>Planar control-speed threshold below which the character is considered "idle-seeking" (m/frame, raw control, spec §4).</summary>
        public const float DefaultMoveSpeedThreshold = 0.005f;

        /// <summary>Seconds of below-threshold control required before entering idle (spec §4).</summary>
        public const float DefaultIdleEntrySeconds = 0.25f;

        /// <summary>Cross-fade duration between the two states (spec §4).</summary>
        public const float DefaultCrossFadeSeconds = 0.2f;

        public float MoveSpeedThreshold { get; }
        public float IdleEntrySeconds { get; }
        public float CrossFadeSeconds { get; }

        /// <summary>True when the state machine is currently in (or fading toward) the idle state.</summary>
        public bool IsIdle { get; private set; }

        /// <summary>Current blend weight toward idle: 0 = fully controller pose, 1 = fully idle pose.</summary>
        public float IdleWeight { get; private set; }

        private float _belowThresholdElapsed;
        private float _crossFadeElapsed;
        private float _crossFadeStartWeight;
        private bool _isCrossFading;

        public IdleMoveBlender(
            float moveSpeedThreshold = DefaultMoveSpeedThreshold,
            float idleEntrySeconds = DefaultIdleEntrySeconds,
            float crossFadeSeconds = DefaultCrossFadeSeconds)
        {
            MoveSpeedThreshold = moveSpeedThreshold;
            IdleEntrySeconds = idleEntrySeconds;
            CrossFadeSeconds = crossFadeSeconds;
        }

        /// <summary>
        /// Advance the state machine by one frame given this frame's raw
        /// (unnormalized) requested control <c>(vx, vz)</c>, and return the
        /// idle blend weight to apply this frame.
        /// </summary>
        public float Update(float rawVx, float rawVz, float deltaTimeSeconds)
        {
            var planarSpeed = Mathf.Sqrt(rawVx * rawVx + rawVz * rawVz);
            var isMoveInput = planarSpeed > MoveSpeedThreshold;

            var targetIsIdle = IsIdle;
            if (isMoveInput)
            {
                _belowThresholdElapsed = 0f;
                if (IsIdle)
                {
                    targetIsIdle = false;
                }
            }
            else
            {
                _belowThresholdElapsed += deltaTimeSeconds;
                if (!IsIdle && _belowThresholdElapsed >= IdleEntrySeconds)
                {
                    targetIsIdle = true;
                }
            }

            if (targetIsIdle != IsIdle)
            {
                IsIdle = targetIsIdle;
                _isCrossFading = true;
                _crossFadeElapsed = 0f;
                _crossFadeStartWeight = IdleWeight;
            }

            if (_isCrossFading)
            {
                _crossFadeElapsed += deltaTimeSeconds;
                var t = CrossFadeSeconds > 0f
                    ? Mathf.Clamp01(_crossFadeElapsed / CrossFadeSeconds)
                    : 1f;
                var targetWeight = IsIdle ? 1f : 0f;
                IdleWeight = Mathf.Lerp(_crossFadeStartWeight, targetWeight, t);

                if (t >= 1f)
                {
                    _isCrossFading = false;
                }
            }
            else
            {
                IdleWeight = IsIdle ? 1f : 0f;
            }

            return IdleWeight;
        }

        /// <summary>Reset to the fully-controller, no-history state.</summary>
        public void Reset()
        {
            IsIdle = false;
            IdleWeight = 0f;
            _belowThresholdElapsed = 0f;
            _crossFadeElapsed = 0f;
            _isCrossFading = false;
        }
    }
}
