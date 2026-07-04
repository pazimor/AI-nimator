using UnityEngine;

namespace AInimator.Controller.PostProcess
{
    /// <summary>
    /// Re-derives per-frame foot-contact state from the pose (loss-only
    /// signal — never predicted by the controller), following
    /// <c>apps/spec/footlock_blending.md</c> §2 exactly.
    /// </summary>
    /// <remarks>
    /// Base criterion (same as the data pipeline's
    /// <c>controller_sequences.py::deriveFootContacts</c>): a foot is in
    /// contact when, on the same frame, its world height is below
    /// <see cref="HeightThreshold"/> **and** its planar (XZ) speed is below
    /// <see cref="SpeedThreshold"/>. Additive engine-side hysteresis
    /// (anti-flicker, spec §2): entering contact requires the raw
    /// criterion; leaving contact requires violating it for
    /// <see cref="ExitFrames"/> consecutive frames **or** a height above
    /// <see cref="ReleaseHeightThreshold"/> (1.5x the height threshold).
    /// </remarks>
    public sealed class FootContactDetector
    {
        /// <summary>World height below which a foot may be in contact (meters, Y axis).</summary>
        public const float DefaultHeightThreshold = 0.05f;

        /// <summary>Planar (XZ) speed below which a foot may be in contact (meters/frame).</summary>
        public const float DefaultSpeedThreshold = 0.01f;

        /// <summary>Consecutive violating frames required to release a lock.</summary>
        public const int DefaultExitFrames = 2;

        /// <summary>1.5x the height threshold — an immediate release condition regardless of frame count.</summary>
        public const float DefaultReleaseHeightMultiplier = 1.5f;

        public float HeightThreshold { get; }
        public float SpeedThreshold { get; }
        public int ExitFrames { get; }
        public float ReleaseHeightThreshold { get; }

        private readonly bool[] _isLocked;
        private readonly int[] _consecutiveViolationFrames;
        private readonly Vector3[] _previousPosition;
        private readonly bool[] _hasPreviousPosition;

        /// <param name="footCount">Number of tracked feet (2: left, right — <see cref="Smpl22Skeleton.FootJointIndices"/>).</param>
        public FootContactDetector(
            int footCount,
            float heightThreshold = DefaultHeightThreshold,
            float speedThreshold = DefaultSpeedThreshold,
            int exitFrames = DefaultExitFrames,
            float releaseHeightMultiplier = DefaultReleaseHeightMultiplier)
        {
            HeightThreshold = heightThreshold;
            SpeedThreshold = speedThreshold;
            ExitFrames = exitFrames;
            ReleaseHeightThreshold = heightThreshold * releaseHeightMultiplier;

            _isLocked = new bool[footCount];
            _consecutiveViolationFrames = new int[footCount];
            _previousPosition = new Vector3[footCount];
            _hasPreviousPosition = new bool[footCount];
        }

        /// <summary>Whether foot <paramref name="footIndex"/> is currently considered locked (in contact).</summary>
        public bool IsLocked(int footIndex) => _isLocked[footIndex];

        /// <summary>
        /// Update contact state for one foot from its current world
        /// position; returns the (possibly hysteresis-held) contact state
        /// for this frame.
        /// </summary>
        public bool Update(int footIndex, Vector3 worldPosition)
        {
            var height = worldPosition.y;
            var speed = 0f;
            if (_hasPreviousPosition[footIndex])
            {
                var previous = _previousPosition[footIndex];
                var dx = worldPosition.x - previous.x;
                var dz = worldPosition.z - previous.z;
                speed = Mathf.Sqrt(dx * dx + dz * dz);
            }

            _previousPosition[footIndex] = worldPosition;
            _hasPreviousPosition[footIndex] = true;

            var rawContact = height < HeightThreshold && speed < SpeedThreshold;

            if (!_isLocked[footIndex])
            {
                // Entering contact requires the raw criterion, no hysteresis delay.
                if (rawContact)
                {
                    _isLocked[footIndex] = true;
                    _consecutiveViolationFrames[footIndex] = 0;
                }

                return _isLocked[footIndex];
            }

            // Currently locked: stay locked unless the raw criterion is
            // violated for ExitFrames consecutive frames, or the foot is
            // clearly airborne (height > ReleaseHeightThreshold), which
            // releases immediately regardless of the frame counter.
            if (height > ReleaseHeightThreshold)
            {
                _isLocked[footIndex] = false;
                _consecutiveViolationFrames[footIndex] = 0;
                return false;
            }

            if (!rawContact)
            {
                _consecutiveViolationFrames[footIndex]++;
                if (_consecutiveViolationFrames[footIndex] >= ExitFrames)
                {
                    _isLocked[footIndex] = false;
                    _consecutiveViolationFrames[footIndex] = 0;
                }
            }
            else
            {
                _consecutiveViolationFrames[footIndex] = 0;
            }

            return _isLocked[footIndex];
        }

        /// <summary>Reset all tracked feet to the unlocked, no-history state.</summary>
        public void Reset()
        {
            for (var i = 0; i < _isLocked.Length; i++)
            {
                _isLocked[i] = false;
                _consecutiveViolationFrames[i] = 0;
                _hasPreviousPosition[i] = false;
            }
        }
    }
}
