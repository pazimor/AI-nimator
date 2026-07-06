using UnityEngine;

namespace AInimator.Controller.PostProcess
{
    /// <summary>
    /// Per-leg foot-lock IK (two-bone, analytic), following
    /// <c>apps/spec/footlock_blending.md</c> §3 exactly: capture the world
    /// position on contact entry, resolve the leg chain to hold it every
    /// locked frame, clamp the correction to 0.3m, and fade the correction
    /// out over 0.1s on release (never a snap).
    /// </summary>
    /// <remarks>
    /// This class is purely a **position corrector**: given the
    /// controller's raw (uncorrected) world joint positions for one frame
    /// (hip/knee/ankle for one leg) plus that leg's current contact state
    /// (from <see cref="FootContactDetector"/>), it returns corrected
    /// knee/ankle world positions. It never mutates the controller's state
    /// window — the caller must push the raw state into
    /// <c>StateBuffer</c> **before** calling this (spec §1: "the fenêtre
    /// autorégressive reçoit TOUJOURS l'état brut, jamais la pose
    /// corrigée").
    /// </remarks>
    public sealed class FootLockIk
    {
        /// <summary>Correction clamp: releases the lock outright beyond this distance (meters, spec §3.3).</summary>
        public const float DefaultClampDistance = 0.3f;

        /// <summary>Release fade duration (seconds, spec §3.4).</summary>
        public const float DefaultReleaseFadeSeconds = 0.1f;

        public float ClampDistance { get; }
        public float ReleaseFadeSeconds { get; }

        private readonly float _upperLength;
        private readonly float _lowerLength;

        private bool _wasLockedLastFrame;
        private Vector3 _lockedWorldPosition;
        private float _releaseFadeElapsed;
        private Vector3 _releaseStartCorrection;
        private bool _isFadingOut;

        /// <param name="upperLength">Hip -> knee bone length (meters).</param>
        /// <param name="lowerLength">Knee -> ankle bone length (meters).</param>
        public FootLockIk(
            float upperLength,
            float lowerLength,
            float clampDistance = DefaultClampDistance,
            float releaseFadeSeconds = DefaultReleaseFadeSeconds)
        {
            _upperLength = upperLength;
            _lowerLength = lowerLength;
            ClampDistance = clampDistance;
            ReleaseFadeSeconds = releaseFadeSeconds;
        }

        /// <summary>
        /// Resolve one leg for one frame.
        /// </summary>
        /// <param name="isLocked">This frame's contact state for this foot (from <see cref="FootContactDetector"/>).</param>
        /// <param name="hipWorldPosition">Raw (uncorrected) hip world position this frame.</param>
        /// <param name="kneeWorldPosition">Raw (uncorrected) knee world position this frame.</param>
        /// <param name="ankleWorldPosition">Raw (uncorrected) ankle (effector) world position this frame.</param>
        /// <param name="deltaTimeSeconds">Frame time, seconds (for the release fade).</param>
        /// <returns>Corrected (knee, ankle) world positions to apply to the rendered pose only.</returns>
        public (Vector3 knee, Vector3 ankle) Resolve(
            bool isLocked,
            Vector3 hipWorldPosition,
            Vector3 kneeWorldPosition,
            Vector3 ankleWorldPosition,
            float deltaTimeSeconds)
        {
            if (isLocked)
            {
                if (!_wasLockedLastFrame)
                {
                    // Entering contact this frame: capture the lock position.
                    _lockedWorldPosition = ankleWorldPosition;
                    _isFadingOut = false;
                    _releaseFadeElapsed = 0f;
                }

                var correction = _lockedWorldPosition - ankleWorldPosition;
                if (correction.magnitude > ClampDistance)
                {
                    // Spec §3.3: a correction beyond the clamp releases the
                    // lock outright (the controller "decided" a big move —
                    // do not stretch the leg to compensate).
                    _wasLockedLastFrame = false;
                    _isFadingOut = false;
                    return (kneeWorldPosition, ankleWorldPosition);
                }

                var solved = TwoBoneIkSolver.Solve(
                    hipWorldPosition, kneeWorldPosition, ankleWorldPosition,
                    _lockedWorldPosition, _upperLength, _lowerLength);

                _wasLockedLastFrame = true;
                return (solved.MidPosition, solved.EffectorPosition);
            }

            // Not locked this frame.
            if (_wasLockedLastFrame && !_isFadingOut)
            {
                // Just released: start the 0.1s fade-out from the last
                // applied correction (spec §3.4 — never a snap).
                _isFadingOut = true;
                _releaseFadeElapsed = 0f;
                _releaseStartCorrection = _lockedWorldPosition - ankleWorldPosition;
            }

            _wasLockedLastFrame = false;

            if (_isFadingOut)
            {
                _releaseFadeElapsed += deltaTimeSeconds;
                var t = ReleaseFadeSeconds > 0f
                    ? Mathf.Clamp01(_releaseFadeElapsed / ReleaseFadeSeconds)
                    : 1f;
                var fadedCorrection = Vector3.Lerp(_releaseStartCorrection, Vector3.zero, t);

                if (t >= 1f)
                {
                    _isFadingOut = false;
                }

                var fadedAnklePosition = ankleWorldPosition + fadedCorrection;
                var solved = TwoBoneIkSolver.Solve(
                    hipWorldPosition, kneeWorldPosition, ankleWorldPosition,
                    fadedAnklePosition, _upperLength, _lowerLength);
                return (solved.MidPosition, solved.EffectorPosition);
            }

            return (kneeWorldPosition, ankleWorldPosition);
        }

        /// <summary>Reset to the unlocked, no-fade state (e.g. on respawn/teleport).</summary>
        public void Reset()
        {
            _wasLockedLastFrame = false;
            _isFadingOut = false;
            _releaseFadeElapsed = 0f;
        }
    }
}
