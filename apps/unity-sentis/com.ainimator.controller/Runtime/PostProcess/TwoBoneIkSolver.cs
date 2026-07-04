using UnityEngine;

namespace AInimator.Controller.PostProcess
{
    /// <summary>
    /// Analytic two-bone IK solver (hip -> knee -> ankle), used by
    /// <see cref="FootLockIk"/> to hold a locked foot at its captured
    /// world position (<c>apps/spec/footlock_blending.md</c> §3).
    /// </summary>
    /// <remarks>
    /// Standard law-of-cosines two-bone solve: given root (hip), a target
    /// effector position, an upper/lower bone length and a pole vector (the
    /// knee direction in the controller's own pose, never a fixed pole —
    /// spec §3.2), returns the corrected knee (mid) and effector (ankle)
    /// positions. This does not rotate the actual bone transforms — B4's
    /// consumer (<see cref="FootLockIk"/>) only needs corrected joint
    /// **positions** to feed back into a position-space pose correction;
    /// wiring this onto an actual `Animator`/rig (e.g.
    /// `AnimationRigging.TwoBoneIKConstraint`) is an engine-integration
    /// step left to the host project (README "Known constraints").
    /// </remarks>
    public static class TwoBoneIkSolver
    {
        public readonly struct Result
        {
            public readonly Vector3 MidPosition;
            public readonly Vector3 EffectorPosition;
            public readonly bool TargetReachable;

            public Result(Vector3 midPosition, Vector3 effectorPosition, bool targetReachable)
            {
                MidPosition = midPosition;
                EffectorPosition = effectorPosition;
                TargetReachable = targetReachable;
            }
        }

        /// <summary>
        /// Solve a two-bone chain so the effector reaches
        /// <paramref name="targetPosition"/> as closely as possible.
        /// </summary>
        /// <param name="rootPosition">Hip (chain root) world position — fixed, not moved by this solve.</param>
        /// <param name="midPosition">Current knee world position (used only to derive the pole direction).</param>
        /// <param name="effectorPosition">Current ankle world position (used only to derive the pole direction).</param>
        /// <param name="targetPosition">Desired world position for the effector (the locked foot position).</param>
        /// <param name="upperLength">Hip -> knee bone length (meters).</param>
        /// <param name="lowerLength">Knee -> ankle bone length (meters).</param>
        /// <remarks>
        /// The pole vector is derived from the *current controller pose's*
        /// knee direction (never a fixed world-space pole), preserving the
        /// natural bend orientation per spec §3.2. When the target is
        /// unreachable (farther than <c>upperLength + lowerLength</c>), the
        /// chain is fully extended toward the target along the
        /// root->target direction and <see cref="Result.TargetReachable"/>
        /// is <c>false</c> — callers (<see cref="FootLockIk"/>) use this to
        /// decide whether to still apply the correction or fall back
        /// (the 0.3m clamp in spec §3.3 already prevents this case from
        /// mattering in practice for foot-lock, but the solver itself stays
        /// correct/stable for a target at any distance).
        /// </remarks>
        public static Result Solve(
            Vector3 rootPosition,
            Vector3 midPosition,
            Vector3 effectorPosition,
            Vector3 targetPosition,
            float upperLength,
            float lowerLength)
        {
            var toTarget = targetPosition - rootPosition;
            var targetDistance = toTarget.magnitude;
            var maxReach = upperLength + lowerLength;
            var minReach = Mathf.Abs(upperLength - lowerLength);

            var reachable = targetDistance <= maxReach && targetDistance >= minReach;
            var clampedDistance = Mathf.Clamp(targetDistance, minReach + 1e-6f, maxReach - 1e-6f);

            if (targetDistance < 1e-6f)
            {
                // Degenerate: target coincides with the root. Fall back to
                // the current pose unchanged rather than dividing by zero.
                return new Result(midPosition, effectorPosition, false);
            }

            var forward = toTarget / targetDistance;

            // Pole direction: current knee offset from the root->effector
            // axis, projected out of `forward` and normalized. Preserves
            // the pose's natural bend plane (spec §3.2) instead of a fixed
            // pole vector.
            var poleRaw = (midPosition - rootPosition) - Vector3.Dot(midPosition - rootPosition, forward) * forward;
            var poleLengthSq = poleRaw.sqrMagnitude;
            var pole = poleLengthSq > 1e-10f
                ? poleRaw / Mathf.Sqrt(poleLengthSq)
                : Vector3.Cross(forward, Vector3.up).normalized;
            if (pole.sqrMagnitude < 1e-10f)
            {
                // forward was parallel to Vector3.up: pick any perpendicular.
                pole = Vector3.Cross(forward, Vector3.right).normalized;
            }

            // Law of cosines: angle at the root between "upper bone" and
            // "root -> target" axis.
            var cosRootAngle = (upperLength * upperLength + clampedDistance * clampedDistance - lowerLength * lowerLength)
                                / (2f * upperLength * clampedDistance);
            cosRootAngle = Mathf.Clamp(cosRootAngle, -1f, 1f);
            var rootAngle = Mathf.Acos(cosRootAngle);

            // Rotate `forward` by `rootAngle` around the plane normal
            // (forward x pole) to get the direction from root to mid.
            var planeNormal = Vector3.Cross(forward, pole).normalized;
            var midDirection = RotateAroundAxis(forward, planeNormal, rootAngle);
            var newMidPosition = rootPosition + midDirection * upperLength;

            // Effector sits `lowerLength` further along the mid->target
            // direction constrained by the remaining triangle side; the
            // simplest closed form is to re-derive it as the point at
            // distance `lowerLength` from `newMidPosition` that lies as
            // close as possible to `targetPosition` while keeping the
            // planar triangle: since |mid->target| already accounts for
            // the law of cosines construction (mid is exactly `lowerLength`
            // from a hypothetically-reachable target), for the reachable
            // case newEffector == targetPosition-consistent point:
            var midToTargetDir = (targetPosition - newMidPosition);
            var midToTargetLen = midToTargetDir.magnitude;
            var newEffectorPosition = midToTargetLen > 1e-6f
                ? newMidPosition + midToTargetDir / midToTargetLen * lowerLength
                : newMidPosition + (forward * lowerLength);

            return new Result(newMidPosition, newEffectorPosition, reachable);
        }

        private static Vector3 RotateAroundAxis(Vector3 vector, Vector3 axis, float angleRadians)
        {
            var rotation = Quaternion.AngleAxis(angleRadians * Mathf.Rad2Deg, axis);
            return rotation * vector;
        }
    }
}
