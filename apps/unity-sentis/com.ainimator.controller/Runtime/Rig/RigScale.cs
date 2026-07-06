using UnityEngine;

namespace AInimator.Controller.Rig
{
    /// <summary>
    /// Pure math for the rig-scale factor (<c>apps/spec/rig_binding.md</c>
    /// §2.3): ratio of the rig's rest-pose pelvis height to the SMPL-22
    /// canonical pelvis height, applied to root motion before it drives the
    /// actor transform. Kept free of any <see cref="Object"/>/component
    /// dependency so it is unit-testable without a live rig.
    /// </summary>
    public static class RigScale
    {
        /// <summary>
        /// Canonical SMPL-22 pelvis rest height (meters, mean shape), used as
        /// the denominator of the scale ratio — matches the value documented
        /// in <c>apps/spec/rig_binding.md</c> §2.3.
        /// </summary>
        public const float SmplPelvisRestHeight = 0.91f;

        /// <summary>
        /// Compute <c>rigScale = rigPelvisRestHeight / SmplPelvisRestHeight</c>.
        /// Throws for a non-positive rig height (calibration error — never
        /// silently return a degenerate scale).
        /// </summary>
        public static float FromPelvisHeight(float rigPelvisRestHeight)
        {
            if (rigPelvisRestHeight <= 0f)
            {
                throw new System.ArgumentOutOfRangeException(
                    nameof(rigPelvisRestHeight),
                    "Rig pelvis rest height must be > 0 -- check the RigMap's pelvis mapping and the rig's rest pose.");
            }

            return rigPelvisRestHeight / SmplPelvisRestHeight;
        }

        /// <summary>
        /// Scale a root-local motion delta <c>(Δforward, Δlateral, Δheight,
        /// Δyaw)</c> by <paramref name="rigScale"/>. Δyaw (index 3, when
        /// present) is dimensionless and is never scaled (spec §2.3).
        /// </summary>
        public static void ScaleGlobalDelta(System.ReadOnlySpan<float> rawGlobalDelta, float rigScale, System.Span<float> destination)
        {
            if (rawGlobalDelta.Length != destination.Length)
            {
                throw new System.ArgumentException("rawGlobalDelta and destination must have the same length.");
            }

            for (var i = 0; i < rawGlobalDelta.Length; i++)
            {
                // Channels 0..2 = (Δforward, Δlateral, Δheight): scaled.
                // Channel 3 (Δyaw), if present: dimensionless, untouched.
                destination[i] = i < 3 ? rawGlobalDelta[i] * rigScale : rawGlobalDelta[i];
            }
        }
    }
}
