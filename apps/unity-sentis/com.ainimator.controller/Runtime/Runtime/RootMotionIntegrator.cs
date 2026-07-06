using System;

namespace AInimator.Controller.Runtime
{
    /// <summary>
    /// Integrates root-local motion deltas <c>(Δforward, Δlateral, Δheight,
    /// Δyaw)</c> into a world-space position + yaw, exactly mirroring
    /// <c>ainimator.model.controller_rollout._integrateOneStep</c>.
    /// </summary>
    /// <remarks>
    /// Coordinate system: Y-up, right-handed (manifest.json
    /// <c>coord_system</c>). World X/Z are the ground plane, Y is height.
    /// Given the current yaw (radians, around +Y), a local delta rotates
    /// into world space as:
    /// <code>
    /// worldDx = cos(yaw) * dFwd - sin(yaw) * dLat
    /// worldDz = sin(yaw) * dFwd + cos(yaw) * dLat
    /// </code>
    /// This is intentionally free of any Unity <c>Transform</c>/<c>Quaternion</c>
    /// dependency so it is unit-testable and so the integration math is
    /// byte-for-byte auditable against the Python reference.
    /// </remarks>
    public struct RootMotionIntegrator
    {
        public float PositionX;
        public float PositionY;
        public float PositionZ;
        public float YawRadians;

        public static RootMotionIntegrator AtOrigin()
        {
            return new RootMotionIntegrator
            {
                PositionX = 0f,
                PositionY = 0f,
                PositionZ = 0f,
                YawRadians = 0f,
            };
        }

        /// <summary>
        /// Advance by one root-local delta frame
        /// <c>(dForward, dLateral, dHeight, dYaw)</c>.
        /// </summary>
        public void Integrate(ReadOnlySpan<float> globalDelta)
        {
            if (globalDelta.Length != 4)
            {
                throw new ArgumentException("globalDelta must have length 4.", nameof(globalDelta));
            }

            var dFwd = globalDelta[0];
            var dLat = globalDelta[1];
            var dHeight = globalDelta[2];
            var dYaw = globalDelta[3];

            var cos = MathF.Cos(YawRadians);
            var sin = MathF.Sin(YawRadians);
            var worldDx = cos * dFwd - sin * dLat;
            var worldDz = sin * dFwd + cos * dLat;

            PositionX += worldDx;
            PositionY += dHeight;
            PositionZ += worldDz;
            YawRadians += dYaw;
        }
    }
}
