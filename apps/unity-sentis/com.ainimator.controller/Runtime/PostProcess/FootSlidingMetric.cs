using UnityEngine;

namespace AInimator.Controller.PostProcess
{
    /// <summary>
    /// Accumulates the foot-sliding metric used for the B4 acceptance
    /// criterion (<c>apps/spec/footlock_blending.md</c> §5): average planar
    /// (XZ) displacement of a foot while it is in contact, over a run.
    /// </summary>
    /// <remarks>
    /// Usage: call <see cref="Accumulate"/> once per frame per foot with
    /// that foot's contact state (from <see cref="FootContactDetector"/>)
    /// and its **rendered** world position (i.e. after foot-lock IK, when
    /// measuring "after"; before IK, when measuring "before" — the same
    /// utility measures both sides of the comparison, only the input pose
    /// differs). <see cref="MeanPlanarDisplacementPerContactFrame"/> gives
    /// meters/frame averaged over every frame the foot was in contact,
    /// matching the "avant/après" comparison procedure documented in the
    /// package README.
    /// </remarks>
    public sealed class FootSlidingMetric
    {
        private Vector3 _previousPosition;
        private bool _hasPrevious;

        public double TotalPlanarDisplacement { get; private set; }
        public int ContactFrameCount { get; private set; }

        /// <summary>
        /// Feed one frame's world foot position + contact state. Only
        /// displacement between two **consecutive contact frames** counts
        /// (a transition frame in/out of contact is not counted, matching
        /// the "sliding while planted" definition in spec §5).
        /// </summary>
        public void Accumulate(Vector3 worldPosition, bool isInContact)
        {
            if (isInContact && _hasPrevious)
            {
                var dx = worldPosition.x - _previousPosition.x;
                var dz = worldPosition.z - _previousPosition.z;
                TotalPlanarDisplacement += Mathf.Sqrt(dx * dx + dz * dz);
                ContactFrameCount++;
            }

            _previousPosition = worldPosition;
            _hasPrevious = isInContact;
        }

        /// <summary>Mean planar displacement per contact frame (meters/frame), or 0 if never in contact.</summary>
        public double MeanPlanarDisplacementPerContactFrame =>
            ContactFrameCount > 0 ? TotalPlanarDisplacement / ContactFrameCount : 0.0;

        public void Reset()
        {
            TotalPlanarDisplacement = 0.0;
            ContactFrameCount = 0;
            _hasPrevious = false;
        }
    }
}
