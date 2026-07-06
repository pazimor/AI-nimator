using UnityEngine;

namespace AInimator.Controller.Samples.RigDemo
{
    /// <summary>
    /// Minimal third-person follow camera for the RigDemo sample: keeps a
    /// fixed world-space offset behind/above the mannequin and looks at its
    /// chest height. Added by <see cref="RigDemoBootstrap"/>.
    /// </summary>
    public sealed class RigDemoFollowCamera : MonoBehaviour
    {
        private static readonly Vector3 Offset = new(0f, 1.7f, -3.5f);
        private static readonly Vector3 LookAtHeight = new(0f, 0.9f, 0f);

        [Tooltip("Transform the camera follows (the mannequin root).")]
        [SerializeField] private Transform target;

        /// <summary>The followed Transform (set by the bootstrap).</summary>
        public Transform Target
        {
            get => target;
            set => target = value;
        }

        private void LateUpdate()
        {
            if (target == null)
            {
                return;
            }

            transform.position = target.position + Offset;
            transform.LookAt(target.position + LookAtHeight);
        }
    }
}
