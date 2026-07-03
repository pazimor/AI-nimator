using System;

namespace AInimator.Controller.Runtime
{
    /// <summary>
    /// Rolling autoregressive state window of length <c>context_frames</c>
    /// (manifest.json). Holds the raw (un-normalized) bone rotation6d and
    /// root-local global-motion history the engine feeds to the controller
    /// each frame.
    /// </summary>
    /// <remarks>
    /// Mirrors the Python rollout's <c>boneHistory</c> / <c>globalHistory</c>
    /// deques (<c>ainimator.model.controller_rollout</c>), but as a
    /// pre-allocated ring buffer: no per-frame heap allocation, so it is
    /// safe on the ≥60 fps hot path (Goal B §3 performance rule).
    /// </remarks>
    public sealed class StateBuffer
    {
        private readonly int _contextFrames;
        private readonly int _boneFrameLength;
        private readonly int _globalFrameLength;

        // Flat ring buffers: frame i's bone data lives at
        // [i * _boneFrameLength, (i+1) * _boneFrameLength).
        private readonly float[] _boneRing;
        private readonly float[] _globalRing;
        private int _writeIndex;
        private int _framesWritten;

        public int ContextFrames => _contextFrames;
        public int BoneFrameLength => _boneFrameLength;
        public int GlobalFrameLength => _globalFrameLength;

        /// <summary>True once the window has been fully seeded (context_frames pushes).</summary>
        public bool IsFull => _framesWritten >= _contextFrames;

        public StateBuffer(int contextFrames, int boneFrameLength, int globalFrameLength)
        {
            if (contextFrames < 1)
            {
                throw new ArgumentException("contextFrames must be >= 1.", nameof(contextFrames));
            }

            _contextFrames = contextFrames;
            _boneFrameLength = boneFrameLength;
            _globalFrameLength = globalFrameLength;
            _boneRing = new float[contextFrames * boneFrameLength];
            _globalRing = new float[contextFrames * globalFrameLength];
            _writeIndex = 0;
            _framesWritten = 0;
        }

        /// <summary>
        /// Seed every slot of the window with the same initial frame (used
        /// once at spawn, before any inference has run).
        /// </summary>
        public void SeedUniform(ReadOnlySpan<float> boneFrame, ReadOnlySpan<float> globalFrame)
        {
            for (var i = 0; i < _contextFrames; i++)
            {
                boneFrame.CopyTo(_boneRing.AsSpan(i * _boneFrameLength, _boneFrameLength));
                globalFrame.CopyTo(_globalRing.AsSpan(i * _globalFrameLength, _globalFrameLength));
            }

            _writeIndex = 0;
            _framesWritten = _contextFrames;
        }

        /// <summary>Push one new frame, evicting the oldest (ring semantics).</summary>
        public void Push(ReadOnlySpan<float> boneFrame, ReadOnlySpan<float> globalFrame)
        {
            if (boneFrame.Length != _boneFrameLength)
            {
                throw new ArgumentException(
                    $"boneFrame length must be {_boneFrameLength}; got {boneFrame.Length}.");
            }

            if (globalFrame.Length != _globalFrameLength)
            {
                throw new ArgumentException(
                    $"globalFrame length must be {_globalFrameLength}; got {globalFrame.Length}.");
            }

            boneFrame.CopyTo(_boneRing.AsSpan(_writeIndex * _boneFrameLength, _boneFrameLength));
            globalFrame.CopyTo(_globalRing.AsSpan(_writeIndex * _globalFrameLength, _globalFrameLength));

            _writeIndex = (_writeIndex + 1) % _contextFrames;
            if (_framesWritten < _contextFrames)
            {
                _framesWritten++;
            }
        }

        /// <summary>
        /// Copy the window into <paramref name="destination"/> in oldest-to-newest
        /// order, shape <c>(contextFrames * boneFrameLength)</c> — exactly the
        /// row-major layout the <c>bone_window</c> ONNX input expects.
        /// </summary>
        public void CopyBoneWindowOrdered(Span<float> destination)
        {
            CopyRingOrdered(_boneRing, _boneFrameLength, destination);
        }

        /// <summary>
        /// Copy the window into <paramref name="destination"/> in oldest-to-newest
        /// order, shape <c>(contextFrames * globalFrameLength)</c> — the
        /// <c>global_window</c> ONNX input layout.
        /// </summary>
        public void CopyGlobalWindowOrdered(Span<float> destination)
        {
            CopyRingOrdered(_globalRing, _globalFrameLength, destination);
        }

        /// <summary>The most recently pushed (or seeded) bone frame.</summary>
        public void CopyLatestBoneFrame(Span<float> destination)
        {
            var latestSlot = (_writeIndex - 1 + _contextFrames) % _contextFrames;
            _boneRing.AsSpan(latestSlot * _boneFrameLength, _boneFrameLength).CopyTo(destination);
        }

        private void CopyRingOrdered(float[] ring, int frameLength, Span<float> destination)
        {
            var expectedLength = _contextFrames * frameLength;
            if (destination.Length != expectedLength)
            {
                throw new ArgumentException(
                    $"destination length must be {expectedLength}; got {destination.Length}.");
            }

            // Oldest frame is at _writeIndex when the ring is full (the next
            // slot to be overwritten is the oldest). Read starting there,
            // wrapping around, so the destination is oldest-to-newest.
            var oldestSlot = _framesWritten < _contextFrames ? 0 : _writeIndex;
            for (var i = 0; i < _contextFrames; i++)
            {
                var slot = (oldestSlot + i) % _contextFrames;
                ring.AsSpan(slot * frameLength, frameLength)
                    .CopyTo(destination.Slice(i * frameLength, frameLength));
            }
        }
    }
}
