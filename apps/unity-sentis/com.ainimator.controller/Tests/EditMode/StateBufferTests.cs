using AInimator.Controller.Runtime;
using NUnit.Framework;

namespace AInimator.Controller.Tests
{
    /// <summary>Unit tests for the autoregressive ring buffer.</summary>
    public class StateBufferTests
    {
        [Test]
        public void SeedUniform_FillsEveryFrameWithTheSameSeed()
        {
            var buffer = new StateBuffer(contextFrames: 3, boneFrameLength: 2, globalFrameLength: 1);
            float[] boneSeed = { 1f, 2f };
            float[] globalSeed = { 9f };

            buffer.SeedUniform(boneSeed, globalSeed);

            Assert.IsTrue(buffer.IsFull);
            var boneWindow = new float[3 * 2];
            buffer.CopyBoneWindowOrdered(boneWindow);
            CollectionAssert.AreEqual(new float[] { 1f, 2f, 1f, 2f, 1f, 2f }, boneWindow);

            var globalWindow = new float[3 * 1];
            buffer.CopyGlobalWindowOrdered(globalWindow);
            CollectionAssert.AreEqual(new float[] { 9f, 9f, 9f }, globalWindow);
        }

        [Test]
        public void Push_EvictsOldestFrame_KeepsOldestToNewestOrder()
        {
            var buffer = new StateBuffer(contextFrames: 3, boneFrameLength: 1, globalFrameLength: 1);
            buffer.SeedUniform(new float[] { 0f }, new float[] { 0f });

            buffer.Push(new float[] { 1f }, new float[] { 10f });
            buffer.Push(new float[] { 2f }, new float[] { 20f });

            var boneWindow = new float[3];
            buffer.CopyBoneWindowOrdered(boneWindow);
            // Oldest seed frame (0) has been pushed out twice; window is [0, 1, 2].
            CollectionAssert.AreEqual(new float[] { 0f, 1f, 2f }, boneWindow);
        }

        [Test]
        public void Push_AfterWrapAround_StillOrdersOldestToNewest()
        {
            var buffer = new StateBuffer(contextFrames: 2, boneFrameLength: 1, globalFrameLength: 1);
            buffer.SeedUniform(new float[] { 0f }, new float[] { 0f });

            buffer.Push(new float[] { 1f }, new float[] { 0f });
            buffer.Push(new float[] { 2f }, new float[] { 0f });
            buffer.Push(new float[] { 3f }, new float[] { 0f });

            var boneWindow = new float[2];
            buffer.CopyBoneWindowOrdered(boneWindow);
            CollectionAssert.AreEqual(new float[] { 2f, 3f }, boneWindow);
        }

        [Test]
        public void CopyLatestBoneFrame_ReturnsMostRecentlyPushedFrame()
        {
            var buffer = new StateBuffer(contextFrames: 2, boneFrameLength: 2, globalFrameLength: 1);
            buffer.SeedUniform(new float[] { 0f, 0f }, new float[] { 0f });
            buffer.Push(new float[] { 5f, 6f }, new float[] { 0f });

            var latest = new float[2];
            buffer.CopyLatestBoneFrame(latest);

            CollectionAssert.AreEqual(new float[] { 5f, 6f }, latest);
        }

        [Test]
        public void Push_WrongLength_Throws()
        {
            var buffer = new StateBuffer(contextFrames: 2, boneFrameLength: 2, globalFrameLength: 1);
            buffer.SeedUniform(new float[] { 0f, 0f }, new float[] { 0f });

            Assert.Throws<System.ArgumentException>(() =>
                buffer.Push(new float[] { 1f }, new float[] { 0f }));
        }
    }
}
