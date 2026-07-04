using NUnit.Framework;
using AInimator.Controller.Prompting;

namespace AInimator.Controller.Tests
{
    /// <summary>
    /// Tests for <see cref="PromptCrossFader"/>: linear interpolation in
    /// embedding space over the configured duration, and the "clear ->
    /// null_emb" semantics of <c>apps/spec/rig_binding.md</c> §3.
    /// </summary>
    public class PromptCrossFaderTests
    {
        private const int Channels = 3;

        [Test]
        public void SnapTo_SetsImmediateValue_NoFade()
        {
            var fader = new PromptCrossFader(Channels);
            float[] emb = { 1f, 2f, 3f };

            fader.SnapTo(emb);

            var result = new float[Channels];
            fader.Evaluate(result);

            Assert.That(result, Is.EqualTo(emb));
            Assert.That(fader.IsFading, Is.False);
        }

        [Test]
        public void BeginFadeTo_AtStart_EvaluatesToFromEmbedding()
        {
            var fader = new PromptCrossFader(Channels, crossFadeSeconds: 0.3f);
            float[] fromEmb = { 0f, 0f, 0f };
            fader.SnapTo(fromEmb);

            float[] toEmb = { 1f, 1f, 1f };
            var current = new float[Channels];
            fader.BeginFadeTo(toEmb, current);

            var result = new float[Channels];
            fader.Evaluate(result);

            Assert.That(result, Is.EqualTo(fromEmb).Within(1e-6f));
            Assert.That(fader.IsFading, Is.True);
        }

        [Test]
        public void Tick_Halfway_LerpsLinearlyBetweenFromAndTo()
        {
            var fader = new PromptCrossFader(Channels, crossFadeSeconds: 0.3f);
            fader.SnapTo(new float[] { 0f, 0f, 0f });

            var current = new float[Channels];
            fader.BeginFadeTo(new float[] { 2f, 4f, -2f }, current);
            fader.Tick(0.15f); // halfway through 0.3s

            var result = new float[Channels];
            fader.Evaluate(result);

            Assert.That(result[0], Is.EqualTo(1f).Within(1e-4f));
            Assert.That(result[1], Is.EqualTo(2f).Within(1e-4f));
            Assert.That(result[2], Is.EqualTo(-1f).Within(1e-4f));
        }

        [Test]
        public void Tick_PastDuration_SettlesOnTargetAndStopsFading()
        {
            var fader = new PromptCrossFader(Channels, crossFadeSeconds: 0.3f);
            fader.SnapTo(new float[] { 0f, 0f, 0f });

            var current = new float[Channels];
            float[] toEmb = { 5f, -5f, 0.5f };
            fader.BeginFadeTo(toEmb, current);
            fader.Tick(0.5f); // well past 0.3s

            var result = new float[Channels];
            fader.Evaluate(result);

            Assert.That(result, Is.EqualTo(toEmb).Within(1e-6f));
            Assert.That(fader.IsFading, Is.False);
        }

        [Test]
        public void BeginFadeTo_MidFade_StartsFromCurrentInterpolatedValue_NotOriginalFrom()
        {
            // Retargeting the fade mid-flight (e.g. rapid prompt changes)
            // must not pop back to the very first "from" value.
            var fader = new PromptCrossFader(Channels, crossFadeSeconds: 0.3f);
            fader.SnapTo(new float[] { 0f, 0f, 0f });

            var current = new float[Channels];
            fader.BeginFadeTo(new float[] { 10f, 10f, 10f }, current);
            fader.Tick(0.15f); // halfway: current value is (5, 5, 5)

            // Redirect toward a third embedding mid-fade.
            fader.BeginFadeTo(new float[] { 0f, 0f, 0f }, current);

            var result = new float[Channels];
            fader.Evaluate(result);

            Assert.That(result[0], Is.EqualTo(5f).Within(1e-3f));
            Assert.That(result[1], Is.EqualTo(5f).Within(1e-3f));
            Assert.That(result[2], Is.EqualTo(5f).Within(1e-3f));
        }

        [Test]
        public void Constructor_ZeroOrNegativeChannels_Throws()
        {
            Assert.Throws<System.ArgumentException>(() => new PromptCrossFader(0));
            Assert.Throws<System.ArgumentException>(() => new PromptCrossFader(-1));
        }

        [Test]
        public void SnapTo_WrongLength_Throws()
        {
            var fader = new PromptCrossFader(Channels);
            Assert.Throws<System.ArgumentException>(() => fader.SnapTo(new float[] { 1f, 2f }));
        }
    }
}
