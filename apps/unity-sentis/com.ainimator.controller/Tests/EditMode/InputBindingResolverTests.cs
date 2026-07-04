using System.Collections.Generic;
using AInimator.Controller.Authoring;
using AInimator.Controller.Presets;
using NUnit.Framework;
using UnityEngine;

namespace AInimator.Controller.Tests
{
    /// <summary>
    /// Unit tests for <see cref="InputBindingResolver"/> — the pure logic
    /// extracted from <see cref="AInimatorActionBinder"/> so key→preset
    /// resolution is testable without a live <c>Input</c> subsystem.
    /// </summary>
    public class InputBindingResolverTests
    {
        private static ControlPreset NewPreset(string name)
        {
            var preset = ScriptableObject.CreateInstance<ControlPreset>();
            preset.name = name;
            return preset;
        }

        [Test]
        public void Resolve_ReturnsFallback_WhenNoKeyHeld()
        {
            var forward = NewPreset("forward");
            var idle = NewPreset("idle");
            var bindings = new List<InputBinding> { new(KeyCode.W, forward) };

            var resolved = InputBindingResolver.Resolve(bindings, _ => false, idle);

            Assert.That(resolved, Is.SameAs(idle));
        }

        [Test]
        public void Resolve_ReturnsBoundPreset_WhenKeyHeld()
        {
            var forward = NewPreset("forward");
            var idle = NewPreset("idle");
            var bindings = new List<InputBinding> { new(KeyCode.W, forward) };

            var resolved = InputBindingResolver.Resolve(bindings, key => key == KeyCode.W, idle);

            Assert.That(resolved, Is.SameAs(forward));
        }

        [Test]
        public void Resolve_FirstMatchWins_WhenMultipleKeysHeld()
        {
            var forward = NewPreset("forward");
            var strafeLeft = NewPreset("strafe_left");
            var idle = NewPreset("idle");
            var bindings = new List<InputBinding>
            {
                new(KeyCode.W, forward),
                new(KeyCode.Q, strafeLeft),
            };

            // Both W and Q are "held": the first row (forward) must win.
            var resolved = InputBindingResolver.Resolve(
                bindings, key => key == KeyCode.W || key == KeyCode.Q, idle);

            Assert.That(resolved, Is.SameAs(forward));
        }

        [Test]
        public void Resolve_SkipsInvalidRows_MissingPresetOrKey()
        {
            var forward = NewPreset("forward");
            var idle = NewPreset("idle");
            var bindings = new List<InputBinding>
            {
                new(KeyCode.None, forward), // invalid: no key
                new(KeyCode.E, null),       // invalid: no preset
                new(KeyCode.W, forward),
            };

            var resolved = InputBindingResolver.Resolve(
                bindings, key => key == KeyCode.E || key == KeyCode.W, idle);

            Assert.That(resolved, Is.SameAs(forward));
        }

        [Test]
        public void Resolve_HandlesNullOrEmptyBindingList()
        {
            var idle = NewPreset("idle");

            Assert.That(InputBindingResolver.Resolve(null, _ => true, idle), Is.SameAs(idle));
            Assert.That(InputBindingResolver.Resolve(new List<InputBinding>(), _ => true, idle), Is.SameAs(idle));
        }

        [Test]
        public void FindPresetForKey_ReturnsNull_WhenNoBindingMatches()
        {
            var bindings = new List<InputBinding> { new(KeyCode.W, NewPreset("forward")) };

            Assert.That(InputBindingResolver.FindPresetForKey(bindings, KeyCode.S), Is.Null);
        }

        [Test]
        public void FindPresetForKey_ReturnsBoundPreset()
        {
            var forward = NewPreset("forward");
            var bindings = new List<InputBinding> { new(KeyCode.W, forward) };

            Assert.That(InputBindingResolver.FindPresetForKey(bindings, KeyCode.W), Is.SameAs(forward));
        }

        [Test]
        public void InputBinding_IsValid_RequiresBothKeyAndPreset()
        {
            var withBoth = new InputBinding(KeyCode.W, NewPreset("forward"));
            var missingKey = new InputBinding(KeyCode.None, NewPreset("forward"));
            var missingPreset = new InputBinding(KeyCode.W, null);

            Assert.That(withBoth.IsValid, Is.True);
            Assert.That(missingKey.IsValid, Is.False);
            Assert.That(missingPreset.IsValid, Is.False);
        }

        [Test]
        public void InputBinding_Serializes_KeyPresetAndLabel()
        {
            var preset = NewPreset("forward");
            var binding = new InputBinding(KeyCode.W, preset, "Move Forward");

            var json = JsonUtility.ToJson(binding);
            var roundTripped = JsonUtility.FromJson<InputBinding>(json);

            Assert.That(roundTripped.Key, Is.EqualTo(KeyCode.W));
            Assert.That(roundTripped.Label, Is.EqualTo("Move Forward"));
            // Object references do not round-trip through JsonUtility outside
            // Unity's serialized-object graph (expected — this only asserts
            // the value-type fields, which is what SerializeField promises).
        }
    }
}
