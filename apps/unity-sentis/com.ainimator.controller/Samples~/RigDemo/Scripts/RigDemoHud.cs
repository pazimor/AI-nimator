using System.Collections.Generic;
using AInimator.Controller.Authoring;
using AInimator.Controller.Presets;
using UnityEngine;

namespace AInimator.Controller.Samples.RigDemo
{
    /// <summary>
    /// In-game HUD for the B3-bis acceptance checks (<c>apps/spec/rig_binding.md</c>
    /// §3/§5): number keys 1..9 hot-swap the active prompt embedding
    /// (<see cref="AInimatorCharacter.SetPrompt"/>, embedding-space
    /// cross-fade), 0 returns to the learned null embedding, and two text
    /// fields exercise the free-text paths — B7 prompt encoding
    /// (<see cref="AInimatorCharacter.SetPromptText"/>) and B6 text commands
    /// (<see cref="AInimatorCharacter.SetTextCommand"/>).
    /// </summary>
    /// <remarks>
    /// Added and initialized by <see cref="RigDemoBootstrap"/>; not meant to
    /// be placed by hand. Uses immediate-mode <see cref="OnGUI"/> to keep the
    /// sample free of UI-package dependencies. Note: movement keys keep
    /// firing while a text field has focus (the runtime polls the keyboard
    /// globally via <see cref="KeyInput"/>) — release WASD before typing.
    /// </remarks>
    public sealed class RigDemoHud : MonoBehaviour
    {
        private const int MaxPromptKeys = 9;

        private AInimatorCharacter _character;
        private IReadOnlyList<ControlPreset> _promptPresets;
        private string _promptText = "";
        private string _commandText = "";
        private string _status = "";

        /// <summary>Wire the HUD to the demo character (called by the bootstrap).</summary>
        public void Initialize(AInimatorCharacter character, IReadOnlyList<ControlPreset> promptPresets)
        {
            _character = character;
            _promptPresets = promptPresets;
        }

        private void Update()
        {
            if (_character == null || _promptPresets == null)
            {
                return;
            }

            for (var i = 0; i < _promptPresets.Count && i < MaxPromptKeys; i++)
            {
                if (KeyInput.GetKeyDown(KeyCode.Alpha1 + i))
                {
                    _character.SetPrompt(_promptPresets[i]);
                    _status = $"Prompt -> preset '{_promptPresets[i].PresetName}' (cross-fade)";
                }
            }

            if (KeyInput.GetKeyDown(KeyCode.Alpha0))
            {
                _character.ClearPrompt();
                _status = "Prompt -> learned null embedding";
            }
        }

        private void OnGUI()
        {
            GUILayout.BeginArea(new Rect(10f, 10f, 380f, 380f), GUI.skin.box);
            GUILayout.Label("<b>AI-nimator Rig Demo (B3-bis)</b>", RichLabel());
            GUILayout.Label("Move: WASD / ZQSD / arrows (hold)");
            DrawPromptKeyLegend();
            DrawPromptTextRow();
            DrawTextCommandRow();
            DrawStateLabels();
            GUILayout.EndArea();
        }

        private void DrawPromptKeyLegend()
        {
            if (_promptPresets == null || _promptPresets.Count == 0)
            {
                GUILayout.Label("No preset in this bundle ships a prompt embedding.");
                return;
            }

            for (var i = 0; i < _promptPresets.Count && i < MaxPromptKeys; i++)
            {
                GUILayout.Label($"[{i + 1}] prompt: {_promptPresets[i].PresetName}");
            }

            GUILayout.Label("[0] clear prompt (null embedding)");
        }

        private void DrawPromptTextRow()
        {
            GUILayout.Space(6f);
            GUILayout.Label("Free-text prompt (B7, in-engine encoder):");
            GUILayout.BeginHorizontal();
            _promptText = GUILayout.TextField(_promptText);
            if (GUILayout.Button("Set", GUILayout.Width(60f)))
            {
                _status = _character.SetPromptText(_promptText)
                    ? $"Prompt text -> '{_promptText}'"
                    : "Prompt text rejected (no text encoder in bundle? see Console)";
            }

            GUILayout.EndHorizontal();
        }

        private void DrawTextCommandRow()
        {
            GUILayout.Space(6f);
            GUILayout.Label("Text command (B6, e.g. 'avance' / 'walk forward'):");
            GUILayout.BeginHorizontal();
            _commandText = GUILayout.TextField(_commandText);
            if (GUILayout.Button("Run", GUILayout.Width(60f)))
            {
                _status = _character.SetTextCommand(_commandText)
                    ? $"Command -> '{_commandText}'"
                    : "Command not resolved (see Console)";
            }

            if (GUILayout.Button("Stop", GUILayout.Width(60f)))
            {
                _character.ClearTextCommand();
                _status = "Command cleared";
            }

            GUILayout.EndHorizontal();
        }

        private void DrawStateLabels()
        {
            GUILayout.Space(6f);
            var presetName = _character.ActivePreset != null ? _character.ActivePreset.PresetName : "(none)";
            GUILayout.Label($"Active preset: {presetName}");
            GUILayout.Label($"Active command: {_character.ActiveTextCommand ?? "(none)"}");
            if (!string.IsNullOrEmpty(_status))
            {
                GUILayout.Label(_status);
            }
        }

        private static GUIStyle RichLabel()
        {
            return new GUIStyle(GUI.skin.label) { richText = true };
        }
    }
}
