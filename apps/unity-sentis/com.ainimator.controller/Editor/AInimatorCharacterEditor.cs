using AInimator.Controller.Authoring;
using UnityEditor;
using UnityEngine;

namespace AInimator.Controller.Editor
{
    /// <summary>
    /// Custom Inspector for <see cref="AInimatorCharacter"/>: the default
    /// field drawer plus a Play-mode tester for the B6 free-text mapper
    /// (<c>apps/spec/text_to_control.md</c>) — a text field + "Resolve"/
    /// "Clear" buttons calling <see cref="AInimatorCharacter.SetTextCommand"/>/
    /// <see cref="AInimatorCharacter.ClearTextCommand"/> directly on the live
    /// component.
    /// </summary>
    [CustomEditor(typeof(AInimatorCharacter))]
    public sealed class AInimatorCharacterEditor : UnityEditor.Editor
    {
        private string _textCommandInput = "";

        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            DrawTextCommandTester();
        }

        private void DrawTextCommandTester()
        {
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Free-text command (B6, test in Play)", EditorStyles.boldLabel);

            var character = (AInimatorCharacter)target;
            using (new EditorGUILayout.HorizontalScope())
            {
                _textCommandInput = EditorGUILayout.TextField(_textCommandInput);
                using (new EditorGUI.DisabledScope(!Application.isPlaying))
                {
                    if (GUILayout.Button("Resolve", GUILayout.Width(70)))
                    {
                        character.SetTextCommand(_textCommandInput);
                    }

                    if (GUILayout.Button("Clear", GUILayout.Width(50)))
                    {
                        character.ClearTextCommand();
                    }
                }
            }

            if (!Application.isPlaying)
            {
                EditorGUILayout.HelpBox("Enter Play mode to test text-command resolution.", MessageType.Info);
            }
            else if (!string.IsNullOrEmpty(character.ActiveTextCommand))
            {
                EditorGUILayout.HelpBox($"Active text command: \"{character.ActiveTextCommand}\"", MessageType.None);
            }
        }
    }
}
