using UnityEngine;
#if AINIMATOR_INPUTSYSTEM && !ENABLE_LEGACY_INPUT_MANAGER
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Controls;
#endif

namespace AInimator.Controller.Authoring
{
    /// <summary>
    /// Keyboard polling that works under either of Unity's input backends,
    /// keeping the plugin's <see cref="KeyCode"/>-based authoring surface
    /// (<see cref="InputBinding"/>) valid regardless of the host project's
    /// <c>Active Input Handling</c> Player Setting:
    /// <list type="bullet">
    /// <item>legacy Input Manager (or "Both"): forwards to <see cref="Input"/>;</item>
    /// <item>Input System package only: polls <c>Keyboard.current</c>,
    /// translating the <see cref="KeyCode"/> (legacy <see cref="Input"/>
    /// APIs throw in this mode);</item>
    /// <item>neither available: returns <c>false</c>, never throws from a
    /// per-frame poll.</item>
    /// </list>
    /// </summary>
    /// <remarks>
    /// The Input System dependency is optional: the asmdef only defines
    /// <c>AINIMATOR_INPUTSYSTEM</c> when <c>com.unity.inputsystem</c> is
    /// installed (versionDefines), so projects without the package still
    /// compile on the legacy path.
    /// </remarks>
    public static class KeyInput
    {
#if ENABLE_LEGACY_INPUT_MANAGER
        /// <summary>True while the key is held (equivalent of <see cref="Input.GetKey(KeyCode)"/>).</summary>
        public static bool GetKey(KeyCode key) => Input.GetKey(key);

        /// <summary>True on the frame the key goes down (equivalent of <see cref="Input.GetKeyDown(KeyCode)"/>).</summary>
        public static bool GetKeyDown(KeyCode key) => Input.GetKeyDown(key);
#elif AINIMATOR_INPUTSYSTEM
        /// <summary>True while the key is held (equivalent of <see cref="Input.GetKey(KeyCode)"/>).</summary>
        public static bool GetKey(KeyCode key) => Resolve(key)?.isPressed ?? false;

        /// <summary>True on the frame the key goes down (equivalent of <see cref="Input.GetKeyDown(KeyCode)"/>).</summary>
        public static bool GetKeyDown(KeyCode key) => Resolve(key)?.wasPressedThisFrame ?? false;

        private static KeyControl Resolve(KeyCode keyCode)
        {
            var keyboard = Keyboard.current;
            if (keyboard == null)
            {
                return null;
            }

            var key = ToInputSystemKey(keyCode);
            return key == Key.None ? null : keyboard[key];
        }

        /// <summary>
        /// Translate a legacy <see cref="KeyCode"/> to the Input System's
        /// <see cref="Key"/>. Most names are identical (letters, arrows,
        /// Space, modifiers) and go through <c>Enum.TryParse</c>; digits and
        /// Return are named differently and are special-cased. Unknown keys
        /// map to <see cref="Key.None"/> (poll returns false, never throws).
        /// </summary>
        private static Key ToInputSystemKey(KeyCode keyCode)
        {
            if (keyCode == KeyCode.Alpha0)
            {
                return Key.Digit0;
            }

            if (keyCode >= KeyCode.Alpha1 && keyCode <= KeyCode.Alpha9)
            {
                // Key.Digit1..Digit9 are contiguous (keyboard row order).
                return Key.Digit1 + (keyCode - KeyCode.Alpha1);
            }

            if (keyCode >= KeyCode.Keypad0 && keyCode <= KeyCode.Keypad9)
            {
                return Key.Numpad0 + (keyCode - KeyCode.Keypad0);
            }

            if (keyCode == KeyCode.Return)
            {
                return Key.Enter;
            }

            return System.Enum.TryParse<Key>(keyCode.ToString(), out var key) ? key : Key.None;
        }
#else
        /// <summary>No input backend available: always false, never throws.</summary>
        public static bool GetKey(KeyCode key) => false;

        /// <summary>No input backend available: always false, never throws.</summary>
        public static bool GetKeyDown(KeyCode key) => false;
#endif
    }
}
