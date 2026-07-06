# Authoring a binding — no code required

This page covers the Goal B phase B3 workflow: mapping a keyboard button to
an animation-driving action, entirely from the Unity Inspector. It assumes a
controller bundle is already posed under `StreamingAssets/AInimatorBundle/`
(see the main `README.md`, "Posing a bundle for the demo").

## 1. Create or import `ControlPreset` assets

You need at least one `ControlPreset` asset per action (`forward`,
`backward`, `strafe_left`, `strafe_right`, `idle`, or any custom one your
prompt/control combination represents).

**Import the bundle's defaults** (fastest path — five presets ready to use):

- Menu **AInimator → Import Presets From Bundle...**, or the **"Import
  Presets From Bundle..."** button in the `AInimatorActionBinder` Inspector.
- Pick the bundle directory (the one containing `presets/*.json`).
- Assets land in `Assets/AInimatorPresets/` (`idle.asset`, `forward.asset`,
  `backward.asset`, `strafe_left.asset`, `strafe_right.asset`), fully
  editable afterwards like any other asset.

**Create a brand-new preset** (e.g. a custom prompt-conditioned action):

- In the `AInimatorActionBinder` Inspector, click **"Create New Preset..."**.
- Name it (snake_case recommended, matches
  `apps/spec/control_preset.schema.json`'s `name` pattern) and save it under
  `Assets/AInimatorPresets/`.
- A new binding row is added automatically, already pointing at the fresh
  preset — you only need to pick a key next.
- Fill in `Vx`/`Vz` (meters/frame, root-local frame; `+Vz` = forward) and,
  if the bundle's manifest declares `control_channels == 4`, `Has Aim` +
  `AimX`/`AimZ`. Leave `PromptEmb` empty unless you have a precomputed
  embedding for this action (`apps/spec/inference_contract.md` §4).

## 2. Add `AInimatorActionBinder` to a GameObject

- Add Component → **AInimator Controller → Action Binder**
  (`AInimator.Controller.Authoring.AInimatorActionBinder`).
- Assign **Idle Preset** — the preset applied whenever no bound key is held.
- Leave **Auto-Initialize From Default Bundle** checked to have the
  component load the bundle from `StreamingAssets/AInimatorBundle/` in
  `Awake()`. Uncheck it (and call `Initialize(bundle)` from your own
  bootstrap script) if another component already owns bundle loading (as
  `CapsuleDemoBootstrap` does in the B1 sample).

## 3. Add binding rows

- Click **"+ Add Binding"** to append a row.
- Each row has three fields: **Key** (a `KeyCode` dropdown), **Preset** (a
  `ControlPreset` object picker — drag or select any asset in the project),
  and an optional cosmetic **Label**.
- Rows are evaluated top-to-bottom; the **first held key wins**. A row
  missing either a key or a preset is skipped (never crashes, never
  silently picks a wrong default).
- Click the **"x"** button to remove a row.

A typical five-row table (mirrors `ROADMAP_PLUGINS.md` §1.1's binding
table):

| Key | Preset |
|---|---|
| `W` | `forward` |
| `S` | `backward` |
| `Q` | `strafe_left` |
| `E` | `strafe_right` |
| *(Idle Preset field, not a row)* | `idle` |

## 4. Press Play

With at least one valid binding and an `Idle Preset` assigned, pressing the
mapped key immediately drives the controller with that preset's control
vector; releasing it falls back to `Idle Preset`. `AInimatorActionBinder`
exposes `ActivePreset` (read-only) and `Controller` (the underlying
`AInimatorController`) for any renderer/animator component that needs to
react to the resolved state — e.g. to read `Controller.RootPosition` /
`Controller.RootYawRadians` as `CapsuleDemoController` does, or to feed
`Controller.Tick`'s returned bone frame into a skinned rig.

Continuous aim (mouse/stick) is not a binding row: call
`SetContinuousAim(Vector2)` once per frame from your own input code when the
manifest declares `control_channels == 4`; it overrides the active preset's
static aim for that frame only.

## 5. Free-text control commands (B6) — not the prompt channel

`apps/spec/text_to_control.md` defines a **separate, low-level** mapper: a
free-text sentence ("cours vers la gauche", "run left") resolves
deterministically to a raw `(vx, vz[, aimX, aimZ])` control vector — the
same units and the same write path as a `ControlPreset`
(`ControlPreset.WriteRawControl`). It is a keyword-table lookup (directions
summed + unit-normalized, speed = max of present keywords, stop-family
always wins), **not** an embedding and **not** the `prompt`/`prompt_emb`
channel described in `rig_binding.md` §3 — the two are unrelated features
that happen to both start from text.

- Call `AInimatorActionBinder.SetTextCommand(string)` or
  `AInimatorCharacter.SetTextCommand(string)` — returns `true` once resolved
  and now driving the controller every frame, `false` if the sentence was
  unrecognized or ambiguous (e.g. "gauche droite" — direction words cancel
  out). A failed call **never** changes the current control and logs a
  warning (never a silent fallback).
- A held key binding still takes priority over an active text command; the
  preset/binding path remains the default. Call `ClearTextCommand()` to
  release control back to bindings/idle.
- Test it live: select the `AInimatorActionBinder`/`AInimatorCharacter`
  GameObject in Play mode, type a sentence into the **"Free-text command
  (B6, test in Play)"** field at the bottom of its Inspector, click
  **Resolve**.
- The keyword table (`Runtime/TextCommand/Resources/text_to_control.json`)
  is a verbatim embedded copy of `apps/spec/text_to_control.json` — the
  canonical source of truth; do not edit the embedded copy directly, change
  the spec file and re-copy.

## Not covered here

- Foot-lock IK / idle↔move blending post-processing — see the B4 section of
  the main `README.md` and `apps/spec/footlock_blending.md`.
- Mapping a free-text prompt to a **preset embedding** (`prompt_emb`) — a
  `ControlPreset` can carry a precomputed `prompt_emb` (imported or
  hand-authored), but there is no in-Inspector text→embedding tool; that
  stays a Python-side authoring step (`apps/spec/inference_contract.md` §4).
  This is unrelated to the B6 free-text **control** mapper above.
