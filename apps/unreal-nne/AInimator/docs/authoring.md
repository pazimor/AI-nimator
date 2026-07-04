# Authoring a binding — no code required (phase B3)

This page is the one-page recipe for "I want a key/button to make my
character play an animation", using `UAInimatorActionComponent`. It
assumes a `UAInimatorControllerRuntime` is already loaded on your Pawn
(see the main `README.md` "Placing a bundle" section) — this page only
covers wiring an input to a `ControlPreset`.

## 1. Add the component

1. Open your Pawn Blueprint (or a Blueprint subclass of
   `AAInimatorDemoPawn`, or any C++ actor with a
   `UAInimatorControllerRuntime` property).
2. **Add Component -> AInimator Action Component**.
3. In its Details panel, set **Runtime** to your actor's
   `UAInimatorControllerRuntime` instance (or leave it empty — the
   component auto-detects the first `UAInimatorControllerRuntime`
   property on the owning actor at `BeginPlay`).

## 2. Create a preset (or use a bundled one)

You have three ways to get a `UAInimatorControlPreset` to bind, all
without writing C++:

- **Use a bundle-provided one.** `UAInimatorControllerRuntime::
  GetBundledPresets()` already exposes the five defaults (`idle`,
  `forward`, `backward`, `strafe_left`, `strafe_right`) loaded from
  the bundle's `presets/` folder once `LoadBundle()` succeeds — grab
  one from there in Blueprint, no asset creation needed.
- **Click "+ Create New Preset"** in the Action Component's Details
  panel (the custom section added below the bindings list). This
  creates a new `UAInimatorControlPreset` asset (under
  `/Game/AInimatorPresets` by default — move it wherever you like)
  and appends a binding row already pointing at it.
- **Click "Import Presets From Bundle..."** to pick a bundle
  directory and save its `presets/*.json` as versioned project
  assets in one step (useful once you want to hand-tune the defaults
  and keep them under source control instead of re-reading them from
  the bundle every load).

Edit a preset's `Vx` / `Vz` (meters/frame, root-local — see
`apps/spec/control_preset.schema.json`) and, if the bundle's
`control_channels == 4`, `AimX` / `AimZ` directly in its Details panel.

## 3. Add a binding row

In the Action Component's Details panel:

1. Click **+ Add Binding** (or use the array's own "+" if you are
   using the default array UI instead of the custom row).
2. Pick **either**:
   - an **Action** — a `UInputAction` asset (Enhanced Input). Make
     sure it is mapped in a `UInputMappingContext` added to the
     player's Enhanced Input subsystem as usual (this component does
     not add the mapping context itself — only binds the actions you
     give it); **or**
   - a **Key** — a plain `FKey` (e.g. `W`), if you don't want to set
     up an `InputAction`/`InputMappingContext` yet. Action takes
     priority if both are set on the same row.
3. Pick the **Preset** this row activates.
4. Optionally set **Idle Preset** on the component itself — applied
   automatically when no bound key is currently held (Action-based
   bindings are event-driven and do not participate in this
   fallback; wire an explicit "released" binding or an idle
   `UInputAction` trigger if you need one).

## 4. Press Play

Hold the bound key, or trigger the bound `InputAction` — the
component calls `Runtime->SetPreset(...)` for you every time a
binding resolves, and the already-loaded runtime does the rest (one
NNE forward per `Tick()`, root motion integrated into world
position/yaw). No further code is required to see the character
respond.

## Reference — what each field maps to

| Details panel field | C++ type | Contract source |
|---|---|---|
| `Runtime` | `UAInimatorControllerRuntime*` | n/a (your loaded bundle) |
| `Bindings[i].Action` | `UInputAction*` | Enhanced Input asset |
| `Bindings[i].Key` | `FKey` | legacy fallback, polled in `Tick()` |
| `Bindings[i].Preset` | `UAInimatorControlPreset*` | `control_preset.schema.json` |
| `IdlePreset` | `UAInimatorControlPreset*` | applied when no key binding is held |

See `AInimatorActionComponent.h` / `AInimatorActionBinding.h` for the
full API (also exposed to Blueprint: `ActivatePreset`,
`ResolvePresetForAction`, `ResolvePresetForKey`).
