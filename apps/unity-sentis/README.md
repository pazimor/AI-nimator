# AI-nimator Controller — Unity (Sentis) plugin

Real-time runtime for the AI-nimator deterministic autoregressive
controller inside Unity, powered by [Sentis](https://docs.unity3d.com/Packages/com.unity.sentis@latest).
This package **defines no model** — it loads a frozen controller bundle
(`controller.onnx` + `norm_stats.json` + `manifest.json` + `presets/`)
produced by the `ainimator` Python export pipeline and drives it one
forward pass per frame. The single source of truth for the I/O contract is
`apps/spec/inference_contract.md` — this plugin implements it, it does not
redefine it.

Status: **Phases B1–B4** (`doc/ROADMAP_PLUGINS.md` §4) — runtime + WASD demo
(B1), the authoring `AInimatorActionBinder` (B3), and foot-lock IK + idle/move
blending post-processing (B4) are implemented. The build orchestrator (B5)
is the remaining later phase.

## Package layout

```
apps/unity-sentis/
├── README.md                      (this file)
├── .gitignore                     (bundle resource folder is never committed)
└── com.ainimator.controller/      the UPM package
    ├── package.json
    ├── Runtime/                   engine-agnostic-ish runtime (no UnityEditor deps)
    │   ├── Bundle/                Manifest, NormStats, BundleLoader, ControlPresetData, MiniJson
    │   ├── Normalization/         Normalizer (pure math, unit-testable)
    │   ├── Runtime/               StateBuffer, RootMotionIntegrator, ControllerRuntime, AInimatorController
    │   ├── Presets/               ControlPreset ScriptableObject
    │   ├── Authoring/             InputBinding, InputBindingResolver, AInimatorActionBinder (B3)
    │   └── PostProcess/           Smpl22Skeleton, SmplForwardKinematics, FootContactDetector,
    │                              TwoBoneIkSolver, FootLockIk, IdleMoveBlender, AInimatorPostProcess,
    │                              FootSlidingMetric (B4)
    ├── Editor/                    ControlPresetImporter, BundleDeliveryTools (batch-mode pack target),
    │                              AInimatorActionBinderEditor (B3 custom Inspector)
    ├── docs/authoring.md          1-page "create a binding with no code" walkthrough (B3)
    ├── Tests/EditMode/            NUnit tests (Normalizer, StateBuffer, BundleLoader, RootMotionIntegrator,
    │                              InputBindingResolver, FootContactDetector, TwoBoneIkSolver,
    │                              SmplForwardKinematics, FootLockIk, IdleMoveBlender, FootSlidingMetric)
    └── Samples~/ControllerDemo/   WASD capsule demo (no IK, no skinning — B1 scope)
```

## Installing the package

Add it to a Unity project (2023.2+) as a local/embedded package, e.g. in
the host project's `Packages/manifest.json`:

```json
{
  "dependencies": {
    "com.ainimator.controller": "file:../AI-nimator/apps/unity-sentis/com.ainimator.controller",
    "com.unity.sentis": "1.6.0"
  }
}
```

(Adjust the relative path to wherever the host project sits relative to
this repo checkout.) Unity will pull in Sentis as a transitive dependency
declared in `package.json`.

## Posing a bundle for the demo

The plugin never ships a `.onnx` — the bundle is a **build artifact**,
regenerated fresh from a checkpoint by the (future, B5) `apps/build/`
orchestrator. Until that orchestrator exists, pose a bundle manually:

1. Export a bundle from the `ainimator` repo:
   ```bash
   $AIPY -m ainimator.cli.export_onnx bundle \
     --checkpoint <path/to/checkpoint.pt> \
     --output-dir output/reference_bundle
   ```
   (See `apps/spec/inference_contract.md` §7 for the exact reference
   command used to produce `output/reference_bundle/`.)
2. Copy the bundle directory into your **host Unity project's**
   `Assets/StreamingAssets/AInimatorBundle/`, so the final layout is:
   ```
   Assets/StreamingAssets/AInimatorBundle/
   ├── controller.onnx
   ├── manifest.json
   ├── norm_stats.json
   └── presets/
       ├── idle.json
       ├── forward.json
       ├── backward.json
       ├── strafe_left.json
       └── strafe_right.json
   ```
   This path is `AInimator.Controller.Bundle.BundlePaths.DefaultBundleDirectory`
   (`Application.streamingAssetsPath/AInimatorBundle`), read by
   `BundleLoader.Load` at runtime.
3. **Never commit this folder.** `apps/unity-sentis/.gitignore` (and the
   repo root `.gitignore`, which blanket-ignores `*.onnx`) exclude it.

`BundleLoader.Load` validates the manifest before any inference runs:
incompatible `bundle_version`, wrong frozen dims (`state_channels`,
`num_bones`, …), or a missing/malformed `norm_stats.json` all raise a
`BundleLoadException` with an explicit message — never a silent fallback
(Goal B verite #4).

## Running the WASD demo

1. Create (or open) a Unity scene in the host project.
2. Add an empty `GameObject`, attach
   `AInimator.Controller.Samples.ControllerDemo.CapsuleDemoBootstrap`
   (import the **Controller Demo (WASD capsule)** sample from the
   package's entry in the Package Manager window, or reference the script
   directly from `Samples~/ControllerDemo/Scripts/`).
3. Press Play. The bootstrap spawns a capsule + ground plane, loads the
   bundle from `StreamingAssets/AInimatorBundle/`, and wires up
   `CapsuleDemoController`:

   | Key | Preset |
   |---|---|
   | `W` / `↑` | `forward` |
   | `S` / `↓` | `backward` |
   | `Q` | `strafe_left` |
   | `E` | `strafe_right` |
   | *(nothing held)* | `idle` |

   The capsule's `Transform` follows the controller's integrated
   world-space root trajectory (position + yaw). No foot-lock IK, no
   skinning — the demo only visualizes the root trajectory, which is
   sufficient for the B1 acceptance criteria (fps + trajectory parity + no
   NaN over a 60 s rollout). Wiring the full 22-bone pose onto a skinned
   humanoid rig is a natural follow-up sample, not required here.

## Runtime architecture (mirrors `ROADMAP_PLUGINS.md` §3.3)

```
prompt_emb = preset.PromptEmb ?? bundle.NormStats.PromptNullEmb   // never zeros
control    = preset.ToRawControl() [+ continuous aim override]
             AInimatorController.Tick(preset):
               ControllerRuntime.Tick(stateBuffer, control, promptEmb, phase, out boneDelta, out globalDelta)
                 Normalizer.Encode*  -> Sentis Worker.Schedule() (one forward) -> Normalizer.Decode*
               RootMotionIntegrator.Integrate(globalDelta)   // yaw-cumulated world position
               stateBuffer.Push(newBoneFrame, globalDelta)   // autoregression
```

One Sentis forward pass per `Tick`, no loop inside the ONNX graph. The
autoregressive window (`context_frames`), the z-normalization, and the
`Δstate → state` integration all live in C#, matching
`ainimator.model.controller_rollout` byte-for-byte in intent (yaw-relative
integration formula, control/state/delta normalization).

## Authoring: button → action (phase B3)

`AInimator.Controller.Authoring.AInimatorActionBinder` is the "no-code"
entry point described in `ROADMAP_PLUGINS.md` §1.1/§4: a list of
`InputBinding` rows (`KeyCode` → `ControlPreset`), resolved every frame
(first-held-key-wins) into the preset applied to an owned
`AInimatorController`. A custom Inspector
(`Editor/AInimatorActionBinderEditor.cs`) lets you add/remove rows, pick a
`ControlPreset` asset per row from a project-wide dropdown, create a brand
new preset asset in one click ("Create New Preset..."), or bulk-import the
bundle's five default presets ("Import Presets From Bundle...", the same
`ControlPresetImporter` used by the `AInimator → Import Presets From
Bundle...` menu item). See **`docs/authoring.md`** for the full
click-by-click walkthrough (create a binding, press Play, see the character
respond — no C# file touched).

`InputBindingResolver` holds the pure key→preset resolution logic
separately from the `MonoBehaviour` so it is unit-testable without a live
`Input` subsystem (`Tests/EditMode/InputBindingResolverTests.cs`).

## Post-processing: foot-lock IK + idle/move blending (phase B4)

`AInimator.Controller.PostProcess.AInimatorPostProcess` implements
`apps/spec/footlock_blending.md` verbatim — that document is normative for
every rule/threshold/order below; this package does not redefine any of it.
Pipeline order (spec §1, unchanged):

```
dstate -> state (AInimatorController.Tick, integrates + pushes RAW state into StateBuffer)
pose   = AInimatorPostProcess.Process(rawBoneFrame, rootWorldPosition, rawVx, rawVz, dt)
           SmplForwardKinematics.ComputeJointPositions   // root-local FK (ports ops.py::rot6dToJointXYZ)
           FootContactDetector.Update(footJoint 10/11)    // height<0.05m AND planar speed<0.01 m/frame, +hysteresis
           FootLockIk.Resolve(per leg: chain 1->4->7 / 2->5->8) // capture, two-bone solve, clamp 0.3m, 0.1s fade-out
           IdleMoveBlender.Update(rawVx, rawVz, dt)        // move>0.005 m/frame, idle after 0.25s, 0.2s cross-fade
applyToSkeleton(pose)   // engine-specific, not part of this package
```

The **state window always receives the raw, pre-post-processing state**
(`AInimatorController.Tick` already pushes before returning) — post-process
output is aval-only, exactly per spec §1's hard rule; feeding a corrected
pose back into the window would contaminate the autoregression and is never
done here.

Key types (`Runtime/PostProcess/`):

- `Smpl22Skeleton` — SMPL-22 bone order, parent hierarchy and T-pose
  bone-local offsets, ported from
  `src/ainimator/core/constants/skeletons.py` (see the file header comment
  for the exact source symbols — regenerate this file if those constants
  ever change upstream, never hand-diverge).
- `SmplForwardKinematics` — the strict-necessary FK subset (`rotation6d` →
  root-local joint positions), porting
  `ops.py::sixdToRotationMatrix`/`rot6dToJointXYZ`. Positions are
  root-local; `RootLocalToWorld` only translates by the engine's integrated
  root position — no extra yaw is applied because the pelvis's own
  rotation6d already encodes the skeleton's world-facing orientation
  (`ainimator/geometry/root_local.py`).
- `FootContactDetector` — redérives contacts from joints 10 (`leftFoot`) /
  11 (`rightFoot`); spec's height/speed criterion plus the 2-frame /
  1.5×-threshold exit hysteresis.
- `TwoBoneIkSolver` — analytic law-of-cosines two-bone solve, pole vector
  taken from the controller's own knee direction (never a fixed pole).
- `FootLockIk` — per-leg capture/hold/clamp(0.3m)/fade(0.1s) state machine
  built on `TwoBoneIkSolver`.
- `IdleMoveBlender` — idle↔move cross-fade weight (`0.005` m/frame move
  threshold on the **raw** control, `0.25s` idle-entry delay, `0.2s`
  cross-fade).
- `AInimatorPostProcess` — wires all of the above for one character,
  exposing an `Options` struct (on/off toggles + every threshold above,
  Inspector-serializable, defaults = the spec's defaults) and returning
  corrected left/right leg world joint positions + the idle blend weight.
  Applying those to an actual skinned rig/`Animator` is intentionally left
  to the host project (engine/rig-specific — see "Known constraints"
  below), matching how the B1 sample only visualizes the root transform.
- `FootSlidingMetric` — the before/after measurement utility for the B4
  acceptance criterion (see "Measuring foot-sliding" below).

### Measuring foot-sliding (B4 acceptance criterion)

`apps/spec/footlock_blending.md` §5 requires a measurable (>50%) reduction
in foot-sliding on a 10s `forward` walk with the reference bundle. Procedure:

1. Run the reference bundle's `forward` preset for 10s (`10 * fps` ticks)
   with `AInimatorPostProcess.Options.enableFootLockIk = false` — feed each
   frame's raw ankle/foot world positions (before any IK) into one
   `FootSlidingMetric` per foot, calling `Accumulate(worldPosition,
   isInContact)` with the **uncorrected** ankle position and the
   `FootContactDetector`-derived contact state.
2. Repeat with `enableFootLockIk = true`, this time accumulating the
   **corrected** ankle position (`AInimatorPostProcess.Process`'s
   `LegResult.AnkleWorldPosition`) into a second pair of metrics.
3. Compare `MeanPlanarDisplacementPerContactFrame` before vs. after per
   foot; record both numbers here once measured against a real bundle in
   Unity (this session could not run Unity — see "Known constraints").

Before/after numbers: **not yet measured** (requires a real Unity Editor +
a posed reference bundle — Pazimor's validation step). Record them in this
section once available:

| Foot | Before (m/frame) | After (m/frame) | Reduction |
|---|---|---|---|
| Left  | _TBD_ | _TBD_ | _TBD_ |
| Right | _TBD_ | _TBD_ | _TBD_ |

## Testing

EditMode tests (`Tests/EditMode/`) run inside the Unity Test Framework
(Window → General → Test Runner → EditMode) once the package is added to
a project:

- `NormalizerTests` — numeric parity against concrete values pulled from
  `output/reference_bundle/norm_stats.json` (copied as a fixture under
  `Tests/Fixtures/Resources/reference_bundle/`, JSON only — no `.onnx`).
- `StateBufferTests` — ring-buffer seeding, push/evict ordering, wrap-around.
- `RootMotionIntegratorTests` — yaw-relative integration formula parity
  with `ainimator.model.controller_rollout._integrateOneStep`.
- `BundleLoaderTests` / `BundleVersionTests` — fail-fast validation: a
  bundle with an incompatible version, wrong frozen dims, or a corrupted
  `prompt.null_emb` must raise `BundleLoadException` loudly.
- `InputBindingResolverTests` (B3) — first-match-wins key resolution,
  invalid-row skipping, fallback-to-idle, binding serialization.
- `FootContactDetectorTests` (B4) — planted/lifted synthetic foot
  trajectories, speed-criterion rejection, the 2-frame exit hysteresis and
  its counter reset, the 1.5×-threshold immediate release.
- `TwoBoneIkSolverTests` (B4) — reachable target (effector lands exactly on
  target, bone lengths preserved), unreachable target (reports
  `TargetReachable = false`, fully extends), pole-direction preservation,
  degenerate target-at-root fallback.
- `SmplForwardKinematicsTests` (B4) — identity-pose sanity (pelvis at
  origin, chained hip→knee→ankle offsets), `RootLocalToWorld` translation-only
  semantics, input-length validation.
- `FootLockIkTests` (B4) — capture-and-hold under small pose drift, clamp
  release beyond 0.3m, the 0.1s fade-out curve, pass-through when never
  locked.
- `IdleMoveBlenderTests` (B4) — immediate move-exit, the 0.25s idle-entry
  delay, the 0.2s cross-fade timing, exact-threshold boundary behavior.
- `FootSlidingMetricTests` (B4) — the before/after measurement utility:
  zero sliding while stationary, accumulation while sliding, non-contact
  frames excluded, entry-frame not counted, reset semantics.

These tests were written and reviewed but **could not be executed in this
environment** (no Unity Editor available here) — run them once inside a
real Unity project before relying on the "green" status.

## Known constraints / things only verifiable with real Unity

- **Sentis API surface**: `ControllerRuntime` was written against the
  documented Sentis 1.x API (`ModelLoader.Load`, `Worker`, `Tensor<float>`,
  `SetInput`/`Schedule`/`PeekOutput`). Sentis' API changed across versions
  (Barracuda → Sentis, and within Sentis 1.x minors) — **verify every
  method/type name against the installed package** before running the
  demo; the exact tensor upload/readback calls
  (`Tensor<float>.Upload`, `PeekOutput` + a synchronous CPU download) are
  the most likely to need a small adjustment. This is called out with
  inline `NOTE (verify against installed Sentis)` comments at the call
  sites.
- **≥60 fps + trajectory parity acceptance** (`ROADMAP_PLUGINS.md` §4, B1):
  requires an actual play-mode run on the target machine with a real
  bundle — this is Pazimor's validation step, not verifiable from this
  session.
- **60 s NaN/explosion smoke test**: a PlayMode test was scoped by the
  roadmap but not authored here (no way to execute PlayMode tests without
  Unity running); a straightforward implementation is to call
  `AInimatorController.Tick` in a tight loop under a fixed preset for
  `60 * targetFps` iterations and assert `!float.IsNaN`/`!float.IsInfinity`
  on every output channel each step.
- **StreamingAssets on Android/WebGL**: `BundlePaths`/`BundleLoader` assume
  a plain filesystem path (fine on Editor/Windows/macOS/Linux — the B1
  demo target). Mobile/web platforms serve `StreamingAssets` from a
  compressed archive or URL and need a `UnityWebRequest`-based loader
  swapped in; out of scope for B1.
- **`AInimatorPostProcess` does not apply the corrected pose to a rig**
  (B4): it returns corrected world joint **positions** for the two legs
  plus an idle blend weight; wiring those onto an actual skinned
  `Animator`/humanoid rig (e.g. driving
  `AnimationRigging.TwoBoneIKConstraint` targets from the returned
  positions, or blending idle↔move on an `Animator` layer with the
  returned weight) is host-project/rig-specific and left as an integration
  step, exactly like the B1 sample only visualizing the root transform on a
  bare capsule.
- **Foot-sliding before/after numbers not yet measured** (see "Measuring
  foot-sliding" above) — requires a real Unity Editor + posed reference
  bundle, Pazimor's validation step.
- **Custom Inspector (`AInimatorActionBinderEditor`) untested against a
  live `SerializedObject`**: the property-drawer code (list add/remove,
  the "Create New Preset..."/"Import Presets From Bundle..." buttons) could
  not be exercised without a running Editor in this session; the
  unit-tested surface is `InputBindingResolver` (pure logic), not the
  Inspector GUI code itself.

## Contract compatibility

`BundleLoader.SupportedBundleVersion = "A7.0"`. A loaded bundle must share
the same version prefix + major number (see `BundleVersion.IsCompatibleWith`)
or loading fails loudly. Bump this constant (and re-validate the frozen
dims in `BundleLoader.ValidateManifest`) only when `apps/spec/manifest.schema.json`
bumps its contract — never silently.
