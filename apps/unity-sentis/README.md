# AI-nimator Controller — Unity (Sentis) plugin

Real-time runtime for the AI-nimator deterministic autoregressive
controller inside Unity, powered by [Sentis](https://docs.unity3d.com/Packages/com.unity.sentis@latest).
This package **defines no model** — it loads a frozen controller bundle
(`controller.onnx` + `norm_stats.json` + `manifest.json` + `presets/`)
produced by the `ainimator` Python export pipeline and drives it one
forward pass per frame. The single source of truth for the I/O contract is
`apps/spec/inference_contract.md` — this plugin implements it, it does not
redefine it.

Status: **Phase B1** (`doc/ROADMAP_PLUGINS.md` §4) — runtime + WASD demo.
Foot-lock IK/blending (B4), the authoring `AInimatorActionBinder` (B3), and
the build orchestrator (B5) are later phases.

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
    │   └── Presets/               ControlPreset ScriptableObject
    ├── Editor/                    ControlPresetImporter, BundleDeliveryTools (batch-mode pack target)
    ├── Tests/EditMode/            NUnit tests (Normalizer, StateBuffer, BundleLoader, RootMotionIntegrator)
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

## Contract compatibility

`BundleLoader.SupportedBundleVersion = "A7.0"`. A loaded bundle must share
the same version prefix + major number (see `BundleVersion.IsCompatibleWith`)
or loading fails loudly. Bump this constant (and re-validate the frozen
dims in `BundleLoader.ValidateManifest`) only when `apps/spec/manifest.schema.json`
bumps its contract — never silently.
