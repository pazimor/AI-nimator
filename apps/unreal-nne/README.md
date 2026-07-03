# AInimator — Unreal (NNE) runtime plugin

Real-time Unreal Engine plugin for the AI-nimator deterministic
autoregressive motion **controller** (Goal B, phase B2 —
`doc/ROADMAP_PLUGINS.md`). It loads a **controller bundle** produced by
the `ainimator` Python export pipeline and runs one NNE forward per
frame; it defines no model and never re-learns normalization.

This plugin is the Unreal twin of `apps/unity-sentis/`. Both consume
the exact same bundle contract (`apps/spec/inference_contract.md`),
so a given bundle + control sequence must produce the **same
trajectory** in both engines within a documented tolerance (Goal B
vérité §2.7 — the parity test, see below).

## What's in the bundle

Produced by
`python -m ainimator.cli.export_onnx bundle --checkpoint <path> --output-dir <dir>`
(see `apps/spec/inference_contract.md`):

```
bundle/
├── controller.onnx        # one forward: state+control[+prompt_emb][+phase] → Δstate
├── norm_stats.json         # mean/std for state, delta, control; prompt.null_emb
├── manifest.json           # the contract's serialization — validated before load
├── resolved_config.yaml    # provenance (not consumed by the plugin)
└── presets/                # idle / forward / backward / strafe_left / strafe_right
```

## Installing the plugin in a project

1. Copy (or symlink) `apps/unreal-nne/AInimator/` into your Unreal
   project's `Plugins/` directory, e.g.
   `YourGame/Plugins/AInimator/`.
2. Enable the **NNE** plugin (Edit > Plugins > search "Neural Network
   Engine") if it is not already enabled — `AInimator.uplugin`
   declares it as a dependency but Unreal still needs it available in
   the engine install.
3. Enable **Enhanced Input** (bundled with the engine since 5.1) — the
   demo pawn's keyboard bindings use it.
4. Regenerate project files and build.

## Placing a bundle (no build orchestrator yet — B5)

Phase B5 (`apps/build/`) will automate "export a fresh bundle from a
checkpoint and drop it in `Content/`" as part of a single build
command. **Until B5 ships**, place a bundle manually:

```bash
$AIPY -m ainimator.cli.export_onnx bundle \
  --checkpoint <your_checkpoint.pt> \
  --output-dir /tmp/ainimator_bundle

# Copy (do not commit — Content/ is gitignored, see apps/unreal-nne/.gitignore):
cp -r /tmp/ainimator_bundle/* \
  YourGame/Plugins/AInimator/Content/Bundle/
```

Point `AAInimatorDemoPawn::BundleDirectory` (Details panel, or set it
from Blueprint before `BeginPlay`) at the **absolute path** of the
folder containing `manifest.json` — e.g.
`.../Plugins/AInimator/Content/Bundle`. The runtime reads
`manifest.json`, `norm_stats.json`, `controller.onnx` and `presets/`
directly from that folder at `LoadBundle()` time; nothing is
pre-imported as a versioned Unreal asset in this phase.

A **reference bundle** for parity testing lives at
`output/reference_bundle/` in the main `ainimator` repo (generated
from a validated A6 checkpoint, `apps/spec/inference_contract.md §7`).

## Running the demo

1. Set `BundleDirectory` on the `AAInimatorDemoPawn` (or a Blueprint
   subclass of it) placed in a test level.
2. Assign `MappingContext`, `MoveForwardAction`, `MoveBackwardAction`,
   `StrafeLeftAction`, `StrafeRightAction` — standard Enhanced Input
   assets (WASD is a natural default, but any binding works).
3. Press Play. WASD selects the matching bundled preset every frame
   (`forward` / `backward` / `strafe_left` / `strafe_right`); release
   to fall back to `idle`. The pawn's actor transform is driven
   directly from `UAInimatorControllerRuntime::GetWorldRootPosition()`
   / `GetWorldYaw()` — there is **no** foot-lock IK or skeletal mesh
   binding yet (that's phase B4); this demo only proves the runtime
   loop end-to-end.

## Architecture (mirrors B1/Sentis 1:1 for parity)

| Type | Responsibility |
|---|---|
| `FBundleLoader` | Parses + **validates** `manifest.json`/`norm_stats.json`; fail-fast on any contract mismatch (never a silent partial load). |
| `FNormalizer` | Pure z-norm math (state/control encode, Δstate decode) from the manifest's stats — no engine dependency, unit-testable. |
| `FStateBuffer` | Flat, pre-allocated `context_frames` ring buffer for the bone + global (root-local motion) window. |
| `UAInimatorControllerRuntime` | Owns the NNE model instance; `LoadBundle` / `SetControl` / `SetPreset` / `SetPromptEmbedding` / `Tick` (one forward per call); integrates `Δstate → state` (world yaw + position) exactly like `ainimator/model/controller_rollout.py`. |
| `UAInimatorControlPreset` | `UDataAsset` hydrated from `presets/*.json`; builds a raw control vector in the manifest's declared channel order. |
| `AAInimatorDemoPawn` | Keyboard demo content (WASD → preset), phase B2 scope only. |

Foot-lock IK / physics blending (B4) and the authoring component with
Details-panel bindings (B3) are **not** in this phase's scope.

## Contract fidelity notes (read before touching the math)

- Control is **normalized before** being fed to the model
  (`vx, vz` z-normed with `norm_stats.json["control"]`; `aim_x, aim_z`
  passed through unit-norm, never z-normed — matches
  `ainimator/cli/generate_controller_v2.py`'s
  `controlNorm = (batch.control - controlMean) / controlStd` done by
  the *caller*, not inside the model's `forward()`).
- The predicted `Δstate` is normalized on the way out of the model;
  `FNormalizer::DenormalizeBoneDelta` / `DenormalizeGlobalDelta` invert
  it with the **delta** stats before the engine integrates it —
  mirrors `denormalizeStepDelta` in
  `ainimator/model/motion_normalizer.py`.
- The next raw bone frame is `lastRawBoneFrame + rawBoneDelta` (added
  to the **raw**, not normalized, last frame) — mirrors
  `controller_rollout.py::_stepOnce`.
- World position/yaw integration:
  `worldDx = cos(yaw)·Δfwd − sin(yaw)·Δlat`,
  `worldDz = sin(yaw)·Δfwd + cos(yaw)·Δlat`,
  `pos += (worldDx, Δheight, worldDz)`, `yaw += Δyaw` — mirrors
  `controller_rollout.py::_integrateOneStep` bit-for-bit (see
  `AInimatorControllerRuntime.cpp::IntegrateRootLocalDelta`).
- No prompt selected → the runtime feeds `norm_stats.json`'s
  `prompt.null_emb` (a **trained** parameter), never a zero vector
  (`inference_contract.md §4`). See
  `UAInimatorControllerRuntime::SetPromptEmbedding`.
- `phase_channels == 0` in the current reference bundle: the `phase`
  ONNX input is not bound in phase B2. If a future bundle declares
  `phase_channels == 2`, `RunOneForward()` needs a `(cos, sin)` tensor
  appended after `prompt_emb` (or directly after `global_window` when
  there is no prompt) — flagged as a TODO in the code, not yet
  implemented (no phase-conditioned reference bundle exists to test
  against).

## Parity test procedure (Unity ↔ Unreal ↔ Python reference)

The acceptance criterion unique to B2 (`ROADMAP_PLUGINS.md` §4): **at
bundle + control-sequence identical, Unity/Sentis and Unreal/NNE must
reproduce the same trajectory as the Python/ONNXRuntime reference**,
within a documented numerical tolerance.

1. Use `output/reference_bundle/` (or any bundle exported the same
   way) in **both** engines.
2. Drive the **same deterministic control sequence** in all three
   runtimes for the same number of steps — e.g. `forward` held for
   150 steps (5 s @ 30 fps) starting from the same rest-pose seed
   (identity rotation6d, zero root motion) that
   `UAInimatorControllerRuntime::InitializeSeedState()` uses.
3. Record, per runtime, per step: world position (X, Y, Z), world yaw,
   and the full bone rotation6d frame.
4. Compare Unreal's trace against:
   - the Python/ONNXRuntime reference from
     `test/ainimator/export/test_bundle.py` (tolerance already fixed
     there at `1e-3` on the normalized forward + one normalization
     step — B0's contract);
   - Unity/Sentis's trace from the same control sequence and bundle
     (`apps/unity-sentis/`, once B1 lands).
5. Document the **measured** tolerance (do not assume it): NNE and
   Sentis may differ from each other and from ONNXRuntime in op
   ordering / precision (`ROADMAP_PLUGINS.md §6` risk). A reasonable
   starting budget, to be confirmed once both engines actually run
   inference, is the same `1e-3` per-step normalized-space tolerance
   used in B0, tracked cumulative drift separately (position error
   over 150 steps) since per-step error compounds under
   autoregression.
6. Any divergence beyond the documented tolerance is a **plugin bug**,
   not an acceptable parity gap — investigate op-level differences
   (e.g. `MultiheadAttention` numerics) before loosening the
   tolerance.

This procedure requires a real Unreal Engine install with NNE enabled
and a real exported bundle — **it cannot be executed in this
development environment** (no engine installed here); running it is
the next actionable step once this code lands in an Unreal project.

## Known points that need a real Unreal/NNE install to validate

- Exact NNE C++ API names/signatures
  (`UE::NNE::GetRuntime<INNERuntimeCPU>`, `IModelCPU`,
  `IModelInstanceCPU::RunSync`, `FTensorBindingCPU`,
  `UNNEModelData::Init`) — written against the well-documented UE 5.3+
  NNE surface, but **must be checked against the installed engine
  version** before compiling; flagged with inline comments at every
  call site in `AInimatorControllerRuntime.cpp`.
- Whether `UNNEModelData::Init(...)` accepts raw ONNX bytes at runtime
  in the installed NNE version, or whether the bundle's `.onnx` must
  instead be imported as an editor asset (a `.uasset` wrapping
  `UNNEModelData`) as part of the future B5 build orchestrator step —
  if runtime `Init()` is unavailable, `InitializeModel()` needs an
  editor-time import path instead.
- Real frame-rate / performance measurement (`>= 60 fps` acceptance
  criterion) — needs profiling in an actual project.
- The full Unity ↔ Unreal parity numeric comparison (see procedure
  above) — needs both plugins running against the same bundle.
