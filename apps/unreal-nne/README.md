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
| `UAInimatorActionComponent` | **B3** — `[Input] -> [ControlPreset]` bindings (Enhanced Input `UInputAction` or legacy `FKey` fallback), applies the resolved preset to a `UAInimatorControllerRuntime` every frame. See `docs/authoring.md`. |
| `FAInimatorActionComponentDetails` (editor module) | **B3** — Details panel customization: "Create New Preset" / "Import Presets From Bundle..." buttons alongside the editable bindings list. |
| `UAInimatorControlPresetFactory` (editor module) | **B3** — Content Browser asset factory for `UAInimatorControlPreset`. |
| `FFootContactDetector` | **B4** — re-derives foot contact from FK pose + hysteresis (`footlock_blending.md` §2). |
| `FSmplForwardKinematics` (`AInimatorForwardKinematics.h`) / `AInimatorSmpl22Skeleton` | **B4** — minimal SMPL-22 FK (rotation6d + bone offsets) needed for foot-lock IK; offsets ported from `ainimator/core/constants/skeletons.py`. |
| `AInimatorTwoBoneIkSolver` | **B4** — analytic two-bone IK (hip→knee→ankle), pole vector from the current pose. |
| `FFootLockIK` | **B4** — capture/clamp/fade foot-lock orchestration (`footlock_blending.md` §3), applied **after** `FStateBuffer::PushFrame` (never contaminates the autoregressive window). |
| `FIdleMoveBlender` | **B4** — idle↔move pose-space cross-fade state machine (`footlock_blending.md` §4). |
| `FFootSlidingMetric` | **B4** — debug-only utility measuring mean planar foot displacement during contact frames (before/after comparison, spec §5). |
| `UAInimatorPostProcessComponent` | **B4** — wires the above into one component with on/off switches + spec-default thresholds as `UPROPERTY`s; call `TickPostProcess()` **after** `Runtime->Tick()`. |
| `UAInimatorRigMap` | **B3-bis** — `UDataAsset`, 22 SMPL bones → target rig `FName`s (`rig_binding.md` §2.1), with editor auto-map by common bone names (incl. UE5 Mannequin). |
| `FRigBinder` | **B3-bis** — pure-math retargeting: bind-time calibration + per-frame `worldRot_target = R_smpl_world * worldRot_rig_rest`, `localRot_target = worldRot_target(parent)⁻¹ · worldRot_target` (`rig_binding.md` §2.2, literal), root-motion scale (§2.3). |
| `AInimatorPoseableMeshApplier` | **B3-bis** — applies `FRigBinder`'s output onto a `UPoseableMeshComponent` (v1 design decision, see `docs/rig_binding.md` §1). |
| `UAInimatorCharacterComponent` | **B3-bis** — orchestrates rig calibration/retarget + scaled root motion + the prompt channel (`SetPrompt`/`SetPromptEmbedding`/`ClearPrompt`, embedding-space cross-fade). See `docs/rig_binding.md`. |
| `FAInimatorRigMapDetails` / `FAInimatorControlPresetDetails` (editor module) | **B3-bis** — "Auto-Map From Skeletal Mesh..." button on `UAInimatorRigMap`; "Import Prompt Embedding (JSON)..." button on `UAInimatorControlPreset` (consumes `ainimator.cli.encode_prompt` output). |

## Authoring a binding without code (phase B3)

See `docs/authoring.md` for the one-page recipe: add
`UAInimatorActionComponent`, create/import a `ControlPreset`, add a
binding row (`Action` or `Key` → `Preset`), press Play. The Details
panel customization (`FAInimatorActionComponentDetails`, editor module
only) adds "+ Create New Preset" and "Import Presets From Bundle..."
buttons on top of the plain editable `Bindings` array.

## Post-processing: foot-lock IK + idle/move blending (phase B4)

Design is **shared and normative** with the Unity plugin —
`apps/spec/footlock_blending.md` is the single source of truth for
every threshold, hysteresis rule, IK chain and fade duration; this
plugin does not deviate from it (see "Deviations from the shared
design" below — currently none).

**Hard rule preserved by construction**: `UAInimatorPostProcessComponent`
only ever *reads* `UAInimatorControllerRuntime::GetLatestBoneFrame()` /
`GetWorldRootPosition()` / `GetWorldYaw()` **after** `Runtime->Tick()`
has already pushed the raw state into `FStateBuffer`. It has no path
back into the runtime's state — the autoregressive window always sees
the controller's raw output, never the IK-corrected pose
(`footlock_blending.md` §1).

Call order, once per frame:

```cpp
Runtime->Tick();
PostProcessComponent->TickPostProcess(DeltaSeconds, RawControlVx, RawControlVz);
// PostProcessComponent->GetCorrectedAnklePosition(0 /*left*/) / (1 /*right*/)
// PostProcessComponent->GetCorrectedKneePosition(...)
// PostProcessComponent->GetMoveBlendWeight()
```

Every spec threshold is exposed as a `UPROPERTY` on
`UAInimatorPostProcessComponent`, defaulted to the spec's value
(`ContactHeightThreshold` 0.05m, `ContactSpeedThreshold` 0.01 m/frame,
`ContactExitFrames` 2, `ContactReleaseHeightMultiplier` 1.5,
`MaxCorrectionMeters` 0.3m, `ReleaseFadeSeconds` 0.1s,
`MoveSpeedThreshold` 0.005 m/frame, `IdleDelaySeconds` 0.25s,
`CrossFadeSeconds` 0.2s), plus `bEnableFootLockIK` /
`bEnableIdleMoveBlend` on/off switches for the mandatory before/after
comparison.

### Foot-sliding metric (before/after)

`FFootSlidingMetric` (enabled via `bRecordFootSlidingMetric`) measures
the mean planar (XZ) displacement of a foot across consecutive
contact-to-contact frames — exactly the "déplacement planaire moyen
des pieds pendant leurs frames de contact" acceptance metric
(`footlock_blending.md` §5). Procedure to reproduce:

1. Drive the `forward` preset for 10s (300 steps @ 30 fps) from the
   reference bundle's rest-pose seed.
2. Run once with `bEnableFootLockIK = false` (records the raw FK
   ankle's sliding) and once with `bEnableFootLockIK = true` (records
   `GetCorrectedAnklePosition`'s sliding), calling
   `GetAverageFootSlidingMeters()` / `GetSampleCount()` at the end of
   each run.
3. Record both numbers here:

   | Run | Avg. planar sliding (m/frame) | Contact-frame pairs sampled |
   |---|---|---|
   | Before (IK off) | _pending — needs a real UE + bundle run_ | _pending_ |
   | After (IK on) | _pending — needs a real UE + bundle run_ | _pending_ |

   **Not yet measured in this environment** (no Unreal Engine install,
   no NNE inference available here — see "Known points that need a
   real Unreal/NNE install to validate" below). `AInimatorPostProcessSmokeTest.cpp`
   exercises the same code path against a synthetic walk cycle and
   asserts the after-IK average is `<=` the before average as a
   regression guard, but that synthetic result is **not** a substitute
   for the real bundle measurement above.

### Deviations from the shared design (`footlock_blending.md`)

**None.** Every threshold, hysteresis rule, IK chain (hip 1/2 → knee
4/5 → ankle 7/8, foot-contact joints 10/11), clamp (0.3m), fade
duration (0.1s release, 0.2s idle↔move cross-fade) and ordering rule
(state buffer receives the raw state, post-processing is strictly
downstream) match the spec exactly, as ported. If a future change
needs to deviate, update `apps/spec/footlock_blending.md` first (it is
outside this plugin's write scope) and only then this file.

## Binding a rigged character + piloting by prompt (phase B3-bis)

See `docs/rig_binding.md` for the one-page recipe: attach a
`UPoseableMeshComponent` + `UAInimatorCharacterComponent`, create/auto-map
a `UAInimatorRigMap` against your character's `SkeletalMesh` (UE5
Mannequin naming supported out of the box), calibrate at bind time, and
drive the prompt channel (`SetPrompt`/`SetPromptEmbedding`/
`ClearPrompt`) with its 0.3s embedding-space cross-fade. Design is
**shared and normative** with the Unity plugin —
`apps/spec/rig_binding.md` is the single source of truth for the
retargeting math and prompt semantics; this plugin applies it
literally (no invented numeric choices — see "Deviations" below).

### Deviations from the shared design (`apps/spec/rig_binding.md`)

**None** for the retargeting math or prompt semantics (§2/§3, ported
literally). One **v1 engineering choice**, explicitly allowed by the
spec (§2.2 "Application au squelette: v1 via UPoseableMeshComponent OU
un FAnimNode custom minimal — choisis le plus robuste"): this plugin
uses **`UPoseableMeshComponent`**, not a custom `FAnimNode` — see
`docs/rig_binding.md` §1 / `AInimatorPoseableMeshApplier.h` for the
full rationale and trade-offs. The native **IK Retargeter** path (§2.4)
is explicitly optional/non-canonical per the spec and is **not
implemented** here (budget was spent on the canonical `RigMap` path +
its tests instead).

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
- **B3**: `AInimatorActionComponent.cpp` uses
  `UEnhancedInputComponent::BindAction`'s variadic payload-forwarding
  overload (`BindAction(Action, TriggerEvent, Object, Func, ExtraArgs...)`,
  available since UE 4.25/4.26's Enhanced Input plugin) to pass the
  triggering `UInputAction*` through to
  `OnEnhancedInputTriggered(Value, SourceAction)`. This is a real,
  documented overload, but **verify it against the installed engine's
  `EnhancedInputComponent.h`** before compiling — the exact template
  constraints on the payload type(s) have shifted slightly across UE
  minor versions.
- **B3**: `FAInimatorActionComponentDetails`'s use of
  `IAssetTools::CreateAsset` / `CreateUniqueAssetName` and
  `IDesktopPlatform::OpenDirectoryDialog` — written against the
  well-documented UE 5.x `AssetTools`/`DesktopPlatform` surface, but
  not compiled/tested here; verify signatures before building.
- **B4**: the real (non-synthetic) foot-sliding before/after
  measurement (see "Foot-sliding metric" above) — needs a real Unreal
  project with the reference bundle loaded and `UAInimatorPostProcessComponent`
  wired to an actual pawn/tick loop; only a synthetic walk-cycle smoke
  test (`AInimatorPostProcessSmokeTest.cpp`) runs in this environment.
- **B4**: this plugin does not bind `FFootLockIK`'s corrected joint
  positions onto an actual `USkeletalMeshComponent`/AnimGraph two-bone
  IK node — that wiring (e.g. via an `AnimGraph` node reading
  `UAInimatorPostProcessComponent::GetCorrectedAnklePosition`) is host
  project integration work, left open exactly as the Unity plugin
  leaves `TwoBoneIkSolver`'s consumer-side rig binding open.
- **B3-bis**: `UAInimatorCharacterComponent`'s retargeting/root-motion
  path, `UPoseableMeshComponent::GetBoneQuaternion`/
  `SetBoneRotationByName`/`GetBoneLocationByName` call signatures, and
  `FAInimatorRigMapDetails`'s `FContentBrowserModule::CreateAssetPicker`
  usage are written against the well-documented UE 5.x surface but not
  compiled/tested here — verify against the installed engine version.
- **B3-bis**: the prompt embedding cross-fade's actual visual quality
  (the "⚠ heuristic, not validated" callout in `docs/rig_binding.md`
  §5) needs a real bundle + a real character in a running Unreal
  project to judge; only the pure lerp/alpha math is unit-tested here
  (`AInimatorPromptCrossFadeTest.cpp`).
- **B3-bis**: `FRigBinder`'s retarget math is unit-tested against a
  synthetic 3-bone mini-rig with known offsets
  (`AInimatorRigBinderTest.cpp`); it has not been validated against a
  real character mesh (visual "marche upright, membres cohérents, pas
  de twist d'os aberrant" acceptance criterion, `rig_binding.md` §5)
  since no Unreal Engine install is available here.
