# Binding a rigged character + piloting by prompt (phase B3-bis)

This page is the one-page recipe for "I have a rigged character (e.g.
the UE5 Mannequin) and I want AI-nimator to drive it, and to be able to
switch its behavior by changing the *prompt* alone". Design source of
truth: `apps/spec/rig_binding.md` (shared with the Unity plugin — this
page only documents the Unreal-side workflow, it does not redefine any
math).

Prerequisite: a `UAInimatorControllerRuntime` already loaded on your
actor (see the main `README.md` "Placing a bundle" section) — this page
only covers attaching it to a rigged mesh and the prompt channel.

## 1. Add the components

1. On your Actor/Pawn, alongside the existing
   `UAInimatorControllerRuntime` (and optionally
   `UAInimatorPostProcessComponent` for B4 foot-lock IK), **Add
   Component -> Poseable Mesh** and assign your character's
   `SkeletalMesh` to it.
   - **Why `UPoseableMeshComponent` and not a custom AnimGraph node?**
     v1 uses `UPoseableMeshComponent` because it lets C++ drive bone
     rotations directly (`SetBoneRotationByName`), with no
     AnimBlueprint asset to author/compile/keep in sync — the lowest
     risk path to validate under "Unreal not executable in this
     environment". Trade-off: it bypasses the normal AnimGraph/Montage
     blending pipeline. A future revision may add a custom `FAnimNode`
     for projects that need to blend AI-nimator output with
     hand-authored AnimBlueprint content; not implemented here (see
     `AInimatorPoseableMeshApplier.h`'s class comment for the full
     rationale).
2. **Add Component -> AInimator Character Component**.
3. Set its **Runtime** (and **PostProcess**, if used) to your loaded
   runtime/post-process components, and **PoseableMesh** to the
   component you just added.

## 2. Create a RigMap

1. **Content Browser -> Add -> Miscellaneous -> Data Asset ->
   AInimator Rig Map** (or right-click and pick it from the New Asset
   menu if your project registers a shortcut for it).
2. In the new asset's Details panel, click **Auto-Map From Skeletal
   Mesh...** and pick your character's `SkeletalMesh` asset. This
   resolves as many of the 22 SMPL bones as it can by common naming
   conventions — the **UE5 Mannequin** names (`pelvis`, `thigh_l`,
   `calf_l`, `foot_l`, `spine_01/02/03`, `clavicle_l`, `upperarm_l`,
   `lowerarm_l`, `hand_l`, mirrored `_r`, `neck_01`, `head`, ...) are
   tried first, then generic Mixamo/Biped-style names.
3. Review the `Entries` array: any row left blank (`TargetBoneName ==
   None`) means that SMPL bone's rotation is simply **ignored** for
   this rig (v1 rule, `rig_binding.md` §2.1) — fill in manually if your
   rig uses an unusual naming convention. The **pelvis** row MUST be
   mapped (it is the retargeting root); everything else is optional.
4. Assign this `UAInimatorRigMap` asset to your
   `UAInimatorCharacterComponent::RigMap` property.

## 3. Calibrate at bind time

Call **`CalibrateRig()`** (Blueprint-callable) once, right after your
character's `PoseableMesh` is in its reference/rest pose — typically
right after `BeginPlay`, right after `Runtime->LoadBundle()` succeeds
and before the first `TickCharacter` call. This captures each mapped
rig bone's rest-pose WORLD rotation and computes the root-motion scale
(`rigScale` = rig rest pelvis height / canonical SMPL pelvis height
~0.91m — `rig_binding.md` §2.3; the component handles the UE
centimeters-vs-SMPL-meters unit conversion internally).

`IsRigCalibrated()` / `GetRigScale()` are exposed to Blueprint for
diagnostics. If calibration fails (no RigMap, no PoseableMesh, or the
pelvis is unmapped), it is logged and `TickCharacter` simply skips
retargeting every frame — **the capsule/pawn (B2) path keeps working
unaffected** (`rig_binding.md` §5 acceptance).

## 4. Tick order

Every frame, in this order:

```cpp
Runtime->Tick();
PostProcess->TickPostProcess(Dt, Vx, Vz);   // optional, B4
CharacterComponent->TickCharacter(Dt);
```

`TickCharacter`:
1. Advances the prompt cross-fade (see §5 below) and pushes the
   resulting embedding to `Runtime` every frame while a fade is active.
2. Applies **scaled root motion** to the owning Actor
   (`SetActorLocation`/`SetActorRotation` from
   `Runtime->GetWorldRootPosition() * RigScale` /
   `Runtime->GetWorldYaw()`).
3. If calibrated: forward-kinematics the controller's current pose
   (reusing `AInimatorForwardKinematics`, the same B4 machinery), then
   retargets every calibrated bone via `FRigBinder::RetargetFrame` and
   applies the result onto `PoseableMesh`.

## 5. Piloting by prompt

`UAInimatorCharacterComponent` exposes, Blueprint-callable:

- **`SetPrompt(UAInimatorControlPreset* Preset)`** — uses the preset's
  precomputed `PromptEmb` (see §6 below for how to get one).
- **`SetPromptEmbedding(const TArray<float>& Embedding)`** — a raw
  embedding vector (length must equal the bundle's
  `prompt_emb_channels`).
- **`ClearPrompt()`** — reverts to the bundle's **learned** null
  embedding (never a zero vector, `inference_contract.md` §4).

Every change starts a **cross-fade in embedding space**: a linear lerp
`emb_old -> emb_new` over `PromptCrossFadeSeconds` (`UPROPERTY`, default
**0.3s**, matches `rig_binding.md` §3). The control vector (`vx, vz,
aim`) keeps being applied every frame independently — prompt and
control are additive.

> ⚠ **Heuristic, not validated.** The controller was never trained on
> interpolated prompt embeddings, so intermediate frames during a fade
> sample an input distribution the model has never seen. Visual quality
> during transitions needs validation in a real Unreal project before
> shipping. If transitions look degraded, the documented fallback
> (`rig_binding.md` §3) is: a hard prompt switch + a **pose-space**
> cross-fade instead, reusing `UAInimatorPostProcessComponent`'s
> existing `FIdleMoveBlender` machinery as the mechanism. This fallback
> is **not implemented** in this pass — it needs Pazimor's visual call
> first (see the session's open questions).

## 6. Getting a prompt embedding

The runtime consumes **embeddings**, never raw text
(`inference_contract.md` §4). Two ways to get one, without writing any
in-engine text encoder:

1. **CLI (repo-side, recommended):**
   ```bash
   $AIPY -m ainimator.cli.encode_prompt \
     --prompt "a person walks forward" \
     --encoder-artifact output/clip_text_artifact \
     --output output/presets/walk_prompt.json
   ```
   This writes `{"prompt": "...", "prompt_emb": [float, ...]}`.
2. **Import it into a `UAInimatorControlPreset` asset**: open the
   preset in the editor and click **Import Prompt Embedding
   (JSON)...** in its Details panel — picks the JSON file above and
   fills `Prompt`/`PromptEmb`/`bHasPromptEmb` from it. Then call
   `SetPrompt(ThatPreset)` at runtime (or from a
   `UAInimatorActionComponent` binding, exactly like a locomotion
   preset).
3. Alternatively, feed the same `prompt_emb` array straight to
   `SetPromptEmbedding` at runtime (e.g. loaded from your own save data
   or streamed from a server) — no asset required.

Free-text prompt resolution **in-engine** (typing a sentence at
runtime and having the game encode it on the fly) is **out of scope**
for this bundle version — it would require the BPE tokenizer + text
encoder ported to C++ (`rig_binding.md` §4, an explicitly deferred
extension).

## Reference — what each API maps to

| API | Type | Contract source |
|---|---|---|
| `UAInimatorRigMap::Entries[i].TargetBoneName` | `FName` | `rig_binding.md` §2.1 |
| `FRigBinder::Calibrate` / `RetargetFrame` | static methods | `rig_binding.md` §2.2 formula (literal) |
| `FRigBinder::ComputeRigScaleFromCentimeters` | static method | `rig_binding.md` §2.3 |
| `UAInimatorCharacterComponent::SetPrompt/SetPromptEmbedding/ClearPrompt` | `UFUNCTION(BlueprintCallable)` | `rig_binding.md` §3 |
| `UAInimatorCharacterComponent::PromptCrossFadeSeconds` | `UPROPERTY` (default 0.3) | `rig_binding.md` §3 |
| `python -m ainimator.cli.encode_prompt` | CLI | `rig_binding.md` §4 |

See `AInimatorRigMap.h`, `AInimatorRigBinder.h`,
`AInimatorCharacterComponent.h`, `AInimatorPoseableMeshApplier.h` for
the full API and design-decision comments.
