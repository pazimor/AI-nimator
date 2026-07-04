# Rig binding — attach to a character and pilot it by prompt

This page covers the Goal B phase B3-bis workflow: retargeting the
controller's SMPL-22 output onto an arbitrary rigged character (`Rig/`) and
driving it with a prompt in addition to (or instead of) button bindings. The
canonical design lives in `apps/spec/rig_binding.md` — this page is the
practical Unity walkthrough of that design; it implements it, it does not
redefine it. It assumes a controller bundle is already posed under
`StreamingAssets/AInimatorBundle/` (see the main `README.md`).

## 1. Create a `RigMap` for your character

- **Assets → Create → AInimator → Rig Map** — creates a blank 22-row asset
  (one row per SMPL-22 bone, canonical order: pelvis, leftHip, rightHip,
  spine1, ..., rightWrist).
- Select the asset; its Inspector shows the 22-row table plus an **Auto-Map**
  section:
  - Drag your character's root bone (or the whole rig root) into **Rig
    Root**, then click **Auto-Map From Root**. This best-effort matches
    common humanoid bone-name conventions (Mixamo, `Hips`/`Spine`/`Head`/
    `LeftArm`/..., Unity Humanoid-exported names) — any row it cannot
    confidently match is left empty rather than guessing wrong.
  - Fix up any row by hand: drag the correct `Transform` onto the row's
    object field, or clear it to leave that SMPL bone unmapped.
- An **unmapped row is a valid, permanent state** — that SMPL bone's
  rotation is simply never applied to the rig (spec §2.1). The 22 SMPL
  bones cover standard humanoid rigs; a custom rig missing e.g. separate
  collar bones can safely leave `leftCollar`/`rightCollar` unmapped.

## 2. Add `RigBinder` to the rigged character

- Add Component → **AInimator Controller → Rig Binder**
  (`AInimator.Controller.Rig.RigBinder`) on the character's root GameObject.
- Assign the **Rig Map** you just built.
- Leave **Calibrate On Awake** checked so the component captures the rig's
  rest-pose world rotations (and the pelvis height for `rigScale`) the
  moment the scene starts — **the rig must already be sitting in its rest
  pose at that point** (spec §2.2 step 1). If you attach the rig
  procedurally later, uncheck this and call `Calibrate()` yourself once the
  rig is posed.
- **Actor Root** defaults to the same GameObject's transform; point it at a
  different transform if the rendered mesh's root differs from the
  GameObject driving gameplay logic.

## 3. Add `AInimatorCharacter` (replaces `AInimatorActionBinder` for a rigged character)

- Add Component → **AInimator Controller → Character**
  (`AInimator.Controller.Authoring.AInimatorCharacter`).
- Assign **Idle Preset** and add binding rows exactly like
  `AInimatorActionBinder` (see `authoring.md` §1–3 — the same
  `ControlPreset`/`InputBinding` workflow, unchanged).
- Assign the **Rig Binder** field to the `RigBinder` component from step 2.
  **Leaving it empty keeps the exact B1 capsule behaviour** (this component
  falls back to driving its own root transform with no retargeting at all —
  spec §5 last line: the capsule path always stays functional).
- Press Play: the character's rig follows the controller's pose every
  frame, retargeted through `RigBinder`; root motion (scaled by `rigScale`)
  drives the rig's actor root.

## 4. Pilot by prompt at runtime

Once the bundle's manifest declares `prompt_emb_channels > 0`:

- `character.SetPrompt(preset)` — cross-fades toward `preset`'s precomputed
  `prompt_emb` (`ControlPreset.PromptEmb`) over `PromptCrossFadeSeconds`
  (default 0.3s, editable via `character.Controller.PromptCrossFadeSeconds`).
- `character.SetPromptEmbedding(float[])` — cross-fades toward a raw
  embedding vector supplied by the game (length must equal
  `manifest.prompt_emb_channels`).
- `character.ClearPrompt()` — cross-fades back to the bundle's **learned**
  null embedding (never zeros — `inference_contract.md` §4).
- The control vector (`vx`, `vz`, aim) from the active `ControlPreset`/
  binding keeps applying every frame independently of the prompt — the two
  are additive, not exclusive.

⚠ **Heuristic, validate visually per bundle.** The controller was never
trained on interpolated prompt embeddings; a linear lerp through embedding
space can pass through regions the model never saw. Watch for limb pops or
an unstable gait during the fade. If transitions look degraded, the
documented fallback (not wired here — content-dependent) is a hard prompt
switch combined with a **pose-space** cross-fade, reusing the
`IdleMoveBlender`-style mechanism (B4 `apps/spec/footlock_blending.md` §4)
on the rendered pose rather than the embedding.

## 5. Getting a `prompt_emb` for a `ControlPreset`

The runtime never encodes free text — it only consumes precomputed
embeddings (`inference_contract.md` §4). To turn a sentence into a
`prompt_emb`:

```bash
python -m ainimator.cli.encode_prompt \
    --prompt "a person walks forward" \
    --encoder-artifact output/clip_text_artifact \
    --output output/presets/walk_prompt.json
```

This writes `{"prompt": "...", "prompt_emb": [...]}`. Load it into a
`ControlPreset` asset either:

- from the preset's own Inspector — click **"Load Prompt Embedding
  JSON... (encode_prompt output)"**, or
- right-click the `ControlPreset` asset (or its component context menu) →
  **"Load Prompt Embedding JSON..."**.

Both write the JSON's `prompt`/`prompt_emb` fields into the preset's
`Prompt`/`PromptEmb` fields — no code required.

## Optional: native Humanoid retargeting (`HumanPoseHandler`)

The `RigMap`/`RigBinder` explicit path above is the **canonical** one (the
only one covered by cross-engine parity, spec §2.4). Unity's `Animator`
Humanoid rig + `HumanPoseHandler` could in principle offer a second, native
retargeting path — this is **not implemented** in this package (budget/
scope decision for B3-bis; the explicit `RigMap` path already covers
Humanoid-rigged characters via auto-map, since Mecanim's own bone names are
matched by the same heuristic). Revisit only if a concrete need for
Humanoid-specific features (avatar masks, muscle-space retargeting) arises.

## Not covered here

- Foot-lock IK / idle↔move blending internals — see `docs/authoring.md`
  "Not covered here" and `apps/spec/footlock_blending.md`. Note: those
  corrections are joint-**position** space; they are not folded back into
  the rotation6d frame `RigBinder` retargets, so a renderer that wants
  foot-locked feet on the *skinned* rig must additionally apply
  `AInimatorCharacter.LastPostProcessResult`'s corrected ankle/knee
  positions via its own IK layer.
- In-engine free-text encoding (typing a sentence in-game) — requires
  porting the CLIP tokenizer + `text_encoder.onnx` into the engine; not part
  of the B0 bundle contract today (`rig_binding.md` §4.2).
