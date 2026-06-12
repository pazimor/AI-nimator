# AI_nimator
Research project: train a diffusion model to generate 3D skeletal
animations (SMPL-22) from text prompts, based on the AMASS dataset.

> **READ FIRST: `doc/ROADMAP.md`** is the canonical roadmap (refactor plan,
> scaling protocol, settled decisions, role split). Follow its phases and
> acceptance criteria; do not re-litigate its "Vérités canoniques".

## Stack
- Language: python (poetry — see pyproject.toml for dependencies)
- Framework: pytorch, device MPS (Mac 32GB) — batch 2–8 + grad accumulation
- Dataset paths: src/configs/dataset.yaml
- Testing: pytest

## Canonical decisions (summary — full list in ROADMAP §2)
- **v2 stack only**: v-prediction + cosine β + DDIM 100 + Min-SNR-γ=5.
- **Lean representation**: 135 channels (rotation6d 132 + root_translation 3).
  FK-derivable signals are supervised at the loss, never predicted.
- **Text encoder**: custom BPE 8k + transformer 4L/256d is the target.
  Frozen CLIP (`textEncoderType="clip"`) = diagnostic baseline only.
- **Encoder is DECOUPLED from generation** (ROADMAP §2.9, phase A7):
  standalone module + artifact, consumed via `TextEncoderProtocol`;
  training regime (joint vs pre-trained) is decided experimentally (B2).
- **ONNX exportability is a design constraint** (ROADMAP §2.10, phase A8):
  never block the Mac NPU path. No data-dependent control flow or
  `.item()` in `forward()`; DDIM loop stays outside the graph; dynamic
  axes (batch, frames, text length). CI export test must stay green.
- **Conditioning**: cross-attn + FiLM + per-block AdaLN + learnable null
  embedding — all flags stay ON by default.
- **Z-normalization is mandatory and asserted** (post-norm ≈ N(0,1));
  CFG dropout mandatory when sampling with cfgScale > 1.
- **`health/` is THE single debug tool** (Probe/Contract/HealthHub).
  Score reference table (direction, targets, thresholds): ROADMAP §3.5.
  Sanity criteria: cfg_sim < 0.95, seed_sim < 0.90, encoder
  cond↔uncond sim < 0.5.
- **Capacity↔N scaling law** (established 2026-06-12, memorization
  regime): denoiser params ≈ ∝ N (384/4→N≤50, 512/6→N≤500,
  640/8→N=1000 ✓). Validated recipe: cond-mask-prob 0.10, **CFG 4–6 at
  inference** (mandatory at scale), ~600 exposures/sample. Evaluate
  retrieval/fidelity at cfg ∈ {1,4,6}. Details: ROADMAP §5.2, LOG.md.

## Directory Structure
Current layout (post-A2):
- `src/ainimator/` — single installable package in dependency layers:
    - `core/` (L0) — types, constants, config schema, device, logging
    - `geometry/` (L1) — quaternion, rot6d, FK, SMPL-22 skeleton,
      motion components
    - `data/` (L2) — preprocessed dataset, dataset builder, augmentation
    - `text/` (L3) — BPE tokenizer, custom encoder, CLIP wrapper
    - `diffusion/` (L3) — cosine schedule, DDIM math, noise schedule
    - `model/` (L3, above text/diffusion) — denoiser v2, losses, sampler,
      motion normalizer, layers, v1 CLIP model
    - `health/` (L4) — diagnose_v2 (health/ tooling built in A3)
    - `training/` (L4) — v2 and v1 training loops, dataset manager
    - `export/` (L4) — postprocess/Collada (ONNX in A8)
    - `cli/` (L5) — entrypoints, zero logic
- `src/configs/` — YAML configurations (outside the package)
- `scripts/` — experiment orchestrators (scaling sweeps)
- `output/` — generated files and run outputs
- `test/ainimator/` — mirrors src/ainimator/ layer tree
- `doc/` — internal documentation (ROADMAP.md, experiments/LOG.md)

Import rules enforced by `import-linter` (`.importlinter`):
- Imports point DOWN only (ascending imports are FORBIDDEN).
- `model` never imports `data`.
- `training` is the only module that sees both `data` and `model`.
- `cli` has zero logic.
Legacy v1 moves to `legacy/` (phase A5), importable by nothing.

## Commands
All commands use the new `ainimator.*` package path (phase A2+).
The poetry venv must have `src/` on its path; this is set up via a
`.pth` file created once: `echo "$PWD/src" > $(poetry run python -c
"import site; print(site.getsitepackages()[0])")/ainimator_dev.pth`

- `poetry run python -m ainimator.cli.build_dataset` — match prompts
- `poetry run python -m ainimator.cli.preprocess_dataset` — preprocess
- `poetry run python -m ainimator.cli.train_custom_tokenizer` — BPE
- `poetry run python -m ainimator.cli.train_generation_v2 --profile {overfit,full}`
  — v2 training (overfit = sanity check, full = real run)
- `poetry run python -m ainimator.cli.generate_animation_v2` — sample + export .dae
- `poetry run python -m ainimator.cli.diagnose_generation_v2` — conditioning
  diagnostics (absorbed into `ainimator.cli.health` after phase A3)
- `poetry run python -m ainimator.cli.health {watch|audit|diagnose|report}` —
  global debug tool (available from phase A3)
- `poetry run python -m ainimator.cli.train_text_encoder` — standalone encoder
  training (available from phase A7)
- `poetry run python -m ainimator.cli.export_onnx {encoder|denoiser}` — ONNX
  export, NPU path (available from phase A8)
- `poetry run pytest` — test suite
- `poetry run lint-imports` — verify layer contracts (must be green)

Legacy v1 (do NOT extend, archived in phase A5): `train_clip`,
`train_generation`, `generate_animation` (v1).

## Conventions
- Document every method with full **DOCString** (NumPy Style).
- Be compliant with Pylance (strict types).
- Avoid **magic numbers** and strings: always use named constants or enums.
- Use **explicit functions** (pure and reusable).
- Avoid shortcuts: no `i`, `m`, etc. in anonymous functions or methods.
- No method should exceed 25 lines.
- No column should exceed 80 characters.
- Try to keep max file length around 500 lines.
- Dataclasses go inside `ainimator/core/types`.
- Refactor into private functions when necessary.
- Use **isolated unit tests** (no cross-module dependencies).

## Rules for agents working on this repo
Three project agents are defined in `.claude/agents/`: **implementer**
(Sonnet, phases A1–A8 + data-extraction requests), **experimenter**
(Opus, protocol B0–B4, files requests in `doc/experiments/requests/`),
**reviewer** (read-only acceptance-criteria gate). Role split: ROADMAP §6.

- Follow ROADMAP phases **in order**; a phase is done only when ALL its
  acceptance criteria pass.
- **Never change existing hyperparameter defaults** without an explicit
  instruction from Pazimor.
- Every training run must write `resolved_config.yaml` in its outputDir
  (from phase A1 on).
- Run the overfit-1-sample smoke test before any long run; never launch
  runs > 30 min without asking.
- Atomic commits per coherent change; ask instead of choosing silently
  when a requirement is ambiguous.
- Every concluded experiment adds one line to `doc/experiments/LOG.md`
  (date, run, config, metrics, verdict); new decisions are reported into
  ROADMAP.md with a date.

## Skills & external tools
- **pytorch-auditor** (skill): quantitative checkpoint audit (weights,
  NaN, dead layers, optimizer). `hub.audit()` is its level 3–4 adapter.
- **ai-debugging** (skill): debugging methodology loop (collect → review
  → hypotheses → test) — use it for any "model behaves wrong" issue.
- Blender: manual visual validation of generated .dae on the probe set
  (walk/jump/sit/wave/run) — Pazimor's responsibility.
