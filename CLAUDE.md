# AI_nimator
Research project: train a **deterministic autoregressive controller** (with
an integrated text encoder) to generate 3D skeletal animations (SMPL-22)
from text prompts and/or control signals, based on the AMASS dataset.

> **READ FIRST: `doc/ROADMAP.md`** is the canonical project frame (vérités,
> layered architecture, health, roles). Execution lives in
> `doc/ROADMAP_DETERMINIST.md` (the model: controller + integrated text
> encoder) and `doc/ROADMAP_PLUGINS.md` (Unity/Unreal plugins).
> Do not re-litigate the "Vérités canoniques".

## Stack
- Language: python (poetry — see pyproject.toml for dependencies), CPP (unreal engine plugin), C# (unity plugin)
- Framework: pytorch
- hardware: device MPS (Mac 24GB) — device AMD 7900xtx (GPU 24GB not for developpements) 
- Testing: pytest

> ⚠ **Environnement (MANDATORY — `G-ENV`).** Tout passe par Poetry
> (`poetry install`, `poetry run …`), Python 3.12/3.13. `pyproject.toml` est
> PEP 621 : `[project].requires-python` doit être un spécificateur **PEP 440**
> (`>=3.12,<3.14`), **jamais** un caret Poetry (`^3.12`) — celui-ci casse
> Poetry 2.x. Les `testpaths`/coverage ne pointent que des dossiers existants
> (`src`, `test`, `apps`). Après toute édition de `pyproject.toml` :
> `poetry lock` puis `poetry install`. (Fix appliqué le 2026-06-24 :
> caret→PEP 440, `tools`→`apps`.)

## Canonical decisions (summary — full list in ROADMAP §2)
- **Full deterministic** (pivot 2026-06-23): the engine is the autoregressive
  controller `f(state, control, [prompt], [phase]) → Δstate`. No diffusion,
  no sampling loop. (Diffusion truths kept as history only.)
- **All-in-one model with integrated text encoder** (ROADMAP_DETERMINIST
  §2.4, §3.0 schema): the controller takes a **control signal** (real-time)
  and/or a **text prompt** (high-level), the prompt encoded by the reused
  custom BPE encoder. Text encoder stays **decoupled** (`TextEncoderProtocol`,
  versioned artifact) — never redefined by the controller.
- **Lean representation**: 135 channels (rotation6d 132 + root_translation 3).
  FK-derivable signals are supervised at the loss, never predicted.
- **ONNX exportability is a design constraint**: never block the NPU path.
  No data-dependent control flow or `.item()` in `forward()`; the
  autoregressive loop + text-encoder pass stay outside the per-frame graph
  (one forward = one step); dynamic axes declared. CI export test stays green.
- **Conditioning**: FiLM + per-block AdaLN + learnable null embedding — ON by
  default (control + text embedding both enter the same conditioning block).
- **Z-normalization is mandatory and asserted** (post-norm ≈ N(0,1) on state
  and deltas).
- **`health/` is THE single debug tool** (Probe/Contract/HealthHub).
  Deterministic contracts: `control_sensitivity`, `mean_collapse`,
  `rollout_drift` (ROADMAP_DETERMINIST §4). Score table: ROADMAP §4.1.
- **Capacity↔N law** (hypothesis inherited from the diffusion runs, retested
  on the controller in A6): scale capacity with the number of clips; do a
  capacity probe before any large run. Details: ROADMAP_DETERMINIST §5 (A6).

## Code conventions
Conventions live in **`doc/GUIDELINES.md`**, one self-contained paragraph per
guideline with a stable ID (`G-XXX`). Retrieve a single one without reading
the whole file:

```bash
grep -A8 'G-DOCSTRING' doc/GUIDELINES.md     # or: rg -A8 'G-DOCSTRING'
```

## Directory Structure
**Single recap of the repo tree: `doc/ROADMAP.md` §3.1.** Do not duplicate
it elsewhere. In short: `src/ainimator/` is one package in dependency layers
(core → geometry → data / text / diffusion → model → health / training /
export → cli), `src/configs/` holds YAML, `apps/` holds the engine apps
(spec / build / unity-sentis / unreal-nne) plus the decoupled health
`monitor/`, all outside the Python package; `legacy/` is archived. Import rules (imports DOWN only, `model` never imports
`data`, `training` is the sole data↔model junction, `cli` zero logic,
`legacy/` importable by nothing) are enforced by `import-linter` — see
guideline `G-IMPORTS`.

## Commands
All commands use the new `ainimator.*` package path.
`poetry install` registers `ainimator` as an editable package
(no manual `.pth` setup required).

> **Post-pivot (2026-06-23) : la diffusion n'est plus à l'ordre du jour.**
> Le **contrôleur déterministe est la voie produit** (phases A, voir
> `doc/ROADMAP_DETERMINIST.md`). Les CLI diffusion ci-dessous sont
> **conservées mais inactives** (default `model-type: diffusion` non encore
> rebasculé — arbitrage Pazimor) ; ne pas y investir.

**Voie produit — contrôleur déterministe** (config-first : un run se pilote
par un **profil** YAML `--profile {overfit|full|debug}`, pas par des flags
d'hyperparamètre — cf. `ROADMAP_DETERMINIST.md` §3.2 ; cible A7) **:**
- `poetry run python -m ainimator.cli.train_controller_v2` — entraînement
  contrôleur, overfit 1 clip (smoke test canonique, A1)
- `poetry run python -m ainimator.cli.train_controller_multiclip_v2` —
  validation A2/A4 (multi-clips, sans split) *(à archiver en A7)*
- `poetry run python -m ainimator.cli.train_controller_generalization_v2` —
  A6 : split train / held-out, eval de généralisation *(devient le profil
  `full` unifié en A7)*
- `poetry run python -m ainimator.cli.generate_controller_v2` — rollout +
  export anim
- `poetry run python -m ainimator.cli.export_onnx controller` — export d'un
  **seul forward** contrôleur (A5)

**Partagé (data, encodeur, santé, qualité) :**
- `poetry run python -m ainimator.cli.build_dataset` — match prompts
- `poetry run python -m ainimator.cli.preprocess_dataset` — preprocess
- `poetry run python -m ainimator.cli.train_custom_tokenizer` — BPE
- `poetry run python -m ainimator.cli.train_text_encoder` — standalone encoder
  (encodeur intégré au modèle tout-en-un)
- `poetry run python -m ainimator.cli.health {watch|audit|diagnose|report}` —
  outil de debug global (`diagnose` route selon `model-type`)
- `poetry run pytest test/ainimator/training/test_controller_training_v2.py -v`
  — smoke test overfit contrôleur (sanity check canonique, vérité #7)
- `poetry run streamlit run apps/monitor/app.py` — moniteur santé Streamlit
  (`ROADMAP_MONITOR.md`)
- `poetry run pytest` — test suite
- `poetry run lint-imports` — verify layer contracts (must be green)

> **Makefile supprimé (2026-06-24).** Les raccourcis (`smoke-test`,
> `debug-run`, `monitor`, futurs `plugin-unity/unreal`) étaient périmés
> post-pivot ; ils sont remplacés par les commandes Poetry ci-dessus. Un
> Makefile sera **refait** avec l'orchestrateur de build `apps/build/`
> (Goal B / phase B5, cf. `ROADMAP_PLUGINS.md §3.4`).

> ⚠ **Export NON migré.** `export_onnx` reste centré
> `{encoder|denoiser|controller}` (un forward). Le **bundle contrôleur**
> attendu par le Goal B (`--bundle` : onnx + `norm_stats.json` +
> `manifest.json` + `presets/`, cf. `ROADMAP_PLUGINS.md` B0) **n'existe pas
> encore** — trou de code à combler avant tout packaging plugin.

**Voie diffusion (inactive, non migrée — conservée pour référence) :**
- `poetry run python -m ainimator.cli.train_generation_v2 --profile {overfit,full,debug}`
  (équivalent : flag `--debug`)
- `poetry run python -m ainimator.cli.generate_animation_v2` — sample + .dae
- `poetry run python -m ainimator.cli.diagnose_generation_v2` — diagnostics
  conditionnement (absorbé dans `ainimator.cli.health`)
- `poetry run python -m ainimator.cli.export_onnx {encoder|denoiser}`

Legacy v1 (archived — see `legacy/`): `train_clip`,
`train_generation`, `generate_animation` v1, `precompute_generation_text_cache`.
Do NOT import from `legacy/` in any `src/ainimator/` module — enforced by
`lint-imports` contract `no_legacy_imports`.

## Rules for agents working on this repo
Five project agents are defined in `.claude/agents/`, **scoped by function /
filesystem boundary** (the phase is only a cursor):
- **orchestrator** (Opus, global vision — decomposes a prompt, routes to the right specialist, owns run/regression analysis, and relays data-extraction requests to dev-python-neural; reads everything, writes no code),
- **dev-python-neural** (Sonnet, Python / neural-network specialist — the network under `src/ainimator/`: controller + integrated text encoder + health/ + export; also data-extraction requests in `doc/experiments/requests/`),
- **reviewer** (Sonnet, read-only QA / acceptance + conventions gate),
- **dev-unity-plugin** (C#/Sentis, `apps/unity-sentis/` only),
- **dev-unreal-plugin** (C++/NNE, `apps/unreal-nne/` only).

Role split: ROADMAP §5. Agents are routed by their file `name:`
(`orchestrator`, `dev-python-neural`, `reviewer`, `dev-unity-plugin`,
`dev-unreal-plugin`). The former `experimenter` role was removed (2026-06-25):
run/regression analysis folds onto the orchestrator (Opus) + Pazimor; the
`doc/experiments/requests/` workflow stays, fulfilled by dev-python-neural.

- Follow the phases of the relevant fiche **in order**; a phase is done only
  when ALL its acceptance criteria pass.
- Conventions are in `doc/GUIDELINES.md`, also packaged as the **`guidelines`
  skill** (`.claude/skills/guidelines/`) that agents load on any code /
  review task. See "Code conventions" above for how to retrieve one paragraph. Key ones: `G-HYPERPARAMS` (never change
  defaults without Pazimor), `G-RESOLVEDCONFIG` (every run writes
  `resolved_config.yaml`), `G-RUNS` (smoke test before any long run; never
  > 30 min without asking), `G-COMMITS` (atomic commits, ask when ambiguous),
  `G-LOG` (one LOG.md line per experiment; new decisions dated in ROADMAP*).
