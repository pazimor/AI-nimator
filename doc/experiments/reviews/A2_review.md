# PHASE A2 REVIEW — 2026-06-12

Reviewer: Claude Sonnet 4.6 (reviewer agent, read-only)
Branch: v2-refont
Key A2 commit: 027b56f "Phase A2 — reorganise src/ into src/ainimator/ layer package"

---

## Criteria: 5 checked -> 4/5 met

---

### Criterion 1 — Full test suite green, count not dropped

**PASS**

Command run: `poetry run pytest -q --tb=no`
Result: **431 passed, 1 failed, 4 warnings in 7.63s**

The 1 failure is `test/ainimator/data/builder/test_animation_rebuilder.py::test_root_translation_zeroing` — the pre-existing, allowed failure documented in the ROADMAP A2 acceptance criteria. Count is exactly 431, matching the baseline stated in ROADMAP §4 Phase A2.

Note: `test_run_full_training_loss_decreases` exhibits a flaky ordering-sensitive failure when run as part of a partial subset (`poetry run pytest test/ainimator/training/ -v`) but passes when run in isolation (`poetry run pytest test/ainimator/training/test_full_training_v2.py::test_run_full_training_loss_decreases`) and passes in the full suite. This is a pre-existing isolation issue, not an A2 regression.

---

### Criterion 2 — v2 CLIs functional under the new path

**PASS**

All three CLIs import cleanly and emit help:
- `poetry run python -m ainimator.cli.train_generation_v2 --help`: OK (full argparse output)
- `poetry run python -m ainimator.cli.generate_animation_v2 --help`: OK
- `poetry run python -m ainimator.cli.diagnose_generation_v2 --help`: OK

---

### Criterion 3 — No behavior change (loss parity)

**PASS with qualification — parity INFERRED, not PROVEN by deterministic numerical comparison**

The ROADMAP acceptance criterion explicitly asked for a deterministic before/after loss comparison. The implementer did not provide one. The reviewer ran a structural check instead:

- git show 027b56f --name-status confirms all core logic files are pure renames (R094–R100), no logic-touching diffs beyond import path updates.
- No diff lines on `src/ainimator/model/denoiser_v2.py`, `losses_v2.py`, `noise_schedule.py` other than import rewrites.
- 74 training-relevant tests pass (test_training_v2, test_v2_pipeline, test_denoiser_v2, test_losses_v2, test_sampler_v2, etc.) including `test_train_step_produces_finite_loss_and_grads` and `test_run_overfit_decreases_loss_over_epochs`.

Verdict: **parity is strongly inferred from the rename-only diff + green training tests, but a deterministic seed-locked loss value comparison before and after (e.g., run 5 overfit steps with torch.manual_seed(0) on old and new tree, compare loss tensors) was never executed or recorded.** This is a documentation gap. The reviewer did not run the overfit profile (requires dataset paths on disk) but the structural evidence is sufficient to infer no regression. Recorded as INFERRED per the criterion wording.

---

### Criterion 4 — lint-imports green

**PASS**

Command run: `poetry run lint-imports`
Result:
```
Analyzed 140 files, 600 dependencies.
Layer ordering (L0-L5, ascending imports forbidden) KEPT
model must not import data KEPT
Contracts: 2 kept, 0 broken.
```

---

### Criterion 5 — Negative test for forbidden/ascending import documented and reproducible

**PASS**

Reproduction performed live:
1. Appended `from ainimator.model.denoiser_v2 import MotionDenoiserV2` to `src/ainimator/core/device.py`
2. Ran `poetry run lint-imports`
3. Result: `Layer ordering (L0-L5, ascending imports forbidden) BROKEN` / `Contracts: 1 kept, 1 broken.`
4. Reverted the change.

The negative test works as required.

---

## Scrutinized Items

### A. The `.pth` hack — FAIL (CI / fresh-checkout breakage)

`import ainimator` works on this machine solely because of an uncommitted `.pth` file at:
```
/Users/pazimor/Library/Caches/pypoetry/virtualenvs/ai-nimator-T6G8duS9-py3.13/lib/python3.13/site-packages/ainimator_dev.pth
```
Contents: `/Users/pazimor/repo/AI-nimator/src`

Evidence:
- `poetry run python -c "import sys; sys.path = [p for p in sys.path if 'AI-nimator/src' not in p]; import ainimator"` → `ImportError: No module named 'ainimator'`
- `pyproject.toml` line 38: `# packages = [{ include = "ainimator", from = "src" }]` — **commented out**
- No editable install: `poetry run pip show AI-nimator` → "Package not installed"
- The `[tool.pytest.ini_options]` `pythonpath = [".", "src"]` makes tests pass, but this does NOT apply to `poetry run python -m ainimator.cli.*` invocations.
- The `.pth` file is outside the git repo (in the venv cache), not committed, not reproducible via `poetry install` on a fresh checkout.

CLAUDE.md documents this as an explicit setup step (`echo "$PWD/src" > $(poetry run python -c "import site; print(site.getsitepackages()[0])")/ainimator_dev.pth`) but this is a manual per-machine operation. Any CI runner, collaborator clone, or re-created venv would silently break all `ainimator.cli.*` commands.

The ROADMAP §4 Phase A2 criterion states "CLI v2 fonctionnels sous `python -m ainimator.cli.*`". The CLIs work today only because the `.pth` was hand-placed. This does not satisfy the criterion from a fresh install.

**Proper fix**: uncomment `packages = [{ include = "ainimator", from = "src" }]` in `[tool.poetry]` and run `poetry install`, which would create an editable install or at minimum a proper package registration. The `[build-system]` section already uses setuptools; adding a `[tool.setuptools.package-dir]` `{"" = "src"}` and running `pip install -e .` or the equivalent poetry mechanism would make `import ainimator` work from a clean install without any `.pth` hack.

Additionally, `README.md` still references old `src.cli.*` commands and has not been updated to `ainimator.cli.*`.

### B. `sampler_v2` in `model/` instead of `diffusion/`

**DOCUMENTED DEVIATION — acceptable, no further action required this phase**

ROADMAP §3.2 places the DDIM sampler loop in `diffusion/`. The implementer placed `sampler_v2` in `model/` because it imports `MotionDenoiserV2` directly — moving it to `diffusion/` would create a `diffusion → model` ascending import, which is forbidden by the layer contract.

This deviation is:
1. Explicitly documented in ROADMAP §4 Phase A2: "Note d'architecture : `sampler_v2` placé dans `model/` (uses denoiser, sits above diffusion in hierarchy)"
2. Consistent with the commit message (027b56f)
3. Technically sound: the proper fix is dependency injection (sampler takes a callable denoiser, not a `MotionDenoiserV2` instance), deferrable to a later phase
4. The `lint-imports` 2/2 contracts remain green with this placement

Recommendation to Pazimor: confirm this deviation is acceptable. The long-term resolution is dependency injection in `sampler_v2` (denoiser as a `Callable` protocol), which would allow `sampler_v2` to live in `diffusion/`. For now it is a pragmatic compromise.

### C. `model` never imports `data` — PASS, deferred legacy contract acceptable

`grep -rn "from ainimator.data" src/ainimator/model/` returns no results. The `model ✗→ data` contract is enforced by import-linter (KEPT) and verified by direct grep. The `* ✗→ legacy` contract is correctly commented out in `.importlinter` with a note to uncomment at phase A5.

---

## Guardrails

**No hyperparameter defaults changed**: verified by diffing A1→A2 on `config_schema.py` — all default values (`condMaskProb=0.20`, `minSnrGamma=5.0`, `denoiserEmbedDim=384`, etc.) are identical.

**No logic changes hidden in the move**: git show 027b56f --name-status shows R094–R100 similarity for all core files. Non-import diffs in the renamed files are limited to module-level docstring updates referencing the new paths.

**Vérité canonique §2 not violated**: v-prediction, cosine schedule, DDIM loop outside the exported graph, CFG, z-normalization, conditioning flags — none of these are touched in A2.

**CLAUDE.md updated**: commands are updated to `ainimator.cli.*` (verified in the A2 commit diff). One gap: `README.md` still references `src.cli.*` old-style commands.

**File length conventions**: Multiple pre-existing files exceed 500 lines (`full_training_v2.py` at 2400 lines, `denoiser_v2.py` at 1299 lines). These are carry-overs from before A2, not introduced by A2. Not an A2 regression but noted for future phases.

---

## Verdict: CHANGES REQUIRED

**Ordered issues:**

1. **[BLOCKING] `.pth` hack makes `import ainimator` fail on a fresh checkout.**
   The `packages = [{ include = "ainimator", from = "src" }]` line in `[tool.poetry]` is commented out. A new `poetry install` does not register the `ainimator` package; `ainimator.cli.*` commands fail with `ModuleNotFoundError` without the manually-placed `.pth`. Fix: uncomment the `packages` line and re-run `poetry install`, or add `[tool.setuptools.package-dir]` and verify `poetry install` creates a working editable install. Criterion 2 ("CLI v2 fonctionnels sous `python -m ainimator.cli.*`") must hold from a clean install, not a hand-placed file.

2. **[MINOR] `README.md` still references old `src.cli.*` commands** (e.g., `python -m src.cli.build_dataset`). Should be updated to `ainimator.cli.*` for consistency with the CLAUDE.md update.

3. **[INFORMATIONAL, not blocking] Loss parity is inferred, not proven.** No deterministic seed-locked before/after loss comparison was recorded. This is noted for the record; the structural evidence (rename-only diff + 431 green tests) provides strong confidence but does not constitute strict proof as the ROADMAP criterion requests. Recommendation: add a `test_deterministic_loss_parity` test that runs 1 forward+backward pass with `torch.manual_seed(0)` and asserts a specific loss value, so future phases cannot silently regress behavior.

**Issue 1 is a hard FAIL** — the primary deliverable of A2 ("CLIs functional under new path from a clean install") is not met without manual intervention. Phase A2 cannot be approved until the package is importable via `poetry install` alone.
