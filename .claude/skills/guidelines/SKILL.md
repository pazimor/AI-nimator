---
name: guidelines
description: >-
  AI-nimator code conventions (the G-XXX guidelines). Use whenever writing,
  reviewing, or refactoring code in this repo, or when the user mentions
  conventions, code style, docstrings, typing, magic numbers, method/file
  length, imports/layers, ONNX-exportability, resolved config, profiles, the
  Poetry env, or atomic commits. The single source of truth is
  doc/GUIDELINES.md; this skill only tells you how to pull the exact paragraph
  you need and which guidelines you must never silently break.
---

# AI-nimator — guidelines (conventions G-XXX)

The canonical text lives in **`doc/GUIDELINES.md`** — one self-contained
paragraph per guideline, each tagged with a **stable ID** (`G-XXX`) on its
heading line. IDs never change; cite them in reviews and commit messages.

**Do not duplicate the index here.** Read it from the source when you need it:

```bash
grep -n '^- `G-' doc/GUIDELINES.md     # list all IDs + one-line summaries
grep -A8 'G-DOCSTRING' doc/GUIDELINES.md   # pull one full guideline (or: rg -A8)
```

## Non-negotiable (STOP and escalate, never silently break)

- **G-ENV** *(mandatory)* — environment is Poetry only (Python 3.12/3.13);
  `pyproject.toml` is PEP 621, `requires-python` is a PEP 440 specifier
  (`>=3.12,<3.14`), never a Poetry caret. After editing it: `poetry lock`
  then `poetry install`. Never bypass Poetry with a global `pip`.
- **G-HYPERPARAMS** — never change an existing hyperparameter default without
  an explicit Pazimor instruction.
- **G-PROFILES** *(règle dure)* — a run is configured by a named YAML
  **profile**, never by hyperparameter CLI flags. To test a value, add a
  profile; never re-introduce an `add_argument` for a hyperparameter.
- **G-ONNX** — every `forward()` stays ONNX-traceable: no data-dependent
  control flow, no `.item()` on the graph path; the autoregressive loop and
  the text-encoder pass stay out of the per-frame graph (one forward = one
  step; the text embedding is computed upstream).
- **G-IMPORTS** — imports go down only; `model` never imports `data`;
  `training` is the sole `data`<->`model` junction; `cli` carries no logic;
  `legacy/` importable by nothing. Enforced by `lint-imports` (must be green).
- **G-RESOLVEDCONFIG** — every run writes `resolved_config.yaml`
  (+ git SHA + date) in its `outputDir`.
- **G-RUNS** — smoke test (overfit profile / `--debug`) before any long run;
  never launch a run > 30 min without asking Pazimor.
- **G-COMMITS** — atomic commits; ask when a change spans concerns.

## Usage

Consult the relevant `G-ID` (grep it) before producing or judging code. In a
review, cite the ID and the `file:line` that violates it. If a task seems to
require breaking a non-negotiable above, STOP and report rather than work
around it.
