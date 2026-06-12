# Request 001 — Recover & consolidate June scaling results
status: partially-done (2026-06-12 — B0 consolidated in LOG.md from the
Opus session; this request now only covers archiving the raw per-run
values from /tmp if they still exist. Low priority.)
requested-by: experimenter (pre-filed at roadmap creation)
date: 2026-06-12

## What
The experiment matrix for protocol step B0 (ROADMAP §5.3): one row per
June run with {run name, N samples, encoder type, embed dim, layers,
params count, epochs, exposures/sample, fidelity, retrieval,
distinctness, cfg_sim, best val loss}.

## Sources
- `/tmp/scaling_results.txt`, `/tmp/decisive_n100.txt` (volatile —
  Pazimor must confirm they still exist, else re-evaluate from
  checkpoints below)
- Checkpoints: `output/n100_decisive/`, `output/n500/`,
  `output/n500e600/`, `output/n500_big/`, `output/n1000_big/`,
  `output/n1000_cfg/`, `output/n1000_640/` (v2_full_best.pt each)
- Re-evaluation tooling: `scripts/scaling_curve_v2.py::_evaluate` (same
  5 probes, seed 0)

## Format
CSV at `doc/experiments/data/june_scaling_matrix.csv` + the
"Not yet consolidated" lines of LOG.md updated with verdicts.

## Done when
Every June run has a complete row (or an explicit "checkpoint
unreadable/missing" note), and LOG.md has no remaining
"to be recovered" entry for June.
