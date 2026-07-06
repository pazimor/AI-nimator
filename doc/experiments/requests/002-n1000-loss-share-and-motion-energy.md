# Request 002 — N=1000 loss-share + generated-vs-GT motion-energy readout
status: pending
requested-by: experimenter
date: 2026-06-21

## What
Two tables to settle the "static motion at N=1000" diagnosis without a
new long run. From the most recent N=1000 checkpoint + its training logs:

1. **Loss-share table** (the `loss_share_*` / `loss_*` metrics the health
   hub already emits in `hub.step()`), averaged over the last ~10 logged
   training steps of the run, columns:
   `component, unweighted_value, weight, weighted_value, share_pct`
   for every component in `_weightedComponents`
   (bone, global, vel_xyz, acceleration, rotation_jerk, joint_xyz,
   foot_contact, clip_guidance, clip_aux_pool, x0_contrastive).
   Goal: confirm/deny that `clip_guidance` (w=0.6) dominates and that the
   FK/smoothness terms sit < 1 % (dead) per §3.5 `loss_share` thresholds.

2. **Motion-energy table** — for the held-in probe set, per cfg ∈ {1,4,6}:
   `cfg, gen_frame_velocity_mean, GT_frame_velocity_mean,
   gen_total_displacement, GT_total_displacement, ratio_gen_over_GT`
   where frame_velocity = mean ‖Δ(FK_joint_xyz)‖ over time, and
   total_displacement = root_translation span over the clip.
   Goal: quantify "near-static" — is gen motion-energy << GT (collapse to
   mean) or comparable but visually muted (a smoothing/orientation issue)?
   **Compute on RAW generations (smooth-sigma = 0)** so the readout is not
   confounded by inference-time damping (LOG 2026-06-18 noted smooth-sigma
   2.0 alone cut stride 0.43→0.22).

## Sources
- Latest N=1000 checkpoint: **`output/recipe_n1000/checkpoints/v2_full_best.pt`**
  (CONFIRMED 2026-06-21 as the most recent N=1000 run — gens dated today;
  resolved_config: `dataset_preprocessed_canon`, 640/8, cond-mask 0.20,
  clipGuidanceWeight 0.6, velocityXyzWeight 0.0, jointPositionWeight 0.0,
  velocitySchedule timestep → this IS the starving-mix baseline to quantify).
- Probe split: `output/recipe_n1000/probe_selection.json` (train 1000,
  held-in 22, held-out 22; max train idx ~13.4k = 3-folder canon set).
- Dataset: `/Users/pazimor/dataset_preprocessed_canon` (CONFIRMED on disk,
  2.6 GB, upright 3-folder set — matches the run's resolved_config).
- Loss-share: training stdout/health log of that run, or re-run a single
  forward batch through `trainStepBatch` with the run's resolved_config.

## Format
JSON + CSV at `doc/experiments/data/002_n1000_loss_share.csv` and
`doc/experiments/data/002_n1000_motion_energy.csv`.

## Done when
Both CSVs exist with the columns above, and the loss-share rows sum to
~100 %. No model/default changes — read-only extraction.
