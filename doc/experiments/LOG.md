# Experiments LOG

One line per concluded experiment. Format:
`| date | run | config (delta vs default) | metrics | verdict |`

| Date | Run | Config | Metrics | Verdict |
|---|---|---|---|---|
| 2026-06 | scaling sweep N=5/20/100/400 (`scaling_curve_v2.py`) | constant step budget, ACCAD, frozen CLIP | fidelity N=100: −0.151, retrieval 0.20, distinct +0.394 | Confounded (exposure 700→56) — cliff was under-training, see decisive test |
| 2026-06-07 | `n100_decisive` (600 exposures/sample) | same 100 samples/probes as sweep | fidelity recovered (vs −0.151 @56 exp) | Under-training confirmed as the N=100 cliff cause |
| 2026-06-07→10 | `n500`, `n500e600`, `n500_big`, `n1000_big`, `n1000_cfg` | capacity ladder probes | consolidated into the capacity law below | Intermediate points: 512/6 (~45M) holds ~N≤500; N=1000 needs more |
| 2026-06-12 | `n1000_640` (640d/8L, ~90M params) | N=1000, cond-mask-prob 0.10, grad-accum 1, no epoch cap, ~600 exp/sample, frozen CLIP (diagnostic encoder) | cfg=1: fid +0.648 / ret 0.20 / dist +0.832 — cfg=4: +0.591 / 0.80 / +0.339 — cfg=6: +0.607 / **1.00 (5/5)** / +0.034 | ✅ **640/8 solves N=1000.** CFG 4–6 essential at inference (model contains the conditioning, CFG expresses it) |
| 2026-06-12 | **Capacity↔N law (3 clean points)** | 384/4 ~17M → N≤~50 ; 512/6 ~45M → N≤~500 ; 640/8 ~90M → N=1000 ✓ | — | **params ≈ ∝ N** (~double params when N doubles) — measured in memorization regime (train==val); generalization bend unknown → step B1-ter |

## Validated recipe (2026-06-12)
- Denoiser capacity ∝ number of distinct motions (double params when N doubles).
- `cond-mask-prob 0.10` (else degenerate collapse + no usable CFG).
- **CFG 4–6 at inference** — mandatory at scale (cfg=1 → retrieval 0.20, cfg=6 → 1.00).
- Sufficient exposure: grad-accum 1, no `maxSamplesPerEpoch` cap, ~600 exposures/sample.

Root cause of the original mystery ("model ignores the prompt at scale"):
under-sized denoiser AND sampling without CFG. Both fixed.
