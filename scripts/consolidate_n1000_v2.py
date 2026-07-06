"""B1-bis consolidation run — lock the validated N=1000 recipe.

This script reproduces the recipe that solved N=1000 in June 2026
(LOG.md 2026-06-12, run ``n1000_640``) as a single, documented,
reproducible run — the ROADMAP §5.3 step B1-bis deliverable.

Recipe (frozen here, authoritative — ``network.yaml`` is NOT read by the
v2 training path, see that file's note):

* denoiser 640d / 8L  (~90M params — the capacity↔N law point for N=1000)
* frozen CLIP text tower (continuity with the law; custom BPE is B2)
* ``cond-mask-prob 0.20``  (default; overridable).  The June LOG used
  0.10, but on a diverse multi-folder N=1000 that collapsed the denoiser
  (run b1bis_n1000_640, 2026-06-16: cfg_sim 0.971, seed_sim 0.966).  0.20
  is the dataclass value tuned against exactly this failure mode.
* no ``maxSamplesPerEpoch`` cap, grad-accum 1
* ~200 epochs — the loss plateaus by ~epoch 100-150 (the 600-epoch run
  was ~3-4× wasteful)
* evaluation at cfg ∈ {1, 4, 6} (CFG 4–6 is mandatory at inference)

Two improvements over the original sweep, per B1-bis:

* probe set widened to ~24 motion verbs (5/5 on 5 probes was a weak
  signal) — reported as ``held-in``;
* a disjoint ``held-out`` probe set (prompts NEVER seen in training) for
  a first generalization read — absent from the protocol until now.

Usage (poetry is broken on this box — call the venv python directly)::

    AIPY=~/Library/Caches/pypoetry/virtualenvs/ai-nimator-T6G8duS9-py3.13/bin/python
    $AIPY scripts/consolidate_n1000_v2.py                 # train + eval
    $AIPY scripts/consolidate_n1000_v2.py --eval-only      # re-eval only
    $AIPY scripts/consolidate_n1000_v2.py --epochs 50      # quick smoke

⚠ Run is LONG: N=1000, batch 8 → ~125 steps/epoch; 200 epochs ≈ 25k
steps on a ~90M model → ~12-15h on MPS (vs 47h for the original 600).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ainimator.data.preprocessed_dataset import PreprocessedLinkDataset
from ainimator.health.evaluation import (
    CANONICAL_CFG_SCALES,
    evaluateMultiCfg,
    pickProbes,
)
from ainimator.model.sampler_v2 import DDIMSamplerV2
from ainimator.training.full_training_v2 import (
    V2FullTrainingConfig,
    runFullTraining,
)
from ainimator.training.training_v2 import EMPTY_PROMPT, loadCheckpointV2

# --- Paths -----------------------------------------------------------
# 2026-06-21 — default to the UPRIGHT canonicalised set.  The old
# ``/Users/pazimor/dataset_preprocessed`` is mis-oriented (AMASS Z-up
# root_orient on a Y-up rest skeleton → bodies stored lying down; FK
# metrics degraded — see memory ``preprocessed-data-misoriented`` and
# LOG 2026-06-17).  ``dataset_preprocessed_canon`` is the 3-folder
# (ACCAD/BMLmovi/BioMotionLab) upright set; override with --dataset-root.
DATASET = Path("/Users/pazimor/dataset_preprocessed_canon")
OUT_DIR_DEFAULT = Path("output/b1bis_n1000_640")
BEST_CHECKPOINT = "checkpoints/v2_full_best.pt"
LATEST_CHECKPOINT = "checkpoints/v2_full_latest.pt"
RUN_COMMAND_FILE = "run_command.txt"
# Probe selection is deterministic but slow (~8 min: pickProbes loads
# each candidate's motion tensor).  Cache it so ``--eval-only`` reuses
# the EXACT same train/probe split instead of recomputing it.
SELECTION_FILE = "probe_selection.json"

# --- Validated recipe (locked) --------------------------------------
N_TRAIN = 1000
# Loss plateaued at epoch ~100-150 on the 47h 600-epoch run
# (b1bis_n1000_640, LOG 2026-06-16) — 600 was ~3-4× wasteful.
EPOCHS_DEFAULT = 200
DENOISER_EMBED_DIM = 640
DENOISER_NUM_LAYERS = 8
DENOISER_NUM_HEADS = 8  # 640 / 8 = 80 dims per head
# cond-mask-prob 0.10 (the June LOG value) collapsed at the denoiser on a
# diverse multi-folder N=1000 (cfg_sim 0.971, seed_sim 0.966).  0.20 is
# the dataclass default tuned against exactly this (full_training_v2.py
# comment).  Overridable via --cond-mask-prob.
COND_MASK_PROB = 0.20
BATCH_SIZE = 8
NORMALIZER_FIT_MAX = 512
VALIDATE_EVERY_EPOCHS = 20

# --- Probe / evaluation ---------------------------------------------
EVAL_FRAMES = 200
EVAL_NUM_STEPS = 100
MIN_PROBE_FRAMES = 80
PROBE_SCAN_LIMIT = 6000  # bound the I/O of probe selection
SHUFFLE_SEED = 0
# Widened verb set (B1-bis): ~24 motions covering the standard probe
# verbs plus common AMASS actions, for a statistically meaningful read.
PROBE_VERBS: tuple[str, ...] = (
    "walk", "run", "jump", "sit", "stand", "wave", "kick", "throw",
    "turn", "dance", "jog", "step", "bend", "raise", "punch", "clap",
    "march", "spin", "climb", "crouch", "squat", "stretch", "balance",
    "walk backward",
)


def _shuffledIndices(dataset: PreprocessedLinkDataset) -> list[int]:
    """Return all link indices in a deterministic shuffled order.

    Parameters
    ----------
    dataset : PreprocessedLinkDataset
        Loaded dataset (provides ``linkEntries``).

    Returns
    -------
    list[int]
        Every link index, shuffled with :data:`SHUFFLE_SEED`.
    """
    indices = list(range(len(dataset.linkEntries)))
    random.Random(SHUFFLE_SEED).shuffle(indices)
    return indices


def _pickProbeSet(
    dataset: PreprocessedLinkDataset, pool: list[int]
) -> list[int]:
    """Pick one probe per verb from a bounded slice of ``pool``.

    Parameters
    ----------
    dataset : PreprocessedLinkDataset
        Dataset to read ``raw_text`` / ``motion`` from.
    pool : list[int]
        Candidate link indices (already shuffled).

    Returns
    -------
    list[int]
        Up to ``len(PROBE_VERBS)`` probe indices.
    """
    return pickProbes(
        dataset,
        pool[:PROBE_SCAN_LIMIT],
        keywords=PROBE_VERBS,
        minFrames=MIN_PROBE_FRAMES,
    )


def _buildTrainIndices(probes: list[int], pool: list[int]) -> list[int]:
    """Assemble the N_TRAIN training set including the held-in probes.

    Parameters
    ----------
    probes : list[int]
        Held-in probe indices that MUST be in the training set.
    pool : list[int]
        Shuffled candidate indices to fill the remainder from.

    Returns
    -------
    list[int]
        Exactly ``min(N_TRAIN, available)`` distinct link indices.
    """
    probeSet = set(probes)
    filler = [index for index in pool if index not in probeSet]
    remaining = N_TRAIN - len(probes)
    return probes + filler[:remaining]


def _buildConfig(
    indices: list[int], epochs: int, outDir: Path, device: str,
    condMaskProb: float, datasetRoot: Path, velXyzWeight: float,
    clipGuidanceWeight: float, auxPoolWeight: float, minSnrGamma: float,
    rotationJerkWeight: float, velocitySchedule: str,
    jointPositionWeight: float, footContactWeight: float,
    denoiserEmbedDim: int, denoiserNumLayers: int,
    denoiserNumHeads: int,
) -> V2FullTrainingConfig:
    """Build the locked recipe config for the consolidation run.

    Parameters
    ----------
    indices : list[int]
        Exact training link indices (train == val).
    epochs : int
        Number of epochs (≈ exposures/sample with no cap).
    outDir : Path
        Output directory for checkpoints / resolved_config.yaml.
    device : str
        Torch device string.
    condMaskProb : float
        CFG-dropout probability (0.20 default; 0.10 collapsed).

    Returns
    -------
    V2FullTrainingConfig
        The frozen B1-bis recipe.
    """
    return V2FullTrainingConfig(
        datasetRoot=datasetRoot,
        tokenizerDir=Path("/tmp"),  # unused on the frozen-CLIP path
        outputDir=outDir,
        sampleLinkIndices=tuple(indices),
        textEncoderType="clip",
        denoiserEmbedDim=denoiserEmbedDim,
        denoiserNumLayers=denoiserNumLayers,
        denoiserNumHeads=denoiserNumHeads,
        condMaskProb=condMaskProb,
        velocityXyzWeight=velXyzWeight,
        clipGuidanceWeight=clipGuidanceWeight,
        # No warmup ramp: keep the contrastive at its (reduced) weight
        # from epoch 0 so the reconstruction loss can drive full-
        # amplitude motion (start must be >= weight per config check).
        clipGuidanceWeightStart=clipGuidanceWeight,
        auxPoolContrastiveWeight=auxPoolWeight,
        minSnrGamma=minSnrGamma,
        rotationJerkWeight=rotationJerkWeight,
        # 2026-06-21 — request 002 / LOG loss-mix rebalance levers.
        # ``velocitySchedule="none"`` supervises smoothness/velocity at
        # ALL noise levels (MDM-style) instead of fading them at high t
        # (the "timestep" default crushed those shares < 1% = dead).
        # jointPosition / footContact are MDM geometric losses; passing
        # them here lets the orchestrator drive a reconstruction-heavy
        # mix WITHOUT touching any src/ default.
        velocitySchedule=velocitySchedule,
        jointPositionWeight=jointPositionWeight,
        footContactWeight=footContactWeight,
        epochs=epochs,
        batchSize=BATCH_SIZE,
        gradientAccumulation=1,
        maxSamplesPerEpoch=0,  # no cap — every sample every epoch
        normalizerFitMaxSamples=NORMALIZER_FIT_MAX,
        validateEveryEpochs=VALIDATE_EVERY_EPOCHS,
        bestMetric="loss_diffusion",
        device=device,
        seed=0,
    )


def _resolveCheckpoint(outDir: Path) -> Path:
    """Return the best checkpoint, falling back to the latest.

    A best checkpoint only exists once validation has run at least once
    (``validateEveryEpochs``); short smoke runs only have the latest.

    Parameters
    ----------
    outDir : Path
        Run directory.

    Returns
    -------
    Path
        Existing checkpoint path (best preferred, else latest).

    Raises
    ------
    FileNotFoundError
        If neither checkpoint exists.
    """
    best = outDir / BEST_CHECKPOINT
    latest = outDir / LATEST_CHECKPOINT
    if best.exists():
        return best
    if latest.exists():
        print(f"[warn] no best checkpoint; evaluating {latest.name}.")
        return latest
    raise FileNotFoundError(f"No checkpoint under {outDir}/checkpoints.")


def _evaluateSet(
    ckptPath: Path, probes: list[int],
    dataset: PreprocessedLinkDataset, device: str,
) -> list[dict[str, float]]:
    """Evaluate a probe set at cfg ∈ {1, 4, 6} from a checkpoint.

    Parameters
    ----------
    ckptPath : Path
        Checkpoint to load (see :func:`_resolveCheckpoint`).
    probes : list[int]
        Probe link indices to regenerate.
    dataset : PreprocessedLinkDataset
        Source of GT motions / prompts.
    device : str
        Torch device string.

    Returns
    -------
    list of dict
        One row per cfg scale (fidelity / retrieval / distinctness).
    """
    tokenizer, encoder, denoiser, schedule, normalizer, _ = (
        loadCheckpointV2(ckptPath, device=device)
    )
    encoder.eval()
    denoiser.eval()
    sampler = DDIMSamplerV2(schedule, predictionMode="v")
    return evaluateMultiCfg(
        probeIndices=probes,
        dataset=dataset,
        tokenizer=tokenizer,
        encoder=encoder,
        denoiser=denoiser,
        sampler=sampler,
        normalizer=normalizer,
        device=device,
        cfgScales=CANONICAL_CFG_SCALES,
        frames=EVAL_FRAMES,
        numSteps=EVAL_NUM_STEPS,
        emptyPrompt=EMPTY_PROMPT,
    )


def _formatRows(label: str, rows: list[dict[str, float]]) -> str:
    """Format evaluation rows as an aligned text block.

    Parameters
    ----------
    label : str
        Probe-set label ("held-in" / "held-out").
    rows : list of dict
        Output of :func:`_evaluateSet`.

    Returns
    -------
    str
        Multi-line table fragment.
    """
    header = f"[{label}]  cfg  fidelity  retrieval  distinctness"
    lines = [header]
    for row in rows:
        lines.append(
            f"           {row['cfg_scale']:>3.0f}  "
            f"{row['fidelity']:+.3f}    {row['retrieval']:.2f}       "
            f"{row['distinctness']:+.3f}"
        )
    return "\n".join(lines)


def _writeRunCommand(
    outDir: Path, epochs: int, condMaskProb: float,
    heldIn: list[int], heldOut: list[int],
) -> None:
    """Record the exact recipe + probe selection for traceability.

    Parameters
    ----------
    outDir : Path
        Run directory.
    epochs : int
        Epochs used.
    condMaskProb : float
        CFG-dropout probability actually used.
    heldIn, heldOut : list[int]
        Probe index lists.
    """
    outDir.mkdir(parents=True, exist_ok=True)
    text = (
        "B1-bis consolidation run (scripts/consolidate_n1000_v2.py)\n"
        f"N_TRAIN={N_TRAIN}  epochs={epochs}  "
        f"(~{epochs} exposures/sample)\n"
        f"denoiser={DENOISER_EMBED_DIM}d/{DENOISER_NUM_LAYERS}L/"
        f"{DENOISER_NUM_HEADS}h  cond_mask_prob={condMaskProb}\n"
        f"text_encoder=clip  batch={BATCH_SIZE}  grad_accum=1  "
        "max_samples_per_epoch=0\n"
        f"eval cfg={CANONICAL_CFG_SCALES}  frames={EVAL_FRAMES}  "
        f"shuffle_seed={SHUFFLE_SEED}\n"
        f"held_in_probes ({len(heldIn)})={heldIn}\n"
        f"held_out_probes ({len(heldOut)})={heldOut}\n"
    )
    (outDir / RUN_COMMAND_FILE).write_text(text)


def _computeSelection(
    dataset: PreprocessedLinkDataset,
) -> tuple[list[int], list[int], list[int]]:
    """Compute training indices + held-in / held-out probe sets.

    Returns
    -------
    tuple
        ``(trainIndices, heldInProbes, heldOutProbes)``.
    """
    shuffled = _shuffledIndices(dataset)
    heldIn = _pickProbeSet(dataset, shuffled)
    trainIndices = _buildTrainIndices(heldIn, shuffled)
    trainSet = set(trainIndices)
    heldOutPool = [index for index in shuffled if index not in trainSet]
    heldOut = _pickProbeSet(dataset, heldOutPool)
    return trainIndices, heldIn, heldOut


def _selectProbes(
    dataset: PreprocessedLinkDataset, outDir: Path,
) -> tuple[list[int], list[int], list[int]]:
    """Load the cached selection, or compute and cache it.

    Parameters
    ----------
    dataset : PreprocessedLinkDataset
        Loaded dataset.
    outDir : Path
        Run directory holding the cached :data:`SELECTION_FILE`.

    Returns
    -------
    tuple
        ``(trainIndices, heldInProbes, heldOutProbes)``.
    """
    cachePath = outDir / SELECTION_FILE
    if cachePath.exists():
        cached = json.loads(cachePath.read_text())
        return cached["train"], cached["held_in"], cached["held_out"]
    trainIndices, heldIn, heldOut = _computeSelection(dataset)
    outDir.mkdir(parents=True, exist_ok=True)
    cachePath.write_text(json.dumps(
        {"train": trainIndices, "held_in": heldIn, "held_out": heldOut}
    ))
    return trainIndices, heldIn, heldOut


def _report(
    outDir: Path, heldIn: list[int], heldOut: list[int],
    dataset: PreprocessedLinkDataset, device: str,
) -> None:
    """Evaluate both probe sets and print the consolidated table."""
    ckptPath = _resolveCheckpoint(outDir)
    inRows = _evaluateSet(ckptPath, heldIn, dataset, device)
    outRows = _evaluateSet(ckptPath, heldOut, dataset, device)
    print("\n========== B1-bis N=1000 / 640-8 ==========")
    print(f"held-in probes:  {len(heldIn)}   "
          f"held-out probes: {len(heldOut)}")
    print(_formatRows("held-in", inRows))
    print(_formatRows("held-out", outRows))
    print("===========================================")


def main() -> None:
    """Run (or re-evaluate) the B1-bis N=1000 consolidation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument(
        "--cond-mask-prob", dest="cond_mask_prob", type=float,
        default=COND_MASK_PROB,
    )
    parser.add_argument(
        "--dataset-root", dest="dataset_root", type=Path, default=DATASET,
    )
    parser.add_argument(
        "--vel-xyz-weight", dest="vel_xyz_weight", type=float,
        default=0.0,
        help="Velocity loss weight for native smoothness (try 1.0).",
    )
    parser.add_argument(
        "--clip-guidance-weight", dest="clip_guidance_weight", type=float,
        default=0.6,
        help=(
            "Text-motion contrastive weight.  Default 0.6 dominates the "
            "loss (~61%%) and mutes motion; try 0.1 for fuller motion."
        ),
    )
    parser.add_argument(
        "--aux-pool-weight", dest="aux_pool_weight", type=float,
        default=0.5,
    )
    parser.add_argument(
        "--min-snr-gamma", dest="min_snr_gamma", type=float,
        default=5.0,
        help=(
            "Min-SNR-gamma loss weighting (default 5.0).  Higher/0 "
            "re-weights the high-noise regime (B1-quater diversity test)."
        ),
    )
    parser.add_argument(
        "--rotation-jerk-weight", dest="rotation_jerk_weight", type=float,
        default=50.0,
        help=(
            "rot6d Δ² smoothness penalty — the lever that cuts trembling "
            "(overfit: framevel 0.094→0.054 at 50, amplitude kept). "
            ">~150 over-smooths. 0 disables."
        ),
    )
    parser.add_argument(
        "--velocity-schedule", dest="velocity_schedule", type=str,
        default="timestep", choices=("none", "timestep"),
        help=(
            "Schedule for vel-xyz / accel / rotation-jerk losses. "
            "'timestep' fades smoothness at high noise (default; crushes "
            "the geometric losses to <1%% share = dead). 'none' "
            "supervises smoothness at ALL noise levels (MDM-style) — the "
            "loss-mix rebalance for near-static motion (LOG 2026-06-21)."
        ),
    )
    parser.add_argument(
        "--joint-position-weight", dest="joint_position_weight",
        type=float, default=0.0,
        help=(
            "MDM-style FK joint-position loss (||FK(pred)-FK(GT)||²). "
            "Default 0.0 (off). Try 1.0 to add reconstruction pressure "
            "on joint trajectories — request 002 rebalance lever."
        ),
    )
    parser.add_argument(
        "--foot-contact-weight", dest="foot_contact_weight", type=float,
        default=0.0,
        help=(
            "MDM-style anti-foot-skating loss. Default 0.0 (off). Try "
            "0.5 once strides return, to suppress sliding."
        ),
    )
    parser.add_argument(
        "--denoiser-embed-dim", dest="denoiser_embed_dim", type=int,
        default=DENOISER_EMBED_DIM,
        help=(
            "Denoiser model width (default 640 = the N=1000 capacity-law "
            "point). 768/896 are the next rungs when 640/8 averages to a "
            "prompt-agnostic motion (conditioning collapse — LOG "
            "2026-06-22). Must be divisible by --denoiser-num-heads."
        ),
    )
    parser.add_argument(
        "--denoiser-num-layers", dest="denoiser_num_layers", type=int,
        default=DENOISER_NUM_LAYERS,
        help="Denoiser depth (default 8). Try 10/12 with a wider dim.",
    )
    parser.add_argument(
        "--denoiser-num-heads", dest="denoiser_num_heads", type=int,
        default=DENOISER_NUM_HEADS,
        help=(
            "Denoiser attention heads (default 8). Keep embed-dim "
            "divisible by this (640/8=80, 768/8=96, 896/8=112 dims/head)."
        ),
    )
    parser.add_argument("--eval-only", action="store_true")
    arguments = parser.parse_args()

    dataset = PreprocessedLinkDataset(arguments.dataset_root)
    trainIndices, heldIn, heldOut = _selectProbes(
        dataset, arguments.out_dir
    )
    print(f"train N={len(trainIndices)}  held-in={len(heldIn)}  "
          f"held-out={len(heldOut)}")

    if not arguments.eval_only:
        _writeRunCommand(arguments.out_dir, arguments.epochs,
                         arguments.cond_mask_prob, heldIn, heldOut)
        config = _buildConfig(trainIndices, arguments.epochs,
                              arguments.out_dir, arguments.device,
                              arguments.cond_mask_prob,
                              arguments.dataset_root,
                              arguments.vel_xyz_weight,
                              arguments.clip_guidance_weight,
                              arguments.aux_pool_weight,
                              arguments.min_snr_gamma,
                              arguments.rotation_jerk_weight,
                              arguments.velocity_schedule,
                              arguments.joint_position_weight,
                              arguments.foot_contact_weight,
                              arguments.denoiser_embed_dim,
                              arguments.denoiser_num_layers,
                              arguments.denoiser_num_heads)
        runFullTraining(config)

    _report(arguments.out_dir, heldIn, heldOut, dataset,
            arguments.device)


if __name__ == "__main__":
    main()
