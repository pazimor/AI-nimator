"""Multi-sample (production) training loop for AI-nimator v2.

This module is the v2 counterpart of ``train_generation.py`` from the
legacy stack — it orchestrates a training run that ingests the full
preprocessed dataset (filtered by folder) instead of overfitting on a
single sample.  The overfit-1-sample loop in :mod:`training_v2` stays
intact for the sanity check; this module is what runs the actual v2
training.

Design choices, kept deliberately lean
--------------------------------------
* **Batch loading**: reuse :class:`PreprocessedLinkDataset` from the
  legacy stack so the shard-cache LRU and the ``link_index`` resolution
  are shared with v1.  We just adapt the payload to the v2 schema
  (rotation6d + root_translation + raw_text + motion_mask).
* **No ``num_workers`` parallelism**: MPS does not benefit from worker
  processes and it complicates RNG seeding.  We loop sequentially —
  IO is dominated by the shard LRU which is already in RAM after a
  few batches.
* **Validation split**: deterministic 10 % split via a stable hash of
  ``linkIndex``.  Replaces the legacy ``validation_indices.json`` file
  with something that round-trips through any environment.
* **Normalizer fit on a streaming subset**: the legacy
  :func:`computeMotionStatistics` walks ``maxBatches`` to keep the cost
  bounded.  We do the same — see :func:`fitNormalizerFromDataset`.
* **Per-component validation**: total loss + bone loss + global loss +
  optional vel-xyz are tracked separately so we can diagnose where the
  model is plateauing.
* **Best checkpoint**: keep the lowest-val_total checkpoint plus the
  latest one.  No EMA in this first iteration to keep the code small;
  it can be retrofitted later via the legacy :class:`ExponentialMovingAverage`.
* **Resume**: optional, reads the same v3 checkpoint format as the
  overfit loop.

Public surface
--------------
* :class:`V2FullTrainingConfig`
* :func:`runFullTraining`
* :func:`fitNormalizerFromDataset`
* :func:`splitTrainVal`
"""

from __future__ import annotations

import gc
import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from ainimator.training.training_v2 import (
    EMPTY_PROMPT,
    TrainingRandomState,
    V2TrainingConfig,
    _denoiserConfigToDict,
    _encoderConfigToDict,
    _scheduleConfigToDict,
    resolveDevice,
)
from ainimator.core.checkpoint_io import saveTorchObjectAtomically
from ainimator.core.resolved_config import writeResolvedConfig
from ainimator.model.ema import ExponentialMovingAverage
from ainimator.model.denoiser_v2 import (
    MotionDenoiserV2,
    MotionDenoiserV2Config,
)
from ainimator.model.losses_v2 import (
    DEFAULT_MIN_SNR_GAMMA,
    ContrastiveMemoryBank,
    diffusionLossV2,
    footContactLossV2,
    jointPositionLossV2,
    poolTextEmbedding,
    textMotionContrastiveLoss,
    velocityXyzLossV2,
)
from ainimator.model.motion_normalizer import MotionNormalizer
from ainimator.diffusion.noise_schedule import (
    NoiseSchedule,
    NoiseScheduleConfig,
    PREDICTION_V,
    SUPPORTED_PREDICTIONS,
)
from ainimator.text import (
    ClipTextEncoder,
    ClipTextEncoderConfig,
    ClipTokenizer,
    CustomTextEncoder,
    CustomTextEncoderConfig,
    CustomTokenizer,
    TextEncoderOutput,
)
from ainimator.data.preprocessed_dataset import PreprocessedLinkDataset
from ainimator.health.hub import HealthHub, buildHealthHub

LOGGER = logging.getLogger(__name__)

# Hash bucket count for the deterministic train/val split.  Using a
# large prime keeps the per-bucket distribution close to uniform.
_VAL_SPLIT_BUCKETS = 10_000


# =====================================================================
# Configuration
# =====================================================================
@dataclass(frozen=True)
class V2FullTrainingConfig:
    """Configuration for :func:`runFullTraining`.

    The config sub-classes the overfit knobs and adds the multi-sample
    specifics (folder filter, batch size, validation split, normalizer
    fitting budget, ...).
    """

    # --- Dataset & tokenizer paths ---------------------------------
    datasetRoot: Path
    tokenizerDir: Path
    outputDir: Path

    # --- Folder filter ---------------------------------------------
    datasetFolders: tuple[str, ...] | None = None  # None = all folders
    # 2026-06-05 — small-N controlled mode.  When set, training uses
    # EXACTLY these link indices (train == val) instead of a folder
    # split.  Bridges the gap between the 1-sample overfit loop and the
    # full multi-sample run: lets you verify the conditioning can map a
    # handful of *distinct* prompts to their own motions before scaling
    # up.  ``None`` keeps the normal folder-filtered behaviour.
    sampleLinkIndices: tuple[int, ...] | None = None

    # --- Training schedule -----------------------------------------
    epochs: int = 100
    batchSize: int = 8
    gradientAccumulation: int = 1
    learningRate: float = 1e-4
    encoderLrMultiplier: float = 3.0
    weightDecay: float = 3e-4
    minSnrGamma: float = DEFAULT_MIN_SNR_GAMMA
    velocityXyzWeight: float = 0.0
    # Phase 1.1 (2026-05-17) — MDM-style geometric loss on FK-derived
    # joint positions.  Supervises the *positions* of the 22 SMPL joints
    # so small rotation errors near the root (hips, spine) cannot
    # compound through the kinematic chain into large end-effector
    # drift.  This is the canonical fix for the "low rotation MSE, but
    # the animation still looks wrong" syndrome that the v2 stack hit
    # at epoch 125 (loss_diffusion=0.0775 yet generations were
    # off-distribution).  Recommended starting value: 1.0 (same order
    # of magnitude as boneLoss + globalLoss).  Scheduled by timestep so
    # high-t samples (where FK-of-noise is meaningless) contribute ~0.
    jointPositionWeight: float = 0.0
    # Phase 1.2 (2026-05-17) — MDM-style foot-contact anti-skating loss.
    # Derives the contact mask on-the-fly from target foot velocities
    # (squared speed < 0.002) and penalises predicted foot velocity
    # where the target says the foot ought to be still.  Reduces the
    # foot-sliding artefacts visible in early v2 generations.  Same
    # timestep scheduling as ``velocityXyzWeight`` /
    # ``jointPositionWeight`` (multiplied by ᾱ_t so FK-of-noise at
    # high t contributes ~0).  Recommended starting value: 0.5.
    footContactWeight: float = 0.0
    # Phase D.1 — text↔motion contrastive (InfoNCE) loss weight.
    # ``0.0`` disables the alignment head and matches the pre-D.1
    # behaviour.  Bumped 0.0 → 0.6 on 2026-05-28 after diagnose_v2 on
    # the 215-epoch CLIP-swap checkpoint showed avg_cfg_sim ≈ 0.999 and
    # σ_rot generated 4× σ_rot train — the prior 0.3 weight (≈23 % of
    # total loss) was insufficient to constrain the encoder pool against
    # the diffusion loss.  0.6 gives the contrastive a comparable budget
    # to the reconstruction signal at convergence; the warmup schedule
    # (clipGuidanceWeightStart=1.0, decaying to 0.6 over 15 epochs)
    # keeps the early epochs even more contrastive-biased.
    clipGuidanceWeight: float = 0.6
    # Temperature of the InfoNCE softmax.  Lower ⇒ sharper distinction
    # between matched and non-matched pairs.  ``0.1`` is the CLIP
    # default and sane for batches of 8–32.
    contrastiveTemperature: float = 0.1
    contrastiveBankSize: int = 256
    clipGuidanceWeightStart: float = 1.0
    clipGuidanceWarmupEpochs: int = 15
    # Phase F iter-2 (2026-05-14) — auxiliary contrastive loss on the
    # *raw* masked-mean pool of the text encoder.  Closes the alignment-
    # head MLP loophole that lets the main contrastive be satisfied
    # without the encoder pool itself discriminating (cond↔uncond raw
    # pool sim = 0.9998 after 49 epochs).  ``0.0`` disables the aux
    # head; ``0.5`` is the recommended starting weight.  Bumped from
    # ``0.0`` → ``0.5`` on 2026-05-28 — the 215-epoch run had this
    # disabled and the diagnostic showed the MLP loophole intact
    # (enc_sim=0.22 vs align_sim=0.12).
    auxPoolContrastiveWeight: float = 0.5
    # 2026-06-01 — generated-motion (x0) contrastive weight.  Closes the
    # generation loophole diagnosed on the 212-epoch CLIP run: the
    # existing contrastives pool the denoiser's hidden state built from
    # the *noisy input*, so the prediction itself could ignore the prompt
    # (cross-prompt cos-sim 0.98 at cfg=1.0).  This contrastive embeds the
    # reconstructed x0 through a dedicated TMR-style encoder and weights
    # the InfoNCE toward high timesteps (where the prediction cannot copy
    # the input).  ``0.0`` disables (and skips building the encoder).
    # ``0.5`` is the recommended starting weight.
    x0ContrastiveWeight: float = 0.0

    # --- Diffusion -------------------------------------------------
    diffusionStepsTraining: int = 1000
    scheduleType: str = "cosine"
    predictionMode: str = PREDICTION_V

    # --- Text encoder selection (Phase 2, 2026-05-18) -------------
    # ``"custom"`` keeps the 5.2M-param BPE transformer trained from
    # scratch.  ``"clip"`` swaps in the frozen OpenAI CLIP ViT-B/32
    # text tower (~63M pretrained, frozen) with only a small trainable
    # 512→embedDim projection + null embedding.  CLIP gives a far
    # sharper semantic space than a 56k-caption from-scratch encoder
    # can build — the canonical fix when contrasted prompts ("jumping"
    # vs "walking") produce near-identical motion.
    textEncoderType: str = "custom"
    clipModelName: str = "openai/clip-vit-base-patch32"
    clipMaxLength: int = 32

    # --- Architecture (matches network.yaml v2 profile) -----------
    encoderHiddenDim: int = 256
    encoderNumLayers: int = 4
    encoderNumHeads: int = 8
    denoiserEmbedDim: int = 384
    denoiserNumLayers: int = 4
    denoiserNumHeads: int = 8
    maxFrames: int = 256

    # --- Regularisation ------------------------------------------
    dropout: float = 0.1
    # CFG dropout — bumped 0.10 → 0.20 in Phase D.2.  The first full
    # 5-folder run with 0.10 gave `cfg_sim ≈ 1.000` (cross-attn dead at
    # inference) which means the unconditional branch never received
    # enough signal to differentiate from the conditional one.  0.20
    # doubles the unconditional pressure.  MDM uses 0.10 on much larger
    # datasets — 0.20 is appropriate for our 56k-sample scale.
    condMaskProb: float = 0.20
    # Phase D.3 — SMPL-22 left↔right mirror augmentation.  Per-sample
    # coin flip, prompts mentioning "left" / "right" / similar are
    # skipped to keep the cross-attention semantics intact.  ``0.0``
    # disables (matches pre-D.3 behaviour).  ``0.5`` is the v2 default
    # — doubles the effective dataset size without losing the original.
    mirrorProb: float = 0.0
    # Phase D.4 — EMA on encoder + denoiser parameters.  Stabilises
    # the val_loss curve, gives a better best checkpoint candidate,
    # and breaks weak attractors that the online weights might
    # otherwise oscillate around.  ``emaDecay <= 0`` disables EMA
    # entirely.  ``0.9999`` is the canonical diffusion value.
    emaDecay: float = 0.0
    emaUseWarmup: bool = True
    # Phase D Levier D — FiLM conditioning shortcut on
    # ``(text_pooled, timestep_embed)``.  Forces a multiplicative
    # path the cross-attention cannot bypass.
    # Phase F (2026-05-12) — default flipped to True after phaseE
    # diagnostic confirmed the cross-attn-only path lets cond/uncond
    # collapse (cfg_sim ≈ 0.9995).
    useFilmConditioning: bool = True
    filmDropout: float = 0.0
    filmInitStd: float = 0.1
    # Phase E (Levier E, 2026-05-08) — per-block AdaLN-style FiLM
    # modulation in every DenoiserBlockV2 sub-layer.  Standard DiT /
    # SD3 conditional architecture, used as the structural fix when
    # the global FiLM alone is insufficient.
    # Phase F (2026-05-12) — default flipped to True; together with the
    # global FiLM this stacks 12+1 multiplicative gradient pathways on
    # the text branch.
    usePerBlockFilm: bool = True
    # Phase F (2026-05-12) — learnable null embedding for CFG dropout.
    # When True the unconditional branch bypasses the encoder entirely
    # and uses ``denoiser.textEncoder.forwardNull()`` instead of
    # re-encoding EMPTY_PROMPT="".  Eliminates the BOS-token-collapse
    # that made encoder cond↔uncond sim = 0.7361.
    useNullEmbedding: bool = True
    # 2026-06-02 — self-conditioning (Analog Bits).  At training, with
    # probability ``selfConditioningProb`` a first no-grad denoiser pass
    # produces a detached x0 estimate that is fed back into the real
    # (gradient) pass.  At sampling each DDIM step feeds the previous
    # step's x0.  Narrows the train/sampling exposure gap that let the
    # model denoise from x_t without ever needing the prompt.  ``0.0``
    # disables and skips building the self-cond projections.
    useSelfConditioning: bool = False
    selfConditioningProb: float = 0.5

    # --- Validation ----------------------------------------------
    validationFraction: float = 0.10
    validationSeed: int = 42
    validateEveryEpochs: int = 1
    # 2026-05-07 — best-checkpoint policy.  ``bestMetric`` chooses
    # which validation key drives the "is this a new best?" check; the
    # default ``"loss_total"`` keeps backward compat but ignores the
    # decomposition ("loss_diffusion" = loss_bone + loss_global lets
    # you ignore a stuck contrastive floor).  ``bestImprovementMin``
    # is the minimum absolute drop (on the chosen metric) that counts
    # as a real improvement — anything smaller is treated as gradient
    # noise and does NOT rewrite the best checkpoint.  Setting both
    # to 0 reproduces the pre-this-change behaviour.
    bestMetric: str = "loss_total"
    bestImprovementMin: float = 0.0
    # Number of consecutive validation passes without a real
    # improvement (per ``bestImprovementMin``) before the trainer
    # logs a stagnation warning.  ``0`` disables the check.
    stagnationPatience: int = 0

    # --- Normalizer ----------------------------------------------
    normalizerFitMaxSamples: int = 2000  # cap for the streaming fit

    # --- Subsampling per epoch -----------------------------------
    # Capping the random sub-sample drawn each epoch keeps wall-clock
    # under control on MPS (the legacy stack uses 5000 for a similar
    # reason).  ``0`` disables the cap (every epoch sees every sample).
    maxSamplesPerEpoch: int = 5000

    # --- Logging / checkpointing ---------------------------------
    logEvery: int = 50  # log every N optimisation steps
    seed: int = 0
    device: str = "auto"
    resumeCheckpoint: Path | None = None

    # --- Health monitoring (A3) ----------------------------------
    # New keys — not changes to existing defaults.
    # healthEnabled=True activates hub.step() inside the training loop.
    # healthEverySteps aligns with logEvery (default 50).
    healthEnabled: bool = True
    healthEverySteps: int = 50

    def __post_init__(self) -> None:
        if self.predictionMode not in SUPPORTED_PREDICTIONS:
            raise ValueError(
                "predictionMode must be one of "
                f"{SUPPORTED_PREDICTIONS}; got {self.predictionMode!r}."
            )
        if self.scheduleType not in ("linear", "cosine"):
            raise ValueError(
                f"scheduleType must be 'linear' or 'cosine'; got "
                f"{self.scheduleType!r}."
            )
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1.")
        if self.batchSize < 1:
            raise ValueError("batchSize must be >= 1.")
        if self.gradientAccumulation < 1:
            raise ValueError("gradientAccumulation must be >= 1.")
        if not (0.0 < self.validationFraction < 1.0):
            raise ValueError(
                "validationFraction must be in (0, 1) exclusive."
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")
        if not (0.0 <= self.condMaskProb <= 1.0):
            raise ValueError("condMaskProb must be in [0, 1].")
        if self.maxSamplesPerEpoch < 0:
            raise ValueError("maxSamplesPerEpoch must be >= 0.")
        if self.encoderLrMultiplier <= 0.0:
            raise ValueError("encoderLrMultiplier must be > 0.")
        if self.auxPoolContrastiveWeight < 0.0:
            raise ValueError("auxPoolContrastiveWeight must be >= 0.")
        if self.clipGuidanceWeight < 0.0:
            raise ValueError("clipGuidanceWeight must be >= 0.")
        if self.contrastiveTemperature <= 0.0:
            raise ValueError("contrastiveTemperature must be > 0.")
        if self.contrastiveBankSize < 0:
            raise ValueError("contrastiveBankSize must be >= 0.")
        if self.clipGuidanceWeightStart < self.clipGuidanceWeight:
            raise ValueError(
                "clipGuidanceWeightStart must be >= clipGuidanceWeight."
            )
        if self.clipGuidanceWarmupEpochs < 0:
            raise ValueError("clipGuidanceWarmupEpochs must be >= 0.")
        if not (0.0 <= self.mirrorProb <= 1.0):
            raise ValueError("mirrorProb must be in [0, 1].")
        if self.emaDecay < 0.0 or self.emaDecay >= 1.0:
            raise ValueError("emaDecay must be in [0, 1).")
        if not (0.0 <= self.filmDropout < 1.0):
            raise ValueError("filmDropout must be in [0, 1).")
        if self.bestMetric not in (
            "loss_total",
            "loss_diffusion",
            "loss_bone",
            "loss_global",
            "loss_vel_xyz",
            "loss_joint_xyz",
            "loss_foot_contact",
            "loss_clip_guidance",
            "loss_clip_aux_pool",
            "loss_x0_contrastive",
        ):
            raise ValueError(
                f"bestMetric must be one of loss_total, loss_diffusion, "
                f"loss_bone, loss_global, loss_vel_xyz, loss_joint_xyz, "
                f"loss_foot_contact, loss_clip_guidance, "
                f"loss_clip_aux_pool, loss_x0_contrastive; got "
                f"{self.bestMetric!r}."
            )
        if self.jointPositionWeight < 0.0:
            raise ValueError("jointPositionWeight must be >= 0.")
        if self.footContactWeight < 0.0:
            raise ValueError("footContactWeight must be >= 0.")
        if self.x0ContrastiveWeight < 0.0:
            raise ValueError("x0ContrastiveWeight must be >= 0.")
        if not (0.0 <= self.selfConditioningProb <= 1.0):
            raise ValueError("selfConditioningProb must be in [0, 1].")
        if self.textEncoderType not in ("custom", "clip"):
            raise ValueError(
                "textEncoderType must be 'custom' or 'clip'; got "
                f"{self.textEncoderType!r}."
            )
        if not (1 <= self.clipMaxLength <= 77):
            raise ValueError("clipMaxLength must be in [1, 77].")
        if self.bestImprovementMin < 0.0:
            raise ValueError("bestImprovementMin must be >= 0.")
        if self.stagnationPatience < 0:
            raise ValueError("stagnationPatience must be >= 0.")
        if self.healthEverySteps < 1:
            raise ValueError("healthEverySteps must be >= 1.")


# =====================================================================
# Train/val split
# =====================================================================
def splitTrainVal(
    linkIndices: Sequence[int],
    validationFraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Deterministic train/val split keyed on the link index.

    The hash bucket guarantees the same set of validation samples
    across sessions (and machines) for a given ``seed``.  This sidesteps
    the legacy ``validation_indices.json`` file which had to be carried
    around on disk.
    """
    if not (0.0 < validationFraction < 1.0):
        raise ValueError(
            "validationFraction must be in (0, 1) exclusive."
        )
    threshold = int(_VAL_SPLIT_BUCKETS * validationFraction)
    train: list[int] = []
    val: list[int] = []
    for index in linkIndices:
        digest = hashlib.sha256(
            f"{int(seed)}:{int(index)}".encode("utf-8")
        ).digest()
        bucket = int.from_bytes(digest[:4], "little") % _VAL_SPLIT_BUCKETS
        (val if bucket < threshold else train).append(int(index))
    return train, val


# =====================================================================
# Dataset filtering
# =====================================================================
def selectLinkIndices(
    dataset: PreprocessedLinkDataset,
    datasetFolders: tuple[str, ...] | None,
) -> list[int]:
    """Return link indices kept after folder filtering."""
    if datasetFolders is None:
        return list(range(len(dataset.linkEntries)))
    allowed = set(datasetFolders)
    return [
        index
        for index, entry in enumerate(dataset.linkEntries)
        if entry.datasetFolder in allowed
    ]


# =====================================================================
# Batch payload
# =====================================================================
@dataclass(frozen=True)
class V2Batch:
    """A padded batch ready to feed the v2 training step.

    Attributes
    ----------
    rotation6d : torch.Tensor
        ``(B, F, 22, 6)`` float32 — already on the target device.
    rootTranslation : torch.Tensor
        ``(B, F, 3)`` float32.
    motionMask : torch.Tensor
        ``(B, F)`` bool, ``True`` on real frames, ``False`` on padding.
    rawTexts : tuple[str, ...]
        Source prompt for each sample.  Tokenised inside the training
        step so we can apply ``cond-mask-prob`` at the string level.
    """

    rotation6d: torch.Tensor
    rootTranslation: torch.Tensor
    motionMask: torch.Tensor
    rawTexts: tuple[str, ...]


def _toTensor(value: object, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"Sample payload missing tensor field {label!r}.")
    return value


def collateV2Batch(
    payloads: Sequence[dict[str, object]],
    maxFrames: int,
    device: torch.device,
) -> V2Batch:
    """Pad a list of raw sample payloads into a :class:`V2Batch`.

    The function clips each sample to ``maxFrames`` first, then pads
    every sample to the **batch maximum**.  Padding uses zeros for the
    motion tensors and ``False`` in the mask.
    """
    if not payloads:
        raise ValueError("collateV2Batch received an empty list.")

    rotations: list[torch.Tensor] = []
    rtrans: list[torch.Tensor] = []
    rawTexts: list[str] = []
    for payload in payloads:
        rotation = _toTensor(payload.get("motion"), "motion")
        rt = _toTensor(payload.get("root_translation"), "root_translation")
        rawText = payload.get("raw_text")
        if not isinstance(rawText, str):
            # Legacy text shards may key the prompt under a different
            # field; fall back to the recorded source identifier rather
            # than crashing — the training step skips empty strings.
            rawText = ""
        rotation = rotation[:maxFrames]
        rt = rt[:maxFrames]
        rotations.append(rotation)
        rtrans.append(rt)
        rawTexts.append(rawText)

    batchFrames = max(rotation.shape[0] for rotation in rotations)
    batchSize = len(rotations)
    numBones = rotations[0].shape[1]
    motionChannels = rotations[0].shape[2]
    globalChannels = rtrans[0].shape[1]

    paddedRotation = torch.zeros(
        batchSize, batchFrames, numBones, motionChannels,
        dtype=torch.float32,
    )
    paddedRtrans = torch.zeros(
        batchSize, batchFrames, globalChannels, dtype=torch.float32,
    )
    motionMask = torch.zeros(
        batchSize, batchFrames, dtype=torch.bool,
    )
    for index, (rotation, rt) in enumerate(zip(rotations, rtrans)):
        frames = rotation.shape[0]
        paddedRotation[index, :frames] = rotation.float()
        paddedRtrans[index, :frames] = rt.float()
        motionMask[index, :frames] = True

    return V2Batch(
        rotation6d=paddedRotation.to(device),
        rootTranslation=paddedRtrans.to(device),
        motionMask=motionMask.to(device),
        rawTexts=tuple(rawTexts),
    )


def iterBatches(
    dataset: PreprocessedLinkDataset,
    indices: Sequence[int],
    batchSize: int,
    maxFrames: int,
    device: torch.device,
    shuffle: bool,
    rngSeed: int,
) -> Iterator[V2Batch]:
    """Yield :class:`V2Batch` batches over ``indices``.

    Indices may be a sub-sample of the full link table; shuffling is
    deterministic given ``rngSeed`` so the loss curve is reproducible.
    """
    workOrder: list[int] = list(indices)
    if shuffle:
        rng = torch.Generator(device=torch.device("cpu"))
        rng.manual_seed(int(rngSeed))
        order = torch.randperm(
            len(workOrder), generator=rng
        ).tolist()
        workOrder = [workOrder[index] for index in order]

    for start in range(0, len(workOrder), batchSize):
        chunk = workOrder[start:start + batchSize]
        if not chunk:
            continue
        payloads = [dataset[index] for index in chunk]
        yield collateV2Batch(payloads, maxFrames=maxFrames, device=device)


# =====================================================================
# Normalizer fitting from a streaming dataset
# =====================================================================
def fitNormalizerFromDataset(
    dataset: PreprocessedLinkDataset,
    indices: Sequence[int],
    normalizer: MotionNormalizer,
    maxSamples: int,
    seed: int = 0,
) -> MotionNormalizer:
    """Compute mean/std over up to ``maxSamples`` random training samples.

    The selected samples are concatenated into a single tensor along
    the frame axis, then the normalizer's :meth:`fitFromTensors`
    handles the per-channel reduction.  Padding is irrelevant here
    because we use ``rotation6d`` / ``root_translation`` directly from
    the shard payload before any batch padding.
    """
    if not indices:
        raise ValueError("Cannot fit a normalizer on zero samples.")
    rng = torch.Generator(device=torch.device("cpu"))
    rng.manual_seed(int(seed))
    available = list(indices)
    sampleCount = min(maxSamples, len(available))
    permutation = torch.randperm(
        len(available), generator=rng
    ).tolist()[:sampleCount]
    selectedIndices = [available[index] for index in permutation]

    boneSamples: list[torch.Tensor] = []
    globalSamples: list[torch.Tensor] = []
    LOGGER.info(
        "Fitting normalizer over %d / %d available samples.",
        sampleCount,
        len(available),
    )
    for linkIndex in selectedIndices:
        payload = dataset[linkIndex]
        boneSamples.append(_toTensor(payload.get("motion"), "motion"))
        globalSamples.append(
            _toTensor(payload.get("root_translation"), "root_translation")
        )
    normalizer.fitFromTensors(boneSamples, globalSamples)
    return normalizer


# =====================================================================
# Component bundle (mirrors V2TrainingComponents from training_v2)
# =====================================================================
@dataclass
class V2FullTrainingComponents:
    """Bundle returned by :func:`buildFullTrainingComponents`."""

    tokenizer: CustomTokenizer | ClipTokenizer
    encoder: CustomTextEncoder | ClipTextEncoder
    denoiser: MotionDenoiserV2
    schedule: NoiseSchedule
    normalizer: MotionNormalizer
    optimizer: torch.optim.Optimizer
    device: torch.device
    # Phase D.4 — EMA shadow for encoder + denoiser parameters, or
    # ``None`` when ``config.emaDecay <= 0``.
    ema: ExponentialMovingAverage | None = None
    contrastiveBank: ContrastiveMemoryBank | None = None


def buildFullTrainingComponents(
    config: V2FullTrainingConfig,
) -> V2FullTrainingComponents:
    """Instantiate the v2 stack for a full training run.

    The normalizer is left at identity init — the caller is expected
    to call :func:`fitNormalizerFromDataset` once the link indices are
    known and before training starts.
    """
    device = resolveDevice(config.device)

    tokenizer: CustomTokenizer | ClipTokenizer
    encoder: CustomTextEncoder | ClipTextEncoder
    if config.textEncoderType == "clip":
        # Phase 2 — frozen CLIP ViT-B/32 text tower.  The tokenizer is
        # CLIP's own 49k BPE (loaded from the HF hub, no local dir) and
        # the encoder exposes a trainable 512→embedDim projection.
        tokenizer = ClipTokenizer(
            modelName=config.clipModelName,
            maxLength=config.clipMaxLength,
        )
        encoder = ClipTextEncoder(
            ClipTextEncoderConfig(
                modelName=config.clipModelName,
                maxLength=config.clipMaxLength,
                outputDim=config.denoiserEmbedDim,
                dropout=config.dropout,
                useNullEmbedding=config.useNullEmbedding,
                l2NormalizeOutput=True,
            )
        ).to(device)
    else:
        tokenizer = CustomTokenizer.load(config.tokenizerDir)
        encoder = CustomTextEncoder(
            CustomTextEncoderConfig(
                vocabSize=tokenizer.vocabSize,
                maxLength=tokenizer.config.maxLength,
                hiddenDim=config.encoderHiddenDim,
                numLayers=config.encoderNumLayers,
                numHeads=config.encoderNumHeads,
                outputDim=config.denoiserEmbedDim,
                padTokenId=tokenizer.padTokenId,
                dropout=config.dropout,
                # Phase F — learnable null embedding for the CFG
                # dropout's unconditional branch, plus L2-norm on the
                # per-token output so cross-attention K/V stay at unit
                # magnitude.
                useNullEmbedding=config.useNullEmbedding,
                l2NormalizeOutput=True,
            )
        ).to(device)

    denoiser = MotionDenoiserV2(
        MotionDenoiserV2Config(
            embedDim=config.denoiserEmbedDim,
            numHeads=config.denoiserNumHeads,
            numLayers=config.denoiserNumLayers,
            numBones=22,
            motionChannels=6,
            globalChannels=3,
            textEmbedDim=config.denoiserEmbedDim,
            maxFrames=config.maxFrames,
            dropout=config.dropout,
            # Phase D.1 — alignment head only built when the
            # contrastive loss is actually used.  Saves ~150k params
            # and avoids state_dict surprises on older checkpoints.
            alignmentEnabled=config.clipGuidanceWeight > 0.0,
            # Phase D Levier D — FiLM conditioning.  Wired via
            # ``--use-film-conditioning`` so it can be combined with
            # any other Phase D fix.
            useFilmConditioning=config.useFilmConditioning,
            filmDropout=config.filmDropout,
            usePerBlockFilm=config.usePerBlockFilm,
            filmInitStd=config.filmInitStd,
            # Phase F iter-2 — auxiliary raw-pool alignment head, built
            # only when the aux contrastive loss is actually active.
            auxPoolAlignmentEnabled=config.auxPoolContrastiveWeight > 0.0,
            # 2026-06-01 — generated-motion (x0) alignment encoder, built
            # only when the x0 contrastive loss is active.
            x0AlignmentEnabled=config.x0ContrastiveWeight > 0.0,
            # 2026-06-02 — self-conditioning input projections.
            useSelfConditioning=config.useSelfConditioning,
        )
    ).to(device)

    schedule = NoiseSchedule(
        NoiseScheduleConfig(
            numSteps=config.diffusionStepsTraining,
            scheduleType=config.scheduleType,
        )
    ).to(device)

    normalizer = MotionNormalizer(
        numBones=22, motionChannels=6, globalChannels=3
    ).to(device)

    # Only trainable params go to the optimizer — the frozen CLIP tower
    # (textEncoderType="clip") would otherwise allocate ~63M slots of
    # unused AdamW moment state.
    optimizer = AdamW(
        [
            {
                "params": [
                    p for p in encoder.parameters() if p.requires_grad
                ],
                "lr": config.learningRate * config.encoderLrMultiplier,
            },
            {
                "params": list(denoiser.parameters()),
                "lr": config.learningRate,
            },
        ],
        weight_decay=config.weightDecay,
    )

    # Phase D.4 — build EMA shadow only when explicitly enabled.  The
    # parameter list combines encoder + denoiser so a single EMA tracks
    # the entire trainable stack.  The EMA helper requires the **same**
    # parameter ordering on every call, hence the helper below.
    ema: ExponentialMovingAverage | None = None
    if config.emaDecay > 0.0:
        ema = ExponentialMovingAverage(
            parameters=_emaParameters(encoder, denoiser),
            decay=config.emaDecay,
            useWarmup=config.emaUseWarmup,
        )

    contrastiveBank: ContrastiveMemoryBank | None = None
    if config.clipGuidanceWeight > 0.0 and config.contrastiveBankSize > 0:
        contrastiveBank = ContrastiveMemoryBank(
            embeddingDim=denoiser.config.alignmentDim,
            bankSize=config.contrastiveBankSize,
        ).to(device)

    return V2FullTrainingComponents(
        tokenizer=tokenizer,
        encoder=encoder,
        denoiser=denoiser,
        schedule=schedule,
        normalizer=normalizer,
        optimizer=optimizer,
        device=device,
        ema=ema,
        contrastiveBank=contrastiveBank,
    )


# =====================================================================
# Best-metric helper (2026-05-07)
# =====================================================================
def _readBestMetric(
    valMetrics: dict[str, float],
    metricName: str,
) -> float:
    """Return the validation value tracked for best-checkpoint logic.

    Standard keys are read directly from ``valMetrics``.  The synthetic
    ``"loss_diffusion"`` key is the sum of ``loss_bone + loss_global``
    — useful when the contrastive head is stuck and you want to track
    diffusion improvements only.
    """
    if metricName == "loss_diffusion":
        return float(
            valMetrics.get("loss_bone", 0.0)
            + valMetrics.get("loss_global", 0.0)
        )
    if metricName not in valMetrics:
        # Defensive fallback — return inf so an unknown metric never
        # registers as a "best".
        return float("inf")
    return float(valMetrics[metricName])


# =====================================================================
# EMA parameter helper
# =====================================================================
def _emaParameters(
    encoder: CustomTextEncoder | ClipTextEncoder,
    denoiser: MotionDenoiserV2,
) -> list[torch.nn.Parameter]:
    """Return the parameter list tracked by the EMA shadow.

    Stable, deterministic ordering — encoder parameters first, then
    denoiser parameters, both via ``parameters()`` (which iterates in
    submodule definition order in PyTorch).  Calling this helper from
    every site that interacts with the EMA guarantees the shadow
    indices line up exactly with the online weights.

    Only trainable parameters are tracked: a frozen CLIP text tower
    (textEncoderType="clip") must not have an EMA shadow — it would
    waste ~63M floats and the shadow would be a no-op anyway.
    """
    return [
        p for p in encoder.parameters() if p.requires_grad
    ] + list(denoiser.parameters())


# =====================================================================
# Training & validation steps
# =====================================================================
def _encodePromptBatch(
    components: V2FullTrainingComponents,
    rawTexts: Sequence[str],
    condMaskProb: float,
    generators: TrainingRandomState,
    useNullEmbedding: bool = False,
) -> tuple[TextEncoderOutput, torch.Tensor]:
    """Tokenise prompts, run the encoder, apply CFG dropout.

    Returns the encoder output **and** a bool ``conditionedMask`` of
    shape ``(B,)`` that is ``True`` for samples whose real prompt was
    kept (``False`` for CFG-dropped samples).  The mask lets the x0
    contrastive zero-weight dropped rows whose text is the shared null
    embedding (they would otherwise be contradictory positives).

    With probability ``condMaskProb`` per sample, the conditional input
    is replaced by the unconditional signal so the unconditional branch
    sees real training signal — required for inference-time CFG to
    behave.

    Phase F (2026-05-12) — when ``useNullEmbedding=True`` the dropped
    samples bypass the encoder entirely and are replaced after-the-fact
    by ``encoder.forwardNull(...)``.  This eliminates the BOS-token
    collapse that made encoder cond↔uncond similarity 0.7361 on phaseD
    and produced the cfg_sim ≈ 0.9995 result on phaseE.

    When ``useNullEmbedding=False`` the legacy behaviour applies:
    dropped samples are replaced by ``EMPTY_PROMPT`` *before* tokenization
    and re-encoded.
    """
    dropFlags: list[bool] = []
    promptList: list[str] = []
    for text in rawTexts:
        if condMaskProb > 0.0:
            coin = torch.rand(
                (1,),
                generator=generators.cpuGenerator,
                device=torch.device("cpu"),
            ).item()
            if coin < condMaskProb:
                dropFlags.append(True)
                # When using the null embedding we still tokenize the
                # original prompt so the conditional encoder produces a
                # valid (T, D) tensor we can splice into; the dropped
                # rows are then overwritten downstream.
                promptList.append(
                    text if useNullEmbedding else EMPTY_PROMPT
                )
                continue
        dropFlags.append(False)
        promptList.append(text)

    encoded = components.tokenizer.encode(promptList)
    inputIds = encoded.inputIds.to(components.device)
    attentionMask = encoded.attentionMask.to(components.device)
    encoderOutput = components.encoder(inputIds, attentionMask)

    if useNullEmbedding and any(dropFlags):
        encoderOutput = _applyNullEmbedding(
            encoder=components.encoder,
            output=encoderOutput,
            dropFlags=dropFlags,
            device=components.device,
        )

    conditionedMask = torch.tensor(
        [not flag for flag in dropFlags],
        device=components.device,
        dtype=torch.bool,
    )
    return encoderOutput, conditionedMask


def _applyNullEmbedding(
    encoder: CustomTextEncoder | ClipTextEncoder,
    output: TextEncoderOutput,
    dropFlags: Sequence[bool],
    device: torch.device,
) -> TextEncoderOutput:
    """Replace the hidden states of dropped samples with the null embedding.

    The conditional encoder output has shape ``(B, T, D)`` and a key-padding
    mask of shape ``(B, T)``.  For each dropped sample ``i`` we overwrite
    ``hiddenStates[i, 0]`` with the learnable null token and set
    ``keyPaddingMask[i] = [False, True, ..., True]`` so the downstream
    cross-attention sees a single real position carrying the null signal.
    """
    numDropped = sum(dropFlags)
    if numDropped == 0:
        return output

    nullOutput = encoder.forwardNull(
        batchSize=numDropped,
        device=device,
        dtype=output.hiddenStates.dtype,
    )
    nullToken = nullOutput.hiddenStates  # (numDropped, 1, D)

    hiddenStates = output.hiddenStates.clone()
    keyPaddingMask = output.keyPaddingMask.clone()

    # Mark every position as padding for dropped samples, then write the
    # null token at position 0 and unmask it.  This keeps the shape
    # ``(B, T, D)`` consistent across cond/uncond samples so the
    # downstream cross-attention does not need to handle ragged batches.
    dropIdx = torch.tensor(
        [i for i, flag in enumerate(dropFlags) if flag],
        device=device,
        dtype=torch.long,
    )
    keyPaddingMask[dropIdx] = True
    keyPaddingMask[dropIdx, 0] = False
    hiddenStates[dropIdx, 0] = nullToken.squeeze(1)

    return TextEncoderOutput(
        hiddenStates=hiddenStates,
        keyPaddingMask=keyPaddingMask,
    )


def _meanOffDiagonalCosine(
    embeddings: torch.Tensor,
) -> tuple[float, int]:
    """Return the mean pairwise cosine similarity (off-diagonal).

    ``embeddings`` is expected to be already L2-normalized along the
    last axis — the alignment-head and aux-pool outputs both satisfy
    that.  The function computes the upper-triangular off-diagonal
    entries of ``E @ E.T`` and returns ``(mean, pair_count)``.  A value
    close to 1.0 means the embeddings are nearly identical across the
    batch (collapse); values around 0.0 mean they discriminate well.
    Batches of size 1 contribute no pairs and are skipped by the caller.
    """
    batchSize = embeddings.shape[0]
    if batchSize < 2:
        return 0.0, 0
    gram = embeddings @ embeddings.transpose(0, 1)
    triu = torch.triu(torch.ones_like(gram), diagonal=1).bool()
    offDiag = gram[triu]
    pairCount = int(offDiag.numel())
    return float(offDiag.mean().detach().item()), pairCount


def effectiveClipWeight(config: V2FullTrainingConfig, epoch: int) -> float:
    """Linearly anneal ``clipGuidanceWeight`` from the warmup start value."""
    if config.clipGuidanceWarmupEpochs <= 0 or epoch >= config.clipGuidanceWarmupEpochs:
        return config.clipGuidanceWeight
    t = epoch / config.clipGuidanceWarmupEpochs
    return config.clipGuidanceWeightStart + t * (
        config.clipGuidanceWeight - config.clipGuidanceWeightStart
    )


def trainStepBatch(
    components: V2FullTrainingComponents,
    batch: V2Batch,
    config: V2FullTrainingConfig,
    generators: TrainingRandomState,
    backwardScale: float = 1.0,
    clipWeight: float | None = None,
) -> dict[str, float]:
    """Forward + backward + optimiser step for a single batch.

    ``backwardScale`` scales the loss before backward — used by gradient
    accumulation to average across micro-batches.  The optimiser step
    itself is left to the caller (so accumulation can flush after N
    micro-batches).
    """
    components.encoder.train()
    components.denoiser.train()
    schedule = components.schedule

    # Phase D.3 — SMPL-22 mirror augmentation.  Imported lazily to
    # avoid a circular dependency (mirror_v2 imports V2Batch from this
    # module).  When ``mirrorProb == 0`` the helper returns ``batch``
    # unchanged with zero overhead.
    if config.mirrorProb > 0.0:
        from ainimator.data.mirror_v2 import mirrorBatch

        batch = mirrorBatch(
            batch,
            probability=config.mirrorProb,
            generator=generators.cpuGenerator,
        )

    rotationRaw = batch.rotation6d
    rootTranslationRaw = batch.rootTranslation

    # Z-normalise the targets so the diffusion sees unit-variance x_0.
    rotation = components.normalizer.normalizeBone(rotationRaw)
    rootTranslation = components.normalizer.normalizeGlobal(
        rootTranslationRaw
    )

    textOutput, conditionedMask = _encodePromptBatch(
        components,
        batch.rawTexts,
        config.condMaskProb,
        generators,
        useNullEmbedding=config.useNullEmbedding,
    )

    timesteps = torch.randint(
        0,
        schedule.numSteps,
        (rotation.shape[0],),
        generator=generators.cpuGenerator,
        device=torch.device("cpu"),
    ).to(components.device)
    noiseRotation = generators.sampleNormal(
        rotation.shape, device=components.device
    )
    noiseGlobal = generators.sampleNormal(
        rootTranslation.shape, device=components.device
    )
    xtRotation, _ = schedule.qSample(
        rotation, timesteps, noise=noiseRotation
    )
    xtGlobal, _ = schedule.qSample(
        rootTranslation, timesteps, noise=noiseGlobal
    )
    targetRotation = schedule.predictionTarget(
        rotation, noiseRotation, timesteps, config.predictionMode
    )
    targetGlobal = schedule.predictionTarget(
        rootTranslation, noiseGlobal, timesteps, config.predictionMode
    )

    motionMask = batch.motionMask
    selfCondBone, selfCondGlobal = _maybeSelfCondEstimate(
        components,
        config,
        generators,
        config.selfConditioningProb,
        xtRotation,
        xtGlobal,
        timesteps,
        textOutput,
        motionMask,
    )
    output = components.denoiser(
        noisyMotion=xtRotation,
        timesteps=timesteps,
        textHiddenStates=textOutput.hiddenStates,
        textKeyPaddingMask=textOutput.keyPaddingMask,
        noisyGlobalFeatures=xtGlobal,
        motionKeyPaddingMask=~motionMask,
        selfCondBone=selfCondBone,
        selfCondGlobal=selfCondGlobal,
    )

    boneLoss = diffusionLossV2(
        prediction=output.boneOutput,
        target=targetRotation,
        timesteps=timesteps,
        alphasCumprod=schedule.alphasCumprod,
        predictionMode=config.predictionMode,
        gamma=config.minSnrGamma,
        motionMask=motionMask,
    )
    assert output.globalOutput is not None
    globalLoss = diffusionLossV2(
        prediction=output.globalOutput,
        target=targetGlobal,
        timesteps=timesteps,
        alphasCumprod=schedule.alphasCumprod,
        predictionMode=config.predictionMode,
        gamma=config.minSnrGamma,
        motionMask=motionMask,
    )
    total = boneLoss + globalLoss

    # Phase 1.1 — reconstruct x0 once for both FK-based losses (velocity
    # + joint position) so we do not invert the prediction twice.
    needsX0 = (
        config.velocityXyzWeight > 0.0
        or config.jointPositionWeight > 0.0
        or config.footContactWeight > 0.0
        or config.x0ContrastiveWeight > 0.0
    )
    x0PredRaw: torch.Tensor | None = None
    x0PredNorm: torch.Tensor | None = None
    if needsX0:
        x0PredNorm = schedule.x0FromPrediction(
            output.boneOutput,
            xtRotation,
            timesteps,
            config.predictionMode,
        )
        x0PredRaw = components.normalizer.denormalizeBone(x0PredNorm)

    velLossValue = 0.0
    if config.velocityXyzWeight > 0.0:
        assert x0PredRaw is not None
        velLoss = velocityXyzLossV2(
            predictedRotation6d=x0PredRaw,
            targetRotation6d=rotationRaw,
            timesteps=timesteps,
            alphasCumprod=schedule.alphasCumprod,
            motionMask=motionMask,
        )
        total = total + config.velocityXyzWeight * velLoss
        velLossValue = float(velLoss.detach().item())

    jointPositionLossValue = 0.0
    if config.jointPositionWeight > 0.0:
        assert x0PredRaw is not None
        jointLoss = jointPositionLossV2(
            predictedRotation6d=x0PredRaw,
            targetRotation6d=rotationRaw,
            timesteps=timesteps,
            alphasCumprod=schedule.alphasCumprod,
            motionMask=motionMask,
        )
        total = total + config.jointPositionWeight * jointLoss
        jointPositionLossValue = float(jointLoss.detach().item())

    footContactLossValue = 0.0
    if config.footContactWeight > 0.0:
        assert x0PredRaw is not None
        footLoss = footContactLossV2(
            predictedRotation6d=x0PredRaw,
            targetRotation6d=rotationRaw,
            timesteps=timesteps,
            alphasCumprod=schedule.alphasCumprod,
            motionMask=motionMask,
        )
        total = total + config.footContactWeight * footLoss
        footContactLossValue = float(footLoss.detach().item())

    # Phase D.1 (Levier B) — text↔motion contrastive (InfoNCE) loss.
    # Both modalities go through their own projection MLP inside the
    # alignment head, then L2-normalize.  The text projection is owned
    # by the denoiser's alignment head so the param updates flow into
    # the same module that produces motionEmbedding.
    clipGuidanceLossValue = 0.0
    if (
        config.clipGuidanceWeight > 0.0
        and output.motionEmbedding is not None
    ):
        textEmbedding = components.denoiser.alignmentHead.projectText(
            textOutput.hiddenStates,
            textOutput.keyPaddingMask,
        )
        negTexts, negMotions = None, None
        if components.contrastiveBank is not None:
            bankResult = components.contrastiveBank.dequeue()
            if bankResult is not None:
                negTexts, negMotions = bankResult
        contrastiveLoss = textMotionContrastiveLoss(
            textEmbedding=textEmbedding,
            motionEmbedding=output.motionEmbedding,
            temperature=config.contrastiveTemperature,
            negativeTexts=negTexts,
            negativeMotions=negMotions,
        )
        if components.contrastiveBank is not None:
            components.contrastiveBank.enqueue(
                textEmbedding, output.motionEmbedding
            )
        activeClipWeight = clipWeight if clipWeight is not None else config.clipGuidanceWeight
        total = total + activeClipWeight * contrastiveLoss
        clipGuidanceLossValue = float(contrastiveLoss.detach().item())

    # Phase F iter-2 — auxiliary contrastive loss on the *raw* pool.
    # Forces the encoder's masked-mean pool to discriminate prompts.
    # No memory bank here: within-batch negatives are enough for the
    # regularisation effect, and a second bank would double the state.
    auxPoolLossValue = 0.0
    if (
        config.auxPoolContrastiveWeight > 0.0
        and output.textPooledRaw is not None
        and output.motionPooledRaw is not None
    ):
        auxPoolLoss = textMotionContrastiveLoss(
            textEmbedding=output.textPooledRaw,
            motionEmbedding=output.motionPooledRaw,
            temperature=config.contrastiveTemperature,
        )
        total = total + config.auxPoolContrastiveWeight * auxPoolLoss
        auxPoolLossValue = float(auxPoolLoss.detach().item())

    # 2026-06-01 — generated-motion (x0) contrastive.  Embeds the
    # reconstructed x0 (not the noisy input) so the *prediction* must be
    # classifiable to its prompt, closing the generation loophole.
    # Weighted toward high-t samples and zeroed on CFG-dropped rows.
    x0ContrastiveValue = 0.0
    if (
        config.x0ContrastiveWeight > 0.0
        and components.denoiser.config.x0AlignmentEnabled
    ):
        assert x0PredNorm is not None
        x0Loss = _x0ContrastiveLoss(
            components=components,
            x0PredNorm=x0PredNorm,
            textOutput=textOutput,
            timesteps=timesteps,
            motionMask=motionMask,
            conditionedMask=conditionedMask,
            temperature=config.contrastiveTemperature,
        )
        if x0Loss is not None:
            total = total + config.x0ContrastiveWeight * x0Loss
            x0ContrastiveValue = float(x0Loss.detach().item())

    (total * backwardScale).backward()

    return {
        "loss_total": float(total.detach().item()),
        "loss_bone": float(boneLoss.detach().item()),
        "loss_global": float(globalLoss.detach().item()),
        "loss_vel_xyz": velLossValue,
        "loss_joint_xyz": jointPositionLossValue,
        "loss_foot_contact": footContactLossValue,
        "loss_clip_guidance": clipGuidanceLossValue,
        "loss_clip_aux_pool": auxPoolLossValue,
        "loss_x0_contrastive": x0ContrastiveValue,
    }


def _x0ContrastiveLoss(
    components: V2FullTrainingComponents,
    x0PredNorm: torch.Tensor,
    textOutput: TextEncoderOutput,
    timesteps: torch.Tensor,
    motionMask: torch.Tensor,
    conditionedMask: torch.Tensor | None,
    temperature: float,
) -> torch.Tensor | None:
    """Symmetric InfoNCE between text and the reconstructed-x0 embedding.

    Embeds ``x0PredNorm`` through the denoiser's dedicated TMR-style
    encoder (which never sees the noisy input or the text) and contrasts
    it with the pooled prompt.  The per-sample weight is ``1 - ᾱ_t`` so
    high-timestep samples — where the prediction cannot copy the noisy
    input — drive the gradient; CFG-dropped rows are zeroed.  Returns
    ``None`` when no sample carries positive weight.
    """
    schedule = components.schedule
    genMotionEmb = components.denoiser.encodeGeneratedMotion(
        x0PredNorm, ~motionMask
    )
    pooledText = poolTextEmbedding(
        textOutput.hiddenStates, textOutput.keyPaddingMask
    )
    textEmb = components.denoiser.projectGeneratedMotionText(pooledText)

    alphaT = schedule.alphasCumprod.to(timesteps.device).gather(
        0, timesteps.long()
    ).to(torch.float32)
    sampleWeights = 1.0 - alphaT
    if conditionedMask is not None:
        sampleWeights = sampleWeights * conditionedMask.to(
            sampleWeights.dtype
        )
    if float(sampleWeights.sum().item()) <= 0.0:
        return None
    return textMotionContrastiveLoss(
        textEmbedding=textEmb,
        motionEmbedding=genMotionEmb,
        temperature=temperature,
        sampleWeights=sampleWeights,
    )


def _maybeSelfCondEstimate(
    components: V2FullTrainingComponents,
    config: V2FullTrainingConfig,
    generator: TrainingRandomState,
    probability: float,
    xtRotation: torch.Tensor,
    xtGlobal: torch.Tensor,
    timesteps: torch.Tensor,
    textOutput: TextEncoderOutput,
    motionMask: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Optionally produce a detached self-conditioning x0 estimate.

    With probability ``probability`` runs one no-grad denoiser pass
    (no self-cond input) and returns the reconstructed x0 (bone, global)
    in normalised space; otherwise returns ``(None, None)`` so the real
    pass self-conditions on zeros.
    """
    if not config.useSelfConditioning or probability <= 0.0:
        return None, None
    coin = torch.rand(
        (1,), generator=generator.cpuGenerator, device=torch.device("cpu")
    ).item()
    if coin >= probability:
        return None, None
    schedule = components.schedule
    with torch.no_grad():
        first = components.denoiser(
            noisyMotion=xtRotation,
            timesteps=timesteps,
            textHiddenStates=textOutput.hiddenStates,
            textKeyPaddingMask=textOutput.keyPaddingMask,
            noisyGlobalFeatures=xtGlobal,
            motionKeyPaddingMask=~motionMask,
        )
        scBone = schedule.x0FromPrediction(
            first.boneOutput, xtRotation, timesteps, config.predictionMode
        ).detach()
        scGlobal = None
        if first.globalOutput is not None:
            scGlobal = schedule.x0FromPrediction(
                first.globalOutput, xtGlobal, timesteps,
                config.predictionMode,
            ).detach()
    return scBone, scGlobal


@torch.no_grad()
def validateEpoch(
    components: V2FullTrainingComponents,
    dataset: PreprocessedLinkDataset,
    valIndices: Sequence[int],
    config: V2FullTrainingConfig,
) -> dict[str, float]:
    """Run a full validation pass and return the averaged per-sample losses."""
    components.encoder.eval()
    components.denoiser.eval()
    schedule = components.schedule

    counters: dict[str, float] = {
        "loss_total": 0.0,
        "loss_bone": 0.0,
        "loss_global": 0.0,
        "loss_vel_xyz": 0.0,
        "loss_joint_xyz": 0.0,
        "loss_foot_contact": 0.0,
        "loss_clip_guidance": 0.0,
        "loss_clip_aux_pool": 0.0,
        "loss_x0_contrastive": 0.0,
        # Phase F iter-2 diagnostics — pair-mean cosine sim within
        # each val batch.  Counters use a parallel "_count" key so the
        # average is per-pair not per-sample.
        "align_pair_sim": 0.0,
        "_align_pair_count": 0.0,
        "enc_pair_sim": 0.0,
        "_enc_pair_count": 0.0,
    }
    sampleCount = 0
    valGenerator = TrainingRandomState.fromSeed(
        seed=config.validationSeed, device=components.device
    )
    for batch in iterBatches(
        dataset=dataset,
        indices=valIndices,
        batchSize=config.batchSize,
        maxFrames=config.maxFrames,
        device=components.device,
        shuffle=False,
        rngSeed=config.validationSeed,
    ):
        rotation = components.normalizer.normalizeBone(batch.rotation6d)
        rootTranslation = components.normalizer.normalizeGlobal(
            batch.rootTranslation
        )
        encoded = components.tokenizer.encode(list(batch.rawTexts))
        textOutput = components.encoder(
            encoded.inputIds.to(components.device),
            encoded.attentionMask.to(components.device),
        )
        timesteps = torch.randint(
            0,
            schedule.numSteps,
            (rotation.shape[0],),
            generator=valGenerator.cpuGenerator,
            device=torch.device("cpu"),
        ).to(components.device)
        xtRotation, noiseRotation = schedule.qSample(
            rotation,
            timesteps,
            noise=valGenerator.sampleNormal(
                rotation.shape, device=components.device
            ),
        )
        xtGlobal, noiseGlobal = schedule.qSample(
            rootTranslation,
            timesteps,
            noise=valGenerator.sampleNormal(
                rootTranslation.shape, device=components.device
            ),
        )
        targetRotation = schedule.predictionTarget(
            rotation, noiseRotation, timesteps, config.predictionMode
        )
        targetGlobal = schedule.predictionTarget(
            rootTranslation, noiseGlobal, timesteps, config.predictionMode
        )

        valSelfCondBone, valSelfCondGlobal = _maybeSelfCondEstimate(
            components,
            config,
            valGenerator,
            1.0 if config.useSelfConditioning else 0.0,
            xtRotation,
            xtGlobal,
            timesteps,
            textOutput,
            batch.motionMask,
        )
        output = components.denoiser(
            noisyMotion=xtRotation,
            timesteps=timesteps,
            textHiddenStates=textOutput.hiddenStates,
            textKeyPaddingMask=textOutput.keyPaddingMask,
            noisyGlobalFeatures=xtGlobal,
            motionKeyPaddingMask=~batch.motionMask,
            selfCondBone=valSelfCondBone,
            selfCondGlobal=valSelfCondGlobal,
        )
        bone = diffusionLossV2(
            output.boneOutput,
            targetRotation,
            timesteps,
            schedule.alphasCumprod,
            config.predictionMode,
            gamma=config.minSnrGamma,
            motionMask=batch.motionMask,
        )
        assert output.globalOutput is not None
        global_ = diffusionLossV2(
            output.globalOutput,
            targetGlobal,
            timesteps,
            schedule.alphasCumprod,
            config.predictionMode,
            gamma=config.minSnrGamma,
            motionMask=batch.motionMask,
        )
        total = bone + global_

        # Phase 1.1 — share x0 reconstruction between FK losses (val).
        needsX0 = (
            config.velocityXyzWeight > 0.0
            or config.jointPositionWeight > 0.0
            or config.footContactWeight > 0.0
            or config.x0ContrastiveWeight > 0.0
        )
        x0PredRaw: torch.Tensor | None = None
        x0PredNorm: torch.Tensor | None = None
        if needsX0:
            x0PredNorm = schedule.x0FromPrediction(
                output.boneOutput, xtRotation, timesteps, config.predictionMode
            )
            x0PredRaw = components.normalizer.denormalizeBone(x0PredNorm)

        velContribution = 0.0
        if config.velocityXyzWeight > 0.0:
            assert x0PredRaw is not None
            velLoss = velocityXyzLossV2(
                predictedRotation6d=x0PredRaw,
                targetRotation6d=batch.rotation6d,
                timesteps=timesteps,
                alphasCumprod=schedule.alphasCumprod,
                motionMask=batch.motionMask,
            )
            total = total + config.velocityXyzWeight * velLoss
            velContribution = float(velLoss.detach().item())

        jointContribution = 0.0
        if config.jointPositionWeight > 0.0:
            assert x0PredRaw is not None
            jointLoss = jointPositionLossV2(
                predictedRotation6d=x0PredRaw,
                targetRotation6d=batch.rotation6d,
                timesteps=timesteps,
                alphasCumprod=schedule.alphasCumprod,
                motionMask=batch.motionMask,
            )
            total = total + config.jointPositionWeight * jointLoss
            jointContribution = float(jointLoss.detach().item())

        footContribution = 0.0
        if config.footContactWeight > 0.0:
            assert x0PredRaw is not None
            footLoss = footContactLossV2(
                predictedRotation6d=x0PredRaw,
                targetRotation6d=batch.rotation6d,
                timesteps=timesteps,
                alphasCumprod=schedule.alphasCumprod,
                motionMask=batch.motionMask,
            )
            total = total + config.footContactWeight * footLoss
            footContribution = float(footLoss.detach().item())

        clipContribution = 0.0
        alignSimContribution = 0.0
        alignSimSampleCount = 0
        if (
            config.clipGuidanceWeight > 0.0
            and output.motionEmbedding is not None
        ):
            textEmbedding = components.denoiser.alignmentHead.projectText(
                textOutput.hiddenStates,
                textOutput.keyPaddingMask,
            )
            contrastiveLoss = textMotionContrastiveLoss(
                textEmbedding=textEmbedding,
                motionEmbedding=output.motionEmbedding,
                temperature=config.contrastiveTemperature,
            )
            total = total + config.clipGuidanceWeight * contrastiveLoss
            clipContribution = float(contrastiveLoss.detach().item())
            # Diagnostic — mean off-diagonal cosine similarity of the
            # post-MLP text embeddings within the batch.  A low value
            # (< 0.5) means the alignment head discriminates prompts;
            # a high value (close to 1) means the head collapsed.
            alignSimContribution, alignSimSampleCount = (
                _meanOffDiagonalCosine(textEmbedding)
            )

        # Phase F iter-2 — auxiliary pool contrastive loss on the raw
        # encoder pool.  Computed even at val time so the curve is
        # logged consistently with training.
        auxPoolContribution = 0.0
        encSimContribution = 0.0
        encSimSampleCount = 0
        if (
            config.auxPoolContrastiveWeight > 0.0
            and output.textPooledRaw is not None
            and output.motionPooledRaw is not None
        ):
            auxPoolLoss = textMotionContrastiveLoss(
                textEmbedding=output.textPooledRaw,
                motionEmbedding=output.motionPooledRaw,
                temperature=config.contrastiveTemperature,
            )
            total = total + config.auxPoolContrastiveWeight * auxPoolLoss
            auxPoolContribution = float(auxPoolLoss.detach().item())
            # Diagnostic — mean off-diagonal cosine similarity of the
            # raw encoder pool within the batch.  This is the **signal
            # of gold** for diagnosing the cond/uncond collapse: if it
            # stays close to 1, FiLM/AdaLN see near-identical
            # conditioning vectors regardless of the prompt.
            encSimContribution, encSimSampleCount = (
                _meanOffDiagonalCosine(output.textPooledRaw)
            )

        x0Contribution = 0.0
        if (
            config.x0ContrastiveWeight > 0.0
            and components.denoiser.config.x0AlignmentEnabled
        ):
            assert x0PredNorm is not None
            x0Loss = _x0ContrastiveLoss(
                components=components,
                x0PredNorm=x0PredNorm,
                textOutput=textOutput,
                timesteps=timesteps,
                motionMask=batch.motionMask,
                conditionedMask=None,
                temperature=config.contrastiveTemperature,
            )
            if x0Loss is not None:
                total = total + config.x0ContrastiveWeight * x0Loss
                x0Contribution = float(x0Loss.detach().item())

        batchSize = rotation.shape[0]
        counters["loss_total"] += float(total.detach().item()) * batchSize
        counters["loss_bone"] += float(bone.detach().item()) * batchSize
        counters["loss_global"] += float(global_.detach().item()) * batchSize
        counters["loss_vel_xyz"] += velContribution * batchSize
        counters["loss_joint_xyz"] += jointContribution * batchSize
        counters["loss_foot_contact"] += footContribution * batchSize
        counters["loss_clip_guidance"] += clipContribution * batchSize
        counters["loss_clip_aux_pool"] += auxPoolContribution * batchSize
        counters["loss_x0_contrastive"] += x0Contribution * batchSize
        if alignSimSampleCount > 0:
            counters["align_pair_sim"] += (
                alignSimContribution * alignSimSampleCount
            )
            counters["_align_pair_count"] += alignSimSampleCount
        if encSimSampleCount > 0:
            counters["enc_pair_sim"] += (
                encSimContribution * encSimSampleCount
            )
            counters["_enc_pair_count"] += encSimSampleCount
        sampleCount += batchSize

    # Pair-sim metrics use their own pair counters; everything else is
    # averaged per sample.  Hidden "_*_count" keys are stripped.
    alignCount = counters.pop("_align_pair_count")
    encCount = counters.pop("_enc_pair_count")
    result: dict[str, float] = {}
    for key, value in counters.items():
        if key == "align_pair_sim":
            # Default to 0.0 when no pairs (single-sample batches or
            # alignment head disabled) so the metric stays finite for
            # downstream finiteness checks.
            result[key] = value / alignCount if alignCount > 0 else 0.0
        elif key == "enc_pair_sim":
            result[key] = value / encCount if encCount > 0 else 0.0
        else:
            result[key] = (
                value / sampleCount if sampleCount > 0 else float("nan")
            )
    return result


# =====================================================================
# Memory management helpers
# =====================================================================
def _releaseDeviceMemory(device: torch.device) -> None:
    """Release cached allocator memory back to the OS.

    Without this, MPS accumulates memory across epochs even after the
    Python references are dropped — the allocator hangs onto the
    pre-allocated blocks and the OS eventually starts swapping, which
    causes a brutal ~100× slowdown after a few hours of training (this
    is the canonical "epochs get slower over time" pattern on Apple
    Silicon).  CUDA has the same issue but ``empty_cache`` is more
    widely known there.

    Always followed by an explicit ``gc.collect`` so PyTorch references
    held by reference cycles in user code are freed before the cache
    flush runs.
    """
    gc.collect()
    if device.type == "mps":
        # ``torch.mps.empty_cache`` is the public API; ``synchronize``
        # is required to ensure all pending kernels have released their
        # input buffers before the cache is reclaimed.
        if hasattr(torch, "mps"):
            try:
                torch.mps.synchronize()
                torch.mps.empty_cache()
            except (RuntimeError, AttributeError):
                # Older PyTorch builds may not expose every entry-point;
                # falling back is harmless.
                pass
    elif device.type == "cuda":
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except (RuntimeError, AttributeError):
            pass
    # CPU: nothing to do — the Python allocator handles it.


def _formatMemoryStats(device: torch.device) -> str:
    """Return a one-line memory snapshot for ``device``.

    Uses the device-specific allocator API when available.  Returns an
    empty string on CPU or when stats are unavailable so callers can
    skip the log line gracefully.
    """
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        try:
            current = torch.mps.current_allocated_memory() / (1024 ** 3)
            driver = torch.mps.driver_allocated_memory() / (1024 ** 3)
            return f"mps_alloc={current:.2f}GB driver={driver:.2f}GB"
        except (RuntimeError, AttributeError):
            return ""
    if device.type == "cuda":
        try:
            current = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            return f"cuda_alloc={current:.2f}GB reserved={reserved:.2f}GB"
        except (RuntimeError, AttributeError):
            return ""
    return ""


# =====================================================================
# Top-level loop
# =====================================================================
def runFullTraining(
    config: V2FullTrainingConfig,
) -> tuple[V2FullTrainingComponents, list[dict[str, float]]]:
    """Execute the full multi-sample training loop.

    Returns the components (with the best-val_loss weights loaded) and
    the per-epoch history dicts.
    """
    config.outputDir.mkdir(parents=True, exist_ok=True)
    writeResolvedConfig(config, config.outputDir)
    device = resolveDevice(config.device)

    dataset = PreprocessedLinkDataset(
        datasetRoot=config.datasetRoot,
        includeTokenizedText=False,
    )
    LOGGER.info(
        "Loaded preprocessed dataset (%d total links, manifest=%s).",
        len(dataset.linkEntries),
        dataset.manifest.modelName,
    )

    activeIndices = selectLinkIndices(dataset, config.datasetFolders)
    LOGGER.info(
        "Active links after folder filter %s: %d.",
        config.datasetFolders or "ALL",
        len(activeIndices),
    )
    if not activeIndices:
        raise RuntimeError(
            "No samples matched the requested folder filter."
        )

    if config.sampleLinkIndices:
        # Small-N controlled mode: train and validate on exactly the
        # requested indices (no split) so we can watch per-prompt
        # reconstruction directly.
        allowed = set(activeIndices)
        missing = [i for i in config.sampleLinkIndices if i not in allowed]
        if missing:
            raise RuntimeError(
                f"sampleLinkIndices {missing} are outside the folder "
                "filter; pick indices within the selected folders."
            )
        trainIndices = list(config.sampleLinkIndices)
        valIndices = list(config.sampleLinkIndices)
        LOGGER.info(
            "Small-N mode: training on %d explicit samples %s (train==val).",
            len(trainIndices),
            trainIndices,
        )
    else:
        trainIndices, valIndices = splitTrainVal(
            activeIndices,
            validationFraction=config.validationFraction,
            seed=config.validationSeed,
        )
        LOGGER.info(
            "Train / val split: %d / %d (val fraction %.3f).",
            len(trainIndices),
            len(valIndices),
            len(valIndices) / max(1, len(activeIndices)),
        )

    components = buildFullTrainingComponents(config)
    fitNormalizerFromDataset(
        dataset=dataset,
        indices=trainIndices,
        normalizer=components.normalizer,
        maxSamples=config.normalizerFitMaxSamples,
        seed=config.seed,
    )

    resumeState: ResumeState | None = None
    if config.resumeCheckpoint is not None:
        resumeState = _loadResumeCheckpoint(
            components, config.resumeCheckpoint
        )

    encoderParams = sum(
        p.numel()
        for p in components.encoder.parameters()
        if p.requires_grad
    )
    denoiserParams = sum(
        p.numel() for p in components.denoiser.parameters()
    )
    LOGGER.info(
        "Trainable params: encoder=%d, denoiser=%d, total=%d.",
        encoderParams,
        denoiserParams,
        encoderParams + denoiserParams,
    )

    generators = TrainingRandomState.fromSeed(
        seed=config.seed, device=components.device
    )

    history: list[dict[str, float]] = []
    if resumeState is not None:
        startEpoch = resumeState.lastEpoch + 1
        bestValTotal = resumeState.bestValTotal
        bestEpoch = resumeState.bestEpoch
    else:
        startEpoch = 1
        bestValTotal = float("inf")
        bestEpoch = -1
    # 2026-05-07 — stagnation counter for the new best-checkpoint
    # policy.  Reset on every meaningful improvement; warning logged
    # when it crosses ``config.stagnationPatience``.
    stagnationEpochs = 0
    bestPath = config.outputDir / "v2_full_best.pt"
    latestPath = config.outputDir / "v2_full_latest.pt"

    # --- Health hub (A3) ----------------------------------------
    healthHub = _buildHealthHub(config) if config.healthEnabled else None
    if healthHub is not None:
        healthHub.attach(components.denoiser)

    if startEpoch > config.epochs:
        LOGGER.warning(
            "Resume requested but checkpoint already at epoch %d ≥ "
            "config.epochs=%d.  Increase --epochs to continue training.",
            resumeState.lastEpoch if resumeState else 0,
            config.epochs,
        )
        return components, history

    if startEpoch > 1:
        LOGGER.info(
            "Resuming training at epoch %d / %d (best so far: "
            "val_total=%.4f at epoch %d).",
            startEpoch,
            config.epochs,
            bestValTotal,
            bestEpoch,
        )

    epochStart = time.time()
    for epoch in range(startEpoch, config.epochs + 1):
        epochSeed = (
            int(config.seed) * 10_000 + epoch
        )  # rotates every epoch
        epochIndices = _selectEpochIndices(
            trainIndices,
            maxSamples=config.maxSamplesPerEpoch,
            seed=epochSeed,
        )

        components.optimizer.zero_grad()
        runningLoss = 0.0
        microStep = 0
        completedSteps = 0
        for batch in iterBatches(
            dataset=dataset,
            indices=epochIndices,
            batchSize=config.batchSize,
            maxFrames=config.maxFrames,
            device=components.device,
            shuffle=True,
            rngSeed=epochSeed,
        ):
            metrics = trainStepBatch(
                components=components,
                batch=batch,
                config=config,
                generators=generators,
                backwardScale=1.0 / config.gradientAccumulation,
                clipWeight=effectiveClipWeight(config, epoch),
            )
            runningLoss += metrics["loss_total"]
            microStep += 1
            if microStep % config.gradientAccumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(components.encoder.parameters())
                    + list(components.denoiser.parameters()),
                    max_norm=1.0,
                )
                components.optimizer.step()
                components.optimizer.zero_grad()
                if components.ema is not None:
                    components.ema.update(
                        _emaParameters(
                            components.encoder, components.denoiser
                        )
                    )
                completedSteps += 1
                if healthHub is not None:
                    healthHub.step(
                        globalStep=completedSteps,
                        metrics=metrics,
                    )
                if completedSteps % config.logEvery == 0:
                    LOGGER.info(
                        "epoch=%d step=%d  train_loss=%.4f",
                        epoch,
                        completedSteps,
                        runningLoss / config.logEvery,
                    )
                    runningLoss = 0.0

        # Flush any remaining accumulated gradients at epoch end.
        if microStep % config.gradientAccumulation != 0:
            torch.nn.utils.clip_grad_norm_(
                list(components.encoder.parameters())
                + list(components.denoiser.parameters()),
                max_norm=1.0,
            )
            components.optimizer.step()
            components.optimizer.zero_grad()
            if components.ema is not None:
                components.ema.update(
                    _emaParameters(
                        components.encoder, components.denoiser
                    )
                )

        # Validation pass.
        valMetrics: dict[str, float] | None = None
        if epoch % config.validateEveryEpochs == 0:
            # Phase D.4 — evaluate on EMA shadow weights when EMA is
            # active.  This is the canonical diffusion best-practice:
            # the val_loss series uses the smoothed weights so a noisy
            # online step can not register a misleading new "best".
            useEma = components.ema is not None
            if useEma:
                components.ema.storeAndSwap(
                    _emaParameters(
                        components.encoder, components.denoiser
                    )
                )
            try:
                valMetrics = validateEpoch(
                    components=components,
                    dataset=dataset,
                    valIndices=valIndices,
                    config=config,
                )
            finally:
                if useEma:
                    components.ema.restore(
                        _emaParameters(
                            components.encoder, components.denoiser
                        )
                    )
            elapsed = time.time() - epochStart
            LOGGER.info(
                "epoch=%d  val_total=%.4f  val_bone=%.4f  "
                "val_global=%.4f  val_vel=%.4f  val_jxyz=%.4f  "
                "val_foot=%.4f  val_clip=%.4f  val_aux=%.4f  "
                "val_x0=%.4f  "
                "enc_sim=%.4f  align_sim=%.4f  clip_w=%.3f  "
                "elapsed=%.1fs",
                epoch,
                valMetrics["loss_total"],
                valMetrics["loss_bone"],
                valMetrics["loss_global"],
                valMetrics["loss_vel_xyz"],
                valMetrics.get("loss_joint_xyz", 0.0),
                valMetrics.get("loss_foot_contact", 0.0),
                valMetrics.get("loss_clip_guidance", 0.0),
                valMetrics.get("loss_clip_aux_pool", 0.0),
                valMetrics.get("loss_x0_contrastive", 0.0),
                valMetrics.get("enc_pair_sim", float("nan")),
                valMetrics.get("align_pair_sim", float("nan")),
                effectiveClipWeight(config, epoch),
                elapsed,
            )
            # 2026-05-07 — best-checkpoint policy with significance
            # threshold.  The metric of interest is configurable so a
            # user with a stuck contrastive floor can track diffusion
            # alone via ``--best-metric loss_diffusion``.  The min
            # improvement avoids rewriting the best on 1e-4 noise
            # drops which used to fire every single epoch.
            currentMetric = _readBestMetric(valMetrics, config.bestMetric)
            improvement = bestValTotal - currentMetric
            isMeaningful = improvement > config.bestImprovementMin
            if isMeaningful:
                bestValTotal = currentMetric
                bestEpoch = epoch
                stagnationEpochs = 0
                # The best checkpoint is what the generation CLI will
                # load — write the EMA shadow as the canonical state
                # so inference uses the smoothed weights by default.
                _saveCheckpoint(
                    components, config, bestPath, epoch=epoch,
                    valMetrics=valMetrics,
                    bestValTotal=bestValTotal, bestEpoch=bestEpoch,
                    useEmaForState=True,
                )
                LOGGER.info(
                    "New best at epoch %d (%s=%.4f, Δ=%.4f); saved to %s.",
                    epoch,
                    config.bestMetric,
                    bestValTotal,
                    improvement,
                    bestPath,
                )
            else:
                stagnationEpochs += 1
                if (
                    config.stagnationPatience > 0
                    and stagnationEpochs >= config.stagnationPatience
                ):
                    LOGGER.warning(
                        "Stagnation: no improvement of %s by ≥ %.4f "
                        "for %d consecutive validations (best=%.4f at "
                        "epoch %d, current=%.4f).  Consider stopping "
                        "the run or revisiting hyperparameters.",
                        config.bestMetric,
                        config.bestImprovementMin,
                        stagnationEpochs,
                        bestValTotal,
                        bestEpoch,
                        currentMetric,
                    )
                    # Reset the counter so we don't spam the warning
                    # every epoch — re-fire only after another full
                    # patience window.
                    stagnationEpochs = 0

        history.append(
            {
                "epoch": float(epoch),
                **(valMetrics or {}),
            }
        )
        _saveCheckpoint(
            components, config, latestPath, epoch=epoch,
            valMetrics=valMetrics,
            bestValTotal=bestValTotal, bestEpoch=bestEpoch,
        )

        # End-of-epoch memory hygiene.  Without this, MPS hangs onto
        # buffers from the training/validation forward passes and the
        # epoch wall-clock balloons after a few hours (the user-reported
        # 1000s → 128000s pattern).  The shard cache is also cleared
        # because the next epoch shuffles the link order, so the LRU
        # contents from the previous epoch are unlikely to be reused.
        dataset.clearCache()
        _releaseDeviceMemory(components.device)
        memStats = _formatMemoryStats(components.device)
        if memStats:
            LOGGER.info("epoch=%d memory %s", epoch, memStats)

    LOGGER.info(
        "Training done. Best val_total=%.4f at epoch %d (saved to %s).",
        bestValTotal,
        bestEpoch,
        bestPath,
    )
    if healthHub is not None:
        healthHub.detach()
        healthHub.close()
    return components, history


# =====================================================================
# Helpers — epoch sub-sampling & checkpoint I/O
# =====================================================================
def _selectEpochIndices(
    trainIndices: Sequence[int],
    maxSamples: int,
    seed: int,
) -> list[int]:
    """Return up to ``maxSamples`` indices for a single epoch."""
    if maxSamples <= 0 or maxSamples >= len(trainIndices):
        return list(trainIndices)
    rng = torch.Generator(device=torch.device("cpu"))
    rng.manual_seed(int(seed))
    permutation = torch.randperm(
        len(trainIndices), generator=rng
    )[:maxSamples].tolist()
    return [trainIndices[index] for index in permutation]


def _clipEncoderConfigToDict(
    config: ClipTextEncoderConfig,
) -> dict[str, Any]:
    """Serialise a :class:`ClipTextEncoderConfig` for the checkpoint payload.

    The dual of :func:`training_v2._clipEncoderConfigFromDict` — the
    schema is distinct from the custom encoder's so ``loadCheckpointV2``
    must branch on ``text_encoder_type``.
    """
    return {
        "modelName": config.modelName,
        "maxLength": config.maxLength,
        "outputDim": config.outputDim,
        "clipHiddenDim": config.clipHiddenDim,
        "dropout": config.dropout,
        "useNullEmbedding": config.useNullEmbedding,
        "l2NormalizeOutput": config.l2NormalizeOutput,
    }


def _trainingConfigForCheckpoint(
    config: V2FullTrainingConfig,
) -> dict[str, Any]:
    """Convert the full-training config to the checkpoint payload format.

    The schema mirrors :func:`training_v2._trainingConfigToDict` so a
    full-training checkpoint can be loaded by the same generation CLI
    used for overfit checkpoints.
    """
    return {
        "datasetRoot": str(config.datasetRoot),
        "tokenizerDir": str(config.tokenizerDir),
        "outputDir": str(config.outputDir),
        "datasetFolders": list(config.datasetFolders or []),
        "sampleLinkIndex": -1,
        "epochs": config.epochs,
        # Bug fix 2026-05-07 — batchSize, gradientAccumulation,
        # maxSamplesPerEpoch were missing from the checkpoint payload
        # and showed up as ``None`` when re-loaded.  These fields are
        # critical to reproduce a run / understand a saved checkpoint
        # so we serialise them now.  ``validationFraction`` and
        # ``validationSeed`` are added too because they affect the
        # train/val split deterministically.
        "batchSize": config.batchSize,
        "gradientAccumulation": config.gradientAccumulation,
        "maxSamplesPerEpoch": config.maxSamplesPerEpoch,
        "validationFraction": config.validationFraction,
        "validationSeed": config.validationSeed,
        "validateEveryEpochs": config.validateEveryEpochs,
        "normalizerFitMaxSamples": config.normalizerFitMaxSamples,
        "learningRate": config.learningRate,
        "weightDecay": config.weightDecay,
        "minSnrGamma": config.minSnrGamma,
        "velocityXyzWeight": config.velocityXyzWeight,
        "jointPositionWeight": config.jointPositionWeight,
        "footContactWeight": config.footContactWeight,
        "clipGuidanceWeight": config.clipGuidanceWeight,
        "auxPoolContrastiveWeight": config.auxPoolContrastiveWeight,
        "x0ContrastiveWeight": config.x0ContrastiveWeight,
        "useSelfConditioning": config.useSelfConditioning,
        "selfConditioningProb": config.selfConditioningProb,
        "contrastiveTemperature": config.contrastiveTemperature,
        "diffusionStepsTraining": config.diffusionStepsTraining,
        "scheduleType": config.scheduleType,
        "predictionMode": config.predictionMode,
        "textEncoderType": config.textEncoderType,
        "clipModelName": config.clipModelName,
        "clipMaxLength": config.clipMaxLength,
        "encoderHiddenDim": config.encoderHiddenDim,
        "encoderNumLayers": config.encoderNumLayers,
        "encoderNumHeads": config.encoderNumHeads,
        "denoiserEmbedDim": config.denoiserEmbedDim,
        "denoiserNumLayers": config.denoiserNumLayers,
        "denoiserNumHeads": config.denoiserNumHeads,
        "maxFrames": config.maxFrames,
        "framesPerStep": 0,
        "seed": config.seed,
        "logEvery": config.logEvery,
        "device": config.device,
        "dropout": config.dropout,
        "condMaskProb": config.condMaskProb,
        "mirrorProb": config.mirrorProb,
        "emaDecay": config.emaDecay,
        "emaUseWarmup": config.emaUseWarmup,
        "useFilmConditioning": config.useFilmConditioning,
        "filmDropout": config.filmDropout,
        "usePerBlockFilm": config.usePerBlockFilm,
        "bestMetric": config.bestMetric,
        "bestImprovementMin": config.bestImprovementMin,
        "stagnationPatience": config.stagnationPatience,
    }


def _saveCheckpoint(
    components: V2FullTrainingComponents,
    config: V2FullTrainingConfig,
    path: Path,
    epoch: int,
    valMetrics: dict[str, float] | None,
    bestValTotal: float = float("inf"),
    bestEpoch: int = -1,
    useEmaForState: bool = False,
) -> None:
    """Persist a v3 checkpoint.

    When ``useEmaForState`` is True (typical for the *best* checkpoint),
    the canonical ``encoder_state_dict`` / ``denoiser_state_dict`` keys
    hold the EMA shadow weights so the generation CLI loads the
    smoothed model by default.  The online weights are persisted under
    ``encoder_online_state_dict`` / ``denoiser_online_state_dict`` for
    exact resume semantics.

    When ``useEmaForState`` is False (typical for the *latest* /
    resumable checkpoint), the canonical keys hold the online weights
    and the EMA shadow is persisted under ``ema_state_dict``.
    """
    encoderState = components.encoder.state_dict()
    denoiserState = components.denoiser.state_dict()
    onlineEncoder: dict[str, torch.Tensor] | None = None
    onlineDenoiser: dict[str, torch.Tensor] | None = None
    if components.ema is not None and useEmaForState:
        # Snapshot the online weights, swap in EMA, capture, restore.
        components.ema.storeAndSwap(
            _emaParameters(components.encoder, components.denoiser)
        )
        try:
            encoderState = {
                k: v.clone() for k, v in components.encoder.state_dict().items()
            }
            denoiserState = {
                k: v.clone() for k, v in components.denoiser.state_dict().items()
            }
        finally:
            components.ema.restore(
                _emaParameters(components.encoder, components.denoiser)
            )
        # The original online weights are recorded too — needed for
        # resume so the next session continues training with the
        # post-step weights, not the smoothed ones.
        onlineEncoder = {
            k: v.clone() for k, v in components.encoder.state_dict().items()
        }
        onlineDenoiser = {
            k: v.clone() for k, v in components.denoiser.state_dict().items()
        }

    # Phase 2 — the encoder config schema differs between the custom
    # BPE transformer and the frozen CLIP tower.  ``text_encoder_type``
    # tells :func:`loadCheckpointV2` which branch to rebuild.
    isClipEncoder = isinstance(components.encoder, ClipTextEncoder)
    if isClipEncoder:
        encoderConfigDict = _clipEncoderConfigToDict(
            components.encoder.config
        )
        # Drop the ~63M frozen CLIP weights from the checkpoint — they
        # are reloaded from the HF hub by ``ClipTextEncoder.__init__``.
        # Only the trainable projection + null embedding are persisted.
        encoderState = {
            k: v for k, v in encoderState.items()
            if not k.startswith("clip.")
        }
        if onlineEncoder is not None:
            onlineEncoder = {
                k: v for k, v in onlineEncoder.items()
                if not k.startswith("clip.")
            }
    else:
        encoderConfigDict = _encoderConfigToDict(components.encoder.config)

    payload: dict[str, Any] = {
        "version": 3,
        "text_encoder_type": "clip" if isClipEncoder else "custom",
        "encoder_state_dict": encoderState,
        "encoder_config": encoderConfigDict,
        "denoiser_state_dict": denoiserState,
        "denoiser_config": _denoiserConfigToDict(
            components.denoiser.config
        ),
        "schedule_config": _scheduleConfigToDict(
            components.schedule.config
        ),
        "schedule_state_dict": components.schedule.state_dict(),
        "normalizer_config": components.normalizer.configToDict(),
        "normalizer_state_dict": components.normalizer.state_dict(),
        "optimizer_state_dict": components.optimizer.state_dict(),
        "tokenizer_dir": str(config.tokenizerDir.resolve()),
        "training_config": _trainingConfigForCheckpoint(config),
        "training_sample": {
            "sampleId": -1,
            "textId": -1,
            "rawText": "",
            "frames": -1,
        },
        "training_meta": {
            "epoch": int(epoch),
            "val_metrics": valMetrics,
            "best_val_total": float(bestValTotal),
            "best_epoch": int(bestEpoch),
            "datasetFolders": list(config.datasetFolders or []),
            "ema_active": components.ema is not None,
            "state_is_ema": bool(
                useEmaForState and components.ema is not None
            ),
        },
    }
    if onlineEncoder is not None:
        payload["encoder_online_state_dict"] = onlineEncoder
    if onlineDenoiser is not None:
        payload["denoiser_online_state_dict"] = onlineDenoiser
    if components.ema is not None:
        payload["ema_state_dict"] = components.ema.stateDict()
    saveTorchObjectAtomically(payload, path)


@dataclass(frozen=True)
class ResumeState:
    """Bookkeeping recovered from a checkpoint to continue training.

    Attributes
    ----------
    lastEpoch : int
        Last epoch that completed BEFORE the crash / save.  The training
        loop resumes at ``lastEpoch + 1``.  Falls back to ``0`` when the
        checkpoint pre-dates the metadata field — in that case the next
        run starts at epoch 1 (full retraining).
    bestValTotal : float
        Best ``val_total`` observed so far.  Restored so a poor
        post-resume validation can not silently overwrite the
        ``v2_full_best.pt`` checkpoint with a worse one.  Falls back to
        ``+inf`` when the metadata is missing.
    bestEpoch : int
        Epoch at which ``bestValTotal`` was reached.  Used in logs only.
    """

    lastEpoch: int
    bestValTotal: float
    bestEpoch: int


def _loadResumeCheckpoint(
    components: V2FullTrainingComponents,
    path: Path,
) -> ResumeState:
    """Load training state from disk and return resume bookkeeping.

    Restores encoder + denoiser + schedule + normalizer + optimizer
    state in-place on ``components``.  Reads ``training_meta`` to
    rebuild the ``ResumeState`` (last completed epoch, best val_total,
    best epoch) so the outer training loop can pick up exactly where
    the previous run stopped.
    """
    payload = torch.load(
        path, map_location=components.device, weights_only=False
    )
    # Resume prefers the online weights when both are available — the
    # online state is the post-step trajectory we need to continue.
    # When the checkpoint was saved with ``useEmaForState=True`` (the
    # best.pt path) the canonical state_dict holds the EMA shadow and
    # the online weights live under ``*_online_state_dict``.
    onlineEncoder = payload.get("encoder_online_state_dict")
    onlineDenoiser = payload.get("denoiser_online_state_dict")
    # CLIP checkpoints persist only the trainable projection + null
    # embedding (the frozen ``clip.*`` tower is filtered at save time
    # and reloaded from the HF hub by ``ClipTextEncoder.__init__``), so
    # the load must tolerate the missing ``clip.*`` keys.
    encoderStrict = not isinstance(components.encoder, ClipTextEncoder)
    components.encoder.load_state_dict(
        onlineEncoder
        if onlineEncoder is not None
        else payload["encoder_state_dict"],
        strict=encoderStrict,
    )
    components.denoiser.load_state_dict(
        onlineDenoiser
        if onlineDenoiser is not None
        else payload["denoiser_state_dict"]
    )
    if "schedule_state_dict" in payload:
        components.schedule.load_state_dict(payload["schedule_state_dict"])
    if "normalizer_state_dict" in payload:
        components.normalizer.load_state_dict(payload["normalizer_state_dict"])
    if "optimizer_state_dict" in payload:
        try:
            components.optimizer.load_state_dict(
                payload["optimizer_state_dict"]
            )
        except (ValueError, KeyError) as error:
            LOGGER.warning(
                "Optimizer state could not be restored (%s); "
                "starting from a fresh optimiser.",
                error,
            )
    # Phase D.4 — restore EMA shadow when available and active.
    if components.ema is not None and "ema_state_dict" in payload:
        try:
            components.ema.loadStateDict(payload["ema_state_dict"])
        except RuntimeError as error:
            LOGGER.warning(
                "EMA state could not be restored (%s); re-initialising "
                "from the loaded online weights.",
                error,
            )

    meta = payload.get("training_meta") or {}
    lastEpoch = int(meta.get("epoch", 0) or 0)
    bestValTotal = float(meta.get("best_val_total", float("inf")))
    bestEpoch = int(meta.get("best_epoch", -1))
    LOGGER.info(
        "Resumed components from %s (lastEpoch=%d, bestValTotal=%.4f "
        "at epoch %d).",
        path,
        lastEpoch,
        bestValTotal,
        bestEpoch,
    )
    return ResumeState(
        lastEpoch=lastEpoch,
        bestValTotal=bestValTotal,
        bestEpoch=bestEpoch,
    )



def _buildHealthHub(config: "V2FullTrainingConfig") -> HealthHub:
    """Construct a HealthHub from the run config.

    Parameters
    ----------
    config : V2FullTrainingConfig
        Training configuration (provides outputDir, healthEverySteps).

    Returns
    -------
    HealthHub
    """
    hub = buildHealthHub(config.outputDir)
    hub._everySteps = config.healthEverySteps
    return hub
