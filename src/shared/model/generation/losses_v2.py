"""Loss functions for AI-nimator v2 diffusion training.

The v2 stack collapses the legacy 11-loss orchestration into **three
canonical losses**, all consumed from this module:

1. :func:`diffusionLossV2` — primary MSE on the prediction target
   (``x0`` or ``v``), with Min-SNR-γ reweighting calibrated for the
   active prediction mode.
2. :func:`velocityXyzLossV2` — auxiliary loss on the per-frame velocity
   of the FK-derived joint positions.  Scheduled via ``timestep`` to
   dodge the FK-of-noise problem at high t.
2bis. :func:`jointPositionLossV2` — MDM-style geometric loss on the
   FK-derived joint positions themselves.  Same timestep scheduling
   as the velocity variant.  This is the canonical fix for the
   "good rotation MSE, bad animation" syndrome — supervises positions
   so small rotation errors near the root cannot compound into large
   end-effector drift.
3. :func:`textMotionAlignmentLoss` — cosine-similarity loss between the
   text and motion embeddings produced by the v2 text/motion encoders.
   (The text-motion alignment uses the same custom text encoder used
   for cross-attention; the *motion* encoder side is built in Phase C.)

The legacy module ``losses.py`` continues to back the v1 training loop
until migration completes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.shared.model.components.ops import (
    maskedMean,
    rot6dToJointXYZ,
    temporalDifference,
)
from src.shared.model.generation.noise_schedule import (
    PREDICTION_EPSILON,
    PREDICTION_V,
    PREDICTION_X0,
    SUPPORTED_PREDICTIONS,
)

# Default Min-SNR-γ — same value as the Étape 1 stack (5.0 from
# Hang et al. 2023).  The *interpretation* of γ changes with
# predictionMode (see ``minSnrLossWeights``), but γ=5.0 is a reasonable
# default for both x0 and v.
DEFAULT_MIN_SNR_GAMMA = 5.0

# Default schedule for the auxiliary velocity-XYZ loss.  ``timestep``
# multiplies the loss by ᾱ_t so high-t samples (where FK-of-noise is
# meaningless) are zeroed out and only the low-t samples drive the
# joint-velocity gradient.
VELOCITY_SCHEDULE_NONE = "none"
VELOCITY_SCHEDULE_TIMESTEP = "timestep"
SUPPORTED_VELOCITY_SCHEDULES: tuple[str, ...] = (
    VELOCITY_SCHEDULE_NONE,
    VELOCITY_SCHEDULE_TIMESTEP,
)


# ---------------------------------------------------------------------
# Min-SNR weighting
# ---------------------------------------------------------------------
def minSnrLossWeights(
    timesteps: torch.Tensor,
    alphasCumprod: torch.Tensor,
    predictionMode: str,
    gamma: float = DEFAULT_MIN_SNR_GAMMA,
) -> torch.Tensor:
    """Compute per-sample Min-SNR loss weights.

    The formulation follows Hang et al. 2023 ("Efficient Diffusion
    Training via Min-SNR Weighting Strategy") with the per-target
    correction from Salimans & Ho 2022 ("Progressive Distillation"):

    * ``predictionMode="x0"``       →  ``min(SNR, γ) / SNR``
    * ``predictionMode="epsilon"``  →  ``min(SNR, γ)`` (per Hang et al.)
    * ``predictionMode="v"``        →  ``min(SNR, γ) / (SNR + 1)``

    All three reduce to the trivial weight 1 when ``γ == 0``.

    Parameters
    ----------
    timesteps : torch.Tensor
        Long tensor of shape ``(B,)`` in ``[0, numSteps)``.
    alphasCumprod : torch.Tensor
        ``ᾱ`` buffer from :class:`NoiseSchedule`, shape ``(numSteps,)``.
    predictionMode : str
        Active prediction target — must be one of
        :data:`SUPPORTED_PREDICTIONS`.
    gamma : float
        SNR clipping threshold.  ``0`` disables Min-SNR (returns ones).

    Returns
    -------
    torch.Tensor
        Per-sample weights of shape ``(B,)``, dtype float32.
    """
    if predictionMode not in SUPPORTED_PREDICTIONS:
        raise ValueError(
            f"predictionMode must be one of {SUPPORTED_PREDICTIONS}; "
            f"got {predictionMode!r}."
        )
    if gamma <= 0.0:
        return torch.ones_like(timesteps, dtype=torch.float32)

    alphasCumprod = alphasCumprod.to(device=timesteps.device)
    alphaT = alphasCumprod.gather(0, timesteps.long())
    snr = alphaT / torch.clamp(1.0 - alphaT, min=1e-8)
    gammaTensor = torch.full_like(snr, float(gamma))
    capped = torch.minimum(snr, gammaTensor)

    if predictionMode == PREDICTION_X0:
        weights = capped / torch.clamp(snr, min=1e-8)
    elif predictionMode == PREDICTION_EPSILON:
        weights = capped
    else:  # PREDICTION_V
        weights = capped / (snr + 1.0)
    return weights.to(torch.float32)


# ---------------------------------------------------------------------
# Per-sample MSE
# ---------------------------------------------------------------------
def perSampleMse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    motionMask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean-squared error reduced to one scalar per batch item.

    Used as the building block for :func:`diffusionLossV2` so the
    Min-SNR weights can scale the per-sample contribution **before**
    the final batch average.

    Returns a tensor of shape ``(B,)``.
    """
    squaredError = (predicted - target) ** 2
    if motionMask is not None:
        # Broadcast (B, F) over trailing dims of squaredError.
        expandMask = motionMask
        while expandMask.dim() < squaredError.dim():
            expandMask = expandMask.unsqueeze(-1)
        expandMask = expandMask.float()
        numerator = (squaredError * expandMask).flatten(1).sum(dim=1)
        denom = expandMask.expand_as(squaredError).flatten(1).sum(dim=1)
        return numerator / torch.clamp(denom, min=1.0)
    return squaredError.flatten(1).mean(dim=1)


# ---------------------------------------------------------------------
# Primary diffusion loss
# ---------------------------------------------------------------------
def diffusionLossV2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    timesteps: torch.Tensor,
    alphasCumprod: torch.Tensor,
    predictionMode: str,
    gamma: float = DEFAULT_MIN_SNR_GAMMA,
    motionMask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Min-SNR-weighted MSE for v-prediction (or x0/ε) diffusion training.

    The loss is the **batch mean** of ``w_b · ||prediction − target||²``
    where ``w_b`` is the Min-SNR weight for sample ``b`` (see
    :func:`minSnrLossWeights`).

    Parameters
    ----------
    prediction : torch.Tensor
        Network output, same shape as ``target``.
    target : torch.Tensor
        Supervised target produced by
        :meth:`NoiseSchedule.predictionTarget`.
    timesteps : torch.Tensor
        Long tensor of shape ``(B,)`` of the diffusion timesteps used
        for the forward sampling.
    alphasCumprod : torch.Tensor
        ``ᾱ`` buffer from :class:`NoiseSchedule`.
    predictionMode : str
        Active prediction target.
    gamma : float, optional
        Min-SNR clipping (default 5.0; 0 disables).
    motionMask : torch.Tensor or None, optional
        Boolean mask of valid frames, shape ``(B, F)``.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must share the same shape; got "
            f"{tuple(prediction.shape)} vs {tuple(target.shape)}."
        )
    perSample = perSampleMse(prediction, target, motionMask=motionMask)
    weights = minSnrLossWeights(
        timesteps=timesteps,
        alphasCumprod=alphasCumprod,
        predictionMode=predictionMode,
        gamma=gamma,
    )
    return (perSample * weights).mean()


# ---------------------------------------------------------------------
# Auxiliary velocity-XYZ loss (FK)
# ---------------------------------------------------------------------
def velocityXyzLossV2(
    predictedRotation6d: torch.Tensor,
    targetRotation6d: torch.Tensor,
    timesteps: torch.Tensor,
    alphasCumprod: torch.Tensor,
    schedule: str = VELOCITY_SCHEDULE_TIMESTEP,
    motionMask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-frame velocity loss in joint XYZ space.

    Computes ``|| Δ(FK(predicted)) − Δ(FK(target)) ||²`` averaged over
    valid frames.  The schedule controls how the loss is weighted by
    timestep:

    * ``"none"``     — uniform weighting across timesteps.
    * ``"timestep"`` — multiply each sample by ``ᾱ_t`` so high-t
      samples contribute ~0 (FK applied to noise is meaningless).

    The 22-bone SMPL skeleton is hard-coded via :func:`rot6dToJointXYZ`.
    """
    if schedule not in SUPPORTED_VELOCITY_SCHEDULES:
        raise ValueError(
            f"schedule must be one of {SUPPORTED_VELOCITY_SCHEDULES}; "
            f"got {schedule!r}."
        )
    if predictedRotation6d.shape != targetRotation6d.shape:
        raise ValueError(
            "predicted and target rotation6d shapes must match; got "
            f"{tuple(predictedRotation6d.shape)} vs "
            f"{tuple(targetRotation6d.shape)}."
        )
    predictedXyz = rot6dToJointXYZ(predictedRotation6d)
    targetXyz = rot6dToJointXYZ(targetRotation6d)
    predictedVel = temporalDifference(predictedXyz)
    targetVel = temporalDifference(targetXyz)

    perSample = perSampleMse(
        predictedVel, targetVel, motionMask=motionMask
    )
    if schedule == VELOCITY_SCHEDULE_NONE:
        return perSample.mean()

    alphasCumprod = alphasCumprod.to(device=timesteps.device)
    alphaT = alphasCumprod.gather(0, timesteps.long()).to(torch.float32)
    return (perSample * alphaT).mean()


# ---------------------------------------------------------------------
# Auxiliary joint-position loss (FK)
# ---------------------------------------------------------------------
def jointPositionLossV2(
    predictedRotation6d: torch.Tensor,
    targetRotation6d: torch.Tensor,
    timesteps: torch.Tensor,
    alphasCumprod: torch.Tensor,
    schedule: str = VELOCITY_SCHEDULE_TIMESTEP,
    motionMask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-frame joint-position loss in XYZ space (MDM-style geometric loss).

    Computes ``|| FK(predicted) − FK(target) ||²`` averaged over valid
    frames.  This is the **single most important** auxiliary loss for
    text-to-motion diffusion: small errors on the upstream rotations
    (e.g. hips) compound through the kinematic chain into very large
    errors at end effectors (feet, hands).  Supervising rotations alone
    (``boneLoss``) produces low MSE in rotation space while still
    yielding implausible joint trajectories — the exact "good metrics,
    bad animation" syndrome.

    The schedule controls how the loss is weighted by timestep:

    * ``"none"``     — uniform weighting across timesteps.
    * ``"timestep"`` — multiply each sample by ``ᾱ_t`` so high-t
      samples contribute ~0 (FK applied to noise is meaningless).

    The 22-bone SMPL skeleton is hard-coded via :func:`rot6dToJointXYZ`.

    Parameters
    ----------
    predictedRotation6d, targetRotation6d : torch.Tensor
        Shape ``(B, F, 22, 6)``.  Both should be in the **raw**
        (denormalised) space because the FK utility assumes
        orthonormalisable 6D vectors.  Callers should denormalise the
        x0 reconstructed from a v/eps prediction before passing it in.
    timesteps : torch.Tensor
        Long tensor of shape ``(B,)`` of the diffusion timesteps used
        for the forward sampling.
    alphasCumprod : torch.Tensor
        ``ᾱ`` buffer from :class:`NoiseSchedule`.
    schedule : str
        One of :data:`SUPPORTED_VELOCITY_SCHEDULES` — reuses the same
        weighting policy as :func:`velocityXyzLossV2` to keep the two
        FK-based losses scheduled consistently.
    motionMask : torch.Tensor or None
        Boolean mask of valid frames, shape ``(B, F)``.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    if schedule not in SUPPORTED_VELOCITY_SCHEDULES:
        raise ValueError(
            f"schedule must be one of {SUPPORTED_VELOCITY_SCHEDULES}; "
            f"got {schedule!r}."
        )
    if predictedRotation6d.shape != targetRotation6d.shape:
        raise ValueError(
            "predicted and target rotation6d shapes must match; got "
            f"{tuple(predictedRotation6d.shape)} vs "
            f"{tuple(targetRotation6d.shape)}."
        )
    predictedXyz = rot6dToJointXYZ(predictedRotation6d)
    targetXyz = rot6dToJointXYZ(targetRotation6d)

    perSample = perSampleMse(
        predictedXyz, targetXyz, motionMask=motionMask
    )
    if schedule == VELOCITY_SCHEDULE_NONE:
        return perSample.mean()

    alphasCumprod = alphasCumprod.to(device=timesteps.device)
    alphaT = alphasCumprod.gather(0, timesteps.long()).to(torch.float32)
    return (perSample * alphaT).mean()


# ---------------------------------------------------------------------
# Auxiliary foot-contact loss (anti-skating, MDM-style)
# ---------------------------------------------------------------------
# SMPL-22 indices of the four contact points used by HumanML3D / MDM
# (matches FOOT_CONTACT_INDICES in
# src/shared/model/components/feature_builder.py).
FOOT_CONTACT_BONE_INDICES: tuple[int, int, int, int] = (7, 10, 8, 11)

# Default squared-speed threshold (m²/frame²).  Matches
# FOOT_CONTACT_THRESHOLD used by the dataset feature builder so the
# contact mask derived at training-loss time aligns with the labels
# produced during preprocessing.
DEFAULT_FOOT_CONTACT_THRESHOLD = 0.002


def footContactLossV2(
    predictedRotation6d: torch.Tensor,
    targetRotation6d: torch.Tensor,
    timesteps: torch.Tensor,
    alphasCumprod: torch.Tensor,
    schedule: str = VELOCITY_SCHEDULE_TIMESTEP,
    motionMask: torch.Tensor | None = None,
    footIndices: tuple[int, ...] = FOOT_CONTACT_BONE_INDICES,
    contactThreshold: float = DEFAULT_FOOT_CONTACT_THRESHOLD,
) -> torch.Tensor:
    """Anti-foot-skating loss derived on-the-fly from target velocities.

    Implements the MDM-style foot-contact loss without requiring a
    learned contact head or any new batch field.  The procedure:

    1. FK the predicted and target rotations to joint XYZ.
    2. Extract the four foot contact points
       ``(leftAnkle, leftFoot, rightAnkle, rightFoot)``.
    3. Build the **target contact mask** from the target foot speeds:
       a foot is "in contact" when its squared per-frame speed is below
       ``contactThreshold`` — same convention as the dataset's
       precomputed ``foot_contact`` labels.
    4. Penalise the predicted foot velocity wherever the target says
       the foot should be still: ``loss = mean(pred_vel² · mask)``.

    The schedule controls how the loss is weighted by timestep:

    * ``"none"``     — uniform weighting across timesteps.
    * ``"timestep"`` — multiply each sample by ``ᾱ_t`` so high-t
      samples contribute ~0 (FK applied to noise is meaningless).

    Parameters
    ----------
    predictedRotation6d, targetRotation6d : torch.Tensor
        Shape ``(B, F, 22, 6)`` in the **raw** (denormalised) space.
    timesteps : torch.Tensor
        Long tensor of shape ``(B,)``.
    alphasCumprod : torch.Tensor
        ``ᾱ`` buffer from :class:`NoiseSchedule`.
    schedule : str
        One of :data:`SUPPORTED_VELOCITY_SCHEDULES`.
    motionMask : torch.Tensor or None
        Boolean mask of valid frames, shape ``(B, F)``.
    footIndices : tuple[int, ...]
        SMPL-22 bone indices treated as foot contact points.  Defaults
        to ``(leftAnkle, leftFoot, rightAnkle, rightFoot) = (7, 10, 8,
        11)`` to match the dataset's foot_contact labels.
    contactThreshold : float
        Squared per-frame foot speed below which a foot is considered
        in contact.  Matches the dataset feature builder default.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    if schedule not in SUPPORTED_VELOCITY_SCHEDULES:
        raise ValueError(
            f"schedule must be one of {SUPPORTED_VELOCITY_SCHEDULES}; "
            f"got {schedule!r}."
        )
    if predictedRotation6d.shape != targetRotation6d.shape:
        raise ValueError(
            "predicted and target rotation6d shapes must match; got "
            f"{tuple(predictedRotation6d.shape)} vs "
            f"{tuple(targetRotation6d.shape)}."
        )
    if predictedRotation6d.shape[1] < 2:
        # Need at least two frames to compute a velocity.
        return predictedRotation6d.new_zeros(())

    predictedXyz = rot6dToJointXYZ(predictedRotation6d)  # (B, F, 22, 3)
    targetXyz = rot6dToJointXYZ(targetRotation6d)

    indexTensor = torch.tensor(
        list(footIndices), device=predictedXyz.device, dtype=torch.long
    )
    predictedFoot = predictedXyz.index_select(2, indexTensor)
    targetFoot = targetXyz.index_select(2, indexTensor)

    # temporalDifference operates on the time axis (dim=1) and returns
    # zero-padded first frame so the temporal length matches the input.
    predictedFootVel = temporalDifference(predictedFoot)  # (B, F, K, 3)
    targetFootVel = temporalDifference(targetFoot)

    targetSpeedSq = (targetFootVel ** 2).sum(dim=-1)  # (B, F, K)
    contactMask = (targetSpeedSq < contactThreshold).to(
        predictedFootVel.dtype
    )

    predictedSpeedSq = (predictedFootVel ** 2).sum(dim=-1)  # (B, F, K)
    skating = predictedSpeedSq * contactMask

    if motionMask is not None:
        frameMask = motionMask.to(skating.dtype).unsqueeze(-1)
        numerator = (skating * frameMask).flatten(1).sum(dim=1)
        # Denominator counts (frame, foot) cells that are simultaneously
        # valid AND in contact — gives the average squared velocity of a
        # foot that ought to be still.  Floor at 1 to avoid div-by-zero
        # on samples where no foot is ever in contact (e.g. a jump).
        denom = (contactMask * frameMask).flatten(1).sum(dim=1)
        perSample = numerator / torch.clamp(denom, min=1.0)
    else:
        numerator = skating.flatten(1).sum(dim=1)
        denom = contactMask.flatten(1).sum(dim=1)
        perSample = numerator / torch.clamp(denom, min=1.0)

    if schedule == VELOCITY_SCHEDULE_NONE:
        return perSample.mean()

    alphasCumprod = alphasCumprod.to(device=timesteps.device)
    alphaT = alphasCumprod.gather(0, timesteps.long()).to(torch.float32)
    return (perSample * alphaT).mean()


# ---------------------------------------------------------------------
# Text–motion alignment loss
# ---------------------------------------------------------------------
def textMotionAlignmentLoss(
    textEmbedding: torch.Tensor,
    motionEmbedding: torch.Tensor,
    motionMask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cosine-similarity loss between matched text/motion embeddings.

    The loss is ``1 − mean(cos_sim(text, motion))``, so 0 = perfectly
    aligned and 2 = anti-aligned.  Both inputs are expected to be 2-D
    pooled embeddings of shape ``(B, D)`` (the upstream caller is
    responsible for masked-mean pooling on the motion side).

    The ``motionMask`` argument is accepted for API symmetry with the
    other v2 losses but is unused here — pooling already collapsed the
    temporal dimension.  Kept so callers can pass the same mask
    everywhere.
    """
    del motionMask  # for symmetry, intentionally unused
    if textEmbedding.shape != motionEmbedding.shape:
        raise ValueError(
            "textEmbedding and motionEmbedding must share the same "
            f"shape; got {tuple(textEmbedding.shape)} vs "
            f"{tuple(motionEmbedding.shape)}."
        )
    if textEmbedding.ndim != 2:
        raise ValueError(
            "alignment loss expects 2-D pooled embeddings (B, D); "
            f"got shape {tuple(textEmbedding.shape)}."
        )
    similarity = F.cosine_similarity(
        textEmbedding, motionEmbedding, dim=-1
    )
    return 1.0 - similarity.mean()


# Default temperature for the InfoNCE contrastive loss.  ``0.1`` is the
# standard value from CLIP / SimCLR — softmax sharpening that makes the
# positive pair stand out clearly from negatives without making
# gradients explode.
DEFAULT_CONTRASTIVE_TEMPERATURE = 0.1


class ContrastiveMemoryBank:
    """FIFO queue of detached (text, motion) embeddings for extra negatives.

    Stored embeddings carry **no gradient** — they only serve as
    additional negatives in the InfoNCE denominator, à la MoCo.  With a
    micro-batch of 8 and ``bankSize=256`` the effective negative count
    jumps from 7 to 263, dramatically improving the contrastive signal.
    """

    def __init__(self, embeddingDim: int, bankSize: int = 256) -> None:
        if bankSize <= 0:
            raise ValueError("bankSize must be > 0.")
        self._bankSize = bankSize
        self._embeddingDim = embeddingDim
        self._textBank = torch.zeros(bankSize, embeddingDim)
        self._motionBank = torch.zeros(bankSize, embeddingDim)
        self._pointer = 0
        self._count = 0

    def to(self, device: torch.device) -> "ContrastiveMemoryBank":
        self._textBank = self._textBank.to(device)
        self._motionBank = self._motionBank.to(device)
        return self

    @torch.no_grad()
    def enqueue(
        self, textEmb: torch.Tensor, motionEmb: torch.Tensor
    ) -> None:
        textNormed = F.normalize(textEmb.detach(), dim=-1)
        motionNormed = F.normalize(motionEmb.detach(), dim=-1)
        batchSize = textNormed.shape[0]
        for i in range(batchSize):
            self._textBank[self._pointer] = textNormed[i]
            self._motionBank[self._pointer] = motionNormed[i]
            self._pointer = (self._pointer + 1) % self._bankSize
            self._count = min(self._count + 1, self._bankSize)

    @torch.no_grad()
    def dequeue(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self._count == 0:
            return None
        valid = self._count
        return (
            self._textBank[:valid].detach(),
            self._motionBank[:valid].detach(),
        )

    @property
    def size(self) -> int:
        return self._count


def textMotionContrastiveLoss(
    textEmbedding: torch.Tensor,
    motionEmbedding: torch.Tensor,
    temperature: float = DEFAULT_CONTRASTIVE_TEMPERATURE,
    negativeTexts: torch.Tensor | None = None,
    negativeMotions: torch.Tensor | None = None,
    sampleWeights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Symmetric InfoNCE loss between text and motion pooled embeddings.

    For a batch of ``B`` samples, each row is its own positive (the
    matched text/motion pair) and the other ``B − 1`` rows are
    negatives.  The loss is the average of two cross-entropy terms —
    text→motion retrieval and motion→text retrieval — making it
    symmetric à la CLIP.

    When ``negativeTexts`` / ``negativeMotions`` are provided (from a
    :class:`ContrastiveMemoryBank`), they are appended as extra
    negatives.  The logits matrix becomes ``(B, B+K)`` where ``K`` is
    the bank size, but the targets still point at the diagonal
    positions ``0..B-1``.

    ``sampleWeights`` (shape ``(B,)``, non-negative) re-weights the
    per-sample cross-entropy of the **positive** rows before averaging.
    The generated-motion (x0) contrastive uses this to emphasise high-
    timestep samples — where the prediction cannot copy the noisy input
    — so the loss actually teaches text conditioning.  ``None`` keeps
    the uniform CLIP behaviour.
    """
    if textEmbedding.shape != motionEmbedding.shape:
        raise ValueError(
            "textEmbedding and motionEmbedding must share the same "
            f"shape; got {tuple(textEmbedding.shape)} vs "
            f"{tuple(motionEmbedding.shape)}."
        )
    if textEmbedding.ndim != 2:
        raise ValueError(
            "contrastive loss expects 2-D pooled embeddings (B, D); "
            f"got shape {tuple(textEmbedding.shape)}."
        )
    if textEmbedding.shape[0] < 2:
        return textEmbedding.new_zeros(())
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0.")

    textNormed = F.normalize(textEmbedding, dim=-1)
    motionNormed = F.normalize(motionEmbedding, dim=-1)

    if negativeMotions is not None:
        motionAll = torch.cat([motionNormed, negativeMotions], dim=0)
    else:
        motionAll = motionNormed

    if negativeTexts is not None:
        textAll = torch.cat([textNormed, negativeTexts], dim=0)
    else:
        textAll = textNormed

    B = textNormed.shape[0]
    targets = torch.arange(B, device=textNormed.device, dtype=torch.long)

    # text→motion: (B, B+K) — each text queries all motions
    logitsT2M = textNormed @ motionAll.t() / temperature
    # motion→text: (B, B+K) — each motion queries all texts
    logitsM2T = motionNormed @ textAll.t() / temperature

    if sampleWeights is None:
        lossT2M = F.cross_entropy(logitsT2M, targets)
        lossM2T = F.cross_entropy(logitsM2T, targets)
        return 0.5 * (lossT2M + lossM2T)

    if sampleWeights.shape != (B,):
        raise ValueError(
            f"sampleWeights must have shape ({B},); got "
            f"{tuple(sampleWeights.shape)}."
        )
    weights = sampleWeights.to(logitsT2M.dtype)
    denom = torch.clamp(weights.sum(), min=1e-8)
    perT2M = F.cross_entropy(logitsT2M, targets, reduction="none")
    perM2T = F.cross_entropy(logitsM2T, targets, reduction="none")
    weightedT2M = (perT2M * weights).sum() / denom
    weightedM2T = (perM2T * weights).sum() / denom
    return 0.5 * (weightedT2M + weightedM2T)


def poolTextEmbedding(
    textHiddenStates: torch.Tensor,
    keyPaddingMask: torch.Tensor | None,
) -> torch.Tensor:
    """Masked-mean pool the text encoder hidden states to ``(B, D)``.

    Sibling helper to :class:`_MotionPoolingHead` in ``denoiser_v2``.
    Lives here in ``losses_v2`` because it is only used by the
    contrastive loss — the text encoder itself returns the full token
    sequence (consumed by the cross-attention) and the diagnostic CLI
    has a private one-off pool already.
    """
    if textHiddenStates.ndim != 3:
        raise ValueError(
            "textHiddenStates must be 3-D (B, T, D); got shape "
            f"{tuple(textHiddenStates.shape)}."
        )
    if keyPaddingMask is None:
        return textHiddenStates.mean(dim=1)
    realMask = (~keyPaddingMask).to(textHiddenStates.dtype).unsqueeze(-1)
    pooled = (textHiddenStates * realMask).sum(dim=1) / torch.clamp(
        realMask.sum(dim=1), min=1.0
    )
    return pooled


# ---------------------------------------------------------------------
# Loss orchestration
# ---------------------------------------------------------------------
def combinedLossV2(
    diffusion: torch.Tensor,
    velocityXyz: torch.Tensor | None,
    textAlignment: torch.Tensor | None,
    diffusionWeight: float = 1.0,
    velocityXyzWeight: float = 0.5,
    textAlignmentWeight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine the v2 losses into a single scalar.

    Returns the combined loss plus a logging dict with per-component
    scalar values (post-weighting).  ``None`` arguments mean the
    corresponding loss is disabled for the current step.
    """
    components: dict[str, float] = {}
    total = diffusionWeight * diffusion
    components["diffusion"] = float(diffusion.detach().item())
    components["diffusion_weighted"] = float(total.detach().item())

    if velocityXyz is not None:
        weighted = velocityXyzWeight * velocityXyz
        total = total + weighted
        components["vel_xyz"] = float(velocityXyz.detach().item())
        components["vel_xyz_weighted"] = float(weighted.detach().item())

    if textAlignment is not None:
        weighted = textAlignmentWeight * textAlignment
        total = total + weighted
        components["text_alignment"] = float(textAlignment.detach().item())
        components["text_alignment_weighted"] = float(weighted.detach().item())

    components["total"] = float(total.detach().item())
    return total, components
