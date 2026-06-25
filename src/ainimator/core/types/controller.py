"""Dataclasses for the Goal A deterministic controller.

Defines the runtime tensors exchanged with :class:`MotionController`
(:class:`ControllerState`, :class:`ControlSignal`, :class:`StylePreset`)
and the frozen architecture config :class:`ControllerV2Config`.

These types freeze the deterministic state vector (ROADMAP_DETERMINIST
§2.2) and the engine flags (§3.2).  They live in ``core/types`` so the
``model`` layer (and everything above it) can import them without
violating the downward-only import contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ainimator.core.constants.controller import (
    PhaseMode,
    ROOT_LOCAL_MOTION_CHANNELS,
    controlSignalChannels,
    phaseConditioningChannels,
)

# Architecture defaults — chosen as the smallest stack that can overfit
# one sequence at the A1 feasibility gate.  They are NOT the production
# capacity; the capacity↔N law (ROADMAP §5.2) is re-derived empirically
# for the controller in later phases.
_DEFAULT_EMBED_DIM = 256
_DEFAULT_NUM_HEADS = 8
_DEFAULT_NUM_LAYERS = 4
_DEFAULT_NUM_BONES = 22
_DEFAULT_MOTION_CHANNELS = 6
#: Root-local motion channels: (Δforward, Δlateral, Δheight, Δyaw).
#: Replaces the 3-channel absolute root_translation of the diffusion stack.
_DEFAULT_GLOBAL_CHANNELS = ROOT_LOCAL_MOTION_CHANNELS  # 4
_DEFAULT_CONTEXT_FRAMES = 1
_DEFAULT_DROPOUT = 0.0
_DEFAULT_FILM_INIT_STD = 0.02
_DEFAULT_STYLE_LATENT_DIM = 64


@dataclass(frozen=True)
class ControllerState:
    """One (or a window of) deterministic state frame(s).

    The controller regresses ``Δstate`` over this lean representation
    only; FK-derivable signals are supervised at the loss, never carried
    here as redundant channels (ROADMAP_DETERMINIST §2.2.a).

    The state is **136 channels** = rotation6d (132) + root-local motion
    (4).  The root-local motion ``(Δforward, Δlateral, Δheight, Δyaw)``
    replaces the absolute ``root_translation`` (3) of the diffusion lean
    representation — see ROADMAP_DETERMINIST §2.2.a (2026-06-24).

    Attributes
    ----------
    rotation6d : torch.Tensor
        Local 6D joint rotations, shape ``(..., numBones, 6)``.
    rootLocalMotion : torch.Tensor
        Root-local motion delta per frame, shape ``(..., 4)``.
        Channels: ``(Δforward, Δlateral, Δheight, Δyaw)``.
    """

    rotation6d: torch.Tensor
    rootLocalMotion: torch.Tensor

    @property
    def numBones(self) -> int:
        """Number of bones carried by :attr:`rotation6d`."""
        return int(self.rotation6d.shape[-2])


@dataclass(frozen=True)
class ControlSignal:
    """The per-frame control that drives the controller.

    Attributes
    ----------
    desiredPlanarVelocity : torch.Tensor
        Desired root velocity on the ground plane (forward, lateral) in
        the root-local frame, shape ``(..., 2)``.
    aimDirection : torch.Tensor or None
        Desired heading as a unit 2-vector ``(cos θ, sin θ)``, shape
        ``(..., 2)``; ``None`` in the A1 minimal control signal, present
        from A2 onward.
    """

    desiredPlanarVelocity: torch.Tensor
    aimDirection: torch.Tensor | None = None


@dataclass(frozen=True)
class StylePreset:
    """A named, lockable style latent (DEFERRED — phase A3).

    Not consumed by A1/A2: the current dataset has no style labels
    (ROADMAP_DETERMINIST §1, §3.3).  Defined here so the state contract
    is frozen once and A3 only has to wire it.

    Attributes
    ----------
    name : str
        Preset identifier (e.g. ``"ninja"``), matching a file in
        ``configs/styles/`` when A3 ships.
    vector : torch.Tensor
        The style latent ``z_style``, shape ``(styleLatentDim,)``.
    locked : bool
        When ``True`` the latent is frozen (no gradient / no resampling).
    """

    name: str
    vector: torch.Tensor
    locked: bool = True


@dataclass(frozen=True)
class ControllerV2Config:
    """Frozen architecture configuration for :class:`MotionController`.

    Mirrors the role of ``MotionDenoiserV2Config`` for the diffusion
    engine.  Conditioning width (control + phase) is derived from the
    feature flags so the model and the data builder agree on a single
    source of truth.

    Attributes
    ----------
    embedDim : int
        Hidden width of the controller transformer.
    numHeads : int
        Attention heads (must divide ``embedDim``).
    numLayers : int
        Stacked controller blocks.
    numBones : int
        Skeleton size (SMPL-22 → 22).
    motionChannels : int
        Channels per bone (rotation6d → 6).
    globalChannels : int
        Global per-frame channels: root-local motion → 4
        ``(Δforward, Δlateral, Δheight, Δyaw)``.
    contextFrames : int
        Number of past frames the controller sees per forward (A1 → 1,
        extended in A4).
    phaseMode : PhaseMode
        Locomotor phase regime (``none`` | ``explicit`` | ``learned``).
    useAimDirection : bool
        Append the aim-direction control channels (A2 rich control).
    useFilmConditioning : bool
        Global FiLM shortcut on the conditioning vector.
    usePerBlockFilm : bool
        Per-block AdaLN modulation (DiT-style).
    filmInitStd : float
        Std of the FiLM/AdaLN projection init.
    dropout : float
        Dropout inside attention and FFN sublayers.
    maxFrames : int
        Frame cap used to size positional encodings.
    styleLatentEnabled : bool
        DEFERRED (A3) — inject a style latent.  Must stay ``False``
        until a style-labelled dataset exists.
    styleLatentDim : int
        Width of ``z_style`` when style latents are enabled.
    """

    embedDim: int = _DEFAULT_EMBED_DIM
    numHeads: int = _DEFAULT_NUM_HEADS
    numLayers: int = _DEFAULT_NUM_LAYERS
    numBones: int = _DEFAULT_NUM_BONES
    motionChannels: int = _DEFAULT_MOTION_CHANNELS
    globalChannels: int = _DEFAULT_GLOBAL_CHANNELS
    contextFrames: int = _DEFAULT_CONTEXT_FRAMES
    phaseMode: PhaseMode = PhaseMode.EXPLICIT
    useAimDirection: bool = False
    useFilmConditioning: bool = True
    usePerBlockFilm: bool = True
    filmInitStd: float = _DEFAULT_FILM_INIT_STD
    dropout: float = _DEFAULT_DROPOUT
    maxFrames: int = 4096
    styleLatentEnabled: bool = False
    styleLatentDim: int = _DEFAULT_STYLE_LATENT_DIM

    def __post_init__(self) -> None:
        if self.embedDim % self.numHeads != 0:
            raise ValueError(
                f"embedDim ({self.embedDim}) must be divisible by "
                f"numHeads ({self.numHeads})."
            )
        if self.numLayers < 1:
            raise ValueError("numLayers must be >= 1.")
        if self.numBones < 1:
            raise ValueError("numBones must be >= 1.")
        if self.motionChannels < 1:
            raise ValueError("motionChannels must be >= 1.")
        if self.globalChannels < 0:
            raise ValueError("globalChannels must be >= 0.")
        if self.contextFrames < 1:
            raise ValueError("contextFrames must be >= 1.")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")
        if self.styleLatentEnabled and self.styleLatentDim < 1:
            raise ValueError(
                "styleLatentDim must be >= 1 when styleLatentEnabled."
            )

    @property
    def controlChannels(self) -> int:
        """Width of the control signal (planar velocity [+ aim])."""
        return controlSignalChannels(self.useAimDirection)

    @property
    def phaseChannels(self) -> int:
        """Width of the phase conditioning (0 when phase is ``none``)."""
        return phaseConditioningChannels(self.phaseMode)

    @property
    def styleChannels(self) -> int:
        """Width of the style latent contribution (0 unless enabled)."""
        return self.styleLatentDim if self.styleLatentEnabled else 0

    @property
    def conditioningChannels(self) -> int:
        """Total conditioning width fed to the controller per frame."""
        return self.controlChannels + self.phaseChannels + self.styleChannels

    @property
    def boneOutputDim(self) -> int:
        """Predicted Δstate channels for the bone branch."""
        return self.numBones * self.motionChannels

    @property
    def totalOutputDim(self) -> int:
        """Total per-frame Δstate width (bone + global)."""
        return self.boneOutputDim + self.globalChannels
