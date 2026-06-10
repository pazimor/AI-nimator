"""CLI entry-point for the v2 training loop.

Two profiles are exposed via ``--profile``:

* ``overfit`` (default) — trains on a single sample picked by
  ``--sample-link-index``.  Used for the architectural sanity check.
* ``full`` — multi-sample production training over the dataset
  filtered by ``--include-folders``.  Reads
  :func:`runFullTraining` from ``full_training_v2``.

Examples
--------
.. code-block:: bash

    # Overfit on a single sample
    poetry run python -m src.cli.train_generation_v2 \\
        --profile overfit \\
        --dataset-root /Users/pazimor/dataset_preprocessed \\
        --tokenizer-dir output/text/custom_tokenizer \\
        --output-dir output/generation_v2 \\
        --sample-link-index 0 \\
        --epochs 200

    # Full training on all 5 v2 folders
    poetry run python -m src.cli.train_generation_v2 \\
        --profile full \\
        --dataset-root /Users/pazimor/dataset_preprocessed \\
        --tokenizer-dir output/text/custom_tokenizer \\
        --output-dir output/generation_v2 \\
        --include-folders ACCAD,BioMotionLab_NTroje,CMU,BMLmovi,KIT \\
        --epochs 200 \\
        --batch-size 8 \\
        --gradient-accumulation 2 \\
        --cond-mask-prob 0.10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from src.features.generation.full_training_v2 import (
    V2FullTrainingConfig,
    runFullTraining,
)
from src.features.generation.training_v2 import (
    DEFAULT_MIN_SNR_GAMMA,
    V2TrainingConfig,
    runOverfit,
)

LOGGER = logging.getLogger(__name__)


def buildArgumentParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the AI-nimator v2 training loop (overfit or full)."
        ),
    )
    parser.add_argument(
        "--profile",
        dest="profile",
        type=str,
        default="overfit",
        choices=["overfit", "full"],
        help=(
            "Training profile.  ``overfit`` trains on a single sample "
            "(sanity check); ``full`` trains on the whole filtered "
            "dataset. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--sample-link-indices",
        dest="sampleLinkIndices",
        type=str,
        default=None,
        help=(
            "[full only — 2026-06-05] Comma-separated link indices to "
            "train on EXACTLY (train==val, no folder split).  Small-N "
            "controlled mode to verify the conditioning distinguishes a "
            "few distinct prompts before scaling up.  Must lie within "
            "--include-folders.  Example: '12,57'."
        ),
    )
    parser.add_argument(
        "--include-folders",
        dest="includeFolders",
        type=str,
        default=None,
        help=(
            "[full only] Comma-separated AMASS folders to keep, e.g. "
            "'ACCAD,CMU,KIT'.  Default: all folders in the manifest."
        ),
    )
    parser.add_argument(
        "--batch-size",
        dest="batchSize",
        type=int,
        default=8,
        help="[full only] Batch size. (default: %(default)s)",
    )
    parser.add_argument(
        "--gradient-accumulation",
        dest="gradientAccumulation",
        type=int,
        default=1,
        help=(
            "[full only] Number of micro-batches to accumulate before "
            "stepping the optimiser. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--validation-fraction",
        dest="validationFraction",
        type=float,
        default=0.10,
        help=(
            "[full only] Fraction of the filtered links reserved for "
            "validation. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--max-samples-per-epoch",
        dest="maxSamplesPerEpoch",
        type=int,
        default=5000,
        help=(
            "[full only] Cap on the random sub-sample drawn each "
            "epoch (0 = no cap). (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--validate-every-epochs",
        dest="validateEveryEpochs",
        type=int,
        default=1,
        help=(
            "[full only] Run a validation pass every N epochs. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--normalizer-fit-max-samples",
        dest="normalizerFitMaxSamples",
        type=int,
        default=2000,
        help=(
            "[full only] Cap on the number of training samples used to "
            "fit the per-channel mean/std. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--resume-checkpoint",
        dest="resumeCheckpoint",
        type=Path,
        default=None,
        help=(
            "[full only] Optional path to a checkpoint to resume from. "
            "Loads encoder + denoiser + normalizer + optimiser state."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        dest="datasetRoot",
        type=Path,
        required=True,
        help="Root of the V2 preprocessed dataset.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        dest="tokenizerDir",
        type=Path,
        required=True,
        help="Directory holding tokenizer.json + config.json.",
    )
    parser.add_argument(
        "--output-dir",
        dest="outputDir",
        type=Path,
        default=Path("output/generation_v2"),
        help="Where to write checkpoints. (default: %(default)s)",
    )
    parser.add_argument(
        "--sample-link-index",
        dest="sampleLinkIndex",
        type=int,
        default=0,
        help=(
            "Index into link_index.json — picks which (motion, text) "
            "pair is used for the overfit run. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of overfit epochs. (default: %(default)s)",
    )
    parser.add_argument(
        "--learning-rate",
        dest="learningRate",
        type=float,
        default=1e-4,
        help="AdamW learning rate. (default: %(default)s)",
    )
    parser.add_argument(
        "--weight-decay",
        dest="weightDecay",
        type=float,
        default=0.0,
        help="AdamW weight decay. (default: %(default)s)",
    )
    parser.add_argument(
        "--min-snr-gamma",
        dest="minSnrGamma",
        type=float,
        default=DEFAULT_MIN_SNR_GAMMA,
        help=(
            "Min-SNR-γ clipping threshold (0 disables). "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--vel-xyz-weight",
        dest="velocityXyzWeight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary FK-velocity loss weight (0 disables). "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--joint-position-weight",
        dest="jointPositionWeight",
        type=float,
        default=0.0,
        help=(
            "MDM-style FK joint-position loss weight (0 disables). "
            "Recommended starting value: 1.0.  This is the canonical "
            "fix for 'low rotation MSE but bad-looking animation'. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--foot-contact-weight",
        dest="footContactWeight",
        type=float,
        default=0.0,
        help=(
            "MDM-style anti-foot-skating loss weight (0 disables). "
            "Derives the contact mask on-the-fly from target foot "
            "velocities and penalises the predicted foot velocity "
            "wherever the target says the foot ought to be still. "
            "Recommended starting value: 0.5. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--best-metric",
        dest="bestMetric",
        type=str,
        default="loss_total",
        choices=[
            "loss_total",
            "loss_diffusion",
            "loss_bone",
            "loss_global",
            "loss_vel_xyz",
            "loss_joint_xyz",
            "loss_foot_contact",
            "loss_clip_guidance",
        ],
        help=(
            "[full only] Validation key driving the best-checkpoint "
            "selection.  ``loss_diffusion`` (= bone + global) is "
            "useful when the contrastive head is stuck at random "
            "chance and you want best-by-diffusion.  "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--best-improvement-min",
        dest="bestImprovementMin",
        type=float,
        default=0.0,
        help=(
            "[full only] Minimum absolute drop on --best-metric that "
            "qualifies as a real improvement.  ``0.0`` reproduces "
            "pre-this-change behaviour (every tiny drop fires a new "
            "best).  ``1e-3`` to ``1e-2`` is sane for noisy diffusion "
            "loss curves. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--stagnation-patience",
        dest="stagnationPatience",
        type=int,
        default=0,
        help=(
            "[full only] Log a stagnation warning when the chosen "
            "best-metric has not improved by --best-improvement-min "
            "for this many consecutive validations.  ``0`` disables. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--use-film-conditioning",
        dest="useFilmConditioning",
        action="store_true",
        help=(
            "[full only — Phase D Levier D] Add a FiLM modulation "
            "shortcut driven by (text_pooled, timestep_embed) at the "
            "input of the denoiser.  Forces a multiplicative text "
            "path the cross-attention cannot zero out — recommended "
            "when post-Levier-B diagnostics show ``cfg_sim ≈ 1.0``."
        ),
    )
    parser.set_defaults(useFilmConditioning=False)
    parser.add_argument(
        "--film-dropout",
        dest="filmDropout",
        type=float,
        default=0.0,
        help=(
            "Dropout inside the FiLM MLP. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--use-per-block-film",
        dest="usePerBlockFilm",
        action="store_true",
        help=(
            "[full only — Phase E Levier E] Add a DiT-style per-block "
            "AdaLN modulation in every DenoiserBlockV2 sub-layer "
            "(self-attn, cross-attn, FFN).  Recommended when ``cfg_sim "
            "≈ 1.0`` even after Levier D (global FiLM) and a multi-"
            "folder dataset.  Stable-Diffusion-3 grade conditioning."
        ),
    )
    parser.set_defaults(usePerBlockFilm=False)
    parser.add_argument(
        "--ema-decay",
        dest="emaDecay",
        type=float,
        default=0.0,
        help=(
            "[full only — Phase D.4] EMA decay on encoder + denoiser "
            "weights.  ``0.0`` disables.  ``0.9999`` is the canonical "
            "diffusion value.  When active, validation runs on the "
            "EMA shadow and the best checkpoint stores the smoothed "
            "weights as canonical state. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--ema-no-warmup",
        dest="emaUseWarmup",
        action="store_false",
        help=(
            "Disable the EMA decay warmup (effective decay = "
            "min(decay, (1+step)/(10+step))).  Off by default; only "
            "use this on very long training runs."
        ),
    )
    parser.set_defaults(emaUseWarmup=True)
    parser.add_argument(
        "--mirror-prob",
        dest="mirrorProb",
        type=float,
        default=0.0,
        help=(
            "[full only — Phase D.3] SMPL-22 left↔right mirror "
            "augmentation probability per sample.  ``0.5`` doubles "
            "the effective dataset size; prompts with explicit "
            "left/right semantics are auto-skipped to keep the "
            "cross-attention text↔motion mapping intact. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--clip-guidance-weight",
        dest="clipGuidanceWeight",
        type=float,
        default=0.0,
        help=(
            "[full only — Phase D.1] Text↔motion InfoNCE contrastive "
            "loss weight.  Set to 0.5 to recover from posterior "
            "collapse (cross-attention dead at inference).  Adds a "
            "small alignment head (~150k params) that builds gradient "
            "signal forcing the cross-attention to actually use the "
            "prompt.  (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--contrastive-temperature",
        dest="contrastiveTemperature",
        type=float,
        default=0.1,
        help=(
            "Softmax temperature for the InfoNCE alignment loss. "
            "Lower ⇒ sharper distinction.  CLIP default is 0.1. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--contrastive-bank-size",
        dest="contrastiveBankSize",
        type=int,
        default=256,
        help=(
            "FIFO memory bank size for extra contrastive negatives. "
            "0 disables. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--clip-guidance-weight-start",
        dest="clipGuidanceWeightStart",
        type=float,
        default=1.0,
        help=(
            "Initial contrastive weight for linear warmup. "
            "Decays to --clip-guidance-weight over "
            "--clip-guidance-warmup-epochs. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--clip-guidance-warmup-epochs",
        dest="clipGuidanceWarmupEpochs",
        type=int,
        default=15,
        help=(
            "Number of epochs to linearly anneal the contrastive "
            "weight from --clip-guidance-weight-start to "
            "--clip-guidance-weight. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--encoder-lr-multiplier",
        dest="encoderLrMultiplier",
        type=float,
        default=3.0,
        help=(
            "Text encoder learning rate = base LR * this multiplier. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--film-init-std",
        dest="filmInitStd",
        type=float,
        default=0.1,
        help=(
            "Std of the FiLM / AdaLN projection init. Higher = "
            "stronger initial conditioning signal. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--aux-pool-contrastive-weight",
        dest="auxPoolContrastiveWeight",
        type=float,
        default=0.0,
        help=(
            "Weight of the auxiliary InfoNCE loss on the raw encoder "
            "masked-mean pool (no MLP on text side). Forces the encoder "
            "pool itself to discriminate prompts so FiLM/AdaLN see a "
            "non-collapsed conditioning vector. 0.0 disables; 0.5 is the "
            "recommended starting value when cfg_sim is stuck at 1.0. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--x0-contrastive-weight",
        dest="x0ContrastiveWeight",
        type=float,
        default=0.0,
        help=(
            "Weight of the generated-motion (x0) InfoNCE contrastive. "
            "Embeds the reconstructed x0 through a dedicated TMR-style "
            "encoder (never sees the noisy input) and weights the loss "
            "toward high timesteps, forcing the *prediction* to depend "
            "on the prompt. Closes the generation loophole where the "
            "model ignores text (cross-prompt sim ~0.98). 0.0 disables; "
            "0.5 is the recommended starting value. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--use-self-conditioning",
        dest="useSelfConditioning",
        action="store_true",
        help=(
            "[full only — 2026-06-02] Enable self-conditioning (Analog "
            "Bits): the denoiser receives its own previous x0 estimate "
            "as an extra input.  Narrows the train/sampling exposure gap "
            "that let the model denoise from x_t without using the "
            "prompt.  Threaded through the sampler automatically."
        ),
    )
    parser.set_defaults(useSelfConditioning=False)
    parser.add_argument(
        "--self-conditioning-prob",
        dest="selfConditioningProb",
        type=float,
        default=0.5,
        help=(
            "Probability of running the extra no-grad pass that produces "
            "the self-conditioning estimate at training. (default: "
            "%(default)s)"
        ),
    )
    parser.add_argument(
        "--diffusion-steps",
        dest="diffusionStepsTraining",
        type=int,
        default=1000,
        help=(
            "Number of timesteps in the training noise schedule. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--schedule-type",
        dest="scheduleType",
        type=str,
        default="cosine",
        choices=["linear", "cosine"],
        help="β schedule shape. (default: %(default)s)",
    )
    parser.add_argument(
        "--prediction-mode",
        dest="predictionMode",
        type=str,
        default="v",
        choices=["v", "x0", "epsilon"],
        help="Diffusion prediction target. (default: %(default)s)",
    )
    parser.add_argument(
        "--text-encoder-type",
        dest="textEncoderType",
        type=str,
        default="custom",
        choices=["custom", "clip"],
        help=(
            "Text-conditioning encoder. ``custom`` is the BPE "
            "transformer trained from scratch; ``clip`` swaps in a "
            "frozen CLIP text tower (only a small projection + null "
            "embedding are trained). (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--clip-model-name",
        dest="clipModelName",
        type=str,
        default="openai/clip-vit-base-patch32",
        help=(
            "HuggingFace model id for the frozen CLIP text tower "
            "(only used when --text-encoder-type=clip). "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--clip-max-length",
        dest="clipMaxLength",
        type=int,
        default=32,
        help=(
            "Max token length for the CLIP tokenizer (1-77, only "
            "used when --text-encoder-type=clip). (default: "
            "%(default)s)"
        ),
    )
    parser.add_argument(
        "--encoder-hidden-dim",
        dest="encoderHiddenDim",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--encoder-num-layers",
        dest="encoderNumLayers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--encoder-num-heads",
        dest="encoderNumHeads",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--denoiser-embed-dim",
        dest="denoiserEmbedDim",
        type=int,
        default=384,
    )
    parser.add_argument(
        "--denoiser-num-layers",
        dest="denoiserNumLayers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--denoiser-num-heads",
        dest="denoiserNumHeads",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-frames",
        dest="maxFrames",
        type=int,
        default=256,
        help=(
            "Frame cap for the denoiser (longer samples are truncated). "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--frames-per-step",
        dest="framesPerStep",
        type=int,
        default=0,
        help=(
            "Window size used per training step (0 = full sample). "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help=(
            "Dropout in encoder + denoiser.  Keep at 0 for overfit "
            "(MUST memorise the sample exactly); 0.1 for full training. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--cond-mask-prob",
        dest="condMaskProb",
        type=float,
        default=0.0,
        help=(
            "Probability of replacing the prompt with the empty string "
            "during training (CFG dropout).  Keep at 0 for overfit; "
            "0.10–0.15 in full training so inference-time CFG works. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed. (default: %(default)s)",
    )
    parser.add_argument(
        "--log-every",
        dest="logEvery",
        type=int,
        default=10,
        help="Print loss every N epochs. (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "Torch device — 'auto' (mps→cuda→cpu), 'cpu', 'mps', 'cuda'. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--log-level",
        dest="logLevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def _argumentsToOverfitConfig(
    arguments: argparse.Namespace,
) -> V2TrainingConfig:
    return V2TrainingConfig(
        datasetRoot=arguments.datasetRoot,
        tokenizerDir=arguments.tokenizerDir,
        outputDir=arguments.outputDir,
        sampleLinkIndex=int(arguments.sampleLinkIndex),
        epochs=int(arguments.epochs),
        learningRate=float(arguments.learningRate),
        weightDecay=float(arguments.weightDecay),
        minSnrGamma=float(arguments.minSnrGamma),
        velocityXyzWeight=float(arguments.velocityXyzWeight),
        diffusionStepsTraining=int(arguments.diffusionStepsTraining),
        scheduleType=str(arguments.scheduleType),
        predictionMode=str(arguments.predictionMode),
        encoderHiddenDim=int(arguments.encoderHiddenDim),
        encoderNumLayers=int(arguments.encoderNumLayers),
        encoderNumHeads=int(arguments.encoderNumHeads),
        denoiserEmbedDim=int(arguments.denoiserEmbedDim),
        denoiserNumLayers=int(arguments.denoiserNumLayers),
        denoiserNumHeads=int(arguments.denoiserNumHeads),
        maxFrames=int(arguments.maxFrames),
        framesPerStep=int(arguments.framesPerStep),
        seed=int(arguments.seed),
        logEvery=int(arguments.logEvery),
        device=str(arguments.device),
        dropout=float(arguments.dropout),
        condMaskProb=float(arguments.condMaskProb),
    )


def _parseIncludeFolders(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    folders = tuple(
        folder.strip() for folder in raw.split(",") if folder.strip()
    )
    return folders or None


def _argumentsToFullConfig(
    arguments: argparse.Namespace,
) -> V2FullTrainingConfig:
    return V2FullTrainingConfig(
        datasetRoot=arguments.datasetRoot,
        tokenizerDir=arguments.tokenizerDir,
        outputDir=arguments.outputDir,
        datasetFolders=_parseIncludeFolders(arguments.includeFolders),
        sampleLinkIndices=(
            tuple(
                int(x) for x in arguments.sampleLinkIndices.split(",") if x.strip()
            )
            if arguments.sampleLinkIndices
            else None
        ),
        epochs=int(arguments.epochs),
        batchSize=int(arguments.batchSize),
        gradientAccumulation=int(arguments.gradientAccumulation),
        learningRate=float(arguments.learningRate),
        weightDecay=float(arguments.weightDecay),
        minSnrGamma=float(arguments.minSnrGamma),
        velocityXyzWeight=float(arguments.velocityXyzWeight),
        jointPositionWeight=float(arguments.jointPositionWeight),
        footContactWeight=float(arguments.footContactWeight),
        clipGuidanceWeight=float(arguments.clipGuidanceWeight),
        contrastiveTemperature=float(arguments.contrastiveTemperature),
        contrastiveBankSize=int(arguments.contrastiveBankSize),
        clipGuidanceWeightStart=float(arguments.clipGuidanceWeightStart),
        clipGuidanceWarmupEpochs=int(arguments.clipGuidanceWarmupEpochs),
        encoderLrMultiplier=float(arguments.encoderLrMultiplier),
        filmInitStd=float(arguments.filmInitStd),
        auxPoolContrastiveWeight=float(arguments.auxPoolContrastiveWeight),
        x0ContrastiveWeight=float(arguments.x0ContrastiveWeight),
        useSelfConditioning=bool(arguments.useSelfConditioning),
        selfConditioningProb=float(arguments.selfConditioningProb),
        mirrorProb=float(arguments.mirrorProb),
        emaDecay=float(arguments.emaDecay),
        emaUseWarmup=bool(arguments.emaUseWarmup),
        useFilmConditioning=bool(arguments.useFilmConditioning),
        filmDropout=float(arguments.filmDropout),
        usePerBlockFilm=bool(arguments.usePerBlockFilm),
        bestMetric=str(arguments.bestMetric),
        bestImprovementMin=float(arguments.bestImprovementMin),
        stagnationPatience=int(arguments.stagnationPatience),
        diffusionStepsTraining=int(arguments.diffusionStepsTraining),
        scheduleType=str(arguments.scheduleType),
        predictionMode=str(arguments.predictionMode),
        textEncoderType=str(arguments.textEncoderType),
        clipModelName=str(arguments.clipModelName),
        clipMaxLength=int(arguments.clipMaxLength),
        encoderHiddenDim=int(arguments.encoderHiddenDim),
        encoderNumLayers=int(arguments.encoderNumLayers),
        encoderNumHeads=int(arguments.encoderNumHeads),
        denoiserEmbedDim=int(arguments.denoiserEmbedDim),
        denoiserNumLayers=int(arguments.denoiserNumLayers),
        denoiserNumHeads=int(arguments.denoiserNumHeads),
        maxFrames=int(arguments.maxFrames),
        dropout=float(arguments.dropout),
        condMaskProb=float(arguments.condMaskProb),
        validationFraction=float(arguments.validationFraction),
        validateEveryEpochs=int(arguments.validateEveryEpochs),
        normalizerFitMaxSamples=int(arguments.normalizerFitMaxSamples),
        maxSamplesPerEpoch=int(arguments.maxSamplesPerEpoch),
        logEvery=int(arguments.logEvery),
        seed=int(arguments.seed),
        device=str(arguments.device),
        resumeCheckpoint=arguments.resumeCheckpoint,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = buildArgumentParser()
    arguments = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, arguments.logLevel),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    profile = str(arguments.profile)
    try:
        if profile == "overfit":
            config = _argumentsToOverfitConfig(arguments)
            components, history = runOverfit(config)
            finalLoss = (
                history[-1]["loss_total"] if history else float("nan")
            )
            LOGGER.info("Done. Final loss: %.4f", finalLoss)
        elif profile == "full":
            fullConfig = _argumentsToFullConfig(arguments)
            components, history = runFullTraining(fullConfig)
            if history:
                lastEntry = history[-1]
                finalVal = lastEntry.get("loss_total", float("nan"))
                LOGGER.info(
                    "Done. Last validation loss: %.4f (epoch %.0f).",
                    finalVal,
                    lastEntry.get("epoch", 0.0),
                )
            else:
                LOGGER.info("Done. (No validation samples in history.)")
        else:  # pragma: no cover — argparse choices guard this
            raise ValueError(f"Unsupported profile {profile!r}.")
    except (
        FileNotFoundError,
        ValueError,
        IndexError,
        KeyError,
        RuntimeError,
    ) as error:
        LOGGER.error("Training failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
