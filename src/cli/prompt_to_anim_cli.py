"""Command line interface for training and sampling."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from src.features.inferance import Prompt2AnimDiffusionSampler
from src.features.training import Prompt2AnimDiffusionTrainer
from src.shared.constants.training import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CACHE_ON_DEVICE,
    DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_CONTEXT_HISTORY,
    DEFAULT_CONTEXT_JSON_LIST,
    DEFAULT_CONTEXT_TRAIN_MODE,
    DEFAULT_CONTEXT_TRAIN_RATIO,
    DEFAULT_DEVICE_BACKEND,
    DEFAULT_EPOCH_COUNT,
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_LAYER_COUNT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_VALIDATION_SAMPLES,
    DEFAULT_MODEL_DIMENSION,
    DEFAULT_MOE_EXPERTS,
    DEFAULT_MOE_TOPK,
    DEFAULT_QAT_READY,
    DEFAULT_RANDOM_SEED,
    DEFAULT_RECACHE_EVERY_EPOCH,
    DEFAULT_SAMPLING_FRAME_COUNT,
    DEFAULT_SAMPLING_GUIDANCE,
    DEFAULT_SAMPLING_STEPS,
    DEFAULT_SAMPLING_TEXT_MODEL,
    DEFAULT_SAVE_DIRECTORY,
    DEFAULT_SEQUENCE_FRAMES,
    DEFAULT_SUCCESS_DEGREES,
    DEFAULT_TARGET_SUCCESS_RATE,
    DEFAULT_TEXT_MODEL_NAME,
    DEFAULT_VALIDATION_INTERVAL,
    DEFAULT_VALIDATION_SPLIT,
)
from src.shared.types import (
    DeviceSelectionOptions,
    SamplingConfiguration,
    TrainingConfiguration,
)


def buildArgumentParser() -> argparse.ArgumentParser:
    """Construct the root parser for training and sampling commands.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser exposing ``train`` and ``sample`` sub-commands.
    """

    parser = argparse.ArgumentParser(
        description="Prompt-to-animation diffusion training and inference",
    )
    subParsers = parser.add_subparsers(dest="command", required=True)

    trainParser = subParsers.add_parser("train", help="Entraîner le modèle")
    trainParser.add_argument("--data-dir", type=Path, required=True)
    trainParser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIRECTORY)
    trainParser.add_argument("--exp-name", type=str, default=DEFAULT_EXPERIMENT_NAME)
    trainParser.add_argument("--epochs", type=int, default=DEFAULT_EPOCH_COUNT)
    trainParser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    trainParser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    trainParser.add_argument("--seq-frames", type=int, default=DEFAULT_SEQUENCE_FRAMES)
    trainParser.add_argument("--context-last", type=int, default=DEFAULT_CONTEXT_HISTORY)
    trainParser.add_argument(
        "--context-train-mode",
        type=str,
        default=DEFAULT_CONTEXT_TRAIN_MODE,
        choices=["off", "alt", "ratio"],
        help="off: jamais; alt: une époque sur deux; ratio: probabilité",
    )
    trainParser.add_argument(
        "--context-train-ratio",
        type=float,
        default=DEFAULT_CONTEXT_TRAIN_RATIO,
        help="Si mode=ratio, probabilité d'utiliser le contexte",
    )
    trainParser.add_argument("--d-model", type=int, default=DEFAULT_MODEL_DIMENSION)
    trainParser.add_argument("--layers", type=int, default=DEFAULT_LAYER_COUNT)
    trainParser.add_argument("--moe-experts", type=int, default=DEFAULT_MOE_EXPERTS)
    trainParser.add_argument("--moe-topk", type=int, default=DEFAULT_MOE_TOPK)
    trainParser.add_argument("--ckpt-every", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    trainParser.add_argument("--resume", type=Path, default=None)
    trainParser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    trainParser.add_argument(
        "--text-model",
        type=str,
        default=DEFAULT_TEXT_MODEL_NAME,
        help="Nom du modèle HuggingFace pour les embeddings",
    )
    trainParser.add_argument("--qat-ready", action="store_true", default=DEFAULT_QAT_READY)
    trainParser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    trainParser.add_argument("--val-split", type=float, default=DEFAULT_VALIDATION_SPLIT)
    trainParser.add_argument("--val-every", type=int, default=DEFAULT_VALIDATION_INTERVAL)
    trainParser.add_argument("--success-deg", type=float, default=DEFAULT_SUCCESS_DEGREES)
    trainParser.add_argument("--target-success", type=float, default=DEFAULT_TARGET_SUCCESS_RATE)
    trainParser.add_argument(
        "--cache-on-device",
        action="store_true",
        default=DEFAULT_CACHE_ON_DEVICE,
    )
    trainParser.add_argument(
        "--max-val-samples",
        type=int,
        default=DEFAULT_MAX_VALIDATION_SAMPLES,
    )
    trainParser.add_argument(
        "--recache-every-epoch",
        action="store_true",
        default=DEFAULT_RECACHE_EVERY_EPOCH,
    )
    trainParser.add_argument("--device", type=str, default=DEFAULT_DEVICE_BACKEND)
    trainParser.add_argument("--no-cuda", action="store_true")
    trainParser.add_argument("--no-dml", action="store_true")
    trainParser.add_argument("--no-mps", action="store_true")
    trainParser.add_argument("--strict-device", action="store_true")

    sampleParser = subParsers.add_parser("sample", help="Générer une animation")
    sampleParser.add_argument("--ckpt", type=Path, required=True)
    sampleParser.add_argument("--prompts", type=Path, required=True)
    sampleParser.add_argument("--out-json", type=Path, required=True)
    sampleParser.add_argument("--frames", type=int, default=DEFAULT_SAMPLING_FRAME_COUNT)
    sampleParser.add_argument("--steps", type=int, default=DEFAULT_SAMPLING_STEPS)
    sampleParser.add_argument(
        "--guidance",
        type=float,
        default=DEFAULT_SAMPLING_GUIDANCE,
    )
    sampleParser.add_argument(
        "--context-jsons",
        type=str,
        default=DEFAULT_CONTEXT_JSON_LIST,
        help=(
            "Fichiers animation.json permettant de fournir du contexte "
            "(séparés par des virgules)"
        ),
    )
    sampleParser.add_argument(
        "--bones",
        nargs="*",
        default=[],
        help="Liste ordonnée des bones lorsque aucun contexte n'est fourni",
    )
    sampleParser.add_argument("--omit-meta", action="store_true")
    sampleParser.add_argument("--text-model", type=str, default=DEFAULT_SAMPLING_TEXT_MODEL)
    sampleParser.add_argument("--device", type=str, default=DEFAULT_DEVICE_BACKEND)
    sampleParser.add_argument("--no-cuda", action="store_true")
    sampleParser.add_argument("--no-dml", action="store_true")
    sampleParser.add_argument("--no-mps", action="store_true")
    sampleParser.add_argument("--strict-device", action="store_true")
    return parser


def parseDeviceOptions(args: argparse.Namespace) -> DeviceSelectionOptions:
    """Translate CLI flags to :class:`DeviceSelectionOptions`."""

    return DeviceSelectionOptions(
        requestedBackend=args.device,
        allowCuda=not args.no_cuda,
        allowDirectML=not args.no_dml,
        allowMps=not args.no_mps,
        requireGpu=args.strict_device,
    )


def buildTrainingConfiguration(args: argparse.Namespace) -> TrainingConfiguration:
    """Create :class:`TrainingConfiguration` from CLI arguments."""

    deviceOptions = parseDeviceOptions(args)
    resumePath = args.resume if args.resume else None
    return TrainingConfiguration(
        dataDirectory=args.data_dir,
        saveDirectory=args.save_dir,
        experimentName=args.exp_name,
        epochs=args.epochs,
        batchSize=args.batch_size,
        learningRate=args.lr,
        sequenceFrames=args.seq_frames,
        contextHistory=args.context_last,
        contextTrainMode=args.context_train_mode,
        contextTrainRatio=args.context_train_ratio,
        modelDimension=args.d_model,
        layerCount=args.layers,
        expertCount=args.moe_experts,
        expertTopK=args.moe_topk,
        checkpointInterval=args.ckpt_every,
        resumePath=resumePath,
        maximumTokens=args.max_tokens,
        textModelName=args.text_model,
        prepareQat=args.qat_ready,
        randomSeed=args.seed,
        validationSplit=args.val_split,
        validationInterval=args.val_every,
        successThresholdDegrees=args.success_deg,
        targetSuccessRate=args.target_success,
        cacheOnDevice=args.cache_on_device,
        maximumValidationSamples=args.max_val_samples,
        recacheEveryEpoch=args.recache_every_epoch,
        deviceOptions=deviceOptions,
    )


def buildSamplingConfiguration(args: argparse.Namespace) -> SamplingConfiguration:
    """Materialise :class:`SamplingConfiguration` from CLI arguments."""

    deviceOptions = parseDeviceOptions(args)
    contextJsons: List[Path] = [
        Path(item)
        for item in args.context_jsons.split(",")
        if item.strip()
    ]
    return SamplingConfiguration(
        checkpointPath=args.ckpt,
        promptsPath=args.prompts,
        outputPath=args.out_json,
        frameCount=args.frames,
        steps=args.steps,
        guidance=args.guidance,
        contextJsons=contextJsons,
        boneNames=list(args.bones),
        omitMetadata=args.omit_meta,
        textModelName=args.text_model,
        deviceOptions=deviceOptions,
    )


def main() -> None:
    """Entry point executed by the prompt-to-animation CLI."""

    parser = buildArgumentParser()
    args = parser.parse_args()
    if args.command == "train":
        configuration = buildTrainingConfiguration(args)
        trainer = Prompt2AnimDiffusionTrainer(configuration)
        trainer.runTraining()
    elif args.command == "sample":
        configuration = buildSamplingConfiguration(args)
        sampler = Prompt2AnimDiffusionSampler(configuration)
        sampler.runSampling()
    else:
        raise ValueError(f"Commande inconnue: {args.command}")


if __name__ == "__main__":
    main()
