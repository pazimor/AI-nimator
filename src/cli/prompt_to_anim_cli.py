"""Command line interface for training and sampling."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from src.features.inferance import Prompt2AnimDiffusionSampler
from src.features.training import Prompt2AnimDiffusionTrainer
from src.shared.types import (
    DeviceSelectionOptions,
    SamplingConfiguration,
    TrainingConfiguration,
)


def buildArgumentParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prompt-to-animation diffusion training and inference",
    )
    subParsers = parser.add_subparsers(dest="command", required=True)

    trainParser = subParsers.add_parser("train", help="Entraîner le modèle")
    trainParser.add_argument("--data-dir", type=Path, required=True)
    trainParser.add_argument("--save-dir", type=Path, default=Path("./ckpts"))
    trainParser.add_argument("--exp-name", type=str, default="exp")
    trainParser.add_argument("--epochs", type=int, default=5)
    trainParser.add_argument("--batch-size", type=int, default=2)
    trainParser.add_argument("--lr", type=float, default=1e-4)
    trainParser.add_argument("--seq-frames", type=int, default=240)
    trainParser.add_argument("--context-last", type=int, default=0)
    trainParser.add_argument(
        "--context-train-mode",
        type=str,
        default="alt",
        choices=["off", "alt", "ratio"],
        help="off: jamais; alt: une époque sur deux; ratio: probabilité",
    )
    trainParser.add_argument(
        "--context-train-ratio",
        type=float,
        default=0.5,
        help="Si mode=ratio, probabilité d'utiliser le contexte",
    )
    trainParser.add_argument("--d-model", type=int, default=256)
    trainParser.add_argument("--layers", type=int, default=6)
    trainParser.add_argument("--moe-experts", type=int, default=8)
    trainParser.add_argument("--moe-topk", type=int, default=2)
    trainParser.add_argument("--ckpt-every", type=int, default=500)
    trainParser.add_argument("--resume", type=Path, default=None)
    trainParser.add_argument("--max-tokens", type=int, default=128)
    trainParser.add_argument(
        "--text-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Nom du modèle HuggingFace pour les embeddings",
    )
    trainParser.add_argument("--qat-ready", action="store_true")
    trainParser.add_argument("--seed", type=int, default=42)
    trainParser.add_argument("--val-split", type=float, default=0.1)
    trainParser.add_argument("--val-every", type=int, default=1000)
    trainParser.add_argument("--success-deg", type=float, default=5.0)
    trainParser.add_argument("--target-success", type=float, default=0.90)
    trainParser.add_argument("--cache-on-device", action="store_true")
    trainParser.add_argument("--max-val-samples", type=int, default=256)
    trainParser.add_argument("--recache-every-epoch", action="store_true")
    trainParser.add_argument("--device", type=str, default="auto")
    trainParser.add_argument("--no-cuda", action="store_true")
    trainParser.add_argument("--no-dml", action="store_true")
    trainParser.add_argument("--no-mps", action="store_true")
    trainParser.add_argument("--strict-device", action="store_true")

    sampleParser = subParsers.add_parser("sample", help="Générer une animation")
    sampleParser.add_argument("--ckpt", type=Path, required=True)
    sampleParser.add_argument("--prompts", type=Path, required=True)
    sampleParser.add_argument("--out-json", type=Path, required=True)
    sampleParser.add_argument("--frames", type=int, default=240)
    sampleParser.add_argument("--steps", type=int, default=12)
    sampleParser.add_argument("--guidance", type=float, default=2.0)
    sampleParser.add_argument(
        "--context-jsons",
        type=str,
        default="",
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
    sampleParser.add_argument("--text-model", type=str, default=None)
    sampleParser.add_argument("--device", type=str, default="auto")
    sampleParser.add_argument("--no-cuda", action="store_true")
    sampleParser.add_argument("--no-dml", action="store_true")
    sampleParser.add_argument("--no-mps", action="store_true")
    sampleParser.add_argument("--strict-device", action="store_true")
    return parser


def parseDeviceOptions(args: argparse.Namespace) -> DeviceSelectionOptions:
    return DeviceSelectionOptions(
        requestedBackend=args.device,
        allowCuda=not args.no_cuda,
        allowDirectML=not args.no_dml,
        allowMps=not args.no_mps,
        requireGpu=args.strict_device,
    )


def buildTrainingConfiguration(args: argparse.Namespace) -> TrainingConfiguration:
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
