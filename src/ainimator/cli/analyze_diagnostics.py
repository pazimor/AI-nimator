"""Summarize a diagnostics JSONL file produced by src.shared.diagnostics.

Typical usage::

    poetry run python -m src.cli.analyze_diagnostics output/generation/diag/*.jsonl

The script is intentionally dependency-free (stdlib only) so it can run
on a training box without loading torch.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def iterRecords(path: Path) -> Iterable[dict[str, Any]]:
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def summarize(path: Path) -> None:
    trainLosses: list[tuple[int, float]] = []
    lossesByBucket: dict[int, list[float]] = defaultdict(list)
    predStdsByBucket: dict[int, list[float]] = defaultdict(list)
    gradNorms: list[float] = []
    epochSummaries: list[dict[str, Any]] = []
    ddimSteps: list[dict[str, Any]] = []
    generations: list[dict[str, Any]] = []

    bucketWidth = 100  # default assume 1000 timesteps / 10 buckets

    for record in iterRecords(path):
        phase = record.get("phase")
        if phase == "train_batch":
            globalStep = int(record.get("global_step", 0))
            loss = float(record.get("loss", math.nan))
            trainLosses.append((globalStep, loss))
            tMean = record.get("t_mean")
            if tMean is not None:
                bucket = min(int(tMean) // bucketWidth, 9)
                lossesByBucket[bucket].append(loss)
                predStats = (record.get("stats") or {}).get("predicted_motion")
                if predStats and predStats.get("std") is not None:
                    predStdsByBucket[bucket].append(float(predStats["std"]))
            gn = record.get("grad_norm")
            if gn is not None:
                gradNorms.append(float(gn))
        elif phase == "epoch_summary":
            epochSummaries.append(record)
        elif phase == "ddim_step":
            ddimSteps.append(record)
        elif phase == "generation_summary":
            generations.append(record)

    print(f"\n=== File: {path} ===")
    print(f"Train batches logged: {len(trainLosses)}")
    print(f"Epoch summaries: {len(epochSummaries)}")
    print(f"DDIM steps: {len(ddimSteps)}")
    print(f"Generations: {len(generations)}")

    if trainLosses:
        losses = [v for _, v in trainLosses]
        first = statistics.mean(losses[: max(1, len(losses) // 20)])
        last = statistics.mean(losses[-max(1, len(losses) // 20) :])
        print(
            f"\n[Train loss] first5% mean={first:.4f}  last5% mean={last:.4f}  "
            f"min={min(losses):.4f}  max={max(losses):.4f}"
        )

    if lossesByBucket:
        print("\n[Loss by timestep bucket]  (converged uniformly? watch outliers)")
        print("  bucket    t-range     n      avg_loss   pred_std_mean")
        for bucket in sorted(lossesByBucket.keys()):
            values = lossesByBucket[bucket]
            predStdList = predStdsByBucket.get(bucket, [])
            predStdMean = (
                f"{statistics.mean(predStdList):.4f}" if predStdList else "n/a"
            )
            tLo = bucket * bucketWidth
            tHi = (bucket + 1) * bucketWidth - 1
            print(
                f"  {bucket:>3}   [{tLo:>4},{tHi:>4}]  "
                f"{len(values):>5}  {statistics.mean(values):>8.4f}   {predStdMean}"
            )

    if gradNorms:
        print(
            f"\n[Grad norm] mean={statistics.mean(gradNorms):.4f}  "
            f"min={min(gradNorms):.4f}  max={max(gradNorms):.4f}  "
            f"last10 mean="
            f"{statistics.mean(gradNorms[-10:]):.4f}"
        )

    if epochSummaries:
        last = epochSummaries[-1]
        print(
            f"\n[Last epoch {last.get('epoch')}] train_loss="
            f"{last.get('train_loss'):.4f}  lr={last.get('learning_rate')}"
        )
        buckets = last.get("timestep_buckets", {})
        if buckets:
            print("  per-bucket avg_loss:")
            for bucketKey in sorted(buckets.keys(), key=int):
                b = buckets[bucketKey]
                print(
                    f"    bucket {bucketKey}: n={int(b.get('count', 0))} "
                    f"avg_loss={b.get('avg_loss', float('nan')):.4f}"
                )

    if ddimSteps:
        print("\n[DDIM trajectory]")
        print("  step   t     x_std   cond_x0_std  uncond_x0_std  guided_x0_std  cfg_delta")
        for rec in ddimSteps:
            xStd = (rec.get("x_bone") or {}).get("std", float("nan"))
            condStd = (rec.get("cond_x0") or {}).get("std", float("nan"))
            uncStd = (rec.get("uncond_x0") or {}).get("std", float("nan"))
            gStd = (rec.get("guided_x0") or {}).get("std", float("nan"))
            cfgDelta = (rec.get("cfg_delta") or {}).get("norm_mean", float("nan"))
            print(
                f"  {rec.get('step_idx'):>3}  "
                f"{rec.get('timestep'):>4}  {xStd:>6.3f}  "
                f"{condStd:>10.3f}   {uncStd:>12.3f}   {gStd:>12.3f}   {cfgDelta:>8.3f}"
            )

    if generations:
        for gen in generations:
            fd = gen.get("frame_delta") or {}
            mq = gen.get("motion_quat") or {}
            print(
                f"\n[Generation] prompt={gen.get('prompt')!r} "
                f"frames={gen.get('num_frames')} cfg={gen.get('cfg_scale')}"
            )
            print(
                f"  motion_quat  mean={mq.get('mean')}  std={mq.get('std')}  "
                f"abs_mean={mq.get('abs_mean')}"
            )
            print(
                f"  frame_delta  mean_abs={fd.get('mean_abs')}  "
                f"std={fd.get('std')}  max_abs={fd.get('max_abs')}"
            )
            if fd.get("mean_abs") is not None and fd.get("mean_abs") < 1e-4:
                print(
                    "  !! WARNING: frame-to-frame delta is near zero -- "
                    "animation is effectively static."
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a diagnostics JSONL produced by training/inference.",
    )
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        if not path.exists():
            print(f"Missing: {path}", file=sys.stderr)
            continue
        summarize(path)


if __name__ == "__main__":
    main()
