"""TensorBoard writer wrapper — gracefully optional.

TensorBoard is a secondary sink; JSONL is the source of truth.
If ``tensorboard`` is not installed this module still imports cleanly
and all write calls silently no-op with a one-time warning.

Usage::

    from ainimator.health.tb_writer import TbWriter
    writer = TbWriter(outputDir)
    writer.write(step=100, metrics={"cfg_sim": 0.42})
    writer.close()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

# Flag to emit the "TB not available" warning exactly once.
_TB_WARNED = False


def _tryImportSummaryWriter() -> Any:
    """Return SummaryWriter class or None if tensorboard missing."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter
    except ImportError:
        pass
    try:
        from tensorboard.summary.writer.event_file_writer import (
            EventFileWriter as _,
        )
        from tensorboardX import SummaryWriter  # type: ignore
        return SummaryWriter
    except ImportError:
        pass
    return None


class TbWriter:
    """Optional TensorBoard writer.

    Falls back to a no-op if tensorboard is not installed.

    Parameters
    ----------
    outputDir : Path
        Run output directory.  TensorBoard logs go to a ``tb/``
        subdirectory.
    """

    def __init__(self, outputDir: Path) -> None:
        global _TB_WARNED
        WriterClass = _tryImportSummaryWriter()
        if WriterClass is None:
            if not _TB_WARNED:
                LOGGER.warning(
                    "tensorboard not installed — TB metrics disabled. "
                    "JSONL remains the source of truth."
                )
                _TB_WARNED = True
            self._writer = None
        else:
            tbDir = outputDir / "tb"
            tbDir.mkdir(parents=True, exist_ok=True)
            try:
                self._writer = WriterClass(log_dir=str(tbDir))
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "TbWriter: could not create SummaryWriter: %s", exc
                )
                self._writer = None

    def write(
        self,
        step: int,
        metrics: dict[str, float | None],
    ) -> None:
        """Write scalar metrics to TensorBoard.

        Parameters
        ----------
        step : int
            Global training step.
        metrics : dict
            Metric name → float value (None values are skipped).
        """
        if self._writer is None:
            return
        for key, value in metrics.items():
            if value is None:
                continue
            try:
                self._writer.add_scalar(
                    f"health/{key}", value, global_step=step
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("TbWriter: add_scalar failed: %s", exc)

    def close(self) -> None:
        """Flush and close the underlying writer."""
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:  # noqa: BLE001
                pass
