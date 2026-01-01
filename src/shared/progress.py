"""Shared progress bar utilities for training workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import torch
from tqdm import tqdm


@dataclass
class TrainingMetrics:
    """Metrics collected during training epoch."""

    totalLoss: float = 0.0
    batchCount: int = 0
    bestLoss: Optional[float] = None

    @property
    def avgLoss(self) -> float:
        """Return average loss across batches."""
        return self.totalLoss / max(self.batchCount, 1)

    def update(self, loss: float) -> None:
        """Update metrics with new batch loss."""
        self.totalLoss += loss
        self.batchCount += 1


class TrainingProgressBar:
    """
    Unified progress bar for training loops.

    Provides consistent visual feedback for CLIP and generation training.

    Parameters
    ----------
    dataloader : Iterable
        The dataloader to iterate over.
    epoch : int
        Current epoch number (1-based).
    totalEpochs : int
        Total number of epochs.
    desc : str
        Description prefix for the progress bar.
    device : torch.device
        Training device.
    chunkInfo : Optional[str]
        Optional string describing current dataset chunk.

    Examples
    --------
    >>> with TrainingProgressBar(loader, epoch=1, totalEpochs=10, desc="CLIP") as pbar:
    ...     for batch in pbar:
    ...         loss = train_step(batch)
    ...         pbar.update_loss(loss)
    """

    def __init__(
        self,
        dataloader: Iterable,
        epoch: int,
        totalEpochs: int,
        desc: str = "Training",
        device: Optional[torch.device] = None,
        chunkInfo: Optional[str] = None,
    ) -> None:
        self.dataloader = dataloader
        self.epoch = epoch
        self.totalEpochs = totalEpochs
        self.desc = desc
        self.device = device
        self.chunkInfo = chunkInfo
        self.metrics = TrainingMetrics()
        self._pbar: Optional[tqdm] = None
        self._batchIndex = 0

    def _buildDescription(self) -> str:
        """Build progress bar description."""
        parts = [f"{self.desc} [{self.epoch}/{self.totalEpochs}]"]
        if self.chunkInfo:
            parts.append(f"({self.chunkInfo})")
        return " ".join(parts)

    def __enter__(self) -> "TrainingProgressBar":
        """Enter context and create progress bar."""
        self._pbar = tqdm(
            self.dataloader,
            desc=self._buildDescription(),
            leave=True,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and close progress bar."""
        if self._pbar is not None:
            self._pbar.close()

    def __iter__(self) -> Iterable:
        """Iterate over dataloader with progress tracking."""
        if self._pbar is None:
            raise RuntimeError("Must use TrainingProgressBar as context manager")
        
        for batch in self._pbar:
            yield batch
            self._batchIndex += 1

    def updateLoss(self, loss: float) -> None:
        """
        Update progress bar with current loss.

        Parameters
        ----------
        loss : float
            Loss value for current batch.
        """
        self.metrics.update(loss)
        if self._pbar is not None:
            self._pbar.set_postfix(
                loss=f"{loss:.4f}",
                avg=f"{self.metrics.avgLoss:.4f}",
                refresh=False,
            )

    def setPostfix(self, **kwargs: Any) -> None:
        """Set custom postfix values on progress bar."""
        if self._pbar is not None:
            self._pbar.set_postfix(**kwargs, refresh=False)


class EpochProgressBar:
    """
    Progress bar for tracking epochs.

    Parameters
    ----------
    totalEpochs : int
        Total number of epochs.
    startEpoch : int
        Starting epoch (for resume).
    desc : str
        Description prefix.

    Examples
    --------
    >>> with EpochProgressBar(totalEpochs=100, desc="CLIP Training") as epochs:
    ...     for epoch in epochs:
    ...         train_loss = train_one_epoch(...)
    ...         epochs.update_metrics(train_loss=train_loss, val_loss=val_loss)
    """

    def __init__(
        self,
        totalEpochs: int,
        startEpoch: int = 0,
        desc: str = "Training",
    ) -> None:
        self.totalEpochs = totalEpochs
        self.startEpoch = startEpoch
        self.desc = desc
        self._pbar: Optional[tqdm] = None
        self._currentEpoch = startEpoch

    def __enter__(self) -> "EpochProgressBar":
        """Enter context and create progress bar."""
        self._pbar = tqdm(
            total=self.totalEpochs - self.startEpoch,
            desc=self.desc,
            leave=True,
            dynamic_ncols=True,
            position=0,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} epochs [{elapsed}<{remaining}] {postfix}",
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and close progress bar."""
        if self._pbar is not None:
            self._pbar.close()

    def __iter__(self) -> Iterable[int]:
        """Iterate over epochs."""
        for epoch in range(self.startEpoch, self.totalEpochs):
            self._currentEpoch = epoch + 1
            yield self._currentEpoch
            if self._pbar is not None:
                self._pbar.update(1)

    def updateMetrics(
        self,
        trainLoss: Optional[float] = None,
        valLoss: Optional[float] = None,
        bestLoss: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Update epoch progress bar with metrics.

        Parameters
        ----------
        trainLoss : Optional[float]
            Training loss for current epoch.
        valLoss : Optional[float]
            Validation loss for current epoch.
        bestLoss : Optional[float]
            Best validation loss so far.
        **kwargs
            Additional metrics to display.
        """
        if self._pbar is None:
            return
        
        postfix = {}
        if trainLoss is not None:
            postfix["train"] = f"{trainLoss:.4f}"
        if valLoss is not None:
            postfix["val"] = f"{valLoss:.4f}"
        if bestLoss is not None:
            postfix["best"] = f"{bestLoss:.4f}"
        postfix.update(kwargs)
        
        if postfix:
            self._pbar.set_postfix(postfix, refresh=True)

    def log(self, message: str) -> None:
        """Write message without breaking progress bar."""
        if self._pbar is not None:
            self._pbar.write(message)
