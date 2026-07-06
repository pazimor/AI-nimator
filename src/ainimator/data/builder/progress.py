"""Progress reporting helpers for dataset building features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class ProgressReporter:
    """Interface for reporting progress."""

    def advance(self, message: Optional[str] = None) -> None:
        """
        Advance the progress indicator.

        Parameters
        ----------
        message : Optional[str], default=None
            Optional short message displayed alongside the indicator.
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Release any progress resources.

        This is called once the conversion loop has finished.
        """
        raise NotImplementedError


class TqdmProgressReporter(ProgressReporter):
    """`tqdm`-powered progress bar."""

    def __init__(self, total: int, description: str) -> None:
        """
        Create a tqdm-backed progress reporter.

        Parameters
        ----------
        total : int
            Total number of steps.
        description : str
            Progress bar label.
        """
        from tqdm import tqdm

        self._bar = tqdm(total=total, desc=description)

    def advance(self, message: Optional[str] = None) -> None:
        """
        Advance the progress bar by one unit.

        Parameters
        ----------
        message : Optional[str], default=None
            Optional postfix appended to the tqdm bar.
        """
        self._bar.update(1)
        if message:
            self._bar.set_postfix_str(message, refresh=False)

    def close(self) -> None:
        """Close the underlying tqdm progress bar."""
        self._bar.close()
