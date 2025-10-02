"""Device selection and reproducibility utilities."""

from __future__ import annotations

import os
import random
from typing import Optional, Tuple

import numpy as np
import torch

from src.shared.types import DeviceSelectionOptions


class DeviceSelector:
    """Resolve the most appropriate `torch.device` for execution."""

    @staticmethod
    def selectDevice(options: DeviceSelectionOptions) -> Tuple[torch.device, str]:
        """Return a device and backend name matching the provided constraints."""
        forcedDevice = DeviceSelector._selectForcedDevice(options)
        if forcedDevice is not None:
            return forcedDevice
        autoDevice = DeviceSelector._selectBestAvailableDevice(options)
        if autoDevice is not None:
            return autoDevice
        if options.requireGpu:
            raise RuntimeError(
                "Aucun backend GPU disponible (CUDA/ROCm, DirectML ou MPS)."
            )
        return torch.device("cpu"), "cpu"

    @staticmethod
    def _selectForcedDevice(
        options: DeviceSelectionOptions,
    ) -> Optional[Tuple[torch.device, str]]:
        backend = _normalize(options.requestedBackend)
        if backend in (None, "auto"):
            return None
        if backend == "cuda":
            return DeviceSelector._resolveCuda()
        if backend == "dml":
            return DeviceSelector._resolveDirectML()
        if backend == "mps":
            return DeviceSelector._resolveMps()
        if backend == "cpu":
            return torch.device("cpu"), "cpu"
        raise ValueError(
            "--device doit être parmi auto|cuda|dml|mps|cpu, reçu:"
            f" {options.requestedBackend}"
        )

    @staticmethod
    def _selectBestAvailableDevice(
        options: DeviceSelectionOptions,
    ) -> Optional[Tuple[torch.device, str]]:
        if options.allowCuda:
            cudaCandidate = DeviceSelector._resolveCuda()
            if cudaCandidate is not None:
                return cudaCandidate
        if options.allowDirectML:
            dmlCandidate = DeviceSelector._resolveDirectML()
            if dmlCandidate is not None:
                return dmlCandidate
        if options.allowMps:
            mpsCandidate = DeviceSelector._resolveMps()
            if mpsCandidate is not None:
                return mpsCandidate
        return None

    @staticmethod
    def _resolveCuda() -> Optional[Tuple[torch.device, str]]:
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        return None

    @staticmethod
    def _resolveDirectML() -> Optional[Tuple[torch.device, str]]:
        try:
            import torch_directml as torchDirectML  # type: ignore
        except ImportError:
            return None
        return torchDirectML.device(), "dml"

    @staticmethod
    def _resolveMps() -> Optional[Tuple[torch.device, str]]:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps"), "mps"
        return None


class RandomnessController:
    """Helpers guaranteeing deterministic behaviour when possible."""

    @staticmethod
    def seedEverywhere(seedValue: int) -> None:
        """Seed Python, NumPy and PyTorch RNGs with the provided value."""
        os.environ["PYTHONHASHSEED"] = str(seedValue)
        random.seed(seedValue)
        np.random.seed(seedValue)
        torch.manual_seed(seedValue)
        RandomnessController._seedCuda(seedValue)
        RandomnessController._seedMps(seedValue)
        RandomnessController._enforceDeterminism()

    @staticmethod
    def _seedCuda(seedValue: int) -> None:
        if not torch.cuda.is_available():
            return
        torch.cuda.manual_seed(seedValue)
        torch.cuda.manual_seed_all(seedValue)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends, "cuda") and hasattr(
            torch.backends.cuda, "matmul"
        ):
            torch.backends.cuda.matmul.allow_tf32 = False

    @staticmethod
    def _seedMps(seedValue: int) -> None:
        if not hasattr(torch.backends, "mps"):
            return
        if not torch.backends.mps.is_available():
            return
        hasManualSeed = (
            hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed")
        )
        if hasManualSeed:
            torch.mps.manual_seed(seedValue)

    @staticmethod
    def _enforceDeterminism() -> None:
        """Request deterministic algorithms when supported."""
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            pass


def _normalize(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return value.lower().strip()
