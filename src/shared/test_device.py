import random
import sys

import pytest
import torch

from src.shared.device import DeviceSelector, RandomnessController
from src.shared.types import DeviceSelectionOptions


def test_select_device_returns_cpu_when_no_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    options = DeviceSelectionOptions(
        requestedBackend=None,
        allowCuda=False,
        allowDirectML=False,
        allowMps=False,
    )
    device, backend = DeviceSelector.selectDevice(options)
    assert device.type == "cpu"
    assert backend == "cpu"


def test_randomness_controller_produces_deterministic_sequences():
    RandomnessController.seedEverywhere(123)
    first_python = random.randint(0, 10**6)
    first_torch = torch.rand(3)

    RandomnessController.seedEverywhere(123)
    second_python = random.randint(0, 10**6)
    second_torch = torch.rand(3)

    assert first_python == second_python
    assert torch.allclose(first_torch, second_torch)


def test_select_device_forced_invalid():
    options = DeviceSelectionOptions(requestedBackend="invalid")
    try:
        DeviceSelector.selectDevice(options)
    except ValueError as exc:
        assert "--device" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid backend")


def test_select_device_forced_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device, backend = DeviceSelector.selectDevice(DeviceSelectionOptions(requestedBackend="cuda"))
    assert backend == "cuda"
    assert device.type == "cuda"


def test_select_device_directml(monkeypatch):
    class DummyDML:
        @staticmethod
        def device():
            return "dml"

    monkeypatch.setitem(sys.modules, "torch_directml", DummyDML)
    device, backend = DeviceSelector.selectDevice(DeviceSelectionOptions(requestedBackend="dml"))
    assert backend == "dml"
    assert str(device) == "dml"


def test_select_device_require_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    options = DeviceSelectionOptions(
        requestedBackend=None,
        requireGpu=True,
        allowCuda=False,
        allowDirectML=False,
        allowMps=False,
    )
    with pytest.raises(RuntimeError):
        DeviceSelector.selectDevice(options)


def test_select_device_mps(monkeypatch):
    class DummyMps:
        @staticmethod
        def is_available():
            return True

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", DummyMps())
    device, backend = DeviceSelector.selectDevice(DeviceSelectionOptions(requestedBackend="mps"))
    assert backend == "mps"
    assert device.type == "mps"


def test_select_device_auto_uses_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    class DummyBackend:
        @staticmethod
        def is_available():
            return False

    monkeypatch.setattr(torch.backends, "mps", DummyBackend())
    if "torch_directml" in sys.modules:
        del sys.modules["torch_directml"]
    device, backend = DeviceSelector.selectDevice(DeviceSelectionOptions(requestedBackend=None))
    assert backend == "cpu"
    assert device.type == "cpu"
