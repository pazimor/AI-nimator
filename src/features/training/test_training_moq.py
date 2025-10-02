import json
from pathlib import Path
from typing import List

import torch
from torch.amp import GradScaler
import pytest

from src.features.training import (
    AnimationPromptDataset,
    DatasetCacheBuilder,
    CheckpointManager,
    Prompt2AnimDiffusionTrainer,
)
from src.shared.temporal_diffusion import TemporalUNetMoE
from src.shared.types import (
    AnimationPromptSample,
    ClipRecord,
    DatasetCache,
    DeviceSelectionOptions,
    TrainingConfiguration,
)


class DummyEncoder(torch.nn.Module):
    def __init__(self, modelName: str, device: torch.device, trainable: bool = False):
        super().__init__()
        self.linear = torch.nn.Linear(1, 4)
        self.outDimension = 4
        self.trainable = trainable
        self.device = device
        self.to(device)

    def forward(self, texts: List[str]):
        batch = len(texts)
        inputs = torch.ones(batch, 1, device=self.device)
        return self.linear(inputs)


@pytest.fixture()
def mini_dataset(tmp_path: Path) -> Path:
    def write_clip(name: str, values: List[str]):
        clip_dir = tmp_path / name
        clip_dir.mkdir()
        rotation = {"rotations": {"hip": values}}
        prompt = {
            "Simple": "pose",
            "advanced": "simple",
            "tag": "demo",
        }
        (clip_dir / "animation.json").write_text(json.dumps(rotation), encoding="utf-8")
        (clip_dir / "prompts.json").write_text(json.dumps(prompt), encoding="utf-8")

    write_clip("clip_a", ["1|0|0|0", "0.9659|0|0.2588|0"])
    write_clip("clip_b", ["1|0|0|0", "0.8660|0|0.5|0"])
    return tmp_path


def test_animation_prompt_dataset(mini_dataset: Path):
    dataset = AnimationPromptDataset(mini_dataset, sequenceFrames=2, contextHistory=1)
    sample = dataset[0]
    assert isinstance(sample, AnimationPromptSample)
    assert sample.rotation6d.shape == (2, 1, 6)
    assert sample.contextSequence is not None


def test_dataset_cache_builder(mini_dataset: Path):
    dataset = AnimationPromptDataset(mini_dataset, sequenceFrames=2)
    encoder = DummyEncoder("dummy", torch.device("cpu"))
    cache = DatasetCacheBuilder.buildCache(
        dataset=dataset,
        device=torch.device("cpu"),
        textEncoder=encoder,
        cacheOnDevice=False,
    )
    assert isinstance(cache, DatasetCache)
    assert len(cache.rotationSequences) == len(dataset)
    assert cache.rotationSequences[0].shape[-1] == 6


def test_checkpoint_manager_roundtrip(tmp_path: Path, capsys):
    model = TemporalUNetMoE(
        rotationInputDim=6,
        hiddenDim=8,
        layerCount=2,
        moeConfiguration={"expertCount": 2, "topK": 1},
        textDim=4,
    )
    encoder = DummyEncoder("dummy", torch.device("cpu"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler("cuda", enabled=False)
    config_dict = {
        "modelDimension": 8,
        "layerCount": 2,
        "expertCount": 2,
        "expertTopK": 1,
    }
    ckpt_path = tmp_path / "roundtrip.pt"
    CheckpointManager.save(
        ckpt_path,
        model,
        encoder,
        optimizer,
        scaler,
        step=7,
        epoch=2,
        configuration=config_dict,
    )
    new_model = TemporalUNetMoE(
        rotationInputDim=6,
        hiddenDim=8,
        layerCount=2,
        moeConfiguration={"expertCount": 2, "topK": 1},
        textDim=4,
    )
    new_encoder = DummyEncoder("dummy", torch.device("cpu"))
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
    new_scaler = GradScaler("cuda", enabled=False)
    step, epoch, loaded_config = CheckpointManager.load(
        ckpt_path,
        new_model,
        new_encoder,
        optimizer=new_optimizer,
        scaler=new_scaler,
    )
    output = capsys.readouterr().out
    assert "[checkpoint]" in output
    assert "size=" in output and "bytes" in output
    assert step == 7 and epoch == 2
    assert loaded_config == config_dict


def test_trainer_run_training(mini_dataset: Path, monkeypatch):
    monkeypatch.setattr(
        "src.features.training.training.PretrainedTextEncoder",
        DummyEncoder,
    )
    config = TrainingConfiguration(
        dataDirectory=mini_dataset,
        saveDirectory=mini_dataset / "ckpts",
        experimentName="unit",
        epochs=1,
        batchSize=1,
        learningRate=1e-3,
        sequenceFrames=2,
        contextHistory=0,
        contextTrainMode="off",
        contextTrainRatio=0.0,
        modelDimension=8,
        layerCount=2,
        expertCount=2,
        expertTopK=1,
        checkpointInterval=1,
        resumePath=None,
        maximumTokens=32,
        textModelName="dummy",
        prepareQat=False,
        randomSeed=123,
        validationSplit=0.5,
        validationInterval=1,
        successThresholdDegrees=10.0,
        targetSuccessRate=1.0,
        cacheOnDevice=False,
        maximumValidationSamples=1,
        recacheEveryEpoch=False,
        deviceOptions=DeviceSelectionOptions(
            requestedBackend="cpu",
            allowCuda=False,
            allowDirectML=False,
            allowMps=False,
            requireGpu=False,
        ),
    )
    trainer = Prompt2AnimDiffusionTrainer(config)
    trainer.runTraining()
    last_ckpt = config.saveDirectory / "unit_last.pt"
    assert last_ckpt.exists()
