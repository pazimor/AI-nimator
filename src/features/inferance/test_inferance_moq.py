import json
from pathlib import Path

import torch

from src.features.inferance import Prompt2AnimDiffusionSampler
from src.shared.temporal_diffusion import TemporalUNetMoE
from src.shared.types import DeviceSelectionOptions, SamplingConfiguration


class DummyEncoder(torch.nn.Module):
    def __init__(self, modelName: str, device: torch.device, trainable: bool = False):
        super().__init__()
        self.linear = torch.nn.Linear(1, 4)
        self.outDimension = 4
        self.trainable = trainable
        self.device = device
        self.to(device)

    def forward(self, texts):
        batch = len(texts)
        inputs = torch.ones(batch, 1, device=self.device)
        return self.linear(inputs)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_sampler_run_sampling(tmp_path, monkeypatch):
    model = TemporalUNetMoE(
        rotationInputDim=6,
        hiddenDim=8,
        layerCount=2,
        moeConfiguration={"expertCount": 2, "topK": 1},
        textDim=4,
    )
    encoder = DummyEncoder("dummy", torch.device("cpu"))
    checkpoint = {
        "model": model.state_dict(),
        "text_encoder": encoder.state_dict(),
        "configuration": {
            "modelDimension": 8,
            "layerCount": 2,
            "expertCount": 2,
            "expertTopK": 1,
        },
    }
    ckpt_path = tmp_path / "mock.ckpt"
    torch.save(checkpoint, ckpt_path)

    prompt_payload = {
        "Simple": "jump",
        "advanced": "quick",
        "expert": "",
        "tag": "demo",
    }
    prompts_path = tmp_path / "prompt.json"
    _write_json(prompts_path, prompt_payload)

    monkeypatch.setattr(
        "src.features.inferance.inferance.PretrainedTextEncoder",
        DummyEncoder,
    )

    config = SamplingConfiguration(
        checkpointPath=ckpt_path,
        promptsPath=prompts_path,
        outputPath=tmp_path / "out.json",
        frameCount=2,
        steps=3,
        guidance=1.5,
        contextJsons=[],
        boneNames=["hip"],
        omitMetadata=False,
        textModelName="dummy",
        deviceOptions=DeviceSelectionOptions(
            requestedBackend="cpu",
            allowCuda=False,
            allowDirectML=False,
            allowMps=False,
            requireGpu=False,
        ),
    )

    sampler = Prompt2AnimDiffusionSampler(config)
    sampler.runSampling()

    data = json.loads(config.outputPath.read_text(encoding="utf-8"))
    assert "rotations" in data
    assert len(data["rotations"]["hip"]) == config.frameCount
