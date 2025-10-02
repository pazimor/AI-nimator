from types import SimpleNamespace

import torch

from src.shared.text import PretrainedTextEncoder


class DummyTokenizer:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, model_name: str):
        return cls()

    def __call__(self, texts, padding=True, truncation=True, return_tensors="pt"):
        batch_size = len(texts)
        seq_len = 3
        input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.config = SimpleNamespace(hidden_size=4)

    @classmethod
    def from_pretrained(cls, model_name: str):
        return cls()

    def forward(self, input_ids=None, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        last_hidden_state = torch.ones(batch_size, seq_len, 4)
        return SimpleNamespace(last_hidden_state=last_hidden_state, pooler_output=None)


def test_pretrained_text_encoder_forward(monkeypatch):
    monkeypatch.setattr("src.shared.text.AutoTokenizer", DummyTokenizer)
    monkeypatch.setattr("src.shared.text.AutoModel", DummyModel)
    device = torch.device("cpu")
    encoder = PretrainedTextEncoder(
        modelName="dummy/model",
        device=device,
        trainable=False,
    )
    embeddings = encoder(["hello", "world"])
    assert embeddings.shape == (2, encoder.outDimension)
    assert torch.allclose(embeddings[0], embeddings[1])
    assert not any(param.requires_grad for param in encoder.model.parameters())


def test_pretrained_text_encoder_with_pooler(monkeypatch):
    class PoolerModel(DummyModel):
        def forward(self, input_ids=None, attention_mask=None):
            batch_size, _ = input_ids.shape
            return SimpleNamespace(pooler_output=torch.ones(batch_size, 4))

    monkeypatch.setattr("src.shared.text.AutoTokenizer", DummyTokenizer)
    monkeypatch.setattr("src.shared.text.AutoModel", PoolerModel)
    encoder = PretrainedTextEncoder("dummy", torch.device("cpu"))
    output = encoder(["a"])
    assert output.shape == (1, encoder.outDimension)


def test_pretrained_text_encoder_missing_transformers(monkeypatch):
    monkeypatch.setattr("src.shared.text.AutoTokenizer", None)
    monkeypatch.setattr("src.shared.text.AutoModel", None)
    try:
        PretrainedTextEncoder("dummy", torch.device("cpu"))
    except ImportError:
        pass
    else:
        raise AssertionError("Expected ImportError when transformers are absent")
