"""Adapter for PyTorch auditor on AI-nimator v2 checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.generation.full_training_v2 import V2FullTrainingConfig
from src.features.generation.training_v2 import (
    resolveDevice,
)
from src.shared.model.generation.denoiser_v2 import MotionDenoiserV2
from src.shared.model.generation.losses_v2 import (
    diffusionLossV2,
    textMotionContrastiveLoss,
    velocityXyzLossV2,
)
from src.shared.model.generation.motion_normalizer import MotionNormalizer
from src.shared.model.generation.noise_schedule import (
    NoiseSchedule,
    NoiseScheduleConfig,
)
from src.shared.model.text import CustomTextEncoder, CustomTextEncoderConfig, CustomTokenizer
from src.shared.preprocessed_dataset import PreprocessedLinkDataset


class PyTorchAuditorAdapter:
    """Adapter for v2 diffusion model audit."""

    def __init__(self, checkpoint_dir: Path | str = "output/generation_v2_phaseE_5folders_v1"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.dataset_root = Path("/Users/pazimor/dataset_preprocessed")
        self.tokenizer_dir = Path("output/text/custom_tokenizer")

    def build_model(self, state_dict: dict[str, Any], device: str) -> torch.nn.Module:
        """Reconstruct the denoiser from state_dict."""
        # Extract architecture from metadata
        metadata = state_dict.get("_metadata", {})
        denoiser_config = metadata.get("denoiser_config", {})

        # Build denoiser from saved config
        denoiser = MotionDenoiserV2.fromStateDict(state_dict, device=device)
        denoiser.eval()
        return denoiser

    def get_sample_input(self, batch: dict[str, Any], device: str) -> tuple[tuple, dict]:
        """Extract forward() args from a batch."""
        x_t = batch.get("x_t", torch.randn(1, 256, 66)).to(device)
        t = batch.get("t", torch.tensor([500])).to(device)
        text_emb = batch.get("text_embedding", torch.randn(1, 768)).to(device)

        return (x_t, t, text_emb), {}

    def build_dataloader(self) -> DataLoader | None:
        """Build a small DataLoader for diagnostics."""
        try:
            dataset = PreprocessedLinkDataset(
                root=self.dataset_root,
                folders=["CMU"],  # Small folder for speed
                link_indices=list(range(min(100, 100))),  # First 100 links
                max_length=256,
                cache_dir=None,
            )
            return DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        except Exception as e:
            print(f"Warning: Could not build DataLoader: {e}")
            return None

    def compute_loss(
        self, model: torch.nn.Module, batch: dict[str, Any], device: str
    ) -> dict[str, torch.Tensor]:
        """Decompose loss into components."""
        try:
            # Dummy inputs for demonstration
            x_t = torch.randn(2, 256, 66).to(device)
            t = torch.randint(0, 1000, (2,)).to(device)
            text_emb = torch.randn(2, 768).to(device)
            x_0 = torch.randn(2, 256, 66).to(device)

            # Call model
            with torch.no_grad():
                pred = model(x_t, t, text_emb)

            # Basic diffusion loss
            diff_loss = torch.nn.functional.mse_loss(pred, x_0)

            return {
                "diffusion": diff_loss,
                "total": diff_loss,
            }
        except Exception as e:
            print(f"Warning: Could not compute loss: {e}")
            return {}
