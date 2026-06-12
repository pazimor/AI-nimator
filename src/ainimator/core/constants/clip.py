"""Constants shared by clip-related tooling."""

from __future__ import annotations

import math
from typing import Final


BATCH_LOG_FREQUENCY: Final[int] = 10

ROTATION_CHANNELS: Final[int] = 6

DEFAULT_LOGIT_SCALE: Final[float] = 2.6592
LOGIT_SCALE_MAX: Final[float] = math.log(100.0)
EPSILON: Final[float] = 1e-6

DEFAULT_PROMPT_MAX_LENGTH: Final[int] = 64
DEFAULT_BATCH_SIZE: Final[int] = 2
DEFAULT_LEARNING_RATE: Final[float] = 1e-4
DEFAULT_MODEL_NAME: Final[str] = "xlm-roberta-base"
DEFAULT_EMBED_DIM: Final[int] = 512
DEFAULT_VALIDATION_SPLIT: Final[float] = 0.1
DEFAULT_EARLY_STOPPING_PATIENCE: Final[int] = 3
