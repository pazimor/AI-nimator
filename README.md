# AI-nimator

Text-to-motion project with a custom BPE text encoder and a diffusion-based
motion generator (v2 stack: v-prediction, cosine schedule, DDIM, CFG).

## Setup

This repo uses Poetry.

```bash
poetry install
```

The `ainimator` package is registered as an editable install automatically.

## Configuration

Main config files:
- `src/configs/train_generation_v2.yaml` — v2 generation training settings
- `src/configs/dataset.yaml` — dataset build settings

## Common workflow

1) Build/convert dataset
```bash
poetry run python -m ainimator.cli.build_dataset --config src/configs/dataset.yaml
```

2) Train generation (diffusion denoiser v2)
```bash
poetry run python -m ainimator.cli.train_generation_v2 --profile overfit
poetry run python -m ainimator.cli.train_generation_v2 --profile full
```

3) Generate animation
```bash
poetry run python -m ainimator.cli.generate_animation_v2
```

## Tools

### Diagnose conditioning
```bash
poetry run python -m ainimator.cli.diagnose_generation_v2
```

### Run tests
```bash
poetry run pytest
```

### Verify import contracts
```bash
poetry run lint-imports
```

## Notes

- All commands use `ainimator.cli.*` (post phase A2).
- Legacy v1 commands (`src.cli.*`) are archived and should not be extended.
- `doc/ROADMAP.md` is the canonical reference for phases and decisions.
