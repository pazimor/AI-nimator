# AI-nimator

Text-to-motion project with a CLIP-like text encoder and a diffusion-based motion generator.

## Setup

This repo uses Poetry.

```bash
poetry install
```

## Configuration

Main config files:
- `src/configs/network.yaml` - shared network architecture (embed dim, heads, layers, bones)
- `src/configs/train_clip.yaml` - CLIP training settings
- `src/configs/train_generation.yaml` - generation training settings
- `src/configs/dataset.yaml` - dataset build settings

Architecture reference:
- `doc/architecture.md`

## Common workflow

1) Build/convert dataset
```bash
poetry run python -m src.cli.build_dataset --config src/configs/dataset.yaml
```

2) Train CLIP (text-motion alignment)
```bash
poetry run python -m src.cli.train_clip --config src/configs/train_clip.yaml
```

3) Train generation (diffusion denoiser)
```bash
poetry run python -m src.cli.train_generation --config src/configs/train_generation.yaml
```

## Tools

### Inspect checkpoints
```bash
poetry run python -m src.cli.tools inspect output/checkpoints/best_model.pt
```

### Shape check (architecture validation)
```bash
poetry run python -m src.cli.tools shape-check --network-profile default
```

### Convert JSON output to Collada
```bash
poetry run python -m src.cli.json_to_collada -i path/to/animation.json -o out.dae
```

## Notes

- `network.yaml` controls shared dimensions across CLIP and generation. Keep `embed-dim` aligned.
