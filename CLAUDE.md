# AI_nimator
this is a research project.
aim to train an AI to generate 3D motions based on AMASS dataset

## Stack
- Language: python
- Framework: pytorch [check pyproject.toml for fukk dependencies]
- Dataset path: src/configs/dataset.yaml
- Testing: pytest

## Directory Structure
- `src/` — Application source code
    - `cli/` — entry files for different executables
    - `configs/` — configurations in yaml 
    - `features/` - specific sources for it's cotresponding CLI
    - `shared` - shared code between all cli
- `output/` - output files generated
- `test/` — files used for Testing
    - `src/` - re produce the src tree and place the corespondings tests here
- `doc/` — Internal documentation

## Commands
- `poetry run python -m src.cli.build_dataset` — build dataset match prompts and animations
- `poetry run python -m src.cli.preprocess_dataset` — allows the dataset to be corectly interpreted
- `poetry run python -m src.cli.train_clip` — training for CLIP part
- `poetry run python -m src.cli.train_generation` — training for generation

## Conventions
- Document every method with full **DOCString** (NumPy Style).
- Be complient with Pylance
- Avoid **magic numbers** and strings: always use named constants or enums.
- Use **explicit functions** (pure and reusable).
- Avoid shortcuts: no `i`, `m`, etc. in anonymous functions or methods.
- use strict types
- No method should exceed 25 lines.
- No column should exceed 80 characters.
- try to keep a max length file around 500 lines.
- dataclasses should be inside src/shared/types 
- Refactor into private functions when necessary.
- Use **isolated unit tests** (no cross-module dependencies).
