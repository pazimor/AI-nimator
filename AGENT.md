# 🚀 Codex – Project Coding Guidelines

> **Purpose**: Provide Codex with a concise, self-contained reference of my project conventions so that every suggestion matches my standards **by default**.
>
> **Scope**: Coding style, architecture, best practices, testing, performance, and more.

---

## 1 • Project Context

1. **Framework:** python 3.12 + PyTorch + numpy (check `pyproject.toml` for each library used)
2. **Environment** setup specific .venv
3. **Goals:** Scalability, security, maintainability, performance, and great developer experience

---

## 2 • Golden Rules

1. Document every method with full **DOCString** (NumPy Style).
2. Always encapsulate business logic in a dedicated service or private method.
3. Avoid **magic numbers** and strings: always use named constants or enums.
4. Use **explicit functions** (pure and reusable).
5. Avoid shortcuts: no `i`, `m`, etc. in anonymous functions or methods.
6. **Strictly forbid unclear abbreviations**: never use `err`, `db`, `val`, `num`, `str`, etc. Always use full, descriptive names (e.g., `error`, `database`, `value`, `quantity`, `message`).
7. **Every feature or method you develop** must be accompanied by **comprehensive unit tests**, written in parallel and validated before every merge (line coverage ≥90%).

### ✍️ step to realise for every prompt

1. do the task
2. check for redondance with any code
3. extract this code into shared
4. implement tests
5. implement and update Docstring on fonctions you touched

---

## 3 • Strict Typing

1. follow the `pyproject.toml` configuration
2. use strict types
3. allways delete unused imports

---

## 4 • Code Standards

### ✏️ Style & Naming

- Use camelCase for variables, functions, and signals
- Use PascalCase for classes, and types
- **Strictly forbid unclear abbreviations**: never use `err`, `db`, `val`, `num`, `str`, etc. Always prefer explicit and descriptive names: `error`, `database`, `value`, `userCount`, `username`, etc.
- Always use clear, descriptive names to improve readability and symbol search

### ⛔ Complexity

- No method should exceed 25 lines.
- No column should exceed 80 characters.
- try to keep a max length file around 500 lines.
- dataclasses should be inside src/shared/types 
- Refactor into private functions when necessary.

### ✅ Tests

- Use **isolated unit tests** (no cross-module dependencies).
- Framework: `pytest`

---

## 5 • files Structure

- structure:
```aiignore
├── AGENT.md
├── doc
│   └── source
│       ├── cli.rst
│       ├── conf.py
│       ├── index.rst
│       ├── inference.rst
│       ├── overview.rst
│       └── training.rst
├── pyproject.toml
└── src
    ├── __init__.py
    ├── cli
    │   └── prompt2anim.py
    ├── constantes
    │   └── constante.py
    ├── features
    │   ├── inferance
    │   │   ├── __init__.py
    │   │   ├── inferance.py
    │   │   └── test_inferance_moq.py
    │   └── training
    │       ├── __init__.py
    │       ├── test_training_moq.py
    │       └── training.py
    └── shared
        ├── __init__.py
        ├── device.py
        ├── io.py
        ├── quaternion.py
        ├── temporal_diffusion.py
        ├── test_device.py
        ├── test_io.py
        ├── test_quaternion.py
        ├── test_temporal_diffusion.py
        ├── test_text.py
        ├── text.py
        └── types
            ├── __init__.py
            ├── config.py
            └── data.py
```

> **Principle**: Cli `src/cli` triggers scripts from `src/features`, shares data models from `src/shared/types`.
>
> every code that can be used by an other feature of the project should be in `src/shared`

---

## 6 • Performance & Optimization

lots of task available on the CLi is really long to run:
- every task should be optimised for performance and speed

---

## 7 • Use Cases, Features & Execution Pipeline

### Data Transformation :
- transform raw data into structured formats (and all dataset preparation steps)
- transform output data into usable formats in applications (blender, unity, UE5, maya, etc.)

### Neural Net Training :
- tuning for VRAM and GPU usage optimisation
- debug stacks avec dedicated WandB service (self hosted)

  - ### Testing Run :
  - testing on local machine with small dataset and full dataset

  - ### Full Run :
  - full run on cloud with large dataset


### Neural Net Inference (post trained operations) :
- ONNX transformation (MLCore, ROCm, CUDA optimisations etc.)
- inference on local machine

---

## 8 • Testing & Quality

📌 **Test Structure & Placement**

- Every `test_*.py` file must be **next to** the source file it tests (same folder).

📊 **Coverage & Goals**

- Every features should reach **≥ 90% lines coverage**.

---

## 9 • Docstring - Documentation

### Python - Docstring NumPy Style

ruff impose un docstring
utilise le style NumPy
```python
def add(a: int, b: int) -> int:
  """
  Adds two numbers.

    Parameters
    ----------
    a : int
        First number
    b : int
        Second number

    Returns
    -------
    int
        Sum of a and b
  """
  return a + b
```
don't omit optional parametter

### Documentation
Sphinx documentation must be generated
1. global readme about the project 
2. a page for the cli
3. a page for each feature, where you can explain what happen (especialy math)

---
