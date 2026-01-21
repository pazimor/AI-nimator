# 🚀 Project Coding Guidelines

> **Purpose**: Provide Codex a baseline, for my project conventions so that every suggestion matches my standards **by default**.
>
> **Scope**: Coding style, architecture, best practices, testing, performance, and more.

---

## 1 • Project Context

- **Framework:** check `pyproject.toml` for each library used
- **Environment** setup with poetry
- usage of **PyLance** for lint

---

## 2 • Golden Rules

1. Document every method with full **DOCString** (NumPy Style).
2. Be complient with Pylance
3. Avoid **magic numbers** and strings: always use named constants or enums.
4. Use **explicit functions** (pure and reusable).
5. Avoid shortcuts: no `i`, `m`, etc. in anonymous functions or methods.

---

## 3 • Code Standards

### ✏️ Style & Naming

- follow the `pyproject.toml` configuration
- use strict types
- delete unused imports
- Use camelCase for variables, functions, and signals
- Use PascalCase for classes, and types
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

## 4 • files Structure

try to folow the project structure

```
.
├── doc
├── src
│   ├── cli
│   │   ├── [first client]
│   │   ├── [seconde client]
│   │   └── [third client]
│   ├── configs
│   ├── features
│   │   ├── [features for first client]
│   │   ├── [features for seconde client]
│   │   └── [Features for third client]
│   └── shared
│       ├── constants
│       ├── model
│       └── types
└── test //testings files extract from the dataset
```

Inside features, each Clis got there specific Code here,
if the an other Cli needs it you have to move the fonction inside shared

### dataset and trained model

dataset-root: "~/dataset_preprocessed"
model_clip_dir: "src/configs/output/clip"
model_generation_dir: "src/configs/output/generation"

---

## 5 • Testing & Quality

📌 **Test Structure & Placement**

- Every `test_*.py` file must be **next to** the source file it tests (same folder).

if you need to run a python commande use the virtual env in `~/.venv/`
---

## 6 • Documentation

Sphinx documentation must be generated
1. global readme about the project 
2. a page for the cli
3. a page for each feature, where you can explain what happen (especialy math)

---
