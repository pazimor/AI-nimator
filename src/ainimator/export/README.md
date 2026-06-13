# ONNX Export — Forbidden Ops and Design Rules (§2.10)

Exportability to ONNX is a **design constraint**, not a performance goal
(ROADMAP §2.10).  The target is Apple Neural Engine via ONNX Runtime
CoreML Execution Provider.  This document lists what is forbidden in any
`forward()` / `encode()` path so future contributors know the contract.

---

## What is exported

| Module | Exported graph | CLI sub-command |
|---|---|---|
| `CustomTextEncoder` | Full `encode()` path (inputIds → hiddenStates + mask) | `export_onnx encoder` |
| `MotionDenoiserV2` | **One denoising step** (noisy motion + timestep + text → clean) | `export_onnx denoiser` |

The **DDIM sampling loop** stays in Python (`ainimator.model.sampler_v2`).
It calls the denoiser repeatedly and orchestrates CFG.  It is NOT in the
exported graph and must NEVER be put there — the loop contains Python
control flow (iteration over timestep list, CFG conditional) that cannot
be traced to a static ONNX graph.

---

## Forbidden patterns (any `forward()` on the graph path)

### 1. Data-dependent control flow

```python
# FORBIDDEN — branch depends on a tensor value, not a fixed config flag
if x.sum() > 0:
    ...

# OK — branch depends on a Python constant or module config attribute
if self._config.useFilmConditioning:
    ...
```

Config flags are Python constants resolved at trace time.  Tensor-value
branches are baked into the graph at the traced branch — subsequent calls
will silently take the wrong path.

### 2. `.item()` on the graph path

```python
# FORBIDDEN — breaks the trace (synchronizes device to host)
length = mask.sum().item()
if length > 0:
    ...

# OK — use tensor ops that stay in-graph
count = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
```

`.item()` extracts a Python scalar from a tensor.  Inside a traced graph
this always returns the value seen during tracing, so the graph is
wrong for any other input.

### 3. Python lists / variable-length tensors in the graph

```python
# FORBIDDEN — Python list of tensors cannot be traced
outputs = [block(x) for block in self.blocks]
x = torch.stack(outputs)  # shape not statically known

# OK — nn.ModuleList with a plain for-loop (PyTorch traces this)
for block in self.blocks:
    x = block(x)
```

`nn.ModuleList` with a `for` loop is fine — the loop is unrolled at trace
time into a static sequence of ops.

### 4. None inputs at trace time

```python
# FORBIDDEN — tracing with mask=None bakes None-path into graph;
#             a later call with a real mask silently ignores it
def forward(self, x, mask=None):
    if mask is not None:
        x = x * mask

# OK — either always pass a mask (use zeros for "no mask") or create
#       an explicit export-wrapper that fixes the signature
```

All mask inputs to the exported wrappers are **concrete float32 tensors**
(`0.0` = real token, `1.0` = padding).  They are cast to `bool` inside
the wrapper before reaching the model.

### 5. Shape-dependent branches in forward

```python
# FORBIDDEN — shape check uses .item() implicitly
if x.shape[0] != self.expectedBatch:
    raise ValueError(...)  # only safe in __init__ / validate helpers

# OK — validation helpers are called from training/CLI, never inside
#       the graph path.  The exported wrapper skips _validateInputs.
```

All validation (`_validateInputs`) and runtime assertions in `forward()`
are safe because they run **before** the ONNX trace enters the model graph,
but watch for any new validate calls that call `.item()` on intermediate
tensors.

---

## Dynamic axes declared on all exported graphs

Both exports declare dynamic axes so the graph is not shape-locked:

| Axis name | Applies to |
|---|---|
| `batch` | All batch-size dimensions |
| `frames` | Motion sequence length (denoiser) |
| `text_len` | Text token sequence length (both) |

Re-exporting after adding a new input that has a fixed shape will silently
lock that dimension.  Always check that new inputs appear in the
`dynamicAxes` dict in `ainimator/export/onnx.py`.

---

## CI contract

The file `test/ainimator/export/test_onnx_parity.py` must stay green.
It exports both models, runs ONNXRuntime (CPU) vs PyTorch on identical
inputs, and asserts `max_abs_diff <= 1e-3`.

Any PR that causes `pytest -m onnx` to fail is rejected — it means a
`forward()` was modified in a way that is not ONNX-traceable or changes
numerical behaviour.

---

## What is out of scope (do not add without Pazimor's decision)

- Quantization (int8/fp16)
- CoreML Execution Provider activation on ANE
- Exporting the full DDIM sampler loop
- Exporting `forwardNull()` (the CFG unconditional path is assembled
  by the sampler in Python, outside the graph)
