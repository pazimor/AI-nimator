# ONNX Export and Controller Bundle — Design Rules (§2.10 / A7 / B0)

Exportability to ONNX is a **design constraint**, not a performance goal
(ROADMAP §2.10).  The target is Apple Neural Engine via ONNX Runtime
CoreML Execution Provider.  This document lists what is forbidden in any
`forward()` / `encode()` path so future contributors know the contract,
and documents the **controller bundle** format introduced in A7/B0.

---

## What is exported

| Module | Exported graph | CLI sub-command |
|---|---|---|
| `CustomTextEncoder` | Full `encode()` path (inputIds → hiddenStates + mask) | `export_onnx encoder` |
| `MotionDenoiserV2` | **One denoising step** (noisy motion + timestep + text → clean) | `export_onnx denoiser` |
| `MotionController` (Goal A) | **One frame forward** (state window + control [+ phase] → Δstate) | `export_onnx controller` |
| **Controller bundle** (A7/B0) | Full artefact set for the engine plugin | `export_onnx bundle` |

---

## Controller bundle (A7 / Goal B0)

A **controller bundle** is the complete artefact set a plugin needs to
run the controller in production without any Python dependency:

```
<bundle_dir>/
  controller.onnx          — one-frame forward ONNX graph
  norm_stats.json          — state / delta / control z-norm statistics
  manifest.json            — frozen I/O contract (ROADMAP_DETERMINIST §2.2)
  resolved_config.yaml     — provenance (config + git SHA + date) [optional]
  presets/
    walk.json              — example preset: {vx: 0.05, vz: 0.0}
    run.json               — example preset: {vx: 0.15, vz: 0.0}
```

### Assembling a bundle

```bash
python -m ainimator.cli.export_onnx bundle \
    --checkpoint output/controller/controller_overfit_checkpoint.pt \
    --output-dir output/controller_bundle \
    --resolved-config output/controller/resolved_config.yaml
```

### manifest.json format

`manifest.json` **serialises** the frozen I/O contract of
`ROADMAP_DETERMINIST §2.2`.  It is NOT a second definition — it is a
machine-readable snapshot the engine reads at load time to verify
compatibility:

```json
{
  "bundle_version": "A7.0",
  "state_channels": 136,
  "num_bones": 22,
  "rotation_channels_per_bone": 6,
  "root_local_motion_channels": 4,
  "control_channels": 2,
  "control_layout": ["vx", "vz"],
  "phase_channels": 0,
  "prompt_emb_channels": 0,
  "context_frames": 1,
  "output_layout": "bone_delta|global_delta",
  "coord_system": "Y-up right-handed",
  "normalization_note": "vx,vz are z-normalized; aim_x,aim_z are unit-norm",
  "reserved_input_groups": ["interaction","perception","reaction","morphology"]
}
```

### norm_stats.json format

```json
{
  "state":   {"bone_mean": [...], "bone_std": [...], "global_mean": [...], "global_std": [...]},
  "delta":   {"bone_mean": [...], "bone_std": [...], "global_mean": [...], "global_std": [...]},
  "control": {"mean": [...], "std": [...], "channels": ["vx","vz"],
              "note": "aim_x, aim_z are unit-norm by construction; no z-norm stats stored for them."}
}
```

The engine uses `norm_stats.json` to:
1. z-normalize the control input before the ONNX forward.
2. denormalize the `Δstate` output after the ONNX forward.

`aim_x / aim_z` are **not** in `norm_stats.json` — they are unit-norm
by construction and must not be z-normalized.

### Preset files

Optional JSON files under `presets/` carry named control presets for
the editor (Unity / Unreal) UI.  Format is free-form; conventionally:

```json
{"vx": 0.05, "vz": 0.0, "description": "default walk"}
```

---

### Controller export (Goal A, phase A5)

The deterministic controller is the engine ONNX export was *designed*
for (ROADMAP_DETERMINIST §2.1 truth #10): a single forward per frame, no
internal schedule, no debruitage loop.  The exported graph is exactly one
`MotionController.forward`:

| Input | Shape | Notes |
|---|---|---|
| `bone_window` | `(B, K, 22, 6)` | last `K = contextFrames` rotation6d frames |
| `control` | `(B, controlChannels)` | planar velocity [+ aim direction] |
| `global_window` | `(B, K, 3)` | last `K` root_translation frames |
| `phase` | `(B, phaseChannels)` | present only when `phaseMode != none` |

Outputs `bone_delta (B, 22, 6)` and `global_delta (B, 3)` — the
**normalized** next-frame deltas.  De/normalization (state + delta
stats) and the autoregressive accumulation `state += Δ` happen in the
engine (C#/C++ — Unity Sentis / Unreal NNE), **outside** the graph,
exactly like the DDIM loop for diffusion.  Foot-lock IK and physics
blending are post-process, also engine-side.

Dynamic axes: `batch` and `context` (the window length `K`).  The
positional-encoding buffer is sized to `maxFrames`, so a longer context
window does not require re-exporting.

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
| `context` | Controller context-window length `K` |

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
