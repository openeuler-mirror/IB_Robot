---
name: rknn-convert
description: "Convert ONNX models to RKNN format for RK3588 NPU deployment. Use when user needs to 'convert model', 'rknn', 'RK3588', 'NPU deploy', '模型转换', 'rknn转换', 'convert to rknn', 'onnx to rknn', 'deploy to rk3588', 'NPU 推理'. Triggers for model conversion, RKNN format, Rockchip NPU deployment."
---

# RKNN Model Conversion Skill

Convert ONNX models to RKNN format for RK3588 NPU deployment using rknn-toolkit2.

## Critical Workflow Split

For LeRobot `pretrained_model/` checkpoints, the verified workflow is **two-stage**:

1. **Main venv exports ONNX** from `pretrained_model`
2. **`.venv-rknn` converts ONNX -> RKNN**

Do **not** try to force the entire `pretrained_model -> RKNN` pipeline to stay inside `.venv-rknn`.
The exporter depends on the main workspace environment (`lerobot`, `.shrc_local`, workspace Python packages),
while RKNN conversion must stay isolated in `.venv-rknn`.

### Quick Decision Table

| Task | Environment |
|------|-------------|
| Export `pretrained_model` -> ONNX | main workspace `venv` via `source .shrc_local` |
| Convert ONNX -> RKNN | dedicated `.venv-rknn` only |
| Run board inference | board runtime (`rknnlite` + `skh-run`), not host venv |

## Environment

**IMPORTANT**: rknn-toolkit2 requires `torch<=2.4.0` and `numpy<=1.26.4`, which conflicts with lerobot's requirements (`torch>=2.7`, `numpy>=2.0`).

- **MUST NOT** install rknn-toolkit2 into the main venv
- **MUST** use the dedicated `.venv-rknn` for ONNX -> RKNN conversion
- **MUST NOT** `source .shrc_local` before running the conversion step in `.venv-rknn`, or the shell will activate the main `venv` and defeat isolation

### Dedicated venv

- Path: `<project_root>/.venv-rknn/`
- Python interpreter: `<project_root>/.venv-rknn/bin/python`
- Contains: `rknn-toolkit2`, `onnx`, `onnxruntime`, `torch==2.4.0`, `numpy==1.26.4`

#### How to activate `.venv-rknn`

If an interactive shell is preferred, activate it explicitly:

```bash
cd <project_root>
source .venv-rknn/bin/activate
python -V
pip list | grep -E 'rknn-toolkit2|onnx|onnxruntime'
```

When automation must avoid inheriting the main workspace environment, prefer calling the interpreter directly:

```bash
cd <project_root>
./.venv-rknn/bin/python <script.py> ...
```

Do **not** activate `.venv-rknn` after `source .shrc_local` in the same shell and assume everything is clean.
For conversion, the safe pattern is still a clean shell plus direct interpreter invocation.

#### Create venv (one-time setup)

```bash
cd <project_root>
python3 -m venv .venv-rknn
.venv-rknn/bin/pip install rknn-toolkit2 onnx onnxruntime
```

#### Auto-creation

If `.venv-rknn` does not exist when conversion is requested, the agent **MUST** create it first:

```bash
cd <project_root>
python3 -m venv .venv-rknn && .venv-rknn/bin/pip install rknn-toolkit2 onnx onnxruntime
```

### onnx.mapping compatibility patch

rknn-toolkit2 2.3.2 uses `onnx.mapping.TENSOR_TYPE_TO_NP_TYPE` and `onnx.mapping.NP_TYPE_TO_TENSOR_TYPE`, which were removed in onnx>=1.16. The conversion script includes a monkey-patch at import time.

## Verified Split Workflow for `pretrained_model`

### Step A: Export ONNX in the main workspace venv

This step needs `lerobot` and the workspace environment, so it must run in the main venv:

```bash
cd <project_root>
source .shrc_local && python3 src/model_utils/model_utils/export_onnx_rknn.py \
    --policy_path models/502000/pretrained_model
```

This produces:

- `models/502000/pretrained_model/act_ros2_rknn.onnx`

### Step B: Convert ONNX to RKNN in `.venv-rknn`

This step must stay isolated from the main venv:

```bash
cd <project_root>
env -i HOME="$HOME" PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    PYTHONPATH="$PWD/libs/lerobot/src:$PWD/src" PYTHONNOUSERSITE=1 \
    ./.venv-rknn/bin/python .agents/skills/rknn-convert/convert_to_rknn.py \
    --onnx models/502000/pretrained_model/act_ros2_rknn.onnx \
    --output models/502000/pretrained_model/model.rknn \
    --mode float16
```

This produces:

- `models/502000/pretrained_model/model.rknn`

### Relationship to the main venv

- main `venv`: owns `lerobot`, `.shrc_local`, workspace Python dependencies, ONNX export
- `.venv-rknn`: owns `rknn-toolkit2` and ONNX -> RKNN conversion only
- neither venv should be used as a substitute for the other

If the user says "export RKNN from a checkpoint", interpret that as:

1. main `venv` exports ONNX
2. `.venv-rknn` converts that ONNX to RKNN

### Verified Result for Current ACT Model

On the current `models/502000/pretrained_model` checkpoint, the verified output is:

- ONNX: `act_ros2_rknn.onnx` (~221 MB)
- RKNN: `model.rknn` (~121 MB)
- Output shape: `(1, 100, 6)`

## Conversion Script

The main script is at: `<project_root>/.agents/skills/rknn-convert/convert_to_rknn.py`

### Usage

```bash
# Always use the dedicated venv python
VENV_PYTHON="<project_root>/.venv-rknn/bin/python"

# Option 1: float16 (recommended for Transformer/ACT models)
$VENV_PYTHON <project_root>/.agents/skills/rknn-convert/convert_to_rknn.py \
    --onnx <onnx_model_path> \
    --output <output_rknn_path> \
    --mode float16

# Option 2: int8 quantization (smaller model, faster, but may lose accuracy)
$VENV_PYTHON <project_root>/.agents/skills/rknn-convert/convert_to_rknn.py \
    --onnx <onnx_model_path> \
    --output <output_rknn_path> \
    --mode int8

# Option 3: hybrid quantization (auto mix float16 + int8)
$VENV_PYTHON <project_root>/.agents/skills/rknn-convert/convert_to_rknn.py \
    --onnx <onnx_model_path> \
    --output <output_rknn_path> \
    --mode hybrid
```

### Conversion Modes

| Mode | Size | Speed | Accuracy | Use Case |
|------|------|-------|----------|----------|
| `float16` | ~50% of onnx | Good | Best | Transformer/ACT models (recommended) |
| `int8` | ~25% of onnx | Best | May degrade | CNN models with calibration data |
| `hybrid` | Varies | Good | Balanced | When int8 loses accuracy on some layers |

### Supported Platforms

- RK3588 (primary target, 6 TOPS NPU)
- RK3576, RK3566/RK3568, RK3562, RV1103/RV1106

## Agent Workflow

When user requests RKNN conversion:

### Step 1: Identify Source Model

Find the ONNX model in the project:
```bash
find <project_root>/models -name "*.onnx" -type f
```

If the source is a LeRobot `pretrained_model/` directory instead of an ONNX file, first export ONNX in the main workspace venv:

```bash
cd <project_root>
source .shrc_local && python3 src/model_utils/model_utils/export_onnx_rknn.py \
    --policy_path <pretrained_model_dir>
```

### Step 2: Inspect Model Structure

```bash
<project_root>/.venv-rknn/bin/python -c "
import onnx
model = onnx.load('<onnx_path>')
print('Inputs:')
for inp in model.graph.input:
    print(f'  {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}')
print('Outputs:')
for out in model.graph.output:
    print(f'  {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}')
print(f'Opset: {model.opset_import[0].version}')
"
```

### Step 3: Choose Conversion Mode

- **ACT / Transformer models**: Use `float16` (preserves accuracy)
- **CNN models** (ResNet, YOLO, etc.): Use `int8` with calibration data
- **Mixed architecture**: Use `hybrid`

### Step 4: Run Conversion

```bash
<project_root>/.venv-rknn/bin/python <project_root>/.agents/skills/rknn-convert/convert_to_rknn.py \
    --onnx <onnx_path> \
    --output <output_path> \
    --mode <mode>
```

### Step 5: Verify Output

```bash
ls -lh <output_path>.rknn
```

## Current Converted Models

| Model | Source | Output | Mode | Size |
|-------|--------|--------|------|------|
| ACT policy (502000, legacy) | `models/502000/act_ros2_simplified.onnx` | `models/502000/act_ros2_simplified.rknn` | float16 | ~115MB |
| ACT policy (`pretrained_model`, current) | `models/502000/pretrained_model/act_ros2_rknn.onnx` | `models/502000/pretrained_model/model.rknn` | float16 | ~121MB |

## Board-Side Deployment

On RK3588 (e.g., BQ3588HM board), use one of:

1. **rknn-toolkit-lite2** (Python): `pip install rknn-toolkit-lite2`
2. **rknn_runtime** (C API): Link against `librknnrt.so`

For the verified BQ3588HM OpenHarmony workflow in this repo, the board side is already modeled by:

- `ibrobot-bq3588hm-oh` for runtime facts and prerequisites
- `bq3588-oh-rknn` for the full deployment and launch overlay

Inference example (Python, on board):
```python
from rknnlite.api import RKNNLite
rknn = RKNNLite()
rknn.load_rknn('model.rknn')
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2_3)
outputs = rknn.inference(inputs=[state, hand_img, top_img])
rknn.release()
```

## Troubleshooting

### Issue: `onnx.mapping` AttributeError
**Cause**: onnx>=1.16 removed `onnx.mapping` module
**Fix**: Script includes monkey-patch. If missing, add:
```python
import onnx, types, sys
if not hasattr(onnx, "mapping"):
    _t2np = {k: v.np_dtype for k, v in onnx._mapping.TENSOR_TYPE_MAP.items()}
    _np2t = {v: k for k, v in _t2np.items()}
    _m = types.ModuleType("onnx.mapping")
    _m.TENSOR_TYPE_TO_NP_TYPE = _t2np
    _m.NP_TYPE_TO_TENSOR_TYPE = _np2t
    onnx.mapping = _m
    sys.modules["onnx.mapping"] = _m
```

### Issue: torch version conflict with lerobot
**Cause**: rknn-toolkit2 requires `torch<=2.4.0`, lerobot requires `torch>=2.7`
**Fix**: Always use `.venv-rknn` for conversion, never the main venv

### Issue: Export step fails inside `.venv-rknn`
**Cause**: The LeRobot exporter needs the main workspace environment, not the conversion-only `.venv-rknn`
**Fix**: Split the workflow:

1. main venv exports ONNX
2. `.venv-rknn` converts ONNX to RKNN

### Issue: Conversion step accidentally imports the main `venv`
**Cause**: Running `source .shrc_local` before the conversion step activates the workspace `venv`
**Fix**: For the conversion step, run `.venv-rknn` in a clean shell (for example with `env -i ...`) and only provide minimal `PYTHONPATH`

### Issue: BQ3588HM inference reports the wrong number of inputs
**Cause**: The verified `pretrained_model` export currently exposes **3 inputs**:

- `observation.state`
- `observation.images.top`
- `observation.images.wrist`

`observation.current` from `config.json` is not part of the exported RKNN input list.

**Fix**: Validate ONNX input order before board testing and feed exactly the exported inputs.

### Issue: NPU ops not supported
**Cause**: Some Transformer ops may fall back to CPU on RK3588
**Fix**: Check rknn build logs for "CPU" fallback warnings. Use `float16` mode which has better op coverage.

## When to Use This Skill

Invoke this skill when:
- Converting ONNX models to RKNN format
- Deploying models to RK3588 NPU
- Checking NPU compatibility of models
- Questions about rknn-toolkit2 usage

Do NOT invoke for:
- Training models (use lerobot skill)
- ONNX export (use onnx export workflow)
- Board connectivity issues (use ibrobot-hdc)
