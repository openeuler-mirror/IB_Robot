---
name: bq3588-oh-rknn
description: "Runs RKNN-based IB_Robot inference on the Bearkey BQ3588HM OpenHarmony board. Use when user wants to 'BQ3588HM RKNN', 'OpenHarmony RKNN', 'cloud_inference.launch.py', 'device:=rknn', '启动板端 RKNN 推理', '开发板 NPU 推理', or validate /data/ibrobot/models/502000/pretrained_model with ros2 launch inference_service cloud_inference.launch.py."
---

# BQ3588HM OpenHarmony RKNN Runtime Skill

Use this skill when the user wants to **actually run** the RKNN inference path on the BQ3588HM board, not just inspect board facts.

Use `ibrobot-hdc` for transport, `ibrobot-bq3588hm-oh` for board facts, and this skill for the verified **runtime overlay + launch command**.

## Verified Launch Target

- Board: Bearkey BQ3588HM
- Transport: `hdc -t <board-ip>:8710` (example form; ask the user for the actual target)
- Deployed install root: `/data/ibrobot/install`
- Deployed policy path: `/data/ibrobot/models/502000/pretrained_model`
- RKNN model file: `/data/ibrobot/models/502000/pretrained_model/model.rknn`

## What Must Be Installed or Deployed First

The launch command assumes these components already exist on the board:

1. official OpenHarmony ROS runtime under `/data/install` and `/data/out`
2. `/data/ros2ohos.env`
3. `rknnlite` under `/sys_prod/robot/out/lib/python3.12/site-packages`
4. `/usr/lib/librknnrt.so`
5. `skh-run` torch runtime under `/data/local/skh-run/usr`
6. deployed IB_Robot install under `/data/ibrobot/install`
7. deployed policy directory under `/data/ibrobot/models/502000/pretrained_model`

If the user asks "what needs to be installed", this is the minimum verified list.

## Host-Side Deployment Pattern

When the board needs to be prepared from scratch, the verified deployment flow is:

1. host cross-builds the OpenHarmony install tree
2. host packages `install/` as `ibrobot-oh-install.tar.gz`
3. host uploads the tarball to `/data`
4. board extracts it to `/data/ibrobot/install`
5. host uploads the policy directory containing `model.rknn`
6. board extracts it to `/data/ibrobot/models/502000/pretrained_model`

Typical host-side transfer pattern:

```bash
HDC_TARGET=<board-ip>:8710
hdc -t "$HDC_TARGET" file send ibrobot-oh-install.tar.gz /data/ibrobot-oh-install.tar.gz
hdc -t "$HDC_TARGET" file send ibrobot-models-502000.tar.gz /data/ibrobot-models-502000.tar.gz
hdc -t "$HDC_TARGET" shell 'mkdir -p /data/ibrobot && tar -zxpf /data/ibrobot-oh-install.tar.gz -C /data/ibrobot && tar -zxpf /data/ibrobot-models-502000.tar.gz -C /data/ibrobot'
```

If `skh-run` is missing, deploy that runtime payload before launch as well.

Before using this skill, ask the user for the actual TCP target IP or confirm that USB HDC will be
used directly. Do not assume a fixed board IP. Prefer TCP, but if TCP is not enabled yet, first
use USB HDC to run `hdc tmode port 8710`, then switch to `hdc tconn <board-ip>:8710`.

## Why a Dedicated Skill Is Needed

The RKNN cloud inference path needs **three runtime layers at the same time**:

1. official OpenHarmony ROS runtime (`/data/ros2ohos.env`)
2. deployed IB_Robot install (`/data/ibrobot/install`)
3. board-local overlays for:
   - `rknnlite` from `/sys_prod/robot/out/lib/python3.12/site-packages`
   - `torch` from `/data/local/skh-run/usr/lib/python3.12/site-packages`

If any one of these is missing, the launch will fail or partially start.

## Verified Launch Sequence

Run the following in **one shell session** on the board:

```sh
cd /data
. ./ros2ohos.env

cd /data/ibrobot
. install/setup.sh

export ROS_DOMAIN_ID=42
export PYTHONHOME=/data/local/skh-run/usr
export PATH=${PYTHONHOME}/bin:$PATH
export PYTHONPATH=/data/ibrobot/install/lerobot/src:/data/ibrobot/install/inference_service/lib/python3.12/site-packages:/data/ibrobot/install/tensormsg/lib/python3.12/site-packages:/data/ibrobot/install/robot_config/lib/python3.12/site-packages:/data/ibrobot/install/ibrobot_msgs/lib/python3.12/site-packages:${PYTHONHOME}/lib/python3.12/site-packages:/sys_prod/robot/out/lib/python3.12/site-packages:/data/install/lib/python3.12/site-packages
export LD_LIBRARY_PATH=${PYTHONHOME}/lib:${PYTHONHOME}/lib/python3.12/site-packages/torch/lib:${PYTHONHOME}/lib/python3.12/site-packages/torchaudio/lib:/sys_prod/robot/out/lib:/data/install/lib:/vendor/lib64
export LD_PRELOAD=${PYTHONHOME}/lib/libpython3.12.so.1.0:${PYTHONHOME}/lib/libomp.so

ros2 launch inference_service cloud_inference.launch.py \
    policy_path:=/data/ibrobot/models/502000/pretrained_model \
    device:=rknn
```

### Why this environment is required

- `ros2ohos.env` provides the official ROS/OpenHarmony runtime
- `. install/setup.sh` layers in deployed IB_Robot packages
- `PYTHONHOME=/data/local/skh-run/usr` gives the board-side torch runtime
- `PYTHONPATH` must include both `skh-run` and `rknnlite`
- `LD_LIBRARY_PATH` must include torch libs, torchaudio libs, ROS libs, and vendor libs
- `LD_PRELOAD` must expose both `libpython3.12.so.1.0` and `libomp.so`

## Verified Expected Signals

On successful startup, the verified logs include:

- `Loading policy from /data/ibrobot/models/502000/pretrained_model with inference_backend=rknn`
- `Engine loaded: rknn, chunk_size=100`
- `PureInferenceNode ready: input=/preprocessed/batch, output=/inference/action`
- `Waiting for preprocessed batches from edge node...`

## Fast Smoke Tests

### 1. Direct RKNN smoke test

This confirms the board NPU runtime itself works:

```sh
export PYTHONPATH=${PYTHONHOME}/lib/python3.12/site-packages:/sys_prod/robot/out/lib/python3.12/site-packages:$PYTHONPATH
python3 - <<'PY'
import numpy as np
from rknnlite.api import RKNNLite
rknn = RKNNLite()
rknn.load_rknn("/data/ibrobot/models/502000/pretrained_model/model.rknn")
rknn.init_runtime(target=None)
out = rknn.inference(inputs=[
    np.random.randn(1, 6).astype(np.float32),
    np.random.randn(1, 3, 480, 640).astype(np.float32),
    np.random.randn(1, 3, 480, 640).astype(np.float32),
])[0]
print(out.shape)
rknn.release()
PY
```

Verified output shape: `(1, 100, 6)`

### 2. Launch smoke test with timeout

```sh
timeout 20s ros2 launch inference_service cloud_inference.launch.py \
    policy_path:=/data/ibrobot/models/502000/pretrained_model \
    device:=rknn
```

This is useful when you only want to confirm the node loads and enters the waiting state.

## Important Runtime Facts

### Input Count for the Verified ACT RKNN Model

The currently verified RKNN model takes **3 inputs**, in this order:

1. `observation.state`
2. `observation.images.top`
3. `observation.images.wrist`

Do **not** feed a fourth `observation.current` tensor when doing direct RKNN tests.

### NumPy Ordering Matters

`torch` on this board must see the `skh-run` NumPy before the system NumPy:

- good: `${PYTHONHOME}/lib/python3.12/site-packages` before `/sys_prod/robot/out/lib/python3.12/site-packages`
- bad: system NumPy first, which causes `torch.from_numpy` / wrapper code paths to fail

## Troubleshooting

### `ModuleNotFoundError: No module named 'rknnlite'`

Cause: `/sys_prod/robot/out/lib/python3.12/site-packages` is missing from `PYTHONPATH`

Fix: append that directory explicitly

### `RuntimeError: Numpy is not available`

Cause: `torch` saw the system NumPy 2.4.0 instead of the `skh-run` NumPy 1.26.4

Fix: make `${PYTHONHOME}/lib/python3.12/site-packages` come before `/sys_prod/robot/out/lib/python3.12/site-packages`

### `_ctypes ... symbol not found`

Cause: using `skh-run` Python without setting `PYTHONHOME=/data/local/skh-run/usr`

Fix: export `PYTHONHOME` before invoking `${PYTHONHOME}/bin/python3`

### `Query dynamic range failed ... static shape`

Cause: normal warning for this static-shape RKNN model

Fix: can be ignored for current validation

## Scope Boundary

Use this skill for:

- running `device:=rknn` cloud inference on BQ3588HM
- launching `cloud_inference.launch.py` on the board
- validating the deployed policy path under `/data/ibrobot/models/502000/pretrained_model`
- layering `ros2ohos.env` + deployed install + `skh-run` + `rknnlite`

Do **not** use this skill for:

- HDC connect/send/recv logic (`ibrobot-hdc`)
- host-side RKNN export (`rknn-convert`)
- generic board facts or ROS bootstrap only (`ibrobot-bq3588hm-oh`)
