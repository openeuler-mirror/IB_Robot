# BQ3588HM 开发板 RKNN NPU 推理指南

本文档介绍完整的端到端流程：将训练好的 ACT 策略模型转换为 RKNN 格式，并在 Bearkey BQ3588HM (RK3588) OpenHarmony 开发板上通过 NPU 运行推理。

## 前置条件

- 主机：Ubuntu 22.04，Python 3.10+
- 开发板：Bearkey BQ3588HM，OpenHarmony 系统，通过 HDC over TCP 访问
- HDC 工具路径：`/home/xqw/Research/oh_sdk/toolchains/hdc`
- 开发板网络地址：`192.168.136.111:8710`（根据实际网络调整）
- 已训练好的模型 checkpoint（如 `models/502000/`）

## 1. 在主机上将 ONNX 转换为 RKNN

### 1.1 创建专用 RKNN 虚拟环境

rknn-toolkit2 要求 `torch<=2.4.0` + `numpy<=1.26.4`，与 lerobot 的 `torch>=2.7` + `numpy>=2.0` 冲突，因此需要单独的虚拟环境：

```bash
python3 -m venv .venv-rknn
source .venv-rknn/bin/activate
pip install rknn-toolkit2==2.3.2
```

### 1.2 导出 ONNX 并转换为 RKNN

使用项目自带的导出脚本：

```bash
# 在项目根目录下
source .venv-rknn/bin/activate
python src/model_utils/model_utils/export_onnx_rknn.py \
    --onnx models/502000/act_ros2_rknn.onnx \
    --output models/502000/act_ros2_rknn.rknn \
    --dtype float16
```

也可以直接从 policy checkpoint 转换：

```bash
python src/model_utils/model_utils/export_onnx_rknn.py \
    --policy_path models/502000 \
    --output models/502000/act_ros2_rknn.rknn \
    --dtype float16
```

脚本会自动处理 onnx>=1.16 中 `onnx.mapping` 的兼容性问题。

**转换结果：** `act_ros2_rknn.rknn`（约 114 MB，float16），适用于包含两个 480x640 相机输入和 14 维状态向量的 ACT 模型。

## 2. 开发板环境配置

### 2.1 板端 Python 环境信息

| 项目 | 值 |
|------|------|
| Python 版本 | 3.12（musl libc） |
| 扩展模块后缀 | `.cpython-312-aarch64-linux-ohos.so` |
| site-packages 路径 | `/sys_prod/robot/out/lib/python3.12/site-packages/` |
| libpython 路径 | `/sys_prod/robot/out/lib/libpython3.12.so.1.0` |
| RKNN 运行时 | `/vendor/lib64/librknnrt.so`（v2.4.1b0） |
| rknnlite | 已预装在 site-packages 中 |

### 2.2 修复 rknnlite .so 文件后缀不匹配

rknnlite 预装的 `.so` 文件使用 `linux-gnu` 后缀，但板端 Python 期望 `linux-ohos` 后缀。需要逐个目录重命名：

```bash
HDC_BIN=/path/to/hdc
HDC_TARGET=<开发板IP>:8710

# api/ 目录
"$HDC_BIN" -t "$HDC_TARGET" shell 'for f in /sys_prod/robot/out/lib/python3.12/site-packages/rknnlite/api/*.cpython-312-aarch64-linux-gnu.so; do new="${f%-gnu.so}-ohos.so"; cp "$f" "$new"; done'

# api/npu_config/ 目录
"$HDC_BIN" -t "$HDC_TARGET" shell 'for f in /sys_prod/robot/out/lib/python3.12/site-packages/rknnlite/api/npu_config/*.cpython-312-aarch64-linux-gnu.so; do new="${f%-gnu.so}-ohos.so"; cp "$f" "$new"; done'

# utils/ 目录
"$HDC_BIN" -t "$HDC_TARGET" shell 'for f in /sys_prod/robot/out/lib/python3.12/site-packages/rknnlite/utils/*.cpython-312-aarch64-linux-gnu.so; do new="${f%-gnu.so}-ohos.so"; cp "$f" "$new"; done'
```

### 2.3 将 librknnrt.so 放到 rknnlite 查找的路径

rknnlite 会在 `/usr/lib/` 中查找 `librknnrt.so`。根文件系统默认只读，需先重新挂载：

```bash
"$HDC_BIN" -t "$HDC_TARGET" shell 'mount -o rw,remount /'
"$HDC_BIN" -t "$HDC_TARGET" shell 'mkdir -p /usr/lib'
"$HDC_BIN" -t "$HDC_TARGET" shell 'cp /vendor/lib64/librknnrt.so /usr/lib/'
```

> **注意：** 此修改在下次烧录固件前一直有效。如果重启后 `/usr/lib` 被还原，需要重新执行 `cp` 命令。

### 2.4 设置 LD_PRELOAD 解决 Python 符号可见性问题

板端 Python 动态链接了 `libpython3.12.so`，但 musl 的动态链接器不会自动将这些符号暴露给 `dlopen` 加载的扩展模块。需要通过 `LD_PRELOAD` 预加载：

```bash
export LD_PRELOAD=/sys_prod/robot/out/lib/libpython3.12.so.1.0
```

每次使用 rknnlite 运行 Python 前都必须设置此环境变量。

## 3. 部署模型并运行推理

### 3.1 推送 RKNN 模型到开发板

```bash
"$HDC_BIN" -t "$HDC_TARGET" file send models/502000/act_ros2_rknn.rknn /data/local/tmp/act_ros2_rknn.rknn
```

### 3.2 运行推理

```python
import numpy as np
import time

from rknnlite.api import RKNNLite

rknn = RKNNLite()
rknn.load_rknn("/data/local/tmp/act_ros2_rknn.rknn")
rknn.init_runtime(target=None)  # None = 使用本机 NPU

# 准备输入数据 — 注意：RKNN 会重排输入顺序为 [state, cam_high, cam_left]
state = np.random.randn(1, 14).astype(np.float32)        # 2D
cam_high = np.random.randn(1, 3, 480, 640).astype(np.float32)  # 4D NCHW
cam_left = np.random.randn(1, 3, 480, 640).astype(np.float32)  # 4D NCHW

t0 = time.time()
outputs = rknn.inference(inputs=[state, cam_high, cam_left])
t1 = time.time()

print(f"输出 shape: {outputs[0].shape}")  # (1, 100, 6)
print(f"推理耗时: {t1 - t0:.3f}s")        # RK3588 NPU 上约 121ms

rknn.release()
```

快速验证一行命令：

```bash
"$HDC_BIN" -t "$HDC_TARGET" shell 'LD_PRELOAD=/sys_prod/robot/out/lib/libpython3.12.so.1.0 python3 -c "
import numpy as np, time
from rknnlite.api import RKNNLite
rknn = RKNNLite()
rknn.load_rknn(\"/data/local/tmp/act_ros2_rknn.rknn\")
rknn.init_runtime(target=None)
state = np.random.randn(1, 14).astype(np.float32)
cam_high = np.random.randn(1, 3, 480, 640).astype(np.float32)
cam_left = np.random.randn(1, 3, 480, 640).astype(np.float32)
t0 = time.time()
outputs = rknn.inference(inputs=[state, cam_high, cam_left])
print(f\"output: shape={outputs[0].shape}, time={time.time()-t0:.3f}s\")
rknn.release()
"'
```

## 4. 关键技术说明

### 输入顺序

RKNN 编译器可能会重排模型输入。ACT 模型的原始 ONNX 输入为 `[cam_high, cam_left, state]`，转换后的 RKNN 模型期望 `[state, cam_high, cam_left]`。如果重新导出模型，务必通过测试推理验证输入顺序。

### 性能数据

| 指标 | 数值 |
|------|------|
| 模型大小（float16） | 约 114 MB |
| NPU 推理延迟 | 约 121 ms |
| 输出 shape | `(1, 100, 6)` — 100 个 action chunk × 6 自由度 |

### 版本兼容性

| 组件 | 版本 |
|------|------|
| rknn-toolkit2（主机，转换用） | 2.3.2 |
| rknn-toolkit-lite2（板端，推理用） | 2.3.2（预装） |
| librknnrt.so（板端） | 2.4.1b0 |
| RKNN 驱动 | 0.9.5 |

### 常见问题排查

| 现象 | 原因 | 解决方法 |
|------|------|----------|
| `ImportError: symbol not found (PyUnicode_FromFormat)` | Python 符号未暴露给 dlopen | 设置 `LD_PRELOAD=/sys_prod/robot/out/lib/libpython3.12.so.1.0` |
| `ModuleNotFoundError: No module named 'rknnlite.xxx'` | `.so` 文件后缀不匹配 | 重命名 `-gnu.so` 为 `-ohos.so`（见 2.2 节） |
| `Can not find dynamic library on RK3588!` | `/usr/lib/` 中没有 `librknnrt.so` | 从 `/vendor/lib64/` 复制（见 2.3 节） |
| `input[0] need 2dims input, but 4dims` | 输入顺序不匹配 | RKNN 重排了输入，state 需放在最前面（见 3.2 节） |
| `RKNN_ERR_MODEL_INVALID` 动态范围查询报错 | 静态 shape 模型的警告 | 可以安全忽略 |

## 5. 相关文件

| 文件 | 说明 |
|------|------|
| `models/502000/act_ros2_rknn.rknn` | RKNN 模型（float16，114 MB） |
| `models/502000/act_ros2_rknn.onnx` | RKNN 优化后的 ONNX（220 MB） |
| `src/model_utils/model_utils/export_onnx_rknn.py` | 导出脚本（ONNX → RKNN） |
| `.agents/skills/rknn-convert/convert_to_rknn.py` | 通用 ONNX → RKNN 转换器 |
