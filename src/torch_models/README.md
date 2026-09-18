# torch_models

IB-Robot 自研 PyTorch 模型的源码包。使用 `ament_python` 随工作区构建、安装，
模型本身不依赖 ROS API，不包含节点、消息转换或推理服务。

## 放置模型

每个模型独占 `src/torch_models/torch_models/<model_name>/`，相关网络、组件和配置类
都放在自己的目录内，并通过该目录的 `__init__.py` 导出公开模型类。例如：

```text
src/torch_models/
|-- package.xml
|-- setup.py
|-- setup.cfg
|-- resource/torch_models
|-- README.md
`-- torch_models/
    |-- __init__.py
    `-- demo_torch_model/
        |-- __init__.py
        `-- demo_torch_model.py
```

新增模型时照此创建一个带 `__init__.py` 的子目录，`setup.py` 会自动发现并安装它。
不要在顶层 `torch_models/__init__.py` 导入所有模型，以免导入包时加载不需要的依赖。
权重和部署配置不放在源码包中，仍放在 `models/<bundle>/`；设备选择、权重加载、
生命周期管理和 ROS 输入输出由调用方负责。

## 构建与引用

先通过仓库现有 `scripts/setup.sh` 准备环境。Torch 是 `setup.py` 声明的 Python 依赖，
具体版本和 CPU/CUDA/NPU 变体沿用项目的平台安装流程；`colcon build` 不负责安装 pip 依赖。
在工作区根目录执行（git worktree 先按 `ibrobot-worktree-env` 配置环境）：

```bash
source .shrc_local && ./scripts/build.sh -- --packages-select torch_models
source .shrc_local
```

其他模块可以直接导入。以轻量模型为例：

```python
import torch
from torch_models.demo_torch_model import DemoTorchModel

model = DemoTorchModel().eval()
with torch.inference_mode():
    output = model(torch.ones(1, 4))

assert output.shape == (1, 2)
assert torch.isfinite(output).all()
```

消费者真正接入时，在它的 `package.xml` 中声明运行依赖：

```xml
<exec_depend>torch_models</exec_depend>
```

Python 消费者还应将 `torch_models` 加入自己的 `setup.py` 的 `install_requires`。
这样 colcon 能按依赖顺序构建，Python 打包元数据也保持一致。模型 provider 通过稳定的
`(model_type, backend, device)` 组合注册，实际服务接入复用现有 `ModelSession` 和统一
推理运行时。

## 模型索引

| 模型 | 目录 | 运行入口 |
|---|---|---|
| DemoTorchModel | `torch_models/demo_torch_model` | CPU smoke example |
| [PI0.5 Ascend310P](torch_models/pi05_ascend_310p/README.md) | `torch_models/pi05_ascend_310p` | `model_type=pi05` + `backend=torch` + `device=npu` |

每个模型目录负责自己的配置校验、运行时准备和平台约束。统一推理服务只按稳定的
`model_type/backend/device` 组合解析 provider，不在通用 manifest exporter 中维护模型特判。

## Provider 扩展

1. 在模型目录新增 `provider.py`，导出 `create_provider() -> PolicyProvider`。
2. 在 `torch_models/policy_provider.py` 的 `_PROVIDERS` 中增加一条
   `(model_type, backend, device): "torch_models.<model_name>.provider"` 映射。
   模块仅在匹配该组合时导入；不匹配时继续使用 LeRobot factory，匹配后加载失败不会静默回退。
3. 通过 `PolicyProvider` 的 hooks 提供模型行为，不修改通用 session 或 manifest exporter：

| 字段 | 职责 |
|---|---|
| `policy_class` | 实现 LeRobot policy 接口的本地模型类 |
| `configure_config(config, model_dtype=...)` | 加载权重前配置模型并检查固定契约 |
| `validate(config=..., bundle_root=..., tokenizer_path=..., device_name=...)` | 模型内部的平台与依赖兼容检查 |
| `load_options` | 该模型特有的 `from_pretrained` 参数 |
| `prepare(policy=..., deployment_fingerprint=..., torch_module=..., torch_npu_module=..., device_name=...)` | 模型放置设备并转换 dtype 后准备推理优化 |
| `execution_metadata(policy)` | 可选的模型诊断 metadata，默认无 |

`ModelSession` 继续管理权重加载、设备放置、processor、取消和资源释放。Provider 不依赖
ROS，也不导入 `inference_service`。模型源码、兼容矩阵和使用说明归属模型目录。
新增模型时应覆盖 lazy 解析、未匹配路由、加载失败和模型特有约束，并在相应设备上验证。

## 轻量验证

`DemoTorchModel` 仅包含一层 `nn.Linear(4, 2)`，使用随机初始化参数，无需权重文件、
下载、训练或 GPU。构建后可在其他模块的目录下验证正常导入和一次 CPU 前向计算：

```bash
# 从工作区根目录执行。
source .shrc_local
(
    cd src/inference_service
    python3 - <<'PY'
import torch
from torch_models.demo_torch_model import DemoTorchModel

model = DemoTorchModel().eval()
with torch.inference_mode():
    output = model(torch.ones(1, 4))

assert output.shape == (1, 2)
assert output.device.type == "cpu"
assert torch.isfinite(output).all()
print("demo_torch_model import and CPU forward: PASS")
PY
)
```

无需手动添加 `PYTHONPATH` 或修改 `sys.path`，导入路径由工作区安装环境提供。
