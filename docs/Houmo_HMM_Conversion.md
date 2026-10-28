# Houmo HMM 模型转换与板端推理指南

本文说明如何把 PI0.5 或 SmolVLA 策略的 Houmo 编译产物打包为 IB-Robot 统一 HMM deployment，并部署到 Houmo LQ50 / M50（xh2）算力卡。

> **支持范围**：HMM backend 仅支持 PI0.5 和 SmolVLA。ACT HMM 已移除；ACT 应使用 `torch`、`ascend`、`hisilicon` 或 `rknn` deployment。

> **板端环境前提**：已安装 Houmo 1.3.0 OpenHarmony runtime，并使用 OpenHarmony 工具链交叉编译和安装对应的 Python binding。具体安装资料暂不对外提供。

## 架构概览

| 策略 | HMM 模块 | Host 侧辅助制品 | 设备指针链路 |
|------|----------|-----------------|--------------|
| **PI0.5** | vision、prefill、action_in_proj、time_mlp、decode、action_out_proj | `embedding.pt` | prefill output -> decode input |
| **SmolVLA** | vision、prefill、action | `token_embedding.pt`、`state_projection.pt` | prefill output -> action input |
| **ACT** | 不支持 | - | 启动时拒绝 |

运行时由 `inference_service.backends.hmm.HMMBackend` 按 manifest 的 `execution`、`bindings` 和 `device_links` 编排。Runtime 不扫描目录、不猜测文件名，也不读取 `config.hmm.json` 或 `HMM_MODEL_PATH`。

每个策略 bundle 根目录只使用一个 `inference_manifest.json`。LeRobot 的 `config.json`、processor 配置、processor 状态和 tokenizer 文件保持只读，所有 HMM deployment 元数据由 `package-hmm-deployment` 生成。

## 一、准备转换环境

Houmo 工具链的依赖可能与项目主环境冲突，建议使用 Houmo 官方 24.04 工具链镜像或独立 venv。不要把 `xhquant`、`tcim` 安装到 IB-Robot 主 venv。

```bash
# 官方镜像示例；具体镜像名以当前 Houmo SDK 发布为准
docker run -it --gpus all \
    -v "$PWD/models:/work/models" \
    harbor.houmo.ai/toolchain/release:Dadao-xh2-v1.3.0-ubuntu24.04-x86.64 bash
```

22.04 runtime 镜像通常只有 `houmo_tcim_runtime`，没有 `xhquant` 和 `tcim` 编译器，不能用于模型转换。

若使用独立 venv：

```bash
python3 -m venv .venv-hmm
source .venv-hmm/bin/activate
pip install xhquant tcim onnx onnxsim torch
```

## 二、生成 HMM 与 ABI 元数据

PI0.5 和 SmolVLA 使用 Houmo 官方 `houmo-examples-xh2` 导出、PTQ 和编译流程。每个传给 IB-Robot packager 的 `.hmm` 都必须配套编译器生成的 TCIM `model.json`，其中包含准确的 input/output 名称、顺序、dtype 和 shape。

### PI0.5

PI0.5 需要以下角色：

| Role | 示例制品 |
|------|----------|
| vision | `siglip.hmm` + 对应 `model.json` |
| prefill | `gemma_2b_prefill.hmm` + 对应 `model.json` |
| action_in_proj | `action_in_proj.hmm` + 对应 `model.json` |
| time_mlp | `time_mlp.hmm` + 对应 `model.json` |
| decode | `gemma_expert_300m_decode.hmm` + 对应 `model.json` |
| action_out_proj | `action_out_proj.hmm` + 对应 `model.json` |
| embedding | `embedding.pt` |

典型官方流程包含 vision、LLM、expert 和 action projection 的导出脚本，然后通过 `tcim.build_from_hmonnx` 生成 `.hmm` 与 ABI 元数据。自定义策略必须使用与实际 camera、chunk size、action dimension 和 tokenizer 长度一致的导出配置。

PI0.5 的 cache 链路由 packager 根据 prefill/decode ABI 自动生成。native prefill 输出和 decode
输入必须具有相同的交错 `past_key_N` / `past_value_N` 名称、dtype 和 shape，且以 device pointer 连接。

仓库提供不依赖 `xh_model_zoo` 的受管转换入口：

```bash
source .shrc_local
MODEL_BUNDLE_ROOT=models/pi05 \
./scripts/convert_hmm.sh pi05
```

`MODEL_BUNDLE_ROOT` 是必填 workspace 相对路径。输出默认为 `models/pi05_hmm_standard`，可通过
`PI05_HMM_OUTPUT` 覆盖；中间产物（ONNX、HMONNX、TCIM 编译缓存、calibration）写入
`models/_work/<输出名>/`（`PI05_HMM_WORK` 可覆盖），不会进入 bundle。为防止混入旧
HMONNX/TCIM 产物，输出路径已存在时脚本会拒绝运行。入口
严格加载当前 checkpoint 和 patched LeRobot，使用 `transformers==5.3.0` 导出 native PaliGemma
prefill 与 action-expert decode 图。容器中的 `torchao==0.17.0` 会在导入前卸载。work 目录中的
`provenance.json` 记录 checkpoint SHA-256、IB-Robot/LeRobot commit、镜像 ID、Transformers、
xhquant 和 TCIM 版本。`embedding.pt["weight"]` 必须与当前 checkpoint 的
`model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight` 完全一致。

### SmolVLA

SmolVLA 需要以下角色：

| Role | 示例制品 |
|------|----------|
| vision | `smolvla_vision.hmm` + 对应 `model.json` |
| prefill | `smolvla_llm_prefill.hmm` + 对应 `model.json` |
| action | `smolvla_action.hmm` + 对应 `model.json` |
| embedding | `token_embedding.pt` |
| state_projection | `state_projection.pt` |

SmolVLA 的主链是 `vision -> embedding -> prefill -> action`。当前 HMM runtime 不执行独立 decode 模块。`state_projection.pt` 必须包含与 state input 和 hidden size 匹配的 projection weight/bias。

仓库提供完整的标准转换入口：

```bash
source .shrc_local
MODEL_BUNDLE_ROOT=models/smolvla ./scripts/convert_hmm.sh smolvla
```

根目录只保留这一通用 HMM 转换入口；各策略的容器编排和导出实现放在对应的
`model_utils/<policy>_export/` 目录，新增策略时不再增加 `scripts/convert_<policy>_hmm.sh`。

`MODEL_BUNDLE_ROOT` 是必填的 workspace 相对路径，没有默认值。未设置、传入空值、绝对路径或
不存在的目录时，脚本会在启动容器前失败。输出默认为 `models/smolvla_hmm_standard`，可通过
`SMOLVLA_HMM_OUTPUT` 覆盖；ONNX、HMONNX、TCIM 编译缓存和 calibration 等中间产物写入
`models/_work/<输出名>/`（`SMOLVLA_HMM_WORK` 可覆盖），bundle 只保留 manifest 引用的制品
和 LeRobot 元数据。还可通过 `SMOLVLA_EXPORT_DEVICE` 和 `HOUMO_IMAGE` 覆盖其他参数。脚本会校验仓库维护的
LeRobot v0.6.0 patch 分支和 clean 状态，在 Houmo 1.3.0 容器中导出 ONNX、执行 xhquant PTQ、
调用 TCIM 编译三个 HMM，并通过 strict loader 生成和验证 `inference_manifest.json`。

标准流程优先使用 `transformers==5.3.0`，并在加载策略失败时尝试 `4.57.1`。容器预装的
`torchao==0.17.0` 与 torch 2.8.0 不兼容，入口脚本会在导入 LeRobot 前卸载它。导出使用
bundle 内的 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`，不依赖网络下载。

vision 图是固定 `512x512`、全 patch 有效的静态导出。Exporter 会将 SmolVLM 的布尔索引位置
编码替换为等价静态 position IDs，并拒绝仍含 `NonZero` 的 ONNX，避免 xhquant 编译失败。
action 图使用 CUDA float32，避免 denoise 分支在 fp16 tracing 时发生 dtype 混用；vision 和
prefill 在 CUDA 上使用 float16。PTQ calibration 文件直接保存对应 PyTorch dummy inputs，
其中 action KV cache 来自同次 prefill 前向。

work 目录还包含 `provenance.json`，记录 IB-Robot/LeRobot commit、checkpoint SHA-256、容器镜像、
依赖版本、导出 shape 和量化参数。该文件用于追溯转换来源，不替代 runtime manifest。

## 三、组装统一 Deployment

不要手写 `inference_manifest.json`。先准备一个仅包含编译器输出路径的 packaging spec，再运行 `package-hmm-deployment`。Packager 会：

- 从原始 LeRobot `config.json` 读取 policy family 和 feature metadata；
- 读取每个 TCIM `model.json` 的 runtime ABI；
- 校验 execution role、tensor name、dtype、shape 和 image layout；
- 复制外部制品到 bundle 下的受管目录；
- 生成完整 `bindings`、`device_links`、SHA-256 和 bundle digest；
- 写入或更新根目录唯一的 `inference_manifest.json`；
- 用 production strict loader 回读验证生成结果。

### PI0.5 Spec

```json
{
  "vision": {
    "artifact": "/compiler/pi05/siglip.hmm",
    "abi": "/compiler/pi05/siglip/model.json"
  },
  "embedding": "/compiler/pi05/embedding.pt",
  "roles": {
    "prefill": {
      "artifact": "/compiler/pi05/gemma_2b_prefill.hmm",
      "abi": "/compiler/pi05/prefill/model.json"
    },
    "action_in_proj": {
      "artifact": "/compiler/pi05/action_in_proj.hmm",
      "abi": "/compiler/pi05/action_in_proj/model.json"
    },
    "time_mlp": {
      "artifact": "/compiler/pi05/time_mlp.hmm",
      "abi": "/compiler/pi05/time_mlp/model.json"
    },
    "decode": {
      "artifact": "/compiler/pi05/gemma_expert_300m_decode.hmm",
      "abi": "/compiler/pi05/decode/model.json"
    },
    "action_out_proj": {
      "artifact": "/compiler/pi05/action_out_proj.hmm",
      "abi": "/compiler/pi05/action_out_proj/model.json"
    }
  },
  "vision_layout": "NCHW"
}
```

### SmolVLA Spec

```json
{
  "vision": {
    "artifact": "/compiler/smolvla/smolvla_vision.hmm",
    "abi": "/compiler/smolvla/vision/model.json"
  },
  "embedding": "/compiler/smolvla/token_embedding.pt",
  "state_projection": "/compiler/smolvla/state_projection.pt",
  "roles": {
    "prefill": {
      "artifact": "/compiler/smolvla/smolvla_llm_prefill.hmm",
      "abi": "/compiler/smolvla/prefill/model.json"
    },
    "action": {
      "artifact": "/compiler/smolvla/smolvla_action.hmm",
      "abi": "/compiler/smolvla/action/model.json"
    }
  },
  "vision_layout": "NCHW"
}
```

Spec 中的相对路径以 spec 文件所在目录为基准。`vision_layout` 只能是 `NCHW` 或 `NHWC`。

### 运行 Packager

项目命令执行前先加载 `.shrc_local`：

```bash
source .shrc_local

package-hmm-deployment \
    --bundle-root models/<policy_bundle> \
    --deployment hmm_lq50 \
    --target-soc lq50 \
    --target-runtime tcim \
    --spec /path/to/hmm-package-spec.json
```

也可从源码调用：

```bash
source .shrc_local
python3 -m model_utils.hmm_export \
    --bundle-root models/<policy_bundle> \
    --deployment hmm_lq50 \
    --target-soc lq50 \
    --target-runtime tcim \
    --spec /path/to/hmm-package-spec.json
```

输出路径是 `<bundle-root>/inference_manifest.json`。典型 bundle 结构如下：

```text
models/<policy_bundle>/
├── config.json
├── policy_preprocessor.json
├── policy_preprocessor_step_*.safetensors
├── policy_postprocessor.json
├── policy_postprocessor_step_*.safetensors
├── <本地 tokenizer/processor assets>
├── inference_manifest.json
└── artifacts/
    └── hmm/
        └── hmm_lq50/
            └── <packager 管理的 HMM/PT 制品>
```

`config.json` 或 processor JSON 引用的 tokenizer/processor 依赖必须已经 vendored 到 bundle 内；packager 不接受远程 semantic dependency。

## 四、配置 Named Pipeline

HMM 不再通过 `device: hmm` 或全局 model table 选择。直接在 control mode 下配置 named pipeline，并选择 manifest 中的 deployment：

```yaml
control_modes:
  model_inference:
    inference:
      enabled: true
      pipelines:
        policy:
          model_path: models/<policy_bundle>
          deployment: hmm_lq50
          execution_mode: monolithic
          request_timeout: 10.0
          default_task: "pick up the object"
    executor:
      type: topic
      mode: model_inference
      inference_pipeline: policy
      control_frequency: 20.0
```

相对 `model_path` 只相对于 `WORKSPACE` 解析。`WORKSPACE` 未设置时配置加载会失败，不会退回当前工作目录。

启动：

```bash
source .shrc_local
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=model_inference \
    use_sim:=true \
    sim_platform:=mock
```

## 五、板端部署

OpenHarmony 板端没有 systemd，rootfs 默认只读，且脚本应兼容 POSIX `sh`。部署前先阅读 `.agents/skills/oh-constraints/SKILL.md`。

安装 aarch64 runtime SDK 后，运行 ROS/HMM 推理前加载 RoboFrame 和 Houmo 环境：

```bash
. /data/roboframe/scripts/robooh_1.0.1.env
. /data/roboframe/scripts/setup/houmo_hmm_env.sh
```

若 release 中保留仓库目录布局，也可使用对应的 `scripts/setup/houmo_hmm_env.sh`。

关键变量应为：

```bash
TCIM_BACKEND=xh2
HOUMO_TARGET=xh2
```

不要设置 `HDPL_PLATFORM=ASIC`，否则 runtime 可能尝试加载 aarch64 SDK 中不存在的 `libhdplrt_asic.so`。

验证设备：

```bash
python3 -c "import tcim_lite.runtime as r; print('device_num:', r.get_device_num())"
```

预期至少返回一个可用设备。

## 六、校验与排查

### ACT 被拒绝

这是预期行为。HMM support matrix 只包含 PI0.5 和 SmolVLA。不要恢复单模块 ACT HMM wrapper、`config.hmm.json` 或 `HMM_MODEL_PATH` fallback。

### Packager 报 ABI 不匹配

确认传入的是每个 `.hmm` 对应的 compiler-emitted `model.json`，不要手写 tensor 顺序。重点检查：

- PI0.5 prefill cache output 与 decode cache input 名称、dtype、shape 一致；
- PI0.5 action/noise shape 为 `(1, chunk_size, max_action_dim)`；
- SmolVLA prefill cache output 与 action cache input 名称、dtype、shape 一致；
- vision 只有一个 image input 和一个 embedding output；
- image layout 与 compiler ABI 一致。

### Manifest hash mismatch

不要手改 SHA-256。重新运行 owning exporter 或 `package-hmm-deployment`，让工具重新打包制品并生成 digest。

### `device_num: 0` 或 `InitDevice failed`

检查 native Houmo 1.3 runtime 使用 `TCIM_BACKEND=xh2`、`HOUMO_TARGET=xh2`，并确认 `/data/local/houmo/lib` 和 `/data/local/houmo-sdk/lib` 已加入 library path，然后重新加载 `houmo_hmm_env.sh`。`Xh2HalBackend` 是 legacy runtime 的 selector，不适用于此 native OpenHarmony 部署。

### `set_input error: Status.UNINITIALIZED`

PI0.5 使用 prefill output -> decode input 的 device pointer；SmolVLA 使用 prefill output -> action input。不要交换 `get_dev_input` 和 `get_dev_output` 语义。该关系应来自 manifest 的 `device_links`，不应在部署脚本中硬编码。

### PTQ 精度下降或 NPU 超时

随机 calibration 数据通常不足以代表真实图像、状态和语言分布。生产模型应使用真实数据重新校准，并在目标板上执行数值对比。

## 参考

- HMM packager：`src/model_utils/model_utils/hmm_export.py`
- PI0.5 HMM exporter：`src/model_utils/model_utils/pi05_export/`
- 统一 manifest writer：`src/model_utils/model_utils/inference_manifest_export.py`
- HMM backend：`src/inference_service/inference_service/backends/hmm/backend.py`
- HMM backend tests：`src/inference_service/test/test_hmm_backend.py`
- HMM exporter tests：`src/model_utils/test/test_hmm_export.py`
- 已迁移 bundle：`models/pi05_hmm/`、`models/smolvla_hmm/`
- 板端环境：`scripts/setup/houmo_hmm_env.sh`
