# Model Utils

`model_utils` 提供 IB-Robot 模型导出、compiled artifact 打包、精度对比和数据集帧检查工具。
所有部署工具使用统一的 `inference_manifest.json`，并通过共享 writer 生成稳定 UUID、自动
revision、轻量结构摘要、execution roles 和 runtime tensor bindings。

## 统一 Bundle 规则

可部署策略目录必须包含 LeRobot 语义文件和唯一的 manifest：

```text
policy_bundle/
├── config.json
├── model.safetensors                         # Torch deployment 按需存在
├── policy_preprocessor.json
├── policy_preprocessor_step_*.safetensors
├── policy_postprocessor.json
├── policy_postprocessor_step_*.safetensors
├── tokenizer/                                # PI0.5 / SmolVLA 按需存在
├── artifacts/<backend>/<deployment>/...
└── inference_manifest.json
```

`config.json` 和 processor 文件由 LeRobot 拥有，工具只读。不要向其中写入 backend flags 或
artifact paths。运行时只从命名 deployment 选择后端，不扫描目录、不猜测文件名、不用环境
变量覆盖 artifact。

Exporter / packager 负责：

1. 读取 compiler/runtime ABI。
2. 为每个 runtime tensor 生成 semantic、name/index、dtype、shape 和 image layout binding。
3. 将运行时 artifact 固化到唯一 generation 目录，禁止原地覆盖。
4. 为 bundle/deployment 分配稳定 UUID，并在发布变更时自动增加 revision。
5. 仅根据 Manifest 结构计算轻量 bundle digest 和 deployment fingerprint。
6. 使用生产 strict loader 验证路径、metadata、bindings 和 ABI 契约。

不要手工编辑 identity、revision、digest 或 bindings。Runtime 不读取模型文件计算内容 SHA；正式
artifact 更新必须经过 packager。

导出器默认把可重建的 ONNX、compiler output 和 ABI metadata 放在 bundle 外的
`models/_work/<bundle>/<backend>/`。最终 manifest 只引用打包到
`artifacts/<backend>/<deployment>/` 的运行时 artifact。ABI JSON 用于生成 bindings，属于
构建输入或编译器输出，不是运行时 artifact；参数帮助中的 `ABI input` / `ABI output` 明确其方向。
发布或归档 bundle 时不得携带 `_work` 中间产物。

## 环境

项目和 ROS 命令前加载：

```bash
source .shrc_local
```

RKNN Toolkit 与主环境依赖可能冲突，使用独立 `.venv-rknn`。Ascend 工具需要可用的 `atc`
和 CANN 环境。HMM 工具需要 TCIM 产物和对应 `model.json` ABI。

## LeRobot Torch CPU / GPU

标准 LeRobot `save_pretrained()` 策略目录可直接生成原生 Torch deployment：

```bash
source .shrc_local

ros2 run model_utils package-torch-deployment \
    --bundle-root /path/to/policy_bundle
```

默认在同一个 `inference_manifest.json` 中生成 `torch-cpu` 和 `torch-cuda`。Torch deployment
不需要 compiled artifact、runtime tensor bindings、execution roles 或 device links，只声明
`backend: torch` 和目标 device。工具仍会自动发现策略 processor/tokenizer/VLM 本地资产，
校验 `model.safetensors` 存在且为安全普通文件，计算轻量 bundle digest，并用生产 loader 验证结果。

只生成 CPU deployment 或使用自定义名称前缀：

```bash
ros2 run model_utils package-torch-deployment \
    --bundle-root /path/to/policy_bundle \
    --devices cpu \
    --deployment-prefix native
```

上述命令生成 `native-cpu`。`--devices` 也支持 `cuda`、`mps` 和 `npu`；生成 manifest
不要求当前主机具备对应设备，实际加载 deployment 时 runtime 才检查设备可用性。

Repository-owned native Torch policies are selected by the stable combination
of manifest `model_type`, runtime backend, and runtime device. For PI0.5 on
Ascend310P, use the normal native Torch deployment:

```bash
ros2 run model_utils package-torch-deployment \
    --bundle-root /path/to/pi05_bundle \
    --devices npu
```

The PI0.5 provider is selected by `model_type: pi05`, `backend: torch`, and
`device: npu`; the generic packager does not contain model-specific branches.

## ACT Ascend Export

`export_onnx_atc.py` 导出 ACT ONNX、调用 ATC 生成 OM，并将 OM 打包为 `ascend`
deployment。最终打包必须提供 compiler/runtime introspected ABI JSON：

```bash
source .shrc_local

python3 src/model_utils/model_utils/export_onnx_atc.py \
    --pretrained_model /path/to/act_bundle \
    --soc_version Ascend310P3 \
    --om_abi_path /path/to/compiler-introspection/model.om.abi.json \
    --deployment ascend_310p3
```

未显式指定输出时，ONNX 和 ATC OM 工作产物写入
`models/_work/<bundle>/ascend/`；最终 OM 自动复制到
`artifacts/ascend/<deployment>/`。`--om_abi_path` 是已有的 compiler/runtime introspection
JSON 输入，ATC 命令本身不生成该文件。

若 ONNX 已存在：

```bash
python3 src/model_utils/model_utils/export_onnx_atc.py \
    --pretrained_model /path/to/act_bundle \
    --soc_version Ascend310P3 \
    --onnx_model_path /path/to/act.onnx \
    --om_model_path /tmp/act.om \
    --om_abi_path /tmp/act.om.abi.json \
    --deployment ascend_310p3 \
    --skip_onnx_export
```

ABI input names必须与 ACT 实际消费的 `observation.state` 和
`observation.images.*` 顺序一致，output 必须是唯一的 `action`。生成的 deployment backend
为 `ascend`，target runtime 为 `acl`。

## ACT Hisilicon SD3403

`export_onnx_hisilicon.py` 负责导出 vendor toolchain 使用的 ACT ONNX。其 `--device` 仅表示
执行 Torch export 的主机设备，不是运行时 backend selector：

```bash
source .shrc_local

python3 src/model_utils/model_utils/export_onnx_hisilicon.py \
    --policy_path /path/to/act_bundle \
    --policy_type act \
    --device cpu \
    --bundle_output /path/to/compiled_bundle
```

Vendor toolchain 生成 SD3403 OM、worker executable 和 ABI JSON 后，使用通用 packager 完成
deployment：

```bash
ros2 run model_utils package-compiled-deployment \
    --bundle-root /path/to/compiled_bundle \
    --deployment sd3403 \
    --backend hisilicon \
    --target-soc sd3403 \
    --target-runtime hisilicon-worker \
    --spec /path/to/hisilicon-package-spec.json
```

Hisilicon deployment 必须包含：

- `execution: ["policy"]`
- `policy` artifact，format `om`
- 可执行的 `worker` artifact，format `executable`
- `policy` role 的完整 inputs/outputs bindings

Hisilicon ONNX 默认写入 `models/_work/<policy_bundle>/hisilicon/`。

## ACT RKNN

### 创建 Toolkit 环境

```bash
python3 -m venv .venv-rknn
. .venv-rknn/bin/activate
python -m pip install rknn-toolkit2 onnx onnxruntime
```

### 从 Checkpoint 导出并转换

```bash
source .shrc_local

python3 src/model_utils/model_utils/export_onnx_rknn.py \
    --policy_path /path/to/act_bundle \
    --convert_rknn \
    --rknn_output /tmp/act.rknn \
    --rknn_abi_output /tmp/act.rknn.abi.json \
    --rknn_mode float16 \
    --rknn_venv_python "$WORKSPACE/.venv-rknn/bin/python" \
    --deployment rk3588
```

### 从已有 ONNX 转换

```bash
python3 src/model_utils/model_utils/export_onnx_rknn.py \
    --onnx /path/to/act.onnx \
    --bundle_root /path/to/act_bundle \
    --convert_rknn \
    --rknn_output /tmp/act.rknn \
    --rknn_abi_output /tmp/act.rknn.abi.json \
    --rknn_mode float16 \
    --rknn_venv_python "$WORKSPACE/.venv-rknn/bin/python" \
    --deployment rk3588
```

`--output` 是优化后 ONNX 路径；最终 RKNN 路径使用 `--rknn_output`。Compiler 必须生成
`--rknn_abi_output`，否则 exporter 不会写入 deployment。图像 layout 由 ABI 显式声明，
runtime 只对 `NHWC` image bindings 转换布局。

从 checkpoint 导出时，工作产物默认写入 `models/_work/<bundle>/rknn/`；最终 RKNN
自动复制到 `artifacts/rknn/<deployment>/`。`--rknn_abi_output` 是 converter 生成的输出。

完整板端流程见 `docs/OpenHarmony_EmbodiedAI_RKNN_Inference.md`。

## PI0.5 Ascend Split Export

`pi05-export` 将 PI0.5 拆分为 VLM 和 Action Expert，并可完成 ONNX、量化、OM 编译与等价性
验证。VLM 会在模型内部把多相机图像合并为一个临时 vision batch，再恢复原有 camera-major
prefix；外部 VLM ABI 仍是原来的逐相机输入、语言输入和 VLM-to-AE handoff，不需要修改调用方、
batch JSON 或相机契约。默认步骤为 `vlm_onnx,ae_onnx,vlm_om,ae_om`：

```bash
source .shrc_local

ros2 run model_utils pi05-export \
    --policy-path /path/to/pi05_bundle \
    --work-dir /path/to/pi05_export_run \
    --soc-version Ascend310P3 \
    --device cpu \
    --dtype fp16
```

使用 NPU 导出时，Gemma text MLP 默认使用精度保持的 `NPUGeglu` 融合
`gelu(gate_proj(x)) * up_proj(x)`。`--fast-gelu-scope` 可将近似 `NPUFastGelu` 限制到
`vision`、`vlm-text`、`ae` 或显式选择 `all`；默认值为 `none`。旧参数 `--fast-gelu` 保持兼容，
等价于 `--fast-gelu-scope all`。FastGELU 可能降低延迟，但会改变 GELU 数学，必须使用既有
baseline 验证。精度优先时省略这些参数或使用 `--fast-gelu-scope none`。

新导出的 Action Expert OM 在每个 timestep 输出 velocity，而不是已经积分的 action。Ascend
backend 按 schedule 对相邻 timestep 执行 Euler integration：

```text
x_next = x_t + (next_timestep - timestep) * velocity
```

velocity deployment 必须包含 `denoising_schedule` JSON artifact。未指定 `--schedule-file` 时，
exporter 根据 bundle `config.json` 的 `num_inference_steps` 生成 uniform schedule；已有严格 schedule
可在导出时打包：

```bash
ros2 run model_utils pi05-export \
    --policy-path /path/to/pi05_bundle \
    --work-dir /path/to/pi05_export_run \
    --soc-version Ascend310P3 \
    --schedule-file /path/to/selected_schedule.json \
    --steps vlm_onnx,ae_onnx,vlm_om,ae_om
```

`--schedule-file` 必须是无未知字段的 `pi05-denoising-schedule-v1` JSON：

```json
{
  "format": "pi05-denoising-schedule-v1",
  "name": "uniform_4",
  "algorithm": "euler",
  "model_output": "velocity",
  "timesteps": [1.0, 0.75, 0.5, 0.25, 0.0]
}
```

`timesteps` 必须从 `1.0` 严格递减到 `0.0`，step 数为其长度减一。Schedule 被复制到
唯一 generation 目录，作为 versioned、non-execution Manifest artifact；它不出现在
`execution` 或 `bindings` 中，但正式替换会增加 deployment revision 并改变 fingerprint。运行时只读取
Manifest 中选中 deployment 的该 artifact，不扫描 bundle 根目录的 `schedule.json`，也不接受
环境变量指定 schedule。已有 Action Expert ABI 输出名为 `action` 且没有 schedule artifact 的
legacy deployment 继续使用旧的逐步 action-output 行为，不会被自动迁移为 velocity 模式。

这里的 `--device` 仅是 Torch export/verification device。OM 完成后工具通过 ACL model
descriptor 自动生成 `<model>.om.abi.json`（只有实际需要时才导入 `acl`），并在 policy bundle
中更新默认名为 `ascend` 的 unified deployment，写入 VLM/Action Expert artifacts、bindings 和
device-pointer links。`--deployment` 统一指定本次写入的 deployment 名称：名称不存在时添加，
已存在时保留 UUID 并更新 revision，内容不变时为 no-op。量化由 `--quant-profile` 和 steps 决定，
没有单独的量化 deployment 命名参数；一次调用只发布一个 deployment。ACL ABI
检查默认使用 device 0；可通过 `--abi-device-id` 和 `--acl-config-path` 覆盖。`--vlm-abi` /
`--ae-abi` 仍可在直接调用 `convert_om` 时显式覆盖自动生成的 ABI。

选择步骤：

```bash
# 只导出 ONNX
ros2 run model_utils pi05-export \
    --policy-path /path/to/pi05_bundle \
    --work-dir /path/to/run \
    --steps vlm_onnx,ae_onnx

# 导出、编译并验证
ros2 run model_utils pi05-export \
    --policy-path /path/to/pi05_bundle \
    --work-dir /path/to/run \
    --soc-version Ascend310P3 \
    --task "pick up the cup" \
    --steps vlm_onnx,ae_onnx,vlm_om,ae_om,verify

# W8A8 ONNX + OM；写入命名 deployment
ros2 run model_utils pi05-export \
    --policy-path /path/to/pi05_bundle \
    --work-dir /path/to/run \
    --soc-version Ascend310P3 \
    --deployment ascend-w8a8 \
    --batch-path /path/to/calibration_batches.json \
    --steps vlm_onnx,ae_onnx,vlm_quant,vlm_om,ae_om,ae_quant,vlm_quant_om,ae_quant_om
```

量化策略可以保存为版本化 `quantization_profiles`，顶层 CLI 只通过 `--quant-profile` 选择策略，
不直接暴露节点正则。随包提供的 YAML `pi05-q40-v1` 固化当前在 Ascend310P3 上验证过的 136 个 VLM
INT8 MatMul；它要求 NPU VLM export、CPU donor、精确 GeGLU，并明确保持 Action Expert 为 FP16：

```bash
ros2 run model_utils pi05-export \
    --policy-path /path/to/pi05_bundle \
    --work-dir /path/to/q40-run \
    --soc-version Ascend310P3 \
    --device npu \
    --donor-device cpu \
    --batch-path /path/to/calibration_batches.json \
    --quant-profile pi05-q40-v1 \
    --steps vlm_onnx,ae_onnx,vlm_quant,ae_om,vlm_quant_om
```

`pi05-vlm-text-ae-attn-mlp-sq-v1` 组合 FP16 VLM text FastGELU 与 108 节点 W8A8
Action Expert：18 层 q/k/v/o attention projection、fused GeGLU MLP projection 和 down projection，
并携带 SmoothQuant `alpha=0.5`、`epsilon=1e-5`。该组合尚待 checkpoint 实测，因此状态为
`experimental`。选择 `ae_quant` 后，顶层 exporter 会用本轮 fresh FP16 VLM/AE OM 创建临时
calibration deployment，按 `--batch-path`、`--num-calib`、task 和固定 seed 42 自动调用
`pi05-om-dump`，在 `<work-dir>/calibration/ae` 生成完整 uniform-10 AE 轨迹；不再接受顶层
`--calib-dir`，因此不会复用 exact-VLM calibration：

```bash
ros2 run model_utils pi05-export \
    --policy-path /path/to/pi05_bundle \
    --work-dir /path/to/ae-sq-run \
    --soc-version Ascend310P3 \
    --device npu \
    --donor-device cpu \
    --batch-path /path/to/calibration_batches.safetensors \
    --num-calib 20 \
    --quant-profile pi05-vlm-text-ae-attn-mlp-sq-v1 \
    --deployment ascend_310p3_vlm_text_ae108_sq \
    --steps vlm_onnx,ae_onnx,vlm_om,ae_om,ae_quant,ae_quant_om
```

不指定 `--work-dir` 时，顶层 exporter 默认使用
`models/_work/<bundle>/ascend/pi05/`，并在其中固定创建 `onnx/`、`runtime_save/`、`om/` 和
`calibration/`。顶层不再暴露 `--exp-dir`、`--output-dir`、`--runtime-save-dir` 或 `--om-dir`；
需要调试非标准布局时直接调用对应底层 exporter。显式 `--work-dir` 和 `--amp-scratch-dir`
在创建任何文件前都会解析真实路径，并拒绝 policy bundle 本身或其子目录（包括 symlink 绕入）。

该 profile 自动采用 `fast_gelu_scope=vlm-text`；显式选择其他 scope 会在导出前报错。精确 NPU
Gemma 路径始终使用 `NPUGeglu`，不再提供公开的 unfused fallback。量化 deployment 按角色选择
artifact，因此上例会把 FP16 VLM OM 与 W8A8 Action Expert OM 写入同一个 deployment。由于当前
manifest 不记录 quant profile identity，profiled deployment 的发布和更新必须同时列出两个角色的
OM 步骤和 fresh ONNX export，避免从同名但不同 profile 的 deployment 或旧 FastGELU scope
复用不兼容 artifact。为 AE calibration 构建的 FP16 AE OM 是工作产物，不会额外发布 FP deployment。

可用 `--list-quant-profiles` 查看随包 YAML 和配置文件中的策略。运行 profile 可保存
`quant_profile: pi05-q40-v1` 引用；自定义策略放在同一 YAML 的 `quantization_profiles` 下，格式为
`pi05-quant-profile-v1`，并为每组 selector 声明 regex 和预期匹配数量。显式执行 `vlm_quant` 会始终
重新校准；只执行 `vlm_quant_om` 时，工具会检查相邻的 `.quant.json`，拒绝复用来自其他 profile
或 policy bundle 的 W8A8 ONNX。

量化 profile 描述节点选择和可复用的算法参数，不携带 checkpoint 相关权重、校准 scale、SmoothQuant
plan 或 OM。将任一 profile 用于另一个 PI05 checkpoint 时仍需重新采集校准数据、重新量化，并执行
独立精度与性能验证；q40 YAML 不允许 `ae_quant`/`ae_quant_om`，AE SmoothQuant YAML 不允许
`vlm_quant`/`vlm_quant_om`。

请求的步骤会先删除该步骤的旧输出，并在子工具返回后确认新输出确实存在，因此失败的导出或
ATC 调用不会把 stale ONNX/OM/ABI 当作本次成功。只重跑 `vlm_om` 或 `ae_om` 时，另一角色
使用磁盘上的现有 OM/ABI，并重新严格校验和更新完整 deployment；缺少任一角色时不会写入
不完整 manifest。只重跑 ONNX 不会冒充已更新的 OM deployment，后续应重跑对应 OM 步骤。

需要直接调用 OM compiler wrapper 时：

```bash
python3 -m model_utils.pi05_export.convert_om \
    --pretrained-policy-path /path/to/pi05_bundle \
    --soc-version Ascend310P3 \
    --vlm-onnx /path/to/vlm.onnx \
    --ae-onnx /path/to/action_expert.onnx \
    --vlm-abi /path/to/vlm.om.abi.json \
    --ae-abi /path/to/action_expert.om.abi.json \
    --deployment ascend
```

运行 `python3 -m model_utils.pi05_export.convert_om --help` 查看当前参数名称和可选的
`--input-shape` / `--atc-arg` 配置。

## GraspGen Ascend Packaging

从 GraspGen checkpoint 到板端可加载 bundle 共三步，三步都有对应的 console entry point，
不需要 `python3 -m`：checkpoint → `graspgen.onnx.json` → `graspgen.om.json` 与八个 OM →
统一 `inference_manifest.json`。

前两步（导出与编译）是 `model_utils` 的工具，记录在本节；**第 3 步打包由
`perception_service` 提供**——GraspGen 的业务归属是抓取，模型目录固定在 `models/grasp/`；
当前通用模型运行时使用 `interface="tensor_model"` 和具体 `model_type/operation`，bundle 的读者是
perception_service 中现有的 adapter 与 typed `GenerateGrasps` service。
bundle 布局、adapter asset、named deployment 与 conformance 见
`src/perception_service/README.md` 的 GraspGen 章节。GraspGen **不是** policy family，
不出现在本文件末尾的策略支持矩阵中。

除非另有说明，本节所有命令均在 IB_Robot 仓库根目录执行；所有项目内路径均相对于仓库根目录。

### 1. 导出八个 ONNX 子图

```bash
source .shrc_local

ros2 run model_utils graspgen-export-onnx \
    --config models/_work/graspgen_robotiq_2f_140/source/checkpoints/graspgen_robotiq_2f_140.yml \
    --generator-checkpoint models/_work/graspgen_robotiq_2f_140/source/checkpoints/graspgen_robotiq_2f_140_gen.pth \
    --discriminator-checkpoint models/_work/graspgen_robotiq_2f_140/source/checkpoints/graspgen_robotiq_2f_140_dis.pth \
    --output-dir models/_work/graspgen_robotiq_2f_140/model_utils/onnx \
    --grasp-batch-size 1000
```

导出后每个子图都会用相同输入跑一次 PyTorch 与 onnxruntime 对比，超出容差即失败退出：删除
已写出的图和 manifest，不会留下"导出成功但数值不对"的产物。基线容差是
`max_relative<=1e-4`、`mean_relative<=1e-5`、`cosine>=1-1e-6`；确需重新标定时用
`--verify-tolerance-scale` 按倍数放宽（例如 `--verify-tolerance-scale 10`），它用于重新
标定基线，不是用来掩盖回归。`--skip-verify` 只应在没有 onnxruntime 的主机上临时使用。

`--artifacts` 可只重导子集（角色名见 `inference_manifest.GRASPGEN_EXECUTION`）；
`--disable-constant-folding` / `--disable-simplify` 关闭默认开启的图清理。

`graspgen.onnx.json` 会写入 `contract_version`，取自
`inference_manifest.graspgen.GRASPGEN_CONTRACT_VERSION`——执行角色顺序或 PointNet++ 采样
几何变更时该版本号会 bump，packager 拒绝打包版本不一致的导出，避免用旧图编出的 OM 被新
backend 用错误的分组驱动。遇到该报错时按本节重跑导出即可。

### 2. 编译 OM

```bash
ros2 run model_utils graspgen-onnx-to-om \
    --manifest models/_work/graspgen_robotiq_2f_140/model_utils/onnx/graspgen.onnx.json \
    --output-dir models/_work/graspgen_robotiq_2f_140/model_utils/om \
    --soc-version Ascend310P3
```

输出 `<role>.om` 与记录 ATC 参数、来源 manifest 的 `graspgen.om.json`。`--skip-existing`
复用已编译产物，`--dry-run` 只打印 ATC 命令，`--atc-path` / `--atc-arg` 覆盖 CANN 查找和
额外编译参数。

### 3. 打包统一 bundle（perception_service 命令）

GraspGen 的八个 OM 必须使用 ACL runtime introspection 得到的实际 ABI，不能使用 ONNX
tensor name 代替编译后 ABI。默认情况下，packager 会在 sidecar 缺失时使用本机 Ascend
device 检查 OM，并把 `<role>.om.abi.json` 写入 `--om-abi-dir`：

```bash
ros2 run perception_service package_graspgen_ascend_bundle \
    --bundle-root models/graspgen \
    --onnx-manifest models/_work/graspgen_robotiq_2f_140/model_utils/onnx/graspgen.onnx.json \
    --om-dir models/_work/graspgen_robotiq_2f_140/model_utils/om \
    --om-abi-dir models/_work/graspgen_robotiq_2f_140/model_utils/abi \
    --abi-device-id 0 \
    --soc-version Ascend310P3
```

ABI 目录需要包含 `generator_sa1.om.abi.json` 到
`discriminator_head.om.abi.json` 的全部八个 sidecar。已有 sidecar 不会重新生成；在无 Ascend
device 的打包主机上需要提前生成全部 sidecar，并传入 `--no-inspect-missing-abi`。Packager 按
runtime index 绑定 GraspGen semantic，并保留 ACL 实际暴露的 tensor name、dtype 和 shape。
八个角色全部解析成功后才落盘，任一 OM 或 ABI 缺失都不会留下半成品 bundle。

打包产物属于抓取模型域；manifest 使用 `tensor_model/graspgen/generate_grasps` 身份。
bundle 不含任何 LeRobot 资产（没有 `config.json` / `policy_preprocessor.json` /
`policy_postprocessor.json`）；模型常量写在
`assets/adapter.json` 里，由 `perception_service.graspgen_adapter.GraspGenAdapter`
读回。参数含义与运行时配置见 `src/perception_service/README.md`。

Packager 会把运行时 artifact 复制到唯一 generation，并在 manifest 中记录
bundle-relative path。发布和部署时应手动传输完整 bundle 目录；板端不需要保留 `--om-dir`
或 `--om-abi-dir` 指向的构建输入。schema v3 不对大文件记录或启动时计算 SHA-256；identity
由 UUID、revision 和轻量结构摘要组成。

## HMM Packaging

HMM 只支持 PI0.5 与 SmolVLA，不支持 ACT。TCIM 编译完成后准备一个只包含路径和 target
选择的 JSON spec，然后执行：

```bash
source .shrc_local

ros2 run model_utils package-hmm-deployment \
    --bundle-root /path/to/policy_bundle \
    --deployment lq50 \
    --target-soc lq50 \
    --target-runtime tcim \
    --spec /path/to/hmm-package-spec.json
```

Packager 读取各 TCIM `model.json` ABI，生成 execution、bindings、device links、UUID/revision 和
轻量结构摘要。SmolVLA 还要求 `state_projection.pt`。详细 spec 和转换流程见
`docs/Houmo_HMM_Conversion.md`。

## 通用 Compiled Deployment Packager

Ascend、Hisilicon、RKNN 或 HMM 的 vendor compiler 已经产出 artifact 和 ABI 时，可使用：

```bash
ros2 run model_utils package-compiled-deployment \
    --bundle-root /path/to/policy_bundle \
    --deployment <name> \
    --backend <ascend|hisilicon|rknn|hmm> \
    --target-soc <soc> \
    --target-runtime <runtime> \
    --spec /path/to/package-spec.json
```

Spec 顶层字段：

```json
{
  "execution": ["policy"],
  "roles": {
    "policy": {
      "artifact": "./model.rknn",
      "format": "rknn",
      "abi": "./model.rknn.abi.json",
      "abi_format": "runtime",
      "input_semantics": {
        "state": "observation.state"
      },
      "output_semantics": {
        "actions": "action"
      },
      "image_layouts": {}
    }
  },
  "artifacts": {},
  "device_links": []
}
```

每个 execution role 必须有 artifact、ABI、完整 input semantic mapping 和 output semantic
mapping。`abi_format` 为 `runtime` 或 `tcim`。

## Observation Batch

`model_utils` 使用版本化 Safetensors 保存可复用的原始 observation batch。图像统一为 HWC
`uint8 [0,255]`，其他字段保留原始数值 dtype；所有 tensor 的第一维都是 sample 维。该格式不包含
policy normalization、tokenization 或其他 model-ready 变换。旧 `.json` batch 仍可读取，但新 batch
应使用 Safetensors。`safetensors>=0.4.3,<1.0.0` 由项目统一的 `requirements/base.txt` 安装；
`model_utils` 的 Python package metadata 声明相同约束。当前 rosdep 数据库没有可用的
`python3-safetensors` key，因此 `package.xml` 不声明无法解析的 ROS 依赖。

保存数值字段时必须提供显式 `FieldSpec`。图像字段必须声明 `semantic=image` 及输入 layout
`CHW` 或 `HWC`，并严格匹配声明的 shape/dtype；保存层只执行确定性的 CHW-to-HWC 规范化，
不会根据字段名、shape 或 dtype 猜测布局。

随机生成适合 shape/runtime smoke test，不适合量化校准：

```bash
ros2 run model_utils observation-batch random \
    --output /tmp/random.observations.safetensors \
    --samples 32 \
    --seed 42 \
    --field 'observation.state=6,float32,-100,100' \
    --field 'observation.images.top=480x640x3,uint8,0,255,image,HWC' \
    --field 'observation.images.wrist=480x640x3,uint8,0,255,image,HWC'
```

从本地 LeRobot dataset 进行 episode 分层抽样，使用 PyAV 解码视频，并由 policy bundle 的
`config.json.input_features` 提供字段 semantic、shape 和 VISUAL CHW layout 契约：

```bash
ros2 run model_utils observation-batch dataset \
    --dataset-root /path/to/lerobot_dataset \
    --policy-path /path/to/policy_bundle \
    --output /tmp/calibration.observations.safetensors \
    --samples 256 \
    --seed 42 \
    --sampling episode-stratified \
    --field observation.state \
    --field observation.images.top \
    --field observation.images.wrist

ros2 run model_utils observation-batch inspect \
    /tmp/calibration.observations.safetensors
```

显式传入 `--field` 时，它是严格白名单；需要 task 时必须同时传 `--field task`。未传任何
`--field` 时，命令选择 policy 的全部 input feature，并在 dataset 提供 task 时保留它。
`inspect` 会显示 dataset/frame/episode 的样本级 provenance dtype、shape 和范围摘要。

## loss_compare

`loss_compare.py` 使用 `PureInferenceEngine` 和命名 deployment 比较同一 batch 在不同 runtime
上的输出。它不接受 runtime backend `--device`；选择 bundle deployment 使用
`--deployment`。

### 生成基准

```bash
source .shrc_local

python3 src/model_utils/model_utils/loss_compare.py \
    --policy_path /path/to/policy_bundle \
    --deployment cuda \
    --batch_path /path/to/batches.observations.safetensors \
    --exp-dir /path/to/experiment \
    --generate-target
```

`--exp-dir` 自动派生：

```text
target.json
target_raw.json
noises/
```

PI0.5 的 external noise 始终由 `seed + batch_index` 在独立 CPU generator 中确定性生成。
生成 baseline 时必须指定 `--exp-dir` 或 `--noise-dir`，noise 会保存到 `noises/`，使不同机器
和 deployment 使用同一数组；比较时即使未配置目录也会显式生成 deterministic noise，运行时
不会退回 backend 自行采样。

### 比较 Compiled Deployment

```bash
python3 src/model_utils/model_utils/loss_compare.py \
    --policy_path /path/to/policy_bundle \
    --deployment ascend \
    --batch_path /path/to/batches.observations.safetensors \
    --exp-dir /path/to/experiment \
    --metrics-json /path/to/experiment/metrics.json
```

可用 `--model_dtype native|fp16|bf16|fp32` 请求 Torch model dtype；不支持该 runtime option
的 deployment 会拒绝启动。`loss_compare` 每个独立 sample 前重置 engine，并同时比较最终
postprocessed action 和可选 raw action。`--metrics-json` 将 latency、normalized/unnormalized
指标和 PI0.5 distribution aggregates 写为 `loss-compare-metrics-v1` JSON，便于 schedule tuner
比较候选项。

PI0.5 schedule 诊断还支持两个仅限本次调用的参数：

- `--schedule-override-path`：临时使用一个严格 schedule，不修改 Manifest。
- `--curvature-log-path`：把每次推理的 schedule 和相邻 velocity curvature 写为 JSONL。

它们和 `--metrics-json` 都是 compute-only、CLI-only 参数，不能与 `--generate-target` 一起使用，
也不会保存到 `loss_compare` profile。Override 仅用于诊断和调优，不能作为部署配置；最终选定的
schedule 必须通过 `--schedule-file` 重新导出，或由 tuner 安装为 Manifest artifact。这两个
schedule 诊断参数只适用于 Action Expert runtime output 为 `velocity`/`v_t` 的 deployment。
例如使用既有 profile、target 和 noise 采集 dense schedule curvature：

```bash
python3 src/model_utils/model_utils/loss_compare.py \
    --config /path/to/loss_compare.yaml \
    --profile pi05-baseline \
    --policy_path /path/to/pi05_bundle \
    --deployment ascend \
    --schedule-override-path /path/to/dense_uniform_20.json \
    --curvature-log-path /path/to/tuning/curvature.jsonl \
    --metrics-json /path/to/tuning/dense_metrics.json
```

从 curvature JSONL 独立生成若干严格候选 schedule：

```bash
ros2 run model_utils pi05-curvature-schedule \
    --log /path/to/tuning/curvature.jsonl \
    --num-steps 3 4 5 \
    --output-dir /path/to/tuning/schedules
```

完整调优可由 `pi05-tune-schedule` 自动完成 dense curvature run、uniform/curvature 候选比较、
报告生成和最佳 schedule 安装：

```bash
ros2 run model_utils pi05-tune-schedule \
    --config /path/to/loss_compare.yaml \
    --profile pi05-baseline \
    --policy-path /path/to/pi05_bundle \
    --deployment ascend \
    --candidate-steps 3 4 5 \
    --metric raw_l1 \
    --artifacts-dir /path/to/tuning/run
```

所选 `loss_compare` profile 必须已有 `batch_path`、`target_path`、`raw_target_path` 和
`noise_dir`。调优只复用这些 batch、target 和 noise，绝不能从待测 OM 生成或覆盖 target。
默认会把选中的 schedule 替换进 deployment 并重算 Manifest identity；`--no-install` 仅生成
候选和报告，此时仍必须另行将最终 schedule 安装到 Manifest。

### PI0.5 OM 显式诊断 Dump

需要检查一个 manifest 中命名的 PI0.5 Ascend deployment 时，使用独立诊断命令，不要给生产
runtime 增加环境变量 dump 开关：

```bash
source .shrc_local

ros2 run model_utils pi05-om-dump \
    --policy-path /path/to/pi05_bundle \
    --deployment ascend_310p3 \
    --batch-path /path/to/batches.observations.safetensors \
    --batch-index 0 \
    --seed 42 \
    --out-dir /tmp/pi05_om_dump_0
```

该命令把 deployment 名称原样交给 `PureInferenceEngine`，由 unified manifest loader 和 Ascend
backend 选择、校验并执行 OM。它通过显式 `diagnostic_capture` callback 传入确定性 external
noise，并保存 VLM 输入/输出、AE 输入、schedule timesteps、每步 `dt`、velocity、
`x_t_stepNN.npy`、`raw_action.npy`、`action.npy` 和 `diagnostic_capture.json`。engine 在成功或
失败后都会关闭，因此测试可注入 mock engine，不要求 ACL 或 NPU。

配置文件默认位于 `~/.config/model_utils/loss_compare.yaml`。优先级为 CLI、profile、
defaults、last、builtin。查看全部参数：

```bash
python3 src/model_utils/model_utils/loss_compare.py --help
```

## frame_inspect

`frame_inspect` 从 LeRobot dataset 选择一个 frame 或 frame range，运行原生 policy，并导出
预测、label、delta、图像和汇总文件。

```bash
source .shrc_local

ros2 run model_utils frame_inspect \
    --policy-path /path/to/policy \
    --dataset-repo-id organization/dataset \
    --dataset-root /path/to/dataset_root \
    --output-dir /tmp/frame_inspect \
    --episode-index 0 \
    --frame-index 10 \
    --device cpu
```

这里的 `--device` 是离线 Torch policy device，不是 unified runtime backend selector。区间示例
使用 `--frame-index 10:30`；也可以使用 `--global-index` 选择全局 frame。

## Manifest 验证与排障

验证一个 deployment：

```bash
source .shrc_local
PYTHONPATH=src/inference_manifest \
python3 -c "from inference_manifest import load_inference_manifest; print(load_inference_manifest('/path/to/policy_bundle', 'cpu').fingerprint)"
```

常见错误：

| 错误 | 处理 |
| --- | --- |
| deployment 不存在 | 使用 manifest 中实际的 deployment 名称或重新运行 exporter |
| `Bundle digest mismatch` | Manifest identity/路径声明与 digest 不一致；重新运行 owning exporter |
| schema v1/v2 | 旧版 bundle/artifact 不受支持；使用当前 exporter 或 packager 重新生成 schema v3 |
| artifact SDK load failure | artifact 可能损坏或 ABI 不兼容；重新打包并发布新 revision |
| missing semantic dependency | 将 processor/tokenizer dependency vendored 到 bundle 后重新打包 |
| binding name/shape/layout mismatch | 用 compiler/runtime 实际 ABI 重新生成 bindings |
| unsupported policy/backend pair | 使用 registry 支持矩阵中的组合 |

不提供 v1 原地迁移命令。旧版 artifact 缺少稳定 identity 和显式 sharing 信息，必须重新生成。

初始支持矩阵：

| Policy model_type | `torch` | `ascend` | `hisilicon` | `rknn` | `hmm` |
| --- | --- | --- | --- | --- | --- |
| ACT | 支持 | 支持 | 支持 | 支持 | 不支持 |
| PI0.5 | 支持 | 支持 | 不支持 | 不支持 | 支持 |
| SmolVLA | 支持 | 不支持 | 不支持 | 支持 | 支持 |

更完整的运行时架构、digest 算法和 pipeline 配置见 `src/inference_service/README.md`。
