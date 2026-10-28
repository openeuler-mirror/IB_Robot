# SmolVLA RKNN 问题排查与修复记录

本文记录 2026 年 7 月在 RoboPi（RK3588/OpenHarmony EmbodiedAI 1.0.1）上排查
SmolVLA RKNN 推理链路时发现的问题、根因、修复方案、验证结果和后续风险。目标是让后续维护者
不必重新经历 ONNX、RKNN、板端运行时和 ROS 全链路的逐层定位过程。

> 本文是故障记录，不是当前部署指南。正式导出和部署流程仍以
> [`OpenHarmony_EmbodiedAI_RKNN_Inference.md`](OpenHarmony_EmbodiedAI_RKNN_Inference.md)
> 以及当前代码为准。

## 1. 记录范围与代码状态

本轮排查最初在隔离工作副本 `/tmp/opencode/ibrobot-rknn-clean` 中完成，基线提交为
`ef798853`，没有直接修改当时正在进行架构迁移的主工作区。随后主工作区已将旧的
`core/rknn/smolvla/SmolVLARKNNModel.py` 编排迁移到 manifest 驱动的
`inference_service/backends/rknn/backend.py`。

因此本文中的代码路径分为两类：

- **历史修复路径**：隔离工作副本中用于定位和板端验证的旧编排代码。
- **当前迁移路径**：主工作区的新 RKNN backend 和统一 deployment exporter。

不能因为旧编排代码已删除，就认为相应问题不再存在。迁移时必须保留本文描述的模型语义和
运行时保护。

## 2. 结论摘要

| 问题 | 影响 | 隔离分支处理 | 当前主工作区状态 |
|------|------|--------------|------------------|
| Action Expert 含 INT64 `ReduceMin` | RK3588 CPU fallback 失败，action 可能全零 | 从 LeRobot 源头改成减首元素，并在导出时检查 | LeRobot patch 是否已纳管需再次确认；当前 exporter 未见专项检查 |
| RKNN prefix 缺少 state token | 177 个 token 中最后一个变成零 padding，策略忽略机器人状态 | 导出并加载 `state_projection.pt`，追加 state token | 新 backend 已加载 state projection 并追加 state token |
| Prefill 使用全 1 attention mask | padding 和 state block 语义错误，KV cache 偏离原模型 | 实现 cumsum block attention | 新 backend 已实现相同 block attention |
| 视觉输入尺寸不匹配 | 相机 `480x640` 与 RKNN 静态 `512x512` 不一致 | 运行时等比例缩放并在左侧/顶部补零 | 新 backend 依赖 codec/processor；需持续验证与 LeRobot 预处理一致 |
| Flattened KV 多一个维度 | Action RKNN 输入 ABI 不匹配 | 修正为 `[16,2,1,177,5,64]` | 新架构已拒绝 flattened-KV，统一使用逐层 cache 输入 |
| Prefill context 连续推理后出现 NaN/Inf | 大约第 10 次请求后 KV 非有限，后续动作失效 | 检测后 release/reload，重试一次 | 新 backend 尚未看到 finite 检查和自动重载，存在回归风险 |
| Action 输出缺少完整校验 | 错误可能静默传播到机器人 | 增加 shape、finite、全零检查 | 新 backend 主要校验 ABI shape/dtype，尚未看到 finite/全零检查 |
| 推理速度慢于动作消费速度 | 队列反复耗尽，运动不连续 | 未根治，仅确认功能正确 | 未解决，需要性能或调度策略优化 |
| 导出与运行时 action ABI 曾不一致 | 重新导出后可能无法加载 | 排查时发现 4-input 与 35-input 两套契约 | 新 exporter 已明确拒绝 legacy flattened-KV，统一 segmented cache ABI |

## 3. 参考张量契约

本次模型为双相机、48 个语言 token、1 个 state token 的 SmolVLA。关键形状如下：

| 阶段 | 张量 | 形状 |
|------|------|------|
| 相机输入 | 单路 NCHW image | `[1, 3, 480, 640]` |
| Vision 静态输入 | 单路 NCHW image | `[1, 3, 512, 512]` |
| Vision 输出 | 单路 image embedding | `[1, 64, 960]` |
| 双路视觉 token | image embeddings | `[1, 128, 960]` |
| 语言 token | language embedding | `[1, 48, 960]` |
| 原始机器人状态 | normalized state | `[1, 6]` |
| 补齐状态 | state projection input | `[1, 32]` |
| 状态 token | state embedding | `[1, 1, 960]` |
| Prefill prefix | prefix embeddings | `[1, 177, 960]` |
| Prefix pad mask | valid tokens | `[1, 177]` |
| Prefix attention | block mask | `[1, 177, 177]` |
| Position IDs | prefix positions | `[1, 177]` |
| 单个 KV 输出 | key 或 value | `[1, 177, 5, 64]` |
| 历史 flattened KV | 16 层 key/value | `[16, 2, 1, 177, 5, 64]` |
| Action noise/velocity | action chunk | `[1, 50, 32]` |
| 后处理动作 | 机器人实际 action | `[50, 6]` |

Prefix 长度计算必须满足：

```text
2 cameras * 64 image tokens + 48 language tokens + 1 state token = 177
```

任何导出或运行时实现如果只得到 176 个真实 token，再补一个零 token，都是语义错误。

## 4. 问题一：Action Expert 的整数 ReduceMin

### 4.1 现象

Action RKNN 可以完成模型加载和 runtime 初始化，但推理时可能发生 CPU fallback 错误，随后得到
全零 velocity，导致 10 步去噪过程没有有效更新。仅看“RKNN 转换成功”或“模型加载成功”无法发现
该问题。

### 4.2 根因

LeRobot 的 expert position normalization 原本为：

```python
expert_position_id = expert_position_id - torch.min(
    expert_position_id, dim=1, keepdim=True
).values
```

导出 ONNX 后得到：

```text
ReduceMin(INT64[1, 50]) -> INT64[1, 1]
```

板端最小复现表明，RK3588 对浮点 `ReduceMin` 的 fallback 可工作，但 INT32/INT64 输入会在运行时
失败。问题不是 ONNX 语义本身，而是 RKNN runtime/CPU fallback 对该 dtype 与算子组合的支持缺陷。

### 4.3 最终修复

Expert position IDs 在该路径中保证单调不减，因此最小值必然是第一个元素。源码修复为：

```python
def normalize_expert_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
    return position_ids - position_ids[:, :1]
```

这样导出图使用 `Slice` 和 `Sub`，不再生成 `ReduceMin`。

历史修复文件：

- `libs/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py`
- `libs/lerobot/tests/policies/smolvla/test_smolvlm_with_expert.py`

### 4.4 导出阶段保护

隔离分支新增了 `rknn_onnx_compat.py`，在 ONNX 导出后执行 shape inference 并拒绝
INT32/INT64 `ReduceMin`：

- `src/model_utils/model_utils/smolvla_export/rknn_onnx_compat.py`
- `src/model_utils/test/test_rknn_onnx_compat.py`
- `src/model_utils/model_utils/smolvla_export/export_onnx.py`
- `src/model_utils/model_utils/smolvla_export/export_rknn_modules.py`

检查策略是：

- 普通图通过。
- FLOAT/FLOAT16 `ReduceMin` 通过。
- INT32/INT64 `ReduceMin` 立即失败，不允许问题延迟到板端暴露。

### 4.5 试验过但不应作为最终方案的方法

排查中验证过两种 ONNX 后处理：

- 在 `ReduceMin` 前后增加 `Cast(FLOAT32)`。
- 根据具体图拓扑直接用 prefix offset 替换 `ReduceMin` 输出。

这些方案用于证明根因，但不适合作为长期实现。前者对任意大 INT64 不保证精确等价，后者强依赖
特定导出图结构。应优先保留 PyTorch 源码级修复。

### 4.6 后续注意

`libs/lerobot` 是子模块，修改必须通过 `third_party/patches/lerobot/` patch 栈交付。后续提交前应
确认该修复已经导出为 patch，而不是只存在于本地子模块工作树。

## 5. 问题二：State projection 和 state token 完全缺失

### 5.1 现象

旧 RKNN 编排只拼接两路图像与语言：

```text
128 image tokens + 48 language tokens = 176 tokens
```

由于 prefill 固定长度是 177，旧代码在末尾补了一个全零 token。模型表面上满足 shape，但没有使用
`observation.state`，策略对当前关节状态不敏感或明显偏离 PyTorch 基准。

### 5.2 根因

原始 SmolVLA 会执行：

```text
observation.state -> pad/truncate -> state_proj -> one prefix token
```

旧 RKNN 路径同时遗漏了以下环节：

- 输入 adapter 不读取 `observation.state`。
- runtime input 数据结构没有 state 字段。
- exporter 没有保存 `flow_model.state_proj`。
- manifest 没有声明 state projection artifact。
- prefix 构造没有追加 state token。

### 5.3 修复

隔离分支完成了以下修改：

- `VLARuntimeInputs` 增加 `state`。
- `SmolVLACompiledAdapter.prepare_inputs()` 强制读取 `observation.state`。
- exporter 生成 `state_projection.pt`。
- manifest 增加 `state_projection` artifact。
- runtime 加载 `Linear(32, 960)` 的 weight 和 bias。
- 6 维 state 右侧补零到 32 维，再投影为 `[1, 1, 960]`。
- state token 追加在 image 和 language token 之后。

历史修复文件：

- `src/inference_service/inference_service/core/compiled_policy.py`
- `src/inference_service/inference_service/core/rknn/smolvla/SmolVLARKNNModel.py`
- `src/model_utils/model_utils/smolvla_export/export_rknn_modules.py`
- `src/inference_service/test/test_compiled_policy.py`

### 5.4 当前主工作区状态

新架构已将该契约迁移到：

- `src/inference_service/inference_service/backends/rknn/backend.py`
- `src/model_utils/model_utils/smolvla_export/export_rknn_modules.py`
- `src/model_utils/test/test_smolvla_rknn_export.py`

当前 backend 会：

- 要求 manifest 声明 `state_projection`。
- 检查 artifact 格式。
- 加载 token embedding 和 state projection。
- 检查 state 最后一维与 projection 输入一致。
- 追加 state embedding 并生成 state mask。

当前 exporter 也会将 `state_projection.pt` 纳入统一 `inference_manifest.json`。

## 6. 问题三：Prefill attention mask 语义错误

### 6.1 现象

旧代码直接构造全 1 方阵：

```python
attention_mask = np.ones((batch, prefix_length, prefix_length), dtype=np.int64)
```

这会让：

- padding token 参与 attention。
- image/language token 看到后面的 state token。
- prefill KV cache 偏离 SmolVLA 原始 block attention 语义。

### 6.2 正确语义

SmolVLA prefix 同时维护：

- `prefix_pad_masks`：token 是否有效。
- attention marker：是否开启一个新 block。

本模型中 image 和 language marker 为 0，state marker 为 1。正确算法是：

```python
cumulative_blocks = np.cumsum(attention_markers, axis=1)
attention = cumulative_blocks[:, None, :] <= cumulative_blocks[:, :, None]
attention &= pad_mask[:, None, :] & pad_mask[:, :, None]
```

结果为：

- image/language token 只能看到有效的 image/language token。
- state token 可以看到前面的有效 token 和自身。
- padding 对应的行和列全部无效。

### 6.3 修复与测试

隔离分支在 `SmolVLARKNNModel.py` 中新增 `_build_prefix_attention_mask()`，并在
`test_smolvla_rknn.py` 中验证 block 和 padding 行为。

当前新 backend 的 `_execute_embedding()` 已实现相同的 cumsum block attention，并通过 manifest
binding 输出 `internal.attention_mask` 与 `internal.position_ids`。

## 7. 问题四：视觉尺寸和 padding 契约不一致

### 7.1 现象

真实相机输入为 NCHW `[1, 3, 480, 640]`，vision RKNN 为静态 `[1, 3, 512, 512]`。旧运行时
直接送入原尺寸，可能导致输入尺寸错误、buffer 解释错误或视觉 embedding 与 PyTorch 路径严重偏离。

### 7.2 正确预处理

对于 `480x640` 输入，应保持纵横比：

```text
[1,3,480,640]
    -> resize [1,3,384,512]
    -> top padding 128 rows
    -> [1,3,512,512]
```

历史修复新增 `_resize_with_pad_nchw()`：

- 只接受 NCHW RGB。
- 使用 bilinear resize，`align_corners=False`。
- 保持纵横比。
- 在左侧和顶部补零，与 LeRobot `resize_with_pad()` 对齐。
- 输出 contiguous array。

测试验证了 `480x640 -> 512x512` 后顶部 128 行为零，以及输入已经是 `512x512` 时保持不变。

### 7.3 仍需确认的归一化顺序

排查时发现一个需要继续核对的语义细节：标准 LeRobot 路径对图像值域与 padding 的处理顺序，可能
与历史 RKNN runtime 的“先在 `[0,1]` 空间 padding，再转换值域”存在差异。零 padding 在
`[0,1] -> [-1,1]` 后会变成 -1，而直接在归一化后补零表示的是中性值，两者并不相同。

后续应使用同一张真实图片，分别 dump：

- LeRobot PyTorch vision 输入。
- codec/processor 送给 RKNN vision 的输入。

然后逐元素比较，不能仅比较 shape。

### 7.4 当前架构注意

当前 RKNN backend 本身不再包含显式 resize helper，尺寸/layout 主要由 processor、codec 和 manifest
binding 决定。后续修改 codec 时必须保留：

- 等比例缩放。
- padding 方向。
- 值域转换顺序。
- NCHW/NHWC layout。
- FP16/FP32 dtype。

### 7.5 旧两阶段 ONNX exporter 仍使用居中 padding

当前 `src/model_utils/model_utils/smolvla_export/export_onnx.py` 中的
`resize_with_pad_onnx()` 会把高度和宽度差值平均分配到两侧，即使用居中 padding；而 LeRobot
`resize_with_pad()` 与历史 RKNN 修复使用左侧/顶部 padding。该文件还先执行
`img * 2.0 - 1.0`，再以 `pad_value=0.0` 补边。

这与 segmented RKNN 路径不是同一个导出入口，但如果后续继续使用 `export_onnx.py` 生成 VLM
ONNX，就会保留另一套视觉几何和值域语义。后续应统一预处理 contract，不能让两个 exporter 对同一
checkpoint 生成不同的 vision 输入。

## 8. 问题五：KV cache flatten 多出一个维度

### 8.1 现象与根因

历史 prefill 输出 32 个 tensor，每个为 `[1, 177, 5, 64]`。stack 后已经包含 batch 维：

```text
[32, 1, 177, 5, 64]
```

旧 reshape 又手动插入一个 `1`，得到错误形状：

```text
[16, 2, 1, 1, 177, 5, 64]
```

修复后历史 4-input action RKNN 所需形状为：

```text
[16, 2, 1, 177, 5, 64]
```

### 8.2 当前架构处理

当前统一 exporter 不再接受 flattened-KV action ABI，而要求 action 输入为：

```text
x_t, timestep, prefix_pad_masks,
past_key_0, past_value_0, ..., past_key_N, past_value_N
```

`write_smolvla_rknn_deployment()` 会明确拒绝第一个输入为 `past_kv_tensor` 的 legacy 模型，错误为：

```text
Legacy flattened-KV SmolVLA action RKNN is unsupported
```

这是比继续维护 flatten/reshape 更可靠的最终方向。已有旧 RKNN 文件不能直接混入新 manifest，必须
用 segmented action exporter 重新导出和转换。

## 9. 问题六：Prefill RKNN context 连续调用后失稳

### 9.1 现象

板端连续推理时，prefill context 在大约第 10 次请求附近会返回 NaN/Inf。诊断中定位到某个
prefill KV 输出（历史记录为输出索引 24）首先变为非有限值。输入图像本身、vision embedding 和
前几次请求均为有限值。

该问题具有以下特征：

- 不是固定图片内容触发，black、white、checker、random 都可正常工作。
- 新建 RKNN context 后恢复。
- 更像 RKNN runtime/driver context 生命周期问题，而不是模型数学公式必然产生 NaN。

### 9.2 运行时规避

历史编排在 `prefill()` 中执行：

1. 检查每个 prefill 输出是否全部 finite。
2. 第一次发现非有限输出时 release 当前 context。
3. 从同一 `prefill.rknn` 重新 load/init。
4. 原输入重试一次。
5. 第二次仍失败则抛出包含 shape、dtype、finite 数量、min/max 的错误。

单元测试使用 FakeRKNN 验证了 release、reload 和 retry 路径。

### 9.3 状态判断

这只是 workaround，不是根因修复。尚未确定问题与以下哪一项相关：

- RKNNLite 版本。
- `librknnrt`/驱动版本。
- toolkit 与 runtime 版本组合。
- NPU 内存复用或碎片。
- 模型特定 kernel。
- 多模型 context 加载顺序。

### 9.4 当前主工作区回归风险

当前 `backends/rknn/backend.py` 的 `RKNNSession.infer()` 只检查“是否有输出”，并通过 manifest
校验输出 shape/dtype，没有发现 `np.isfinite()` 检查或按 role 重建 context 的恢复逻辑。

因此迁移到新 backend 后，该板端问题可能再次静默出现。建议在新架构中增加通用但受控的保护：

- 对浮点 runtime 输出执行 finite 检查。
- 错误信息包含 role、output index、shape、dtype 和 finite 统计。
- 只对已知可恢复的 prefill role 做一次 session 重建和重试。
- 重建必须保持 lifecycle、共享 session 和并发锁语义正确。
- 记录 reload 计数到 health/diagnostics，不能静默恢复。

## 10. 问题七：错误输出缺少 fail-fast 诊断

历史运行时补充了以下检查：

- Vision 输入和输出必须 finite。
- Prefill 每个输出必须 finite。
- Action KV、time、noise 输入必须 finite。
- Action 输出必须为 `[1, chunk_size, max_action_dim]`。
- Action 输出必须 finite。
- Action velocity 全零时立即报错，提示可能是 runtime CPU fallback 失败。
- 去噪错误包含 step 编号。

这些检查把“机器人不动或动作异常”转换成了具体阶段、具体 tensor 的错误。

当前新 backend 已有严格的 manifest input/output index、shape 和 dtype 校验，但尚未看到 finite 或
全零检查。建议优先恢复 finite 检查。全零检查应谨慎处理，因为理论上模型可能合法地产生全零或
极小 velocity；更稳妥的做法是记录 warning 和统计，或只在已知异常版本组合下提升为错误。

## 11. 问题八：Action 导出 ABI 曾经分裂

排查时存在两套 action 契约：

- 历史 runtime 使用 4 个输入：flattened KV、prefix mask、time、noise。
- segmented exporter 使用 `x_t`、timestep、prefix mask 和 32 个独立 KV，共 35 个输入。

曾发现相邻目录中的 action ONNX 与 action RKNN 并非同一 ABI 链路生成，存在旧 RKNN 文件被复制到
新输出目录的可能。这类错误仅凭文件名无法发现。

当前新 exporter 已通过 compiler 生成的 `*.abi.json` 和统一 manifest 强制执行 segmented ABI，
并拒绝 legacy flattened-KV。后续必须遵守以下规则：

- 不手工复制或重命名旧 RKNN 冒充新模型。
- 不根据文件名猜测输入顺序。
- package 前必须读取 compiler-resolved ABI。
- manifest 中的 bindings 必须与 RKNN runtime 的真实 input/output index 一致。
- 每次转换后用 production strict loader 回读 deployment。

## 12. 问题九：模型制品和 manifest 不完整

修复 state token 后，SmolVLA RKNN bundle 至少需要：

```text
vision RKNN artifact(s)
prefill RKNN artifact
action RKNN artifact
token_embedding.pt
state_projection.pt
config.json
processor files
tokenizer files
inference_manifest.json
compiler-resolved ABI metadata used during packaging
```

隔离工作副本的本地模型目录一度缺少 `state_projection.pt` 和新 manifest，导致代码修复后仍无法直接
加载。板端测试通过手工补齐 artifact 和配置完成，但长期方案必须由 exporter/package 流程生成，
不能依赖板端手改 JSON。

当前 `write_smolvla_rknn_deployment()` 已把 state projection 纳入统一 manifest，后续应坚持使用该
打包入口。

## 13. 问题十：版本信息容易误判

排查初期依据旧说明认为主机 RKNN toolkit 为 2.3.2，实际 `.venv-rknn` 为 `2.4.2a8`，模型 compiler
版本为 `2.4.2a2`。板端观测到：

- RKNNLite Python package：2.3.2。
- `librknnrt`：2.4.1b0。

版本字符串来自不同组件，不能只记录一个“RKNN 版本”。每次复现必须分别记录：

- 主机 `rknn-toolkit2` 版本。
- 模型内 compiler/toolkit 版本。
- 板端 `rknnlite` Python package 版本。
- 板端 `librknnrt.so` 版本。
- NPU driver 版本。
- SoC 和系统镜像版本。

版本不完全一致不一定立即加载失败，但会增加数值和 context 稳定性风险。

## 14. 板端验证结果

### 14.1 分阶段张量诊断

分阶段诊断确认以下张量均为 finite 且非全零：

- Vision 输出：`[1, 64, 960]`。
- Prefix：`[1, 177, 960]`。
- 32 个 KV：每个 `[1, 177, 5, 64]`。
- 历史 flattened KV：`[16, 2, 1, 177, 5, 64]`。
- 每个去噪 step 的 velocity：`[1, 50, 32]`。
- 最终 action chunk：`[1, 50, 32]`。

### 14.2 图像压力测试

使用 black、white、checker、random 四种 `480x640` 图像循环 12 次。日志
`/tmp/opencode/smolvla_480x640_stress.log` 和
`/tmp/opencode/smolvla_image_stress12.log` 显示：

- 12/12 prefix 全部 finite，`169920/169920`。
- 12/12 KV 全部 finite，`1812480/1812480`。
- 12/12 action 全部 finite，`1600/1600`。
- Action 不是全零。

另一轮包含自动 context 重载的直接板端压力测试同样完成 12 次请求；第 10 次附近触发 prefill
重载后恢复。该事件来自当时终端验证记录，现有保存的简化 stress 日志未包含 reload 文本，因此后续
复现时应把 runtime 日志和 tensor 统计统一保存。

### 14.3 ROS 纯推理链路

`PureInferenceNode` 首次成功结果示例：

```text
latency=5049.0ms, action_shape=[50, 6]
```

完整 round-trip 测试通过当时的批处理输入与动作输出接口连续发送 12 次请求，结果为：

- 12/12 请求成功。
- 每次输出 `[50, 6]`。
- 每次 `300/300` 元素为 finite。
- 单次端到端耗时约 6.4 秒量级。

### 14.4 完整 mock simulation

板端使用 `ROS_DOMAIN_ID=53` 启动：

```sh
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=model_inference \
    use_sim:=true \
    default_task:="pick"
```

该配置使用 `hardware_mock/contract_mock`，不是 Gazebo。验证到：

- 当时的 ACT 推理节点、`/action_dispatcher`、`/contract_mock` 正常出现。
- 首次推理总耗时 `7839.8ms`，其中 inference `7674.7ms`。
- Dispatcher 收到 `chunk=50` 的有效动作。
- 动作 topic 输出 `[50, 6]`，机械臂 command topic 持续发布。
- 运行过程中累计完成多次推理，功能链路可恢复并继续运行。

### 14.5 清理结果

测试结束后，launch 父进程退出但部分 ROS 子进程成为 orphan。最终单独停止：

- launch supervisor。
- `contract_mock`。
- `action_dispatcher`。
- RKNN inference worker。
- `ros2 topic hz` 监控进程。

其中 inference worker 未响应 graceful termination，最后对其使用 SIGKILL。清理后
`ROS_DOMAIN_ID=53` 下无测试节点残留。

这也说明板端测试脚本不能只 kill launch shell PID，应管理进程组或在退出后检查 ROS node 和子进程。

## 15. 尚未解决的问题

### 15.1 推理延迟导致队列饥饿

系统控制频率为 20 Hz，50 个 action 的 chunk 只能覆盖：

```text
50 / 20 Hz = 2.5 s
```

但板端推理耗时约 4.2 至 7.8 秒，首轮甚至接近 7.84 秒。因此 action dispatcher 的 queue 会反复
耗尽，日志持续出现：

```text
running but queue empty
```

并观察到一次 10 秒 action-server timeout。系统之后能够恢复并继续推理，但模拟运动会间断或使用
hold action，无法称为平滑闭环。

可评估的方向：

- 降低 `num_steps`，但必须评估策略质量。
- 增大 `chunk_size` 或实际执行步数，但需要重新导出静态 action 模型并评估开环误差。
- 降低控制频率，但会影响控制品质。
- 提前触发下一次推理或使用双缓冲。
- Dispatcher 在队列耗尽时采用明确的 hold/插值策略。
- 优化 RKNN 模型、量化和 NPU core 调度。
- 对 vision/prefill 复用做性能分析，避免不必要的 host copy。

### 15.2 Prefill context 根因未定位

自动重载只能恢复服务，不能证明底层问题已修复。需要在统一新 backend 上重新做 100 次以上压力
测试，并记录 reload 次数、NPU 内存、runtime/driver 日志和触发请求。

### 15.3 新 backend 缺少 finite/retry 保护

当前主工作区迁移已保留 state token、block attention 和 segmented cache ABI，但未看到历史编排中的
finite 检查和 prefill reload。该项应视为高优先级回归修复。

### 15.4 图像预处理数值一致性未完成

尺寸和 padding shape 已验证，但尚需对 LeRobot 与 RKNN 的最终 vision 输入做逐元素对比，特别是
padding 区域和 `[0,1] -> [-1,1]` 的顺序。

### 15.5 全零 velocity 判断需要策略化

历史代码将全零 action velocity 直接视为 runtime failure。该规则对本次 CPU fallback 故障有效，
但不一定适用于所有模型输入。建议新 backend 将其作为可配置诊断或连续多 step 异常判断。

### 15.6 热循环 host copy 开销

10 步 denoise 每一步都需要准备 noise、time 和 cache 输入。历史修复为安全检查增加了 dtype/contiguous
转换，可能带来额外 host copy。新 backend 也应通过 profiling 判断是否重复转换不变 KV cache。

## 16. 已知测试覆盖和缺口

隔离分支已运行并通过：

- 32 个 inference-service 相关测试。
- 5 个 RKNN ONNX compatibility 测试。
- 本轮修改文件的 Ruff lint 和 format。

已有单元测试覆盖：

- `480x640 -> 512x512` top-left padding。
- 已匹配尺寸时保持 contiguous 和数值不变。
- block attention 排除 padding。
- prefill 首次 NaN 后 reload/retry。
- adapter 将 state 传给 runtime。
- FLOAT/FLOAT16 `ReduceMin` 可接受。
- INT32/INT64 `ReduceMin` 被拒绝。
- 新 exporter 拒绝 legacy flattened-KV action ABI。
- 新 manifest 包含多相机 role、embedding、prefill、action 和 state projection。

仍建议补充：

- 完整 prefix 拼接结果，确认 state 恰好位于第 177 个 token。
- state 从 6 维补齐到 32 维并投影的数值测试。
- 多 batch 行为。
- prefill 重试第二次仍失败的错误路径。
- action finite、shape 和全零诊断。
- 新 `RKNNBackend` 的 session reload 测试。
- LeRobot 与 RKNN 图像预处理逐元素一致性测试。
- 新架构下 12 次、100 次连续板端请求测试。
- toolkit/runtime 版本矩阵回归。

## 17. 排查过程中使用的诊断工具

以下脚本位于隔离工作副本根目录，主要用于一次性定位，不应未经整理直接提交到仓库根目录：

| 脚本 | 用途 |
|------|------|
| `analyze_onnx_reductions.py` | 统计 reduction 节点、dtype、shape 和上下游 |
| `trace_onnx_value.py` | 反向追踪 ONNX value producer |
| `generate_reducemin_matrix.py` | 生成不同 dtype/opset 的最小 ReduceMin 模型 |
| `convert_reducemin_matrix.py` | 将最小模型转换为 RKNN |
| `test_reducemin_matrix_board.py` | 板端验证不同 dtype 的实际 runtime 行为 |
| `patch_int64_reducemin.py` | 试验 Cast workaround |
| `patch_reducemin_to_prefix_offset.py` | 试验基于图拓扑的替换 |
| `verify_reducemin_cast_equivalence.py` | 验证小整数范围内 Cast 等价性 |
| `compare_onnx_models.py` | ONNX Runtime 比较原图与 patch 图 |
| `test_action_rknn_board.py` | 独立验证 action RKNN shape、finite 和非零输出 |
| `test_smolvla_action_boundary_board.py` | 验证 KV 编排到 action 输入边界 |
| `diagnose_smolvla_pipeline_board.py` | 分阶段统计 vision/prefix/KV/action |
| `stress_smolvla_images_board.py` | 多图片、多请求压力测试 |
| `dump_smolvla_preprocessor_board.py` | dump processor 最终 batch |

其中 `diagnose_smolvla_pipeline_board.py` 和 `stress_smolvla_images_board.py` 基于历史
`SmolVLARKNNModel` API。当前 backend 已迁移后，这两个脚本不能直接运行，需要改为通过统一 manifest、
codec 和 `RKNNBackend` 发起请求，否则会缺少 state projection 参数和新的 prefix/prefill 接口。

排查还产生了 `check0_base_optimize.onnx`、`check2_correct_ops.onnx`、
`check3_fuse_ops.onnx` 等 RKNN compiler 中间图，单个约 384 至 394 MiB。这些是临时诊断产物，
不应纳入 Git。

## 18. 后续修复优先级

1. 在当前 `RKNNBackend` 增加浮点输出 finite 检查，并为 prefill 实现一次可观测的 session 重建重试。
2. 确认 `normalize_expert_position_ids()` 已通过 LeRobot patch 栈纳管，并恢复导出阶段整数
   `ReduceMin` fail-fast 检查。
3. 在当前统一 exporter 和 backend 上重新生成完整 bundle，不复用历史 flattened-KV RKNN。
4. 新增 LeRobot 与 RKNN vision 输入逐元素一致性测试，解决归一化与 padding 顺序疑问。
5. 统一 `export_onnx.py` 与 segmented exporter 的 padding 方向和值域 contract。
6. 在板端执行至少 100 次连续请求，并记录 finite、reload 次数、延迟分布和内存。
7. 设计 queue starvation 的调度策略，再评估模型侧性能优化。

## 19. 复现时的最小检查清单

- [ ] 记录 toolkit、compiler、RKNNLite、`librknnrt`、driver 和系统版本。
- [ ] 确认 bundle 含 `state_projection.pt`，manifest 声明该 artifact。
- [ ] 确认 prefix 长度等于相机 token + language token + 1 state token。
- [ ] 确认 attention mask 不是全 1，而是 block attention。
- [ ] 确认 action ABI 为当前 segmented cache 契约。
- [ ] 扫描 action ONNX，不允许 INT32/INT64 `ReduceMin`。
- [ ] 对 vision、prefill、action 的输入输出逐阶段检查 finite。
- [ ] 连续执行至少 12 次请求，不只验证首帧。
- [ ] 检查 action 不是静默全零，输出 shape 与 manifest 一致。
- [ ] 测量推理耗时是否小于 action chunk 可消费时长。
- [ ] 测试结束后检查 ROS 节点和 orphan 子进程是否清理完成。

## 20. 最终判断

本轮故障不是单一 RKNN 算子问题，而是多个独立缺陷叠加：

- 导出图中存在板端不支持的整数 reduction。
- Runtime 丢失 state token。
- Attention mask 语义错误。
- Vision 静态尺寸预处理缺失。
- KV ABI 编排错误。
- RKNN prefill context 存在长期运行稳定性问题。
- 推理性能无法覆盖实时动作消费速度。

修复前，模型可能“能加载、能返回 shape”，但语义和数值都不可信。后续验收不能只检查 API 成功，
必须同时验证模型制品契约、每阶段 tensor 数值、连续请求稳定性和 ROS 闭环时序。
