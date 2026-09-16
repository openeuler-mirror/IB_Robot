# inference_manifest

`inference_manifest` 定义 IB-Robot 推理 bundle 的硬件无关 schema v3 契约。加载器只接受
schema version 3，不翻译旧身份或旧 runtime 别名。

模型结构使用 `execution`、`bindings`、`device_links` 和 `execution_contract` 表达。
manifest 不声明异步提交、dataset 隔离或发布正确性等运行时保证；这些能力由加载后的 runtime 验证。
PI0.5 policy adapter 在运行时检查可复用 prefix 的输入依赖和状态契约，backend 检查异步隔离及 priority streams，
executor 负责触发、快照时效和代际隔离。机器人配置的 `scheduling.stages` 仍按执行角色设置优先级。
旧实验 bundle 必须通过转换器重新发布 manifest，清除旧字段并刷新身份指纹；模型产物 ABI 未变时无需重新编译。

## Schema v3

调度身份永远是 `interface`、`model_type`、`operation` 三元组。Policy 使用
`operation: "predict"`，且 `model_type` 必须与 LeRobot `config.json` 中的 `type` 一致。
张量服务使用 `interface: "tensor_model"`；服务变体通过 `operation` 表达。

```json
{
  "schema_version": 3,
  "bundle": {
    "uuid": "6a7e8c42-1031-41f3-8cde-c1bf8738ca31",
    "revision": 3,
    "name": "speech-direction",
    "files": [{"path": "config/assets.json"}],
    "digest": {
      "algorithm": "sha256",
      "scope": "structure",
      "value": "<lightweight structural digest>"
    }
  },
  "model": {
    "interface": "tensor_model",
    "model_type": "speech_direction",
    "operation": "enhance_and_vad",
    "inputs": [{"semantic": "audio", "dtype": "float32", "shape": [1, 512]}],
    "outputs": [{"semantic": "audio", "dtype": "float32", "shape": [1, 512]}]
  },
  "deployments": {
    "ascend-310p": {
      "execution_contract": {
        "state_scope": "stream",
        "execution_structure": "direct",
        "cancellation_granularity": "checkpoint",
        "stateful": true,
        "state_bank_mode": "runtime_exclusive",
        "max_open_streams": 1,
        "state_links": [{
          "role": "enhancer",
          "state_name": "hidden.state",
          "owner": "session",
          "source": "session.state_in",
          "target": "session.state_out",
          "scope": "runtime",
          "state_bank": "enhancer.bank"
        }]
      },
      "role_identities": {
        "enhancer": {
          "interface": "tensor_model",
          "model_type": "fullsubnet",
          "operation": "enhance"
        },
        "vad": {
          "interface": "tensor_model",
          "model_type": "silero_vad",
          "operation": "vad"
        }
      },
      "role_runtime_profiles": {
        "enhancer": {
          "backend": "ascend",
          "target": {"soc": "Ascend310P", "runtime": "acl", "runtime_abi": "cann-8.0"},
          "profile": {"device_id": 0}
        },
        "vad": {
          "backend": "ascend",
          "target": {"soc": "Ascend310P", "runtime": "acl", "runtime_abi": "cann-8.0"},
          "profile": {"device_id": 0}
        }
      },
      "artifacts": {
        "enhancer": {"path": "artifacts/enhancer.om", "format": "om"},
        "vad": {"path": "artifacts/vad.om", "format": "om"}
      },
      "execution": ["enhancer", "vad"],
      "bindings": {
        "enhancer": {
          "inputs": [{"semantic": "audio", "index": 0, "dtype": "float32", "shape": [1, 512]}],
          "outputs": [{"semantic": "audio", "index": 0, "dtype": "float32", "shape": [1, 512]}]
        },
        "vad": {
          "inputs": [{"semantic": "audio", "index": 0, "dtype": "float32", "shape": [1, 512]}],
          "outputs": [{"semantic": "audio", "index": 0, "dtype": "float32", "shape": [1, 512]}]
        }
      }
    }
  }
}
```

`state_links` 只允许稳定的逻辑名称。路径、设备标识、lease 标识、stream 标识、进程 ID、
socket、资源句柄都会被拒绝。请求契约不包含 stream admission 与 state links。迭代式契约必须把
`orchestration_visibility` 声明为 `executor` 或 `session`；direct 契约则省略该字段。

规范模型映射如下：

| Service | Interface | Model type | Operation |
|---|---|---|---|
| ACT | `policy` | `act` | `predict` |
| PI0.5 | `policy` | `pi05` | `predict` |
| SmolVLA | `policy` | `smolvla` | `predict` |
| SAM2 | `tensor_model` | `sam2` | `prompt` or `automatic` |
| Grounding DINO | `tensor_model` | `grounding_dino` | `detect` |
| ZipVoice | `tensor_model` | `zipvoice` | `synthesize` |
| FullSubNet | `tensor_model` | `fullsubnet` | `enhance` |
| Silero VAD | `tensor_model` | `silero_vad` | `vad` |

## 类型化 Runtime Profile

单模型部署使用一个 `runtime_profile`；组合部署按 role 使用 `role_runtime_profiles` 条目。
Profile 是类型化的，分别暴露部署投影与 runtime 实例投影。Ascend profile 只包含
`device_id`；ACL 初始化使用注入的进程 provider，manifest 不再携带任何运行时 ACL 配置路径。

`ValidatedDeployment` 是加载器的不可变快照，包含已解析的 artifact 句柄、role 绑定、语义
契约、完整性状态、可移植的 deployment fingerprint，以及独立的完整 runtime profile /
实例 fingerprint。Deployment fingerprint 排除设备 ID、本地路径、provider 身份和完整实例
profile。

## 产物完整性

常规加载只校验声明的摘要、路径与结构，不对大文件计算 hash。
`verify_deployment_artifacts(bundle_root, deployment_name)` 执行显式内容校验，并在旁边写入
`inference_integrity.json` 报告。不匹配以稳定代码 `artifact_digest_mismatch` 上报；发布
代码必须拒绝处于 `mismatch` 状态的报告。

## 公开 API

| API | 用途 |
|---|---|
| `load_inference_manifest()` | 完整 schema、路径、语义与身份校验 |
| `load_inference_manifest_metadata()` | 边侧元数据校验 |
| `canonical_bundle_digest()` | 轻量 bundle 结构摘要 |
| `deployment_fingerprint()` | 可移植的部署身份摘要 |
| `runtime_profile_fingerprint()` | 完整 runtime 实例摘要 |
| `verify_deployment_artifacts()` | 安装时或按需的显式产物校验 |
| `write_inference_manifest()` | 规范化原子写入器 |
