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

每个 deployment 可携带 `extra_parameters`，默认 `{}`。它是模型拥有的任意 JSON 对象，
可嵌套数组、对象、数字、字符串、布尔值和 null；manifest 只检查 JSON 结构与有限数值，
不认识其中的模型字段。对应 provider 负责解释、默认值与支持范围的校验。

```json
"extra_parameters": {
  "denoising": {"solver": "ab2", "schedule": "uniform", "steps": 3},
  "language_capacity": 48
}
```

参数直接内嵌，无需额外参数文件或用户维护的 SHA-256 校验值。非空参数参与 deployment 与
runtime profile 的缓存身份，使同名部署重新导出后不会复用旧配置的图或性能档案；空对象与
未声明时保持旧身份。参数不是可执行类路径，不从 manifest 动态导入代码。
当前原生 LeRobot Torch session 将其交给 `PolicyProvider.resolve_load_options`；没有消费者时
非空参数会报错。其他模型/后端可实现各自消费入口，manifest 不预设统一参数语义。
新增字段需要支持它的运行时；旧版 strict schema 会拒绝该字段，而不是静默切换算法。

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


## 模型启动与运行身份

模型启动必须提供 bundle 目录和命名 deployment。
目录必须包含有效的 schema-v3 `inference_manifest.json`。
Torch 权重同样通过 bundle 声明。
系统不支持未打包本地文件的 native 身份快照或自动回退加载。

`load_inference_manifest()` 与 `load_inference_manifest_metadata()` 在读取模型资产和
初始化后端前检查 manifest 入口。
入口错误包括空路径、缺失目录、单个模型文件，以及缺失或非普通文件的 manifest。
加载器抛出带 `bundle_manifest_required` 的 `ManifestValidationError`。
错误消息提示先使用项目 exporter/packager 打包。

加载器继续校验现有 manifest 的 schema、命名部署和资产契约。
任何校验失败都阻止模型加载。
远端元数据入口允许部署产物位于远端，但必须提供有效的本地 bundle manifest。

通用模型服务在创建 plugin/session 前执行这些检查。必需服务初始化失败时抛错；可选服务
保持诊断接口，报告 failed / not ready，不创建模型实例、不执行推理。

`runtime_identity` 提供 `physical_device()`、`runtime_composition_fingerprint()`、
`validate_runtime_backend_support()` 和 `validate_runtime_device_visibility()`，不依赖 ROS 或模型 SDK。
`validate_runtime_backend_support()` 只验证 profile 存在及 backend 有映射，不验证单设备放置或可见性。
组合部署允许 CPU 辅助角色，但 `physical_device()` 必须解析到一个主要物理设备。
Torch 的 `cuda` / `npu` 不隐含索引 0：调用方必须通过 `torch_device_index` 显式传入模型加载、
执行上下文所使用的 SDK 当前设备索引，并保持该上下文稳定。未提供或索引非法时拒绝生成身份；
设备可见性与枚举约定仍须单独通过 `validate_runtime_device_visibility()` 检查。
设备标识只接受 `local/cpu`、`local/mps`，或 `local/<backend>:<index>`；backend 为
`cuda`、`npu`、`rknn`、`hmm`、`hisilicon`，index 为无前导零的非负十进制整数。
名称生成与格式校验共用设备种类定义；backend 名称不必等于设备种类，例如 `ascend → npu`。
非法标识先报 `ValueError`，不会被当作无需可见性检查的设备。
CUDA 宿主必须在 SDK 初始化前设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，且不设置
`CUDA_VISIBLE_DEVICES`（包括空值）；默认 FASTEST_FIRST 和其他枚举顺序均拒绝。
所有参与身份比较的宿主须使用同一稳定设备清单；不支持容器设备重映射、热插拔或 SDK
初始化后修改这些环境变量。检查只验证宿主配置，不查询或修正已经初始化的 SDK 状态。
HMM 设备按 typed profile 与 runtime options 解析，ONNX provider 别名规范化后参与身份。
`physical_device()` 的 runtime options 只接受 mapping 或 `None`，空列表、空字符串也拒绝。
RKNN 身份仅支持现有 RKNNLite 的本地默认设备：`device_id` 省略或为 `0`，`device` 省略或为
`rknn:0`。其他设备声明报 `ValueError`，包括组合 deployment 中的角色声明；不推断多设备支持，
不改变已有 schema 或加载链路。
组合指纹包含非空白 adapter 标识、stage 和严格 JSON options，拒绝未解析的空 adapter 以及
隐式字符串化造成的身份冲突；合法标识保持原值，不进行可能合并不同标识的自动裁剪。
`fingerprint_format_version` 由 `RUNTIME_COMPOSITION_FINGERPRINT_VERSION` 定义；组合载荷结构或
规范序列化规则改变时递增。它不是 adapter ABI；adapter 行为不兼容变化须由调用方更新 adapter
标识。ONNX session 与身份计算共用 profile 的 provider 解析，其 SDK 可用性检查仍由 session 负责。
格式 v1 的固定摘要测试包含嵌套 options、中文和 stage，约束 Unicode 转义与键排序规则。
身份函数供后续集成使用，本次不接入模型构造或调度。
HMM 身份函数的设备优先级测试不代表 policy 装配链已经接通：既有装配器仍会在 options
未声明设备时填入 `device_id=0`，与非零 typed profile 可能冲突。后续加载集成需统一设备解析并
补装配契约测试；本次不修改该既有装配行为。

## 发布产物完整性

导出及打包调用方在最终文件落盘后调用 `finalize_deployment_artifacts()`，将 SHA-256 写入
artifact 声明，再通过 `inherit_deployment_version()` 继承 UUID、按声明变化递增 revision。
新产物不带 SHA；保留产物必须保留原 SHA，终结时发现内容不符立即拒绝，不能重新认可未知变化。
`upsert_deployment()` 在发布边界从旧 manifest 建立索引，按当前路径对应的文件身份恢复旧证据；
复用未带 SHA 的旧 candidate、清空或改写 candidate 的 SHA 都不能重新认可已变化的发布文件。
合法更新使用独立 generation；没有旧摘要的 legacy 声明仍只能绑定当前字节。
revision 比较保留 JSON 布尔值与数值的差别。感知 sidecar 复用已计算摘要，避免重复读取大模型文件。
这里改用 manifest 作为 compiled **deployment revision** 的依据，不改变普通／Ascend 入口
既有的 **bundle revision** 策略：修复 sidecar 摘要仍会递增 bundle revision，继而改变 bundle
digest 和最终 fingerprint。普通入口沿用 sidecar 文本比较（格式变化也计入），Ascend 比较解析后的
字典。不能据 deployment revision 不变推断完整运行身份不变；修复后的再次相同发布保持稳定。

感知打包入口在覆盖任何已发布文件前，先完成所有待保留角色的摘要校验；拒绝保留产物时不改变
发布目录。预检查同时覆盖 adapter、摘要 sidecar 和复制的 assets；它们与 artifact 或 OM 复制源
存在符号链接或硬链接别名时，在写入前拒绝发布，避免原地写入使已校验摘要失效。
所有 OM 导入目标及 asset 写入目标先规范化 bundle-relative path，再解析父目录和最终路径；
即使目标尚不存在，也拒绝越过 bundle 根目录的符号链接。asset 写入目标之间不能解析到
相同路径或共享 inode；不支持这种布局，调用方须先提供独立目标。
文件发布目标不能是目录或指向目录的软链接，以免替换目录入口后破坏保留文件的祖先路径。
已存在的文件目标必须是普通文件，已有父路径必须是目录。
所有文件目标在解析后不能互为父子路径。预检拒绝 FIFO 和 socket，避免 asset 写入阻塞。
普通 Torch 必需及保留资产、旧 bundle.files、asset 复制输入、manifest 和 integrity report
也参与读写冲突检查。
保留资产不能通过直接或多跳符号链接依赖本次将替换的 manifest 或将删除的完整性报告；
预检在计算 sidecar 摘要前拒绝，
避免发布后摘要立即失效。指向未改写 bundle 内文件的链接仍受支持。
复制输入可以复用自身目标；与另一个 asset 写入目标别名，或依赖即将替换的 artifact 时，必须
在首次内容写入前拒绝。预检不提供并发路径替换防护。
普通感知入口通过元数据 loader 读取旧声明，允许重建缺失的 adapter、摘要 sidecar，
以及仍有编译源的 OM，然后执行完整加载验证。已声明的 OM 与编译源都缺失时，在写入前拒绝发布，
不会隐式删除 deployment；首次发布仍可省略没有提供的可选部署。
普通、Ascend 和 GraspGen 感知入口共享文件身份及摘要继承规则，包括规范化路径、符号链接和硬链接别名；
`PublishedArtifactIndex` 按文件身份索引整个旧 manifest 的摘要，复制前验证跨角色、跨 deployment
复用的源文件，避免复制或重命名丢失旧证据。`preflight_publication()` 接收每个部署的
`ArtifactPublication`，集中展开角色、验证并回填保留摘要，再返回发布对象与未替换 deployment；
旧 manifest 只从 `PublishedArtifactIndex.manifest` 获取。
调用方把发布对象直接交给复制终结函数，不再拆装映射或重复终结保留角色，
别名输入使用同一索引共享读取。复制后的目标仍独立终结。
多个导入角色经父目录链接解析到同一个目标目录项时，写入前拒绝；不同目录项的硬链接仍可
通过原子替换各自更新，而不改写共享 inode。
替换依赖检查共用一个目标集合；排除当前角色自身目标后，仍检查整条符号链接链。
索引的旧声明映射是在写入前建立的文件身份快照，不随 manifest 或替换后的目标自动更新。
预检证明目标互不冲突，且源和保留路径不依赖替换目标。
复制阶段直接消费预检结果。
准备阶段的每次摘要查找仍核对文件身份、大小、mtime 与 ctime。
只有相同文件状态才能复用内容缓存。
复制后的目标独立终结，不传旧索引；原地修改 artifact 或改变已索引别名会破坏使用前提。
发布结束丢弃索引，下次发布重建；旧摘要冲突时拒绝复用。此约定依赖整个发布期间只有单一写入者，
三个感知 CLI 目前不提供 bundle 写锁，并发调用同一 bundle 不受支持。
`integrity_report_path()` 统一提供 bundle 根目录报告路径，供读取、预检、摘要排除和清理使用。
Ascend 摘要清单仅排除该报告的准确路径，子目录中的 `inference_integrity.json` 仍是普通资产。
三个入口在 manifest 写入和加载校验成功后删除旧完整性报告。
恢复原始字节时，deployment UUID/revision 保持不变，loader 返回声明状态。
新的内容校验报告仍须显式生成。报告失效不通过增加模型版本实现。
索引构造保留缺失旧文件可由独立编译源修复的行为；其他文件系统错误统一抛出
`ManifestIntegrityError` / `artifact_integrity_unverified`，附 deployment、role、path 和原始异常原因。
其 `read_and_verify_digest()` 读取内容或复用相同文件状态的缓存，并核对已有声明；
`verify_declared_digest()` 在没有旧声明时返回 `None`，不会读取内容，不能据此认定文件已验证。

摘要接纳、未核对旧声明的哈希读取和逐块比较是索引内部操作。
调用方使用 `read_and_verify_digest()`、`verify_declared_digest()` 或 `reusable_digest()`。
`bind_artifact_digest()` 在绑定时校验小写 64 位 SHA-256 类型约束。
首次发布到不存在的目标时，独立输入不需要比较用预哈希，但保留关联旧声明的源校验和复制后的
最终目标校验。通用 generation 复用先按文件大小及权限排除候选；有旧声明时用其 SHA 排除
不匹配候选。共同完整性判定由 `PublishedArtifactIndex.reusable_digest()` 返回可复用摘要。
路径、权限、generation 搜索和替换策略仍属于各调用方。

没有旧声明时，索引按当前文件状态查询内容摘要缓存。
只有两侧都有缓存摘要时，索引才直接比较摘要。
否则索引按 64 KiB 逐块比较，遇到首个不同块即退出。
索引同步累计相同字节的 SHA-256，无需额外哈希遍历。
只有一侧缓存时，完整匹配仍需读完两侧文件。
这保留了逐块提前退出能力，不增加分块缓存或失效状态。

匹配后校验两侧读前读后的文件身份、大小、mtime/ctime，再核对已有证据并接纳摘要。
64 KiB 是单块大小，不是整个历史搜索的读取预算；沿用原有全历史搜索，G 个候选都在末尾才
不同的最坏读取量仍为 O(G × 文件大小)，本次未新增全局读取上限。
测试中的读取预算仅统计指定路径经 `Path.open(...).read()` 返回的摘要／比较字节。
该计量不覆盖 `shutil.copy2()`、其他读取 API 或未列入集合的别名，不代表整个发布的 I/O 上限。

三个感知入口共用预检分类结果。
预检把相等导入转为已验证的保留角色，只在 `sources` 中保留待复制角色。
复制阶段直接执行预检结果，不再重复比较源与目标。
调用方须保证单一写入者，并在预检和复制之间保持源与保留文件不变。
预检拒绝本次写入与源或保留文件之间的别名依赖。
更新使用临时文件加原子替换，避免修改其他目标共享的 inode。

复制后仍独立校验最终文件。临时文件使用同目录短名称，不在 artifact 文件名上追加
后缀，支持文件系统允许的长文件名。
独立输入的摘要与旧目标声明不同、且旧声明没有冲突时，直接判定需要替换，不读取旧目标内容；
源证据校验与复制后的最终目标校验保持不变。
通用 compiler 打包链也保留复用产物的旧 SHA；独立输入不会被去重到已知损坏的 generation，
一次 deployment 构造只解析并索引一次旧 manifest。
这不构成通用文件系统事务，
后续复制或 I/O 失败仍不承诺回滚。

`validate_deployment_artifact_integrity()` 只读验证声明及内容，不生成报告；
`verify_content=False` 只检查摘要声明，用于远端元数据。无 artifacts 只允许全部 profile 为 Torch；
携带 compiled artifacts 的 Torch 部署同样接受校验。缺失摘要和内容不符分别抛出
`ManifestIntegrityError`，代码为 `artifact_integrity_unverified` 和 `artifact_digest_mismatch`。
普通加载与远端声明校验共享同一声明状态判定，显式校验额外报告缺失摘要的角色。
这里的 verified 只表示声明与文件字节一致，不证明模型可执行。
完整性检查不新增 compiled 模型的非空或格式有效性约束。
允许为空的辅助 artifact 仍遵循同一摘要契约。
compiled artifact 的符号链接环在路径边界转换为路径错误，再由完整性入口包装为
`ManifestIntegrityError` / `artifact_integrity_unverified`。

显式校验报告绑定完整 deployment 声明，重新发布后旧报告不能证明新产物通过校验；
按需加载也核对被校验声明与加载快照一致。普通加载、写入不会自动刷新 artifact SHA，
bundle 路径沿用已有解析规则。发布方须保证从产物终结到 manifest 发布之间没有其他写入者。
