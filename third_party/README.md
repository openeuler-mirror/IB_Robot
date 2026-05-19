# third_party 目录说明

本目录用于存放 **IB_Robot 对上游第三方依赖的受控定制内容**。

当前目录下主要维护的是 LeRobot 的补丁栈：

```text
third_party/
└── patches/
    └── lerobot/
        ├── INDEX.yaml
        └── v0.5.1/
            ├── 0001-*.patch
            ├── 0002-*.patch
            ├── ...
            ├── manifest.yaml
            ├── series.txt
            ├── series.master-parity-candidates.txt
            └── series.openharmony-5.1.0-musl.txt
```

## 设计目标

`third_party` 不是简单的“源码备份目录”，而是把对上游依赖的本地改造以 **可审计、可选择、可复现** 的方式管理起来。

以 `third_party/patches/lerobot/` 为例，这套机制主要解决三个问题：

1. 在不直接长期分叉上游仓库的前提下，为 `libs/lerobot` 叠加 IB_Robot 所需改造。
2. 针对不同平台或运行场景，按条件选择需要应用的补丁，而不是所有主机都打同一套补丁。
3. 将旧分支中沉淀的功能迁移拆分为独立 patch，便于逐个验证、逐个纳管。

## LeRobot 补丁栈总览

### 1. `INDEX.yaml`

位置：`third_party/patches/lerobot/INDEX.yaml`

作用：

1. 作为 **LeRobot patch stack 的上层索引**。
2. 指定当前激活的上游 tag，例如 `active_tag: v0.5.1`。
3. 将 tag 字符串映射到实际目录，例如 `v0.5.1 -> third_party/patches/lerobot/v0.5.1/`。
4. 记录该 tag 对应的上游 commit 与默认分支名。

它是整个补丁栈的入口。`scripts/setup/lerobot_patches.sh` 不会直接硬编码某个 tag 目录，而是先通过 `scripts/setup/lerobot_resolve_active.py` 解析 `INDEX.yaml`，再进入对应版本目录。

### 2. `<tag>/000x-*.patch`

例如：

- `0001-python-compat-syntax-and-metadata.patch`
- `0004-openharmony-lazy-import-policy-stack.patch`
- `0005-compat-add-npu-device-detection.patch`
- `0011-knowledge-distillation.patch`

这些文件是 **真正修改 LeRobot 源码的补丁单元**。

它们具有以下特点：

1. 由 `git format-patch` 导出，是标准 mailbox patch。
2. 设计给 `git am` 直接应用，而不是普通 `git diff` 文本。
3. 一个 patch 对应一个相对独立的逻辑改动，便于审查、回滚和组合。

按内容大致可以分成几类：

#### Python 兼容性补丁

- `0001`：把 Python 3.12-only 语法和元数据降到 Python 3.11 可用。
- `0002`：继续把最低 Python 版本要求降到 3.10。
- `0003`：为 Python 3.10 回补 `typing.Unpack` 兼容层。

这类 patch 解决的是“能否安装、能否解释执行”的问题，不直接增加业务功能。

#### 平台运行时补丁

- `0004`：为 OpenHarmony 板端推理路径做 lazy import，减少板端运行时依赖压力。

这类 patch 面向特定平台运行时，重点是裁剪不必要的训练期或数据集依赖。

#### 设备能力补丁

- `0005`：给 LeRobot 的 `device_utils.py` 增加 `npu` 识别与选择能力。

这类 patch 主要解决设备发现与设备选择问题，是 Ascend/NPU 接入的基础能力。

#### 训练栈迁移补丁

- `0009`：自适应样本加权的前置改造。
- `0010`：weighted training 与量化约束相关扩展。
- `0011`：知识蒸馏训练支持。
- `0014`：训练过程 TensorBoard 日志。

这类 patch 主要作用在训练配置、训练脚本和 policy training forward 路径上。

#### 模型与工具扩展补丁

- `0012`：恢复 `mt_act` 模型族。
- `0013`：恢复 attention visualization 等研发工具。

这类 patch 更偏模型能力恢复或研发辅助工具，不一定进入默认运行时补丁栈。

### 3. `<tag>/series*.txt`

这些文件是 **补丁装配顺序表**，决定“按什么顺序尝试应用哪些 patch”。

#### `series.txt`

默认补丁序列。

它代表常规 `setup.sh` / 默认环境下使用的补丁候选集合，但是否最终应用，还要继续经过 `manifest.yaml` 的条件过滤。

#### `series.master-parity-candidates.txt`

主线迁移候选序列。

它用于保存从旧 `lerobot_ros2` 历史中抽取出来、但暂时 **不进入默认补丁栈** 的功能 patch。适合做 feature migration 和 master parity 验证。

#### `series.openharmony-5.1.0-musl.txt`

OpenHarmony 5.1.0 musl 运行时专用序列。

目前只包含板端实际验证所需的最小 patch 集合，用于尽可能保持上游 Python 3.12 行为，同时只修复板端推理真正依赖的问题。

### 4. `<tag>/manifest.yaml`

这是 **每个 tag 目录下最核心的元数据文件**。

它负责描述：

1. 当前 patch 栈绑定到哪个上游 tag 和 commit 范围。
2. 这些 patch 的来源分支、参考 commit、应用方式。
3. 支持哪些 profile，例如 `openharmony`、`ascend`、`training`、`distillation`。
4. 每个 patch 的用途与适用条件。

可以把它理解成“补丁栈契约”。

其中最关键的是 `patches:` 段。每个 patch 都会声明：

1. `file`：patch 文件名。
2. `purpose`：这个 patch 为什么存在。
3. `applies_to`：该 patch 在什么条件下才允许应用。

`applies_to` 目前支持：

1. `python_min`
2. `python_max`
3. `profiles`

这意味着：

1. `series.txt` 只定义“候选顺序”。
2. `manifest.yaml` 决定“当前主机是否真的应该应用这个 patch”。

例如：

1. Python 兼容补丁只应在低版本 Python 主机上生效。
2. OpenHarmony lazy-import 补丁只应在 `openharmony` profile 下生效。
3. 训练相关 patch 不应无条件打到纯推理场景。

## 实际应用流程

LeRobot patch stack 的生效路径大致如下：

1. `scripts/setup/lerobot_patches.sh` 读取 `third_party/patches/lerobot/INDEX.yaml`。
2. `scripts/setup/lerobot_resolve_active.py` 解析当前激活 tag。
3. 进入对应目录，例如 `third_party/patches/lerobot/v0.5.1/`。
4. 根据目标 profile 选择一条 `series*.txt`。
5. 使用 `scripts/setup/lerobot_filter_series.py` 按 `manifest.yaml` 中的 `python_min/python_max/profiles` 过滤 patch。
6. 对保留下来的 patch 按顺序执行 `git am`。

因此：

1. 目录中的 patch 不会无条件全部应用。
2. 同一个 tag 目录下可以同时维护多条 patch profile。
3. 平台差异由 `series*.txt + manifest.yaml` 联合表达，而不是复制多份源码。

## 为什么要这样做

相比直接长期维护一个打满补丁的 `libs/lerobot` 分支，这种方案的优点是：

1. **可审计**：每个改动都是独立 patch，有清晰的标题、作者、用途和适用边界。
2. **可裁剪**：不同主机只应用自己需要的 patch。
3. **可迁移**：旧功能可以逐个迁移、逐个验证，而不是一次性整包回灌。
4. **可升级**：未来升级到新的 LeRobot tag 时，可以新建一个 tag 目录并逐步搬运 patch。

## 当前目录的建议理解方式

如果只看 `third_party/patches/lerobot/v0.5.1/`，可以把里面的文件理解为三层：

1. `000x-*.patch`：真正的代码改动。
2. `series*.txt`：针对不同场景的 patch 装配清单。
3. `manifest.yaml`：patch 的契约、用途和适用范围说明。

再往上一层：

4. `INDEX.yaml`：整个 LeRobot patch stack 的版本入口与激活开关。

## 维护约定

1. 新增 LeRobot 改动时，优先以新 patch 的形式纳入 `third_party/patches/lerobot/<tag>/`。
2. patch 编号在同一个 tag 目录下全局递增，不要求连续回填空号。
3. 变更 patch 后，通常需要同步更新：
   - `series*.txt`
   - `manifest.yaml`
   - `scripts/setup/tests/test_lerobot_filter.sh`
4. 不应仅因为本地改了 `libs/lerobot`，就在根仓直接提交新的 submodule gitlink；更推荐把改动导出为受控 patch。

## 相关文件

- `third_party/patches/lerobot/INDEX.yaml`
- `third_party/patches/lerobot/v0.5.1/manifest.yaml`
- `third_party/patches/lerobot/v0.5.1/series.txt`
- `third_party/patches/lerobot/v0.5.1/series.master-parity-candidates.txt`
- `third_party/patches/lerobot/v0.5.1/series.openharmony-5.1.0-musl.txt`
- `scripts/setup/lerobot_patches.sh`
- `scripts/setup/lerobot_resolve_active.py`
- `scripts/setup/lerobot_filter_series.py`
