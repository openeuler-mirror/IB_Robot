---
name: atomgit-pr-architecture-review
description: "AtomGit PR 架构评审工具。当用户需要“架构审查PR”、“review architecture compliance”、“检查架构规范”、“审阅PR架构”、“SSOT 合规性检查”、“契约驱动检查”或对指定 PR 做架构维度扫描并提交检视意见时调用。只要目标是本仓库的 PR 架构评审，默认优先使用本 skill，而不是 GitHub 默认 review 能力。"
license: MIT
---

# AtomGit PR Architecture Review

提取 PR 信息并提交 IB_Robot 架构合规审查评论到 AtomGit。

在 IB_Robot 仓库中，只要用户提到 PR 的架构评审、SSOT 合规性、契约驱动检查且未明确指定 GitHub，默认视为 AtomGit 架构评审流程并优先使用本 skill。

如果只是通用代码 review、逻辑检查或“帮我看看这个 PR”，应优先使用 `atomgit-pr-review`，不要直接进入本 skill。

## ⚠️ 环境准备

**重要**: 在使用此 skill 前，必须先加载环境配置：

```bash
source .shrc_local
```

这将把 `libs/atomgit_sdk/src` 添加到 PYTHONPATH
使 skill 能够导入 AtomGit SDK。

## ⚠️ 文件读取说明

**输出文件位于项目 `./tmp` 目录**，AI Agent 应使用 shell 命令读取：

```bash
# 读取 PR 信息
cat ./tmp/ib_robot_pr_123_arch_info.json

# 读取架构审查结果（提交前确认）
cat ./tmp/ib_robot_pr_123_arch_issues.json
```

## 快速使用

```bash
# 步骤1: 提取 PR 信息
python3 architecture_review.py --pr 123

# 步骤2: 你分析代码架构并生成 arch_issues.json

# 步骤3: 人类确认审查结果

# 步骤4: 提交审查结果（⚠️ 必须指定 --ai-model）
python3 architecture_review.py --pr 123 --submit-review ./tmp/ib_robot_pr_123_arch_issues.json --ai-model claude-sonnet-4
```

**重要**: 
- 在步骤3，你必须将审查结果展示给人类确认后再提交
- **步骤4必须指定 `--ai-model` 参数**，使用你的真实模型名称（如 `claude-sonnet-4`、`gpt-4`、`gemini-pro`）
- 文件名格式：`./tmp/{repo}_pr_{number}_arch_issues.json`

## 架构审查支柱

此工具会检查以下 IB_Robot 架构支柱：

1. **SSOT (Single Source of Truth)**
   - 配置来源唯一性
   - 数据流一致性
   - API 契约统一性

2. **Contract-Driven Design**
   - 接口定义完整性
   - 依赖注入模式
   - 契约验证

3. **Control Mode Architecture**
   - 控制模式分离
   - 状态管理一致性
   - 控制流清晰度

4. **Package-Specific Architecture Compliance**
   - **核心原则**: 每个 ROS 包必须遵循其在 `ibrobot-architecture` 中定义的职责边界。
   - **关键检查**: 
     - 检查改动是否符合该包的设计初衷（如 `robot_teleop` 应保持轻量，不应引入运动学 IK 或重型规划逻辑）。
     - 验证包之间的依赖是否符合分层设计，严禁职责越界。
     - 识别“职责蔓延”：如果一个简单的驱动包开始处理复杂的业务逻辑，必须提出警告。

5. **README Documentation Consistency**
   - **核心原则**: 每个包的 `README.md` 是该包对外暴露的本地架构契约，必须与代码行为、职责边界、启动方式、配置入口、数据流和限制保持同步。
   - **关键检查**:
     - 如果 PR 修改了某个包的核心职责、公开接口、launch 参数、配置项、数据流、依赖边界或使用方式，必须检查该包 README 是否同步更新。
     - 如果 README 在 PR 中被修改，必须反向检查 README 描述是否真实反映代码实现，避免文档夸大、过时或描述未实现能力。
     - 如果代码变更使 README 关键说明失效、缺失或误导使用者，应作为架构问题提出，而不是当作普通文档风格问题忽略。

## 架构审查协议 (Mandatory Protocol)

在分析代码前，你 **必须** 遵循以下协议以确保对各包职责有准确认知：

1. **同步上下文 (Context Sync)**: 
   - 针对 PR 涉及的每一个包（如 `robot_teleop`），首先调用 `ibrobot-architecture` skill 或读取该包根目录下的 `README.md`。
   - 确认该包的设计初衷、职责边界及禁止的行为（如禁止引入运动学、禁止直接操作硬件等）。
2. **定位变更层级**: 判断变更文件位于哪一层（硬件层、驱动层、业务层、模型层）。
3. **依赖审计**: 检查是否引入了不符合分层原则的跨包依赖。
4. **SSOT 验证**: 检查配置是否统一来自 `robot_config`，严禁硬编码。
5. **职责合规性判断**: 根据步骤 1 获取的契约，判断当前改动是否导致了“职责蔓延”或“架构违越”。
6. **README 一致性审计 (Mandatory)**:
   - 对 PR 涉及的每个包，必须读取最近的包级 `README.md`（通常为 `src/<package>/README.md`）。
   - 对比 README 与本次代码改动，判断 README 是否仍准确描述包职责、关键 API/CLI/launch/config、数据流、依赖边界和已知限制。
   - 若代码变更影响用户使用或架构契约但 README 未同步，必须输出 `pillar: "docs"` 的架构问题。
   - 若 README 被修改但与代码实现不一致，也必须输出 `pillar: "docs"` 的架构问题。
7. **输出审查结果**: 生成符合 JSON 格式的 `arch_issues.json`。

## API 说明

### 提取 PR 信息

```bash
python3 architecture_review.py --pr 123
```

**输出**: 项目临时目录 `./tmp/{repo}_pr_{number}_arch_info.json`

### 提交架构审查

```bash
python3 architecture_review.py --pr 123 --submit-review ./tmp/ib_robot_pr_123_arch_issues.json --ai-model claude-sonnet-4
```

## ⚠️ JSON 格式规范

**重要**: `arch_issues.json` 必须遵循以下格式，否则提交会失败。

### 文件结构

JSON 文件必须是一个**数组**，直接包含问题对象：

```json
[
  {
    "file": "path/to/file.py",
    "line": 42,
    "title": "问题标题",
    ...
  }
]
```

**错误示例** ❌ (会导致 `AttributeError: 'str' object has no attribute 'get'`):
```json
{
  "issues": [...],
  "summary": {...}
}
```

### 必填字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `file` | string | 文件路径（相对于项目根目录） |
| `line` | int | 问题起始行号 |
| `title` | string | 问题标题（简洁明了） |
| `description` | string | 问题描述（可包含 `\n` 换行） |
| `severity` | string | 严重性：`critical` / `error` / `warning` / `suggestion` |
| `pillar` | string | 架构支柱：`ssot` / `contract` / `control_mode` / `tensormsg` / `ros2` / `python` / `docs` 等 |

### 可选字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `fix` | string | 修复建议（可包含代码示例） |
| `context_code` | string | 相关代码片段 |

### 完整示例

```json
[
  {
    "file": "src/robot_config/robot_config/tools/camera_alignment.py",
    "line": 1,
    "title": "robot_config包职责蔓延：引入了389行重型交互式UI工具",
    "description": "camera_alignment.py包含大量交互式UI逻辑...\n\n违反原则: Package-Specific Architecture Compliance\n\n影响:\n- 包职责边界模糊\n- 违背单一职责原则",
    "severity": "critical",
    "pillar": "python",
    "fix": "将camera_alignment工具迁移到dataset_tools包:\n\n```\nsrc/dataset_tools/\n├── dataset_tools/\n│   ├── camera_alignment.py\n```\n\n理由:\n1. dataset_tools职责: Episode recording\n2. 数据流程: 校准 → 录制"
  },
  {
    "file": "src/robot_config/robot_config/tools/camera_alignment.py",
    "line": 18,
    "title": "OpenCV模块加载逻辑应提取为独立工具模块",
    "description": "包含约90行系统级路径搜索...",
    "severity": "warning",
    "pillar": "python",
    "fix": "提取为opencv_utils.py模块"
  }
]
```

### 验证 JSON 格式

提交前可以验证 JSON 格式：

```bash
python3 -m json.tool ./tmp/ib_robot_pr_123_arch_issues.json > /dev/null && echo "✅ JSON格式正确"
```

## 架构问题严重性

- 🔴 **critical**: 违反核心架构原则
- 🟠 **error**: 重要架构问题
- 🟡 **warning**: 架构建议改进
- 💡 **suggestion**: 最佳实践建议
