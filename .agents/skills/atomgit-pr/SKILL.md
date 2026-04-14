---
name: atomgit-pr
description: "AtomGit PR 工作流工具。当用户需要在本仓库或 AtomGit 上“创建PR”、“更新PR描述”、“同步PR标题/正文”、“生成PR摘要”、“create pull request”、“open merge request”、“update PR description”、“generate PR summary”或围绕 PR 管理动作工作时调用。它负责 PR 资源的创建和维护，不负责通用代码 review；只要目标是本仓库的 PR / merge request 管理，默认优先使用本 skill，而不是 GitHub 默认能力。"
license: MIT
---

# AtomGit PR Workflow Tool

创建新 PR、提取 PR 管理上下文或更新现有 PR 描述。

如果用户的目标是“**review 一个 PR / 帮我看看这个 PR 有没有问题 / 分析已有评论**”，不要使用本 skill，改用 `atomgit-pr-review`；如果目标是 SSOT / 契约 / 架构职责边界检查，改用 `atomgit-pr-architecture-review`。

在 IB_Robot 仓库中，只要用户提到 PR / merge request 且未明确指定 GitHub，默认视为 AtomGit 工作流并优先使用本 skill。

本 skill 支持对 **任意 AtomGit 仓库** 指定目标：

- `--owner` / `--repo`: 显式覆盖 `config.json` 中的仓库
- `--url`: 从 AtomGit / GitCode 的仓库或 PR 链接自动解析 `owner/repo`

## ⚠️ 环境准备

**重要**: 在使用此 skill 前，必须先加载环境配置：

```bash
source .shrc_local
```

这将把 `libs/atomgit_sdk/src` 添加到 PYTHONPATH
使 skill 能够导入 AtomGit SDK。

## ⚠️ 获取 Fork Owner（必需）

在创建 PR 前，**必须**先通过 `git remote -v` 获取 fork owner：

```bash
git remote -v
```

输出示例：
```
origin    git@atomgit.com:YourName/IB_Robot.git (fetch)
origin    git@atomgit.com:YourName/IB_Robot.git (push)
upstream  git@atomgit.com:openEuler/IB_Robot.git (fetch)
upstream  git@atomgit.com:openEuler/IB_Robot.git (push)
```

从中提取 fork owner（即个人仓库的用户名，如 `YourName`），然后通过 `--fork-owner` 参数传递给脚本。

## 快速使用

### 创建 PR (推荐 Agent 方式)

Agent 在创建 PR 时，**必须**遵循 [PR #32](https://atomgit.com/openeuler/IB_Robot/pull/32) 的极高专业水准。描述文件应采用以下深度结构，并优先使用 **Mermaid 图表** 来解释复杂的逻辑或架构。

**PR 描述强制要求：**

1.  **Markdown 渲染质量**: **必须**确保所有的 Markdown 语法（包括标题、加粗、列表、代码块、Mermaid 图表）都能被正确渲染。避免直接在 shell 命令中使用未处理的换行符。
2.  **超链接使用**: 对相关的 Issue、PR、技术规范或设计文档，**必须**使用 Markdown 超链接进行关联（如 `[PR #32](https://atomgit.com/openeuler/IB_Robot/pull/32)`），以方便审阅者查阅背景。
3.  **深度结构化内容**:
    *   **Background & Motivation (背景与动机)**: 详细说明问题的根源、业务痛点或功能需求。
    *   **Proposed Solution (方案概述)**: 描述解决思路、架构设计决策。如果涉及流程，**必须使用 Mermaid 流程图或时序图**。
    *   **Technical Changes (技术细节)**: 按模块拆解代码级变更，解释关键类、函数或配置的逻辑。
    *   **Impact Assessment (影响范围)**: 评估变更对系统其他部分、现有 API 或性能的潜在影响。
    *   **Verification (验证结果)**: 
        *   **Environment**: 测试环境配置。
        *   **Command**: 具体的运行命令（如 `ros2 launch ...`）。
        *   **Results**: 截图描述或输出日志片段，证明功能符合预期且无回归风险。

```bash
# 1. 获取变更信息
git diff upstream/master..HEAD

# 2. Agent 深度分析并生成专业描述文件 pr_description.md
# 必须包含上述 5 个部分，并包含 Mermaid 图。

# 3. 创建 PR
python3 pr_creation.py --branch feat/my-feature --fork-owner BreezeWu --title "feat(scope): technical summary" --description-file pr_description.md
```

### 基础用法

```bash
# 步骤1: 获取 fork owner
git remote -v

# 步骤2: 创建 PR (必须包含 Summary/Changes/Verification 三大核心板块)
python3 pr_creation.py --branch feat/my-feature --fork-owner BreezeWu --title "fix: specific issue" --body "## Background\n...\n## Changes\n...\n## Verification\n..."

# 跨仓库：直接指定目标仓库
python3 pr_creation.py --branch feat/my-feature --fork-owner BreezeWu --owner some-org --repo some-repo --body "..."

# 跨仓库：从链接自动解析 owner/repo
python3 pr_creation.py --branch feat/my-feature --fork-owner BreezeWu --url https://atomgit.com/some-org/some-repo --body "..."
```

### 生成/更新 PR 描述 (Agent 驱动)

当需要为已有 PR 生成高质量描述时，遵循以下 Agent 工作流：

**步骤 1: 提取 PR 上下文**
```bash
python3 pr_management.py --pr 123 --fetch-info

# 默认会包含 PR 评论；如只看提交和 Diff，可显式关闭
python3 pr_management.py --pr 123 --fetch-info --no-comments
```
Agent 会读取生成的 `tmp/{repo}_pr_123_context.json`，其中默认包含提交记录、修改文件、代码 Diff (patch) 以及 PR 评论。

**步骤 2: Agent 分析与同步**
Agent 分析完 Diff 后，会生成一份 `description.json`:
```json
{
  "title": "feat: 新功能标题",
  "description": "详细的变更逻辑说明..."
}
```
然后运行同步命令：
```bash
python3 pr_management.py --pr 123 --update-pr description.json
```

## API 说明

### pr_creation.py

创建新的 Pull Request。

**参数**:
- `--branch`: 分支名（可选，默认当前分支）
- `--fork-owner`: Fork 仓库的 owner（**必需**，通过 `git remote -v` 获取）
- `--title`: PR 标题（可选，自动生成）
- `--body`: PR 描述（可选，自动生成）
- `--base`: 目标分支（默认：master）
- `--owner`: 目标仓库 owner（可选，覆盖 `config.json`）
- `--repo`: 目标仓库 repo（可选，覆盖 `config.json`）
- `--url`: AtomGit / GitCode 仓库或 PR 链接（可选，自动解析 `owner/repo`）
- `--draft`: 创建草稿 PR（可选）
- `--dry-run`: 仅显示计划，不创建

**示例**:
```bash
# 完整示例
python3 pr_creation.py --branch feat/new-feature --fork-owner BreezeWu

# 指定标题
python3 pr_creation.py --branch feat/new-feature --fork-owner BreezeWu --title "feat: add new feature"
```

### pr_management.py

管理和维护已有 PR 的数据。

**模式**:
1. `--pr <NUM> --fetch-info`: 提取 PR 的完整上下文 (提交、文件、Diff)，Agent 学习用。
2. `--pr <NUM> --update-pr <JSON>`: 将 Agent 生成的描述同步到服务器。

**参数**:
- `--pr`: PR 编号（可由 `--url` 自动解析）
- `--owner`: 目标仓库 owner（可选，覆盖 `config.json`）
- `--repo`: 目标仓库 repo（可选，覆盖 `config.json`）
- `--url`: PR 链接（可选，自动解析 `owner/repo/pr_number`）
- `--output-dir`: JSON 输出目录 (默认: ./tmp)
- `--no-comments`: 在 `--fetch-info` 模式下跳过 PR 评论抓取
- `--ai-model`: 签名使用的 AI 名称 (默认: agent)
- `--dry-run`: 预览生成的描述但不执行更新

## PR 描述格式

PR 描述会自动包含：

- **Summary**: 变更概述
- **Changes**: 详细变更列表
- **Testing**: 测试说明
- **Checklist**: 检查项

## 注意事项

1. **分支命名**: 建议使用 `feat/`, `fix/`, `docs/`, `refactor/` 等前缀
2. **提交信息**: 确保提交信息符合规范
3. **代码审查**: 创建 PR 后等待代码审查
4. **CI 检查**: 确保 CI 通过后再合并
5. **跨仓库前提**: 创建 PR 时当前本地 worktree 仍需与目标仓库代码相匹配；`--owner/--repo/--url` 只负责切换 AtomGit API 目标，不会替你切换本地 Git 工作区
