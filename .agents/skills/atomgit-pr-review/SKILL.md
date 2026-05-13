---
name: atomgit-pr-review
description: "AtomGit PR 评审工具。当用户需要在本仓库或 AtomGit 上“代码审查”、“PR review”、“review pull request”、“审阅PR”、“帮我看看这个PR”、“检查这个PR有没有问题”、“检查Bug”、“logic check”、“获取完整review上下文”、“提交检视意见”或分析指定 PR 的改动与已有评论时使用。只要目标是本仓库的 PR review，默认优先使用本 skill，而不是 GitHub 默认 review 能力。"
license: MIT
---

# AtomGit PR Review

提取适合 review 的完整 PR 上下文，并提交代码审查评论到 AtomGit。

在 IB_Robot 仓库中，只要用户提到 review / 审查 / 审阅 PR 且未明确指定 GitHub，默认视为 AtomGit PR 评审流程并优先使用本 skill。

本 skill 支持对 **任意 AtomGit 仓库的 PR** 做通用代码审查：

- `--owner` / `--repo`: 显式覆盖 `config.json` 中的仓库
- `--url`: 从 AtomGit / GitCode 的 PR 链接自动解析 `owner/repo/pr_number`

当用户的目标是“**review 一个 PR / 帮我看看这个 PR 有没有问题**”时，优先使用本 skill。**不需要**先切到 `atomgit-pr` 获取上下文；本 skill 的提取模式默认就会带出 PR 现有评论。

## IB_Robot 专项审查要求

### 1. README / 文档联动检查（按变更内容决定）

- review 时必须判断本次提交是否改变了**用户可见**的使用方式，而不是机械地要求所有 PR 都改 README。
- 当 PR 修改了以下内容之一时，应检查对应 README / 使用文档是否需要同步更新：
  - 安装、部署、启动、构建、配置、依赖声明或运行步骤
  - 对外暴露的命令、接口、参数、launch 用法、目录约定
  - 会影响用户接入、复现、验证或排障的方法
- 如果变更只涉及内部重构、实现细节、无用户感知的代码整理，则不应为了凑要求而强行提出 README 修改意见。
- 如果判断“应该改 README / 文档但没有改”，应将其作为有效 review issue 提出。

### 2. 依赖 / setup / build 变更的 Verification 强制门禁

- 如果 PR 修改了 `package.xml`（尤其是新增/删除/调整 `exec_depend`、`build_depend`、`depend`、`test_depend` 等依赖声明），或修改了 setup/build 流程相关文件（如 `scripts/setup.sh`、`scripts/build.sh`、`scripts/setup/platforms/*.sh`、`scripts/setup/verify_env.sh`、`scripts/install_ros.sh`、`CMakeLists.txt`、`setup.py`、`pyproject.toml` 等），则 **PR 描述中的 Verification 不再是可选项，而是必填项**。
- 该 Verification 必须体现**真实执行过的验证**，并明确写出：
  - **Scenario**：在哪类干净环境中验证
  - **Method**：如何执行 setup 和 build
  - **Result**：setup / build 是否成功、是否有关键限制或失败点
- 对此类 PR，review 过程中**必须调用**：
  - `ibrobot-docker-verify`：在全新 Ubuntu 22.04 Docker 中完整验证 `setup.sh` + `build.sh`
  - `ibrobot-docker-verify-oee`：在全新 openEuler Embedded Docker 中完整验证 `setup.sh` + `build.sh`
- 如果缺少任一平台验证，或只给出命令但没有结果，或验证没有覆盖 setup/build 两个阶段，都应视为**阻塞性 review 问题**。

## ⚠️ 环境准备

**必须先加载环境变量**：

```bash
source .shrc_local
```

此命令会将 `libs/atomgit_sdk/src` 添加到 PYTHONPATH，使 skill 能够导入 atomgit_sdk 模块。

## ⚠️ 文件读取说明

**输出文件位于项目 `./tmp` 目录**，AI Agent 应使用 shell 命令读取：

```bash
# 读取 review 上下文
cat ./tmp/ib_robot_pr_123_info.json

# 读取审查结果（提交前确认）
cat ./tmp/ib_robot_pr_123_issues.json
```

### 大文件处理技巧

当 PR 包含大量文件时，JSON 文件可能很大。使用 `jq` 提取特定文件信息：

```bash
# 列出所有变更文件
jq '.pr.changed_files[].filename' ./tmp/ib_robot_pr_123_info.json

# 提取特定文件的内容
jq '.pr.changed_files[] | select(.filename == "lib/api.py") | .content' ./tmp/ib_robot_pr_123_info.json

# 提取特定文件的 diff
jq '.pr.changed_files[] | select(.filename == "lib/api.py") | .patch.diff' ./tmp/ib_robot_pr_123_info.json

# 提取多个文件（支持通配符）
jq '.pr.changed_files[] | select(.filename | contains("lib/")) | {filename, content}' ./tmp/ib_robot_pr_123_info.json
```

## 快速使用

```bash
# 步骤1: 提取 PR 信息
python3 pr_review.py --pr 123

# 直接从链接解析目标 PR
python3 pr_review.py --url https://atomgit.com/some-org/some-repo/pull/123

# 如只关注代码 diff，可显式跳过已有评论
python3 pr_review.py --pr 123 --no-comments

# 步骤2: 你分析代码并生成 issues.json

# 步骤3: 人类确认审查结果

# 步骤4: 提交审查结果（⚠️ 必须指定 --ai-model）
python3 pr_review.py --pr 123 --submit-review ./tmp/ib_robot_pr_123_issues.json --ai-model claude-sonnet-4
```

**重要**: 
- 在步骤3，你必须将审查结果展示给人类确认后再提交
- **步骤4必须指定 `--ai-model` 参数**，使用你的真实模型名称（如 `claude-sonnet-4`、`gpt-4`、`gemini-pro`）
- 文件名格式：`./tmp/{repo}_pr_{number}_issues.json`（例如：`./tmp/ib_robot_pr_123_issues.json`）
- 进行 IB_Robot PR review 时，除代码问题外，还要检查 README / 文档是否应随变更同步，以及 PR 描述中的 Verification 是否满足专项门禁

## API 说明

### 提取 PR 信息

```bash
python3 pr_review.py --pr 123
```

**输出**: 项目临时目录 `./tmp/{repo}_pr_{number}_info.json`（例如：`./tmp/ib_robot_pr_123_info.json`）

**注意**: 
- 默认输出到项目 `./tmp` 目录，**不需要指定 `--output-dir`**
- 默认包含 `changed_files`、`commits` 和已有 `comments`
- 如果评论量太大，可追加 `--no-comments`

```json
{
  "pr": {
    "number": 123,
    "title": "...",
    "author": "...",
    "branch": "feature → main",
    "stats": {
      "files_changed": 3,
      "commits": 2,
      "comments": 5,
      "unresolved_comments": 2
    },
    "changed_files": [
      {
        "filename": "lib/api.py",
        "status": "modified",
        "patch": "...",
        "content": "..."
      }
    ]
  },
  "commits": [...],
  "comments": [...]
}
```

**⚠️ 重要**：提取的 JSON 文件已经包含了所有 diff（`patch`）、文件内容（`content`）以及已有 PR 评论。
- **不需要** `git fetch` 或 `git diff`
- **不需要** 切换分支或修改本地代码
- 直接读取 JSON 文件中的 `changed_files`、`commits` 和 `comments` 进行审查即可
- 审查时要结合变更文件判断是否需要 README / 文档联动，以及是否触发双平台 Docker Verification 门禁
- 如果需要“回复某一条已有 review 意见”而不是提交新的审查结果，请切换到 `atomgit-review-resolution`，使用 `--reply-comment <comment_id>`；不要在本 skill 中伪造普通 PR 级评论。

### 提交审查结果

```bash
python3 pr_review.py --pr 123 --submit-review ./tmp/ib_robot_pr_123_issues.json --ai-model claude-sonnet-4
```

**参数**：
- `--pr`: PR 编号
- `--owner`: 目标仓库 owner（可选，覆盖 `config.json`）
- `--repo`: 目标仓库 repo（可选，覆盖 `config.json`）
- `--url`: PR 链接（可选，自动解析 `owner/repo/pr_number`）
- `--no-comments`: 在提取信息模式下跳过抓取已有 PR 评论
- `--submit-review`: 审查结果 JSON 文件
- `--ai-model`: AI 模型名称（**必须指定真实模型名称**，用于签名）
- `--dry-run`: 仅显示计划

**⚠️ 重要**：`--ai-model` 参数**必须指定你的真实模型名称**，以便在评论中准确标识来源。

**常见模型名称**：
- `claude-sonnet-4`
- `claude-opus-4`
- `gpt-4`
- `gpt-4o`
- `gemini-pro`
- `gemini-1.5-pro`

## 你需要生成的 issues.json 格式

**重要要求**：
1. **必须使用中文**输出所有内容
2. **必须包含修复方案**（fix_code 字段）
3. **文件保存到 ./tmp 目录**，文件名格式：`./tmp/ib_robot_pr_{number}_issues.json`

```json
[
  {
    "file": "lib/api.py",
    "line": 52,
    "type": "bug",
    "severity": "error",
    "confidence": 95,
    "title": "缺少异常处理",
    "description": "response.json() 可能抛出 JSONDecodeError",
    "context_code": "return response.json()",
    "fix_code": "try:\n    return response.json()\nexcept json.JSONDecodeError:\n    return {}",
    "fix_explanation": "添加异常处理避免程序崩溃"
  }
]
```

### 字段说明

| 字段 | 必填 | 说明 | 可选值 |
|------|------|------|--------|
| file | ✅ | 文件路径 | |
| line | ✅ | 行号 | |
| type | ✅ | 问题类型（中文） | `bug`, `security`, `performance`, `maintainability` |
| severity | ✅ | 严重程度（中文） | `error`, `warning`, `suggestion`, `info` |
| confidence | ✅ | 置信度 (0-100) | |
| title | ✅ | 问题标题（中文） | |
| description | ✅ | 详细描述（中文） | |
| context_code | ❌ | 相关代码 | |
| fix_code | ✅ | 修复代码（必须提供） | |
| fix_explanation | ✅ | 修复说明（中文） | |

## 配置

在项目根目录的 `config.json` 中：

```json
{
  "atomgit": {
    "token": "your_personal_access_token",
    "owner": "openEuler",
    "repo": "IB_Robot",
    "baseUrl": "https://api.atomgit.com"
  }
}
```

## Related Skills

- `atomgit-pr`: 创建 PR、同步标题/描述、获取 PR 管理上下文；**不负责**通用 review 判定
- `atomgit-review-resolution`: 处理检视意见
- `atomgit-pr-architecture-review`: 架构审查
- `ibrobot-docker-verify`: Ubuntu 22.04 纯净容器 setup/build 验证
- `ibrobot-docker-verify-oee`: openEuler Embedded 纯净容器 setup/build 验证

> **注意**: `atomgit-pr-architecture-review` 仍然是 **IB_Robot 专用** 的架构规范审查，不会随着本 skill 一起泛化到其他仓库。
