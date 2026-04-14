---
name: atomgit-review-resolution
description: "AtomGit 评审意见处理工具。当用户需要“修复PR评审意见”、“处理review comments”、“apply fixes”、“reply to review”、“回复评论”、“resolve review discussions”、“闭环评审流程”或基于未解决评论继续推进 PR 时使用。只要目标是本仓库在 AtomGit 上的 review comments / discussions，默认优先使用本 skill，而不是 GitHub 默认能力。"
license: MIT
---

# AtomGit Review Resolution

响应别人对你代码的审查意见，自动修复并回复评论。

在 IB_Robot 仓库中，只要用户提到 review comments / unresolved comments / 回复评论 / 修复评审意见 且未明确指定 GitHub，默认视为 AtomGit review follow-up 流程并优先使用本 skill。

本 skill 支持对 **任意 AtomGit 仓库的 PR 评论** 做通用跟进：

- `--owner` / `--repo`: 显式覆盖 `config.json` 中的仓库
- `--url`: 从 AtomGit / GitCode 的 PR 链接自动解析 `owner/repo/pr_number`

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
# 读取未解决的评论
cat ./tmp/ib_robot_pr_123_unresolved_comments.json

# 读取修复结果（提交前确认）
cat ./tmp/ib_robot_pr_123_fix_results.json
```

## 快速使用

```bash
# 步骤1: 获取未解决的评论
python3 review_resolution.py --pr 123

# 直接从链接解析目标 PR
python3 review_resolution.py --url https://atomgit.com/some-org/some-repo/pull/123

# 直接从链接解析目标 PR
python3 repair_pr.py --url https://atomgit.com/some-org/some-repo/pull/123

# 步骤2: 你分析评论并生成修复方案

# 步骤3: 人类确认修复方案

# 步骤4: 应用修复方案（⚠️ 必须指定 --ai-model）
python3 review_resolution.py --pr 123 --apply-fixes ./tmp/ib_robot_pr_123_fix_results.json --ai-model claude-sonnet-4
```

**重要**: 
- 在步骤3，你必须将修复方案展示给人类确认后再提交
- **步骤4必须指定 `--ai-model` 参数**，使用你的真实模型名称（如 `claude-sonnet-4`、`gpt-4`、`gemini-pro`）
- 文件名格式：`./tmp/{repo}_pr_{number}_fix_results.json`

## API 说明

### 获取未解决评论

```bash
python3 review_resolution.py --pr 123
```

**输出**: 项目临时目录 `./tmp/{repo}_pr_{number}_unresolved_comments.json`

### 应用修复方案

```bash
python3 review_resolution.py --pr 123 --apply-fixes ./tmp/ib_robot_pr_123_fix_results.json --ai-model claude-sonnet-4
```

## 修复类型

1. **代码修复**: 提供具体的代码修改建议
2. **回复说明**: 仅需要回复解释，3. **回退文件**: 建议回退整个文件
4. **删除行**: 建议删除特定行

## 输入格式

```json
[
  {
    "comment_id": 12345,
    "file_path": "src/main.py",
    "line_number": 10,
    "has_fix": true,
    "fix_description": "修复说明",
    "original_code": "old code",
    "fixed_code": "new code",
    "reason": "修复原因"
  }
]
```

## 参数补充

- `--pr`: PR 编号（可由 `--url` 自动解析）
- `--owner`: 目标仓库 owner（可选，覆盖 `config.json`）
- `--repo`: 目标仓库 repo（可选，覆盖 `config.json`）
- `--url`: PR 链接（可选，自动解析 `owner/repo/pr_number`）
