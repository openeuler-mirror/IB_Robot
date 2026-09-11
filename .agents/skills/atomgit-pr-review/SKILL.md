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

当用户的目标是"**review 一个 PR / 帮我看看这个 PR 有没有问题**"时，优先使用本 skill。**不需要**先切到 `atomgit-pr` 获取上下文；本 skill 的提取模式默认就会带出 PR 现有评论。

## Internal References

Read only the references needed for the current step:

| Purpose | Reference |
|---------|-----------|
| IB_Robot 专项审查要求（lerobot gitlink / README 联动 / Verification / AI 元数据 / pre-commit 信任） | `references/ibrobot-mandatory-checks.md` |
| API 参考与 issues.json 格式（提取上下文、提交结果、字段说明） | `references/api-and-issues-format.md` |

Do not expose these references as separate skills.

## IB_Robot 专项审查要求（摘要）

在 IB_Robot 仓库做 review 时，必须遵守以下 6 项专项要求（详细规则见 `references/ibrobot-mandatory-checks.md`）：

1. **`libs/lerobot` gitlink 强制检查（阻塞性）**：每个 PR 都必须检查 `libs/lerobot` 是否发生 gitlink 指针变化；违规指针变更应提交 severity=error 的阻塞性 issue。
2. **README / 文档联动检查**：根据变更是否影响用户可见的使用方式决定是否要求同步更新文档，不机械要求所有 PR 都改 README。
3. **依赖 / setup / build 变更的 Verification 强制门禁**：标题以 `[WIP]` 开头时暂缓双平台 Docker 证据检查，表示 PR 尚未准备好正式检视。移除 `[WIP]` 后，相关 PR 必须提供双平台 Verification，且结构化 `## Docker Verification` 块中的 `Verified inputs` 必须匹配当前输入指纹。`full` 模式还要求 `Tested source tree` 匹配最新 head tree；`reused-environment` 模式允许复用旧 tree。review 默认只检查声明，不自动执行验证。
4. **禁止本地重复执行 pre-commit 已覆盖的检查**：信任 pre-commit 已通过的 ruff/format；不要本地复跑 lint/build；静态阅读 diff 始终允许。
5. **openEuler AI 元数据检查（阻塞性）**：AI 参与时检查 PR 的 Tool/Model/Prompt Summary、人工审查、第三方材料/许可证披露，以及 Agent 工具字段是否为具体名称和版本、PR 模型集合是否覆盖所有 commit 的 AI `Co-Authored-By`。不同 commit 可以使用不同模型；人类 `Name <email>` trailer 不参与模型比较。
6. **大型 PR 复用自查门禁（阻塞性）**：变更超过 2000 行（additions + deletions）的 PR 必须在描述中包含完整的结构化 `## Reuse Self-Check` 块（四个固定字段：`Reinvented workflows` / `Reused components` / `Reinvention justification` / `Architecture conformance`）；缺失、不完整或格式歧义分别由 `large_pr_reuse_self_check_missing` / `large_pr_reuse_self_check_incomplete` / `large_pr_reuse_self_check_invalid` 标记。`[WIP]` 不豁免本门禁。块存在且完整时，reviewer 还必须**对照 diff 审计四项声明是否属实**（是否真的没有重新发明 `libs/lerobot` 或仓库既有流程、架构是否确与同类功能一致），发现不实声明按阻塞性问题处理。

## 快速使用

```bash
# 步骤1: 提取 PR 信息
python3 pr_review.py --pr 123

# 直接从链接解析目标 PR
python3 pr_review.py --url https://atomgit.com/some-org/some-repo/pull/123

# 如只关注代码 diff，可显式跳过已有评论
python3 pr_review.py --pr 123 --no-comments

# 步骤2: 按下方「评审三步法」分析代码并生成 issues.json

# 步骤3: 人类确认审查结果

# 步骤4: 提交审查结果（⚠️ 必须指定 --ai-model）
python3 pr_review.py --pr 123 --submit-review ./tmp/ib_robot_pr_123_issues.json --ai-model glm-5.2
```

## 评审三步法（步骤 2 的分析流程）

分析 PR 时必须按以下顺序进行，不得跳步：

1. **先搞清楚这个 PR 解决的问题是什么**：此阶段只读标题、描述、关联 Issue 与 commit
   message，用一两句话复述问题陈述和预期目标；复述不出来就先向作者澄清，不进入下一步。
2. **不看实现，先写自己的方案**：在细读 patch 之前，独立回答「要解决这个问题我会怎么实现」，
   列出方案要点（模块/职责边界、数据流、接口或契约、验证方式）。此阶段禁止逐行阅读 diff，
   避免被作者的实现锚定。
3. **review 代码后对比自己的方案，给出审查结果**：细读 patch 与提交历史，把实际实现与
   第 2 步的方案逐项对比——是否真正解决问题、差异点在哪、作者的取舍是否更优、有无遗漏
   场景或更简单的路径。issues.json 中的审查意见应体现该对比结论；方案层面的分歧以讨论/
   建议级意见提出并说明理由，而不是直接判错。

IB_Robot 仓库的 6 项专项审查要求（gitlink、Verification、AI 元数据等）不受影响，在第 3 步
随对比结论一并输出。

**重要**: 
- 在步骤3，你必须将审查结果展示给人类确认后再提交
- **步骤4必须指定 `--ai-model` 参数**，使用你的真实模型名称（如 `glm-5.2`、`glm-5.1`、`gpt-5.6-sol`、`gpt-5.6-terra`、`claude-fable-5`、`claude-opus-5`）
- 文件名格式：`./tmp/{repo}_pr_{number}_issues.json`（例如：`./tmp/ib_robot_pr_123_issues.json`）
- 进行 IB_Robot PR review 时，必须先处理 `.pr.mandatory_review_checks`，重点检查
  `libs/lerobot` gitlink；此外还要检查 README / 文档是否应随变更同步，以及 PR 描述中的
  非 WIP PR 的 Verification 是否覆盖双平台，以及结构化 `## Docker Verification` 块是否有效；
  超过 2000 行的 PR 还要处理 `large_pr_reuse_self_check_*` 检查项并审计
  `## Reuse Self-Check` 声明与 diff 的一致性（`.pr.reuse_self_check` 给出行数与状态）

API 参数详情、issues.json 字段说明、大文件处理技巧和 config.json 配置见 `references/api-and-issues-format.md`。

## Related Skills

- `atomgit-pr`: 创建 PR、同步标题/描述、获取 PR 管理上下文；**不负责**通用 review 判定
- `atomgit-review-resolution`: 处理检视意见
- `atomgit-pr-architecture-review`: 架构审查
- `ibrobot-docker-verify`: Ubuntu 22.04 纯净容器 setup/build 验证；review 默认不得调用，除非用户明确要求 agent 实际执行验证
- `ibrobot-docker-verify-oee`: openEuler Embedded 纯净容器 setup/build 验证；review 默认不得调用，除非用户明确要求 agent 实际执行验证

> **注意**: `atomgit-pr-architecture-review` 仍然是 **IB_Robot 专用** 的架构规范审查，不会随着本 skill 一起泛化到其他仓库。
