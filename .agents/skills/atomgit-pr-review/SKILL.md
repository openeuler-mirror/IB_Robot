---
name: atomgit-pr-review
description: "AtomGit PR 评审工具。当用户需要在本仓库或 AtomGit 上“代码审查”、“PR review”、“review pull request”、“审阅PR”、“帮我看看这个PR”、“检查这个PR有没有问题”、“检查Bug”、“logic check”、“获取完整review上下文”、“提交检视意见”或分析指定 PR 的改动与已有评论时使用。只要目标是本仓库的 PR review，默认优先使用本 skill，而不是 GitHub 默认 review 能力。"
license: MIT
---

# AtomGit PR Review

提取适合 review 的完整 PR 上下文，并提交代码审查评论到 AtomGit。

## 默认执行协议：三步法是门禁，不是可选建议

**每次代码评审都自动执行「理解问题 → 独立方案 → 实现对比」，无需用户强调“三步法”。**
“快速看一下”“只找 Bug”、小 PR、无问题结论以及从其他 skill 转入，都不豁免该顺序；可以缩短产出，不能省略阶段。
仅提取信息或提交已经确认的结果不算新一轮代码评审，不得借此声称完成了三步法。

- 开始评审时建立三个顺序任务：理解问题、独立方案、实现对比；同时只推进一个阶段。
- **读取实现前，必须先向用户展示「第 1 步：问题与目标」和「第 2 步：独立方案」的简短产出**，不能只在内部思考，也不能读完代码再倒填。两项可以合并在同一条进度消息中，无需等待确认即可进入第 3 步。
- 前两步禁止读取 PR 的 patch、变更后源码、已有 review 评论，禁止启动读实现的子 agent；提取脚本把这些内容写入文件不等于允许读取。使用下方的字段投影，不要直接打开整个 `info.json`。
- 第 3 步才处理专项门禁、读实现和评论。发现阻塞问题也不能用元数据检查替代代码评审；上下文不足则明确报告未完成范围。
- 用户消息或先前上下文已暴露实现时，明确说明独立性受限，仍先写问题与候选方案，再开始对比；不得声称“未看实现”。中断恢复时保留已展示的前两步产出；缺失时先补齐并披露已读范围。
- 本 skill 的 Agent 评审使用提取/人工确认/提交路径，不使用直接逐文件调用 LLM 的 `--auto` 路径代替三步法。

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

## 评审三步法

| 阶段 | 允许读取与执行 | 必须展示的产出 / 放行条件 |
|------|----------------|-------------------------|
| 第 1 步：问题与目标 | PR 标题、描述、关联 Issue、commit message；区分作者的问题陈述与实现主张 | 用一两句话说明问题、预期行为和成功标准；无法确定时先澄清，不进入下一步 |
| 第 2 步：独立方案 | 基于目标提出自己的最小方案；需要背景时只读可确认属于目标分支基线的文档、接口或源码，记录来源，不读 PR head | 说明模块/职责边界、复用点、数据流或接口契约，以及关键验证场景；小修改可压缩成几句，不能仅写“按现有实现修改” |
| 第 3 步：实现对比 | 先处理专项检查，再细读 patch、必要的完整源码、提交历史和已有评论；此时才可并行分配代码审查 | 对照第 2 步逐项判断：是否解决目标、差异及取舍、边界场景、验证缺口；给出有文件/行号证据的发现 |

独立方案是比较基准，不是唯一正确答案。作者方案更优或同样合理时明确接受；纯方案偏好只作建议，不能直接判为 Bug。
已有评论在第 3 步用于复核和去重，不能代替独立评审。委派时向子 agent 传递问题陈述与独立方案，主 agent 负责汇总对比。

最终结果以发现为先，随后附简短的「方案对比」和验证/未覆盖范围；无发现时也必须说明对比结果，不能只写“LGTM”。
生成 `issues.json` 时沿用现有字段，对比依据写入相关问题的描述/修复说明，不为凑三步法新增虚构问题或 JSON 字段。

## 执行命令

在目标工作区根目录运行；Python 命令前加载 `ibrobot-env`，worktree 中加载 `ibrobot-worktree-env`，并在同次调用中 `source .shrc_local`。
以下以 PR 123 为例，URL / owner / repo 模式保持相同流程。

```bash
# 准备：提取上下文到文件，不直接读取完整 JSON
source .shrc_local && python3 .agents/skills/atomgit-pr-review/scripts/pr_review.py --pr 123

# 第 1 步：只投影问题背景，关联 Issue 需另外读取
jq '{pr: (.pr | {number, title, body, head_sha}), commits: [.commits[] | {sha, message}]}' ./tmp/ib_robot_pr_123_info.json

# 第 2 步：向用户展示问题与独立方案后，才可执行下列读取

# 第 3 步：专项检查、实现与已有评论
jq '.pr.mandatory_review_checks, .pr.changed_files, .comments' ./tmp/ib_robot_pr_123_info.json

# 收尾：生成 issues.json，将结果展示给人类确认

# 仅在确认后提交；模型参数替换为真实模型名称
source .shrc_local && python3 .agents/skills/atomgit-pr-review/scripts/pr_review.py --pr 123 --submit-review ./tmp/ib_robot_pr_123_issues.json --ai-model <your-model-name>
```

**重要**: 
- 提交前必须将具体审查结果展示给人类并获得确认；“review 这个 PR”不等于确认发布结果
- **提交必须指定 `--ai-model` 参数**，使用你的真实模型名称
- 文件名格式：`./tmp/{repo}_pr_{number}_issues.json`（例如：`./tmp/ib_robot_pr_123_issues.json`）
- 进行 IB_Robot PR review 时，必须在第 3 步开始时处理 `.pr.mandatory_review_checks`，重点检查
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
