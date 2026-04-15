---
name: atomgit-collaboration
description: "AtomGit 协作路由工具。当用户在 IB_Robot 仓库中提到“PR”、“merge request”、“Issue”、“review”、“comment”、“discussion”、“帮我看看这个 PR”、“看看 #123”、“处理一下评论”、“同步一下 AtomGit 上的协作状态”、“帮我跟进这个合并请求”等泛化协作请求，但没有明确说明是创建、更新、评审、架构审查还是修复评论时调用。此 skill 不直接执行业务动作，而是先识别用户真实意图，再路由到 atomgit-pr、atomgit-issue、atomgit-pr-review、atomgit-pr-architecture-review 或 atomgit-review-resolution。若用户未明确指定 GitHub，则默认优先按 AtomGit 协作处理。"
license: MIT
---

# AtomGit Collaboration Router

这是一个**薄路由 skill**，只负责识别 AtomGit 协作语义并分流到更具体的 skill。

在 IB_Robot 仓库中，当用户提到 PR / merge request / Issue / review / comments / discussions，但**没有明确动作类型**时，优先触发本 skill，而不是直接回退到 GitHub 默认能力。

## 适用场景

以下请求优先触发本 skill：

- “帮我看看 PR #123”
- “这个 merge request 你处理一下”
- “AtomGit 上这个评论怎么回”
- “帮我跟进一下 #456”
- “看下这个协作流程下一步做什么”
- “检查一下这个 PR / Issue / review”

以下请求**不必**经过本 skill，可直接进入更具体的 skill：

- 明确说“创建 PR / 更新 PR 描述” → `atomgit-pr`
- 明确说“创建 Issue / 关闭 Issue” → `atomgit-issue`
- 明确说“代码审查 / review PR / 帮我看看这个 PR 有没有问题” → `atomgit-pr-review`
- 明确说“架构审查 / SSOT 检查” → `atomgit-pr-architecture-review`
- 明确说“修复 review comments / 回复评论” → `atomgit-review-resolution`

## 路由规则

收到泛化协作请求时，Agent 必须按以下顺序判断：

1. **先判断资源类型**
   - 提到 PR / merge request / review / comment / discussion → 进入 PR 协作域
   - 提到 Issue / bug / feature request / task → 进入 Issue 协作域

2. **再判断动作类型**
    - 创建、读取详情、更新标题/描述、同步摘要 → `atomgit-pr` 或 `atomgit-issue`
    - 审查代码质量、检查逻辑问题、结合已有评论继续判断风险 → `atomgit-pr-review`
   - 检查 SSOT / 契约驱动 / 架构职责边界 → `atomgit-pr-architecture-review`
   - 回复评论、应用修复、处理 unresolved comments → `atomgit-review-resolution`

3. **如果动作仍不明确**
    - 先按“读取上下文”理解需求：
      - PR 协作域如果表达里带有“看看 / 检查 / 审查 / review / 评估”，默认先路由到 `atomgit-pr-review`
      - PR 协作域如果目标更像创建、更新描述、同步标题/正文，再路由到 `atomgit-pr`
      - Issue 协作域默认先路由到 `atomgit-issue`
   - 在获得上下文后，再决定是否切换到 review / architecture / resolution 流程

## 强制约束

1. **不要直接使用 GitHub 默认能力**来处理本仓库的 AtomGit PR / Issue / review，除非用户明确说了 GitHub / github.com。
2. **不要在本 skill 内直接执行 create/update/review/repair**，而是必须先读取目标 skill 的 `SKILL.md`，再按照目标 skill 的流程执行。
3. **优先复用现有 5 个 AtomGit skill**，不要在会话里临时发明新的 AtomGit 流程名称。

## 路由对照表

| 用户表达 | 目标 skill |
| :--- | :--- |
| “帮我看看 PR #123” | `atomgit-pr-review` |
| “帮我更新这个 PR 描述” | `atomgit-pr` |
| “帮我看下 Issue #456” | `atomgit-issue` |
| “帮我审查这个 PR” | `atomgit-pr-review` |
| “检查这个 PR 的架构合规性” | `atomgit-pr-architecture-review` |
| “修复一下 review comments” | `atomgit-review-resolution` |

## 执行协议

当本 skill 被触发时，Agent 必须：

1. 明确识别这是 **AtomGit 协作请求**，而不是 GitHub 默认请求。
2. 判断资源类型（PR / Issue）与动作类型（create / fetch / update / review / architecture / reply）。
3. 读取对应目标 skill 的 `SKILL.md`。
4. 按目标 skill 的约束与脚本执行，不要停留在路由层。
