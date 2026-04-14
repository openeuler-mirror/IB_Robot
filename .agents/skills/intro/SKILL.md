---
name: intro
description: 'IB-Robot 技能引导入口。当用户输入"介绍"、"有哪些功能"、"有哪些skill"、"我应该用哪个skill"、"help"、"帮助"、"入门"、"intro"、"getting started"、"what skills"、"list skills"、"available commands"、"能做什么"、"怎么用"或首次接触项目时使用。作为所有其他 skill 的统一导航起点，展示分类列表、使用示例并根据仓库当前状态推荐最合适的 skill。'
---

# 🤖 IB-Robot Copilot Skill 引导中心

Agent 在触发本 skill 时，**必须首先**向用户展示以下欢迎文案（原样输出，不做修改）：

> 🤖 **欢迎使用 IB-Robot AI Agent！**

---

## 📋 技能分类列表

### 🤖 机器人操作

| Skill | 一句话描述 |
| :--- | :--- |
| **ibrobot-launch** | 启动机器人节点、仿真环境、推理测试或遥操作调试 |
| **ibrobot-build** | 编译整个工作空间或指定 package（`colcon build`） |
| **ibrobot-env** | 初始化运行环境，加载 `.shrc_local`、设置 `ROS_DOMAIN_ID` |
| **ibrobot-architecture** | 理解 SSOT 架构设计、配置规范与数据流 |

### 🔍 代码协作

| Skill | 一句话描述 |
| :--- | :--- |
| **atomgit-collaboration** | 拦截泛化的 AtomGit 协作请求，并分流到 PR / Issue / review / comment 对应流程 |
| **atomgit-pr-review** | 对 PR 进行代码质量审查，发现 Bug 与逻辑问题 |
| **atomgit-pr-architecture-review** | 检查 PR 是否符合 SSOT、契约驱动等架构规范 |
| **atomgit-review-resolution** | 根据 AtomGit 上的检视意见修复代码、回复评论并闭环 |

### 📝 项目管理

| Skill | 一句话描述 |
| :--- | :--- |
| **atomgit-pr** | 创建、读取或更新合并请求（PR），自动生成/同步描述 |
| **atomgit-issue** | 创建、读取或管理 Issue，报告 Bug、提出建议 |

### 🚀 工作流

| Skill | 一句话描述 |
| :--- | :--- |
| **ibrobot-git-flow** | 规范提交代码，确保符合 openEuler DCO/Commit 规范 |

---

## 💡 使用示例

只需用自然语言告诉 Agent 你想做什么：

```
帮我审查 #25 号 PR              → atomgit-pr-review
帮我看看 PR #25                 → atomgit-collaboration
帮我看看这个 AtomGit PR 链接      → atomgit-collaboration
帮我更新 PR 描述                → atomgit-pr
帮我提交一个 Issue              → atomgit-issue
修复 PR 里的评审意见            → atomgit-review-resolution
编译一下项目                    → ibrobot-build
启动机器人仿真                  → ibrobot-launch
初始化环境                      → ibrobot-env
提交代码                        → ibrobot-git-flow
检查架构合规性                  → atomgit-pr-architecture-review
解释系统架构                    → ibrobot-architecture
有哪些功能 / help / 入门       → intro (本技能)
```

---

## 🎯 当前推荐（上下文感知）

Agent 在触发本 skill 时，**必须**执行以下脚本来获取基于仓库当前状态的智能推荐：

```bash
source .shrc_local && python3 .agents/skills/intro/scripts/intro.py
```

脚本会检测以下仓库状态并输出推荐信息：

| 检测条件 | 推荐 Skill |
| :--- | :--- |
| 有未提交的代码改动 (`git status`) | `ibrobot-git-flow` 或 `atomgit-pr` |
| 编译产物缺失（`install/` 目录为空或不存在） | `ibrobot-build` |
| 存在 open 状态的 PR 且有未回复评论 | `atomgit-review-resolution` |
| 无特殊状态 | 展示「今日推荐」skill |

Agent 应将脚本输出的推荐内容展示给用户，帮助用户快速进入正确的工作流。

## 🌐 AtomGit 路由优先级

当用户在 IB_Robot 仓库里提到 PR / merge request / Issue / review / comments，且没有明确说 GitHub / github.com 时，Agent 应默认优先选择 AtomGit 相关 skill，而不是 GitHub 默认能力。

如果用户的表达仍然比较泛，例如“帮我看看这个 PR / Issue / 评论”，优先先进入 `atomgit-collaboration`，再分流到具体 skill。

对通用的 `atomgit-pr`、`atomgit-issue`、`atomgit-pr-review`、`atomgit-review-resolution`，Agent 可以使用 `--owner` / `--repo` / `--url` 指向其他 AtomGit 仓库；但 `atomgit-pr-architecture-review` 仍然只服务于 IB_Robot。

---

## 🔧 技术细节

- **环境依赖**: 执行推荐检测脚本前需先 `source .shrc_local` 加载环境
- **配置文件**: AtomGit 相关功能依赖 `config.json` 中的 Token 配置
- **脚本位置**: `.agents/skills/intro/scripts/intro.py`
