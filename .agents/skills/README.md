# AI Agent 技能库 (Skills)

此目录包含了专为 AI Agent（如 Claude Code）设计的技能插件，用于自动化 IB-Robot 项目中的各种开发工作流。每个技能都定义了精准的触发条件（Description），并提供了执行复杂任务所需的工具和上下文。

## 技能清单

| 技能名称 | 分类 | 主要触发场景 (Triggers) |
| :--- | :--- | :--- |
| [intro](./intro) | 引导 | 「介绍」「有哪些功能」「help」「入门」「intro」等，作为所有 skill 的导航入口。 |
| [ibrobot-env](./ibrobot-env) | 环境 | 加载 `.shrc_local`、设置 `ROS_DOMAIN_ID`、解决 `ModuleNotFoundError` 等。 |
| [ibrobot-build](./ibrobot-build) | 操作 | 执行项目编译 (`colcon build`)、构建特定 package 或修复编译错误。 |
| [ibrobot-launch](./ibrobot-launch) | 操作 | 启动机器人系统、运行仿真、测试 ACT 推理或进行遥操作调试。 |
| [ibrobot-architecture](./ibrobot-architecture) | 知识 | 理解 SSOT 模式、修改 `robot_config`、解释数据流或契约设计。 |
| [ibrobot-git-flow](./ibrobot-git-flow) | 工作流 | 提交代码、推送至个人仓库、确保符合 openEuler DCO/Commit 规范。 |
| [ibrobot-docker-verify](./ibrobot-docker-verify) | 验证 | 在干净 Ubuntu 22.04 Docker 容器中端到端验证 setup.sh + build.sh。 |
| [ibrobot-docker-verify-oee](./ibrobot-docker-verify-oee) | 验证 | 在 openEuler Embedded (aarch64) Docker 容器中端到端验证 setup.sh + build.sh。 |
| [atomgit-collaboration](./atomgit-collaboration) | AtomGit | 拦截泛化的 PR / Issue / review / comment 请求，并路由到具体 AtomGit skill。 |
| [atomgit-pr](./atomgit-pr) | AtomGit | 管理 PR 生命周期：创建、读取上下文、更新标题/描述、生成摘要。 |
| [atomgit-issue](./atomgit-issue) | AtomGit | 管理 Issue 生命周期：创建、读取详情、更新内容、关闭/重开。 |
| [atomgit-pr-review](./atomgit-pr-review) | AtomGit | 对 PR 进行代码质量审查、逻辑检查、发现潜在 Bug 并提交检视意见。 |
| [atomgit-pr-architecture-review](./atomgit-pr-architecture-review) | AtomGit | 验证 PR 是否符合 SSOT、契约驱动设计等项目架构支柱。 |
| [atomgit-review-resolution](./atomgit-review-resolution) | AtomGit | 处理评审意见：获取未解决评论、修复代码、回复并闭环 review。 |

---

## 技能分类说明

### 🧭 引导入口

- **技能导航 ([intro](./intro))**: 所有 skill 的统一入口，展示分类列表与使用示例，并根据仓库状态智能推荐最合适的 skill。

### 🤖 IB-Robot 核心操作
这些技能旨在处理 IB-Robot 软件栈特有的日常开发任务。

- **环境管理 ([ibrobot-env](./ibrobot-env))**: 确保 shell 上下文正确继承了项目特有的环境变量。
- **编译构建 ([ibrobot-build](./ibrobot-build))**: 封装了 ROS 2 复杂的编译参数，确保构建的一致性。
- **系统启动 ([ibrobot-launch](./ibrobot-launch))**: 机器人系统的总入口，支持一键拉起复杂的节点拓扑。
- **架构顾问 ([ibrobot-architecture](./ibrobot-architecture))**: 充当项目的架构师，解答一切关于设计模式和配置规范的问题。
- **工程规范 ([ibrobot-git-flow](./ibrobot-git-flow))**: 自动化执行开源社区繁琐的提交规范校验。
- **容器验证 ([ibrobot-docker-verify](./ibrobot-docker-verify))**: 在全新 Ubuntu 22.04 Docker 容器中运行 setup.sh 和 build.sh 的完整端到端验证，确保修改不会破坏首次安装体验。
- **openEuler 容器验证 ([ibrobot-docker-verify-oee](./ibrobot-docker-verify-oee))**: 在 openEuler Embedded aarch64 Docker 容器（qemu-user 模拟 chroot）中端到端验证 setup.sh + build.sh，以 root 用户模拟真实开发板操作环境。

### 🌐 AtomGit 自动化工具
这些技能通过集成 AtomGit API，实现了 PR / Issue 生命周期和代码审查的自动化。

> **⚠️ 前置条件：配置 AtomGit Token**
> 
> 使用 AtomGit 相关技能前，必须先配置 Personal Access Token：
> 
> 1. 访问 https://atomgit.com 并登录
> 2. 点击右上角头像 → 个人设置
> 3. 找到「访问令牌」选项
> 4. 点击「新建访问令牌」，勾选 `repo` 和 `pull_request` 权限
> 5. **立即复制保存** Token（只显示一次）
> 
> 设置环境变量：
> ```bash
> export ATOMGIT_TOKEN="your_token_here"
> ```
> 
> Token 配置存储在项目根目录的 `config.json` 中，通过环境变量 `$ATOMGIT_TOKEN` 引用。

- **PR 工作流 ([atomgit-pr](./atomgit-pr))**: 面向 PR 资源本身，覆盖创建、读取管理上下文、更新描述等全生命周期动作；如果目标是通用 review，应改用 `atomgit-pr-review`。
- **Issue 工作流 ([atomgit-issue](./atomgit-issue))**: 面向 Issue 资源本身，覆盖创建、读取、更新与状态流转，支持 `--owner` / `--repo` / `--url` 进行跨仓库调用。
- **通用评审 ([atomgit-pr-review](./atomgit-pr-review))**: 利用 LLM 充当第一道代码防线，默认提取变更、提交和已有评论，支持直接从 PR 链接解析目标仓库与编号。
- **架构扫描 ([atomgit-pr-architecture-review](./atomgit-pr-architecture-review))**: 专门检查是否违背了 SSOT 等核心架构原则。
- **意见处理 ([atomgit-review-resolution](./atomgit-review-resolution))**: 实现从“发现问题”到“修复代码/回复评论”的自动化闭环，支持直接从 PR 链接解析目标仓库与编号。
- **协作路由 ([atomgit-collaboration](./atomgit-collaboration))**: 面向“看看这个 PR / 帮我跟进这个评论”这类泛化协作请求，先识别意图，再分流到具体 AtomGit skill。

> **边界说明**: `atomgit-pr-architecture-review` 仍然是 **IB_Robot 专用** 能力；本次跨仓库能力只开放给通用 PR / Issue / review / review-resolution 流程。

### 命名与拆分原则（Agent Skill Best Practice）

1. **优先按资源/工作流命名，不按单个动作命名**：用 `atomgit-pr`、`atomgit-issue`，不要用 `atomgit-submit-pr` 这类只覆盖一个动词的名字。Agent 看到资源名，更容易把 create / fetch / update / summarize 归到同一个 skill，而不是回退到 GitHub 默认能力。
2. **description 要覆盖完整生命周期动词**：同一个 skill 的 description 应同时包含 create / get / fetch / update / close / reply 等常见动作，避免名字很宽、触发词很窄。
3. **先按“平台 + 资源”拆，再按“专业能力”细分**：`atomgit-pr` 与 `atomgit-issue` 负责资源生命周期；`atomgit-pr-review` 与 `atomgit-pr-architecture-review` 负责不同评审维度；`atomgit-review-resolution` 负责 review follow-up。只有当执行流程、输入输出和成功标准明显不同，才继续拆 skill。
4. **显式写出平台优先级**：在本仓库里，只要目标是 PR / Issue / review comment 且用户未明确指定 GitHub，就应优先触发 AtomGit skill。
5. **为泛化协作请求保留一个薄路由层**：当用户只说“帮我看看这个 PR / 评论 / 协作状态”而未说明动作时，用 `atomgit-collaboration` 先识别资源与意图，再转入具体 skill，避免直接落到 GitHub 默认能力。

---

## 如何增加新技能

若要向本项目添加新技能，请遵循以下步骤：
1. 在 `.agents/skills/` 下创建一个新目录。
2. 添加 `SKILL.md` 文件，确保 `description` 字段采用 **if-then** 条件触发风格（包含中英双语关键词），并在涉及第三方平台时明确写出平台优先级。
3. 编写技能所需的配套脚本（Python/Bash）或库文件。
4. 更新此 `README.md` 文件，将新技能添加到清单表格中。
