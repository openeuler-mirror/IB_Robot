# AI Agent 技能库 (Skills)

此目录包含了专为 AI Agent（如 Claude Code）设计的技能插件，用于自动化 IB-Robot 项目中的各种开发工作流。每个技能都定义了精准的触发条件（Description），并提供了执行复杂任务所需的工具和上下文。

## 技能清单

| 技能名称 | 分类 | 主要触发场景 (Triggers) |
| :--- | :--- | :--- |
| [intro](./intro) | 引导 | 「介绍」「有哪些功能」「help」「入门」「intro」等，作为所有 skill 的导航入口。 |
| [ibrobot-env](./ibrobot-env) | 环境 | 跑测试、运行脚本或 ROS 2 命令前加载 `.shrc_local`、设置 `ROS_DOMAIN_ID`、解决 `ModuleNotFoundError`；禁止手动拼装 ROS/PYTHONPATH 环境。 |
| [ibrobot-worktree-env](./ibrobot-worktree-env) | 环境 | 在 `git worktree` 中复用主仓库 venv，避免主仓库/worktree 混合环境导致测错分支。 |
| [ibrobot-build](./ibrobot-build) | 操作 | 执行项目编译 (`./scripts/build.sh`)、构建特定 package 或修复编译错误。 |
| [ibrobot-launch](./ibrobot-launch) | 操作 | 分平台启动 Ubuntu/openEuler 工作区或 OpenHarmony 板端机器人系统、仿真、mock/契约测试、推理与遥操作。 |
| [ibrobot-robot-skill-design](./ibrobot-robot-skill-design) | 操作 | 交互式设计机器人 skill，澄清 anchor/motion space/safety/catalog 暴露并生成验证计划。 |
| [ibrobot-control](./ibrobot-control) | 操作 | Hermes/Agent 通过 `robot-skill` 操作现有高层技能或异步视觉游戏。 |
| [ibrobot-inference-verify](./ibrobot-inference-verify) | 工作流 | 统一推理运行时端到端验证：本机 CPU/CUDA Torch policy + 感知 ROS mock、板端 Ascend OM + ROS typed service。 |
| [oh-constraints](./oh-constraints) | 板端 | OpenHarmony 板端运行时约束汇总（toybox 命令缺失、musl libc、只读 rootfs、无 systemd、无 /usr/bin/env、LD_PRELOAD 干扰等），板端操作前必读。 |
| [oh-access](./oh-access) | 板端 | 连接 OpenHarmony 开发板，执行 HDC shell / file send / file recv。 |
| [oh-build-roboframe](./oh-build-roboframe) | 板端 | 使用 `build_roboframe_oh.sh` 主机侧交叉编译并打包 IB_Robot 自有 OpenHarmony 运行时。 |
| [om-convert](./om-convert) | 模型 | Ascend OM 唯一入口；执行 ACT/PI05 转换，指导新 LeRobot policy 的实验性支持开发，并提供可选性能优化。 |
| [rknn-convert](./rknn-convert) | 模型 | 将 ONNX 转成 RKNN，并维护主 `venv` 导出 ONNX、`.venv-rknn` 转 RKNN 的流程边界。 |
| [hmm-convert](./hmm-convert) | 模型 | 将 PI0.5 / SmolVLA 编译产物打包为后摩 HMM deployment（xh2 NPU），生成统一 manifest；ACT HMM 不支持。 |
| [ibrobot-architecture](./ibrobot-architecture) | 知识 | 理解 SSOT 模式、修改 `robot_config`、解释数据流或契约设计。 |
| [ibrobot-lerobot-patch](./ibrobot-lerobot-patch) | 工作流 | 将 `libs/lerobot` 的本地改动导出为 `third_party/patches/lerobot/<tag>/*.patch`，并通过辅助脚本同步 `series/manifest/test`。 |
| [ibrobot-git-flow](./ibrobot-git-flow) | 工作流 | 提交代码、推送至个人仓库、确保符合 openEuler DCO/Commit 规范。 |
| [ibrobot-docker-verify](./ibrobot-docker-verify) | 验证 | 在干净 Ubuntu 22.04 Docker 容器中验证 setup.sh + build.sh；作者侧 `[WIP]` PR 暂缓。 |
| [ibrobot-docker-verify-oee](./ibrobot-docker-verify-oee) | 验证 | 在 openEuler Embedded (aarch64) Docker 容器中验证 setup.sh + build.sh；作者侧 `[WIP]` PR 暂缓。 |
| [sync-github](./sync-github) | 工作流 | 将 IB_Robot 仓库的 origin 远端（个人 AtomGit fork）的 master 分支同步推送到 GitHub 远端。 |
| [skill-creator](./skill-creator) | 工作流 | 新建/重构 Agent skill，按 agentskills.io 规范编写 SKILL.md，校验 frontmatter 与渐进式披露。 |
| [oh-cross-build-ros-pkg](./oh-cross-build-ros-pkg) | 板端 | 为 OpenHarmony 板交叉编译移植第三方 ROS 2 包（如 usb_cam）。 |
| [ohloha-build-pkg](./ohloha-build-pkg) | 板端 | 用 `tools_ohloha_pkgs` / `builder.sh` 交叉编译第三方包（bash、zsh、vim、ncurses 等）到 OpenHarmony 板端。 |
| [oh-rebuild-kernel](./oh-rebuild-kernel) | 板端 | 重新编译并刷入 OpenHarmony 内核 (boot_linux.img)，启用 USB ACM 等驱动。 |
| [deepwiki-config](./deepwiki-config) | 文档 | 根据 DeepWiki 目录结构生成或更新 `deepwiki_processor.py` 所需的 `doc_config.json`。 |
| [deepwiki-translator](./deepwiki-translator) | 文档 | 将 DeepWiki 英文 Markdown 端到端转换为可交付的中文文档（翻译、构建目录与索引、校验链接与锚点）。 |
| [doc-link-validator](./doc-link-validator) | 文档 | 校验 Markdown/RST 文档树中的本地、外部与 AtomGit 链接有效性；断链检查、交付前链接体检。 |
| [mermaid-syntax-validation](./mermaid-syntax-validation) | 文档 | 检查、修复并浏览器验证 Markdown/Sphinx Mermaid 图语法，确保发布 HTML 不再渲染 Mermaid 错误 SVG。 |
| [simplified-technical-chinese](./simplified-technical-chinese) | 文档 | 简明技术中文（STC）受控写作：借鉴 ASD-STE100 方法，为中文技术文档提供编号规则、受控词表与零依赖检查脚本。 |
| [atomgit-collaboration](./atomgit-collaboration) | AtomGit | 拦截泛化的 PR / Issue / review / comment 请求，并路由到具体 AtomGit skill。 |
| [atomgit-pr](./atomgit-pr) | AtomGit | 管理 PR 生命周期：创建、读取上下文、选择 WIP/正式检视阶段、更新标题/描述；工具版本由 coding agent 先行执行版本命令后传入。 |
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

- **环境管理 ([ibrobot-env](./ibrobot-env))**: 确保 shell 上下文正确继承了项目特有的环境变量。任何 `python3`/`pytest`/`ros2` 命令前统一走 `source .shrc_local &&`，环境类报错的唯一修复入口，禁止手动 source ROS setup 或 export PYTHONPATH。
- **Worktree 环境 ([ibrobot-worktree-env](./ibrobot-worktree-env))**: 在 `git worktree` 中复用主仓库 venv，避免 source 主仓库 `.shrc_local` 造成的「worktree venv + 主仓库源码」混合环境，并给出验证脚本与已知限制清单。
- **编译构建 ([ibrobot-build](./ibrobot-build))**: 封装了 ROS 2 复杂的编译参数，确保构建的一致性。
- **系统启动 ([ibrobot-launch](./ibrobot-launch))**: 机器人系统的总入口，区分 Ubuntu/openEuler 源码工作区与 OpenHarmony `/data/roboframe` 板端运行时。
- **机器人 Skill 设计 ([ibrobot-robot-skill-design](./ibrobot-robot-skill-design))**: 通过交互式流程把自然语言动作需求转成安全的 SSOT skill 设计，避免把观察位/目标位附近动作误实现为无语义锚点的关节轨迹；明确 anchor、动作空间、安全链路与 catalog 暴露。
- **机器人控制 ([ibrobot-control](./ibrobot-control))**: 约束 Hermes/Agent 经 `robot-skill` 操作现有技能或视觉游戏，并保持运动确认、幂等请求、终态和授权边界。
- **板端约束 ([oh-constraints](./oh-constraints))**: OpenHarmony 板端运行时约束汇总（toybox 命令缺失、musl libc、只读 rootfs、无 systemd、无 /usr/bin/env、LD_PRELOAD 干扰、SSH RemoteCommand 等），凡涉及板端操作前必读。
- **板端连接 ([oh-access](./oh-access))**: 统一封装 OpenHarmony 板的 HDC over TCP 访问与文件传输。
- **OH 主机侧构建 ([oh-build-roboframe](./oh-build-roboframe))**: 通过 `build_roboframe_oh.sh` 交叉编译 `ibrobot_msgs,tensormsg,robot_config,inference_service`，并强制确认 `series.openharmony-5.1.0-musl.txt` 真正进入板端 runtime 产物。
- **Ascend OM 唯一入口 ([om-convert](./om-convert))**: 所有 Ascend OM 转换和优化请求只触发该 skill。它会在开始时集中收集模型、机器拓扑、当前工具支持的精度契约和自动决策偏好；ACT/PI05 使用已有转换流程，其他 LeRobot policy 进入明确标注的实验性支持开发流程。Torch target、精度验证、逐 OM `ais_bench --loop 20` 和可选优化按所选路线执行。PI05 还会确认使用本地 PaliGemma tokenizer 资产还是自动下载并纳入 bundle。
- **RKNN 转换 ([rknn-convert](./rknn-convert))**: 明确 ONNX 导出与 RKNN 转换的边界，避免主 `venv` 与 `.venv-rknn` 污染。
- **HMM 转换 ([hmm-convert](./hmm-convert))**: 将 PI0.5 / SmolVLA 的 xhquant/tcim 编译产物通过 `package-hmm-deployment` 纳入统一 `inference_manifest.json`；ACT HMM 不支持。
- **架构顾问 ([ibrobot-architecture](./ibrobot-architecture))**: 充当项目的架构师，解答一切关于设计模式和配置规范的问题。
- **LeRobot 补丁纳管 ([ibrobot-lerobot-patch](./ibrobot-lerobot-patch))**: 把 `libs/lerobot` 的本地改动回收为受管 patch 栈，而不是直接提交子模块 gitlink。
- **工程规范 ([ibrobot-git-flow](./ibrobot-git-flow))**: 自动化执行开源社区繁琐的提交规范校验。
- **容器验证 ([ibrobot-docker-verify](./ibrobot-docker-verify))**: 在全新 Ubuntu 22.04 Docker 容器中运行 setup.sh 和 build.sh；作者侧门禁先询问 WIP/正式检视，`[WIP]` 暂缓验证。
- **openEuler 容器验证 ([ibrobot-docker-verify-oee](./ibrobot-docker-verify-oee))**: 在 openEuler Embedded aarch64 Docker 容器中验证 setup.sh + build.sh；作者侧 `[WIP]` PR 暂缓。
- **OH 交叉编译移植 ([oh-cross-build-ros-pkg](./oh-cross-build-ros-pkg))**: 将第三方 ROS 2 包（如 usb_cam）通过 Docker 交叉编译工具链移植到 OpenHarmony 板，覆盖克隆、编译、部署、板端验证和 launch 集成全流程。
- **OH 内核重编 ([oh-rebuild-kernel](./oh-rebuild-kernel))**: 重新编译并刷入 OpenHarmony 板的 Linux 内核 (boot_linux.img)，用于启用 USB ACM（SO-101 机械臂）、游戏手柄等内核驱动。

### 📚 DeepWiki 文档工具
这些技能用于生成 DeepWiki 处理配置，把 DeepWiki 输出文档本地化为中文，并保障交付文档质量。

- **配置生成 ([deepwiki-config](./deepwiki-config))**: 从 DeepWiki 目录结构生成或增量更新 `doc_config.json`，保持 `deepwiki_processor.py` 可识别的层级与标签。
- **中文翻译 ([deepwiki-translator](./deepwiki-translator))**: 配置优先流程：本地化配置标题后逐页翻译，再构建目录与索引并校验链接锚点，全程保持 H1、文件名与处理器规则一致。
- **链接校验 ([doc-link-validator](./doc-link-validator))**: 对 Markdown/RST 文档树做只读链接体检——本地相对链接存在性、外部 URL 可达性、AtomGit 源链接（blob/PR/commit/issue）API 级验证；AtomGit 访问由脚本内自含的最小只读 CLI 调用完成（argv/超时/`GC_HOST`/身份探测），技能目录可独立携带到其他 docs 仓库。
- **Mermaid 语法验证 ([mermaid-syntax-validation](./mermaid-syntax-validation))**: 对 Markdown/Sphinx 文档中的 Mermaid 图执行语法风险扫描、最小修复、Sphinx 构建和浏览器端渲染验证。
- **简明技术中文 ([simplified-technical-chinese](./simplified-technical-chinese))**: 受控中文技术写作规范（vendored 自上游 mzopedia/simplified-technical-chinese v0.2，CC BY 4.0 文本 + MIT 脚本，见其 PROVENANCE.md）：40 条编号规则 + 受控词表 + 零依赖 `tools/check.py`；写操作步骤、部署排障等中文文档时执行，仅对本次修改的文件运行检查。

### 🌐 AtomGit 自动化工具
日常 PR / Issue 与评审操作由 Agent 直接调用官方 `gitcode` CLI，参数按需查询 `--help`。
项目检查工具只承担披露、复用自查和验证规则；批量链接检查保留专用脚本（自含最小只读
CLI 调用），intro 为纯本地状态检查，不再远端抽查 PR。

> **⚠️ 前置条件：安装并认证官方 AtomGit CLI（`gitcode` 入口）**
> 
> 使用 AtomGit 相关技能前，必须先完成 CLI 认证：
> 
> 1. 将官方 CLI 安装到 `/usr/local/bin/gitcode`，升级时更新同一文件；
>    保留系统 Graphviz 的 `/usr/bin/gc`。
>    参见[正式安装说明](../../docs/atomgit_cli_migration.md#正式安装linux)，不要依赖工作树 `tmp/`。
> 2. 访问 https://atomgit.com → 右上角头像 → 个人设置 → 「访问令牌」→ 新建访问令牌
>    （勾选 `repo` 和 `pull_request` 权限），**立即复制保存**（只显示一次）
> 3. 在私密终端认证（凭证存储由 CLI 管理；stdin 输入不代表不会持久保存）：
> 
> ```bash
> gitcode auth login --hostname atomgit.com --with-token
> GC_HOST=atomgit.com gitcode auth status
> ```
> 
> 认证与平台访问全部由 CLI 管理；仓库级 `config.json` 及其旧 token 已删除；
> intro 为纯本地检查，不访问平台；CLI 登录配置不受影响。

- **PR 工作流 ([atomgit-pr](./atomgit-pr))**: 面向 PR 资源本身，通过 CLI 创建、读取和更新；发布前用本地工具检查披露、复用自查及 WIP/正式检视规则，`--agent-tool` 使用实际版本命令的输出；如果目标是通用 review，应改用 `atomgit-pr-review`。
- **Issue 工作流 ([atomgit-issue](./atomgit-issue))**: 面向 Issue 资源本身，直接使用 CLI 创建、读取、更新和流转状态，以 `-R owner/repo` 明确目标仓库。
- **通用评审 ([atomgit-pr-review](./atomgit-pr-review))**: Agent 直接读取 PR、diff 和评论，按三步评审流程分析；发布前确认具体发现并复核源仓库和 head SHA。
- **架构扫描 ([atomgit-pr-architecture-review](./atomgit-pr-architecture-review))**: 专门检查是否违背了 SSOT 等核心架构原则。
- **意见处理 ([atomgit-review-resolution](./atomgit-review-resolution))**: Agent 修复代码并按 Git 工作流提交推送，再直接用 CLI 回复和解决线程；不提供自动恢复执行器。
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
1. 调用 [skill-creator](./skill-creator) 作为统一入口，并按其项目约定与校验清单执行。
2. 在 `.agents/skills/` 下创建新目录和 `SKILL.md`；按需添加 `references/`、`scripts/` 等配套内容。
3. 同步更新三份技能索引：`AGENTS.md` 的「Agent 技能索引」、本 `README.md` 的技能清单与分类说明、`.agents/skills/intro/SKILL.md` 的技能分类列表与使用示例。
4. 使用 `skill-creator` 的 Validation Checklist 校验 frontmatter、description、渐进式披露和三索引一致性；任一索引未同步时不得提交。
