# IB_Robot 专项审查要求

## When to Read

- 在 IB_Robot 仓库做 PR review 时
- 提取 PR 上下文后，看到 `.pr.mandatory_review_checks` 含 `lerobot_gitlink_changed` 时
- 判断 PR 是否触发双平台 Docker Verification 门禁时
- PR 或 commit 表明存在 AI 辅助贡献时
- PR 变更超过 2000 行（`.pr.reuse_self_check.required` 为 true）或
  `.pr.mandatory_review_checks` 含 `large_pr_reuse_self_check_*` 时

## 1. `libs/lerobot` gitlink 强制检查（阻塞性）

- **每个 IB_Robot PR 都必须检查 `libs/lerobot` 是否发生 gitlink / submodule
  指针变化**，不能只查看 `libs/lerobot` 目录下是否有普通文件 diff。
- 以下任一信号都表示 gitlink 发生了变化：
  - changed file 路径精确等于 `libs/lerobot`
  - 文件 mode 为 `160000`
  - diff 中出现 `Subproject commit <sha>`
- IB_Robot 默认通过 `third_party/patches/lerobot/<tag>/` 管理 LeRobot 修改。
  如果 PR 只是修改 LeRobot 源码，却直接提交了 `libs/lerobot` 新指针，而没有导出 managed
  patch，应提交 **severity=error 的阻塞性 review issue**。
- 错误 gitlink 提交可能引用仅存在于贡献者本地仓库或个人 fork 的 commit。其他开发者执行
  `git submodule update --init --recursive` 时会出现 `not our ref`、`did not contain` 或无法
  checkout，对 setup、Docker 验证和发布构建造成直接破坏。

### 合法指针升级必须同时满足

- PR 明确声明这是 LeRobot 上游 tag/base 升级，而不是普通源码修复。
- 新 commit 可以从 `.gitmodules` 声明的权威 submodule URL 获取，不能只存在于作者个人
  fork、本地 object database 或临时分支。
- `third_party/patches/lerobot/INDEX.yaml`、对应 tag 目录、`manifest.yaml` 的
  `lerobot_commit_range` 与新基线保持一致。
- 现有 patch stack 已针对新基线重放或迁移，相关 `series*.txt`、manifest 条目和过滤测试
  同步更新。
- PR 描述包含基线升级原因、patch 迁移情况以及开发者提供的 setup/build 验证。

只要上述条件有一项缺失，就不能把 gitlink 变化当作普通实现细节放行。修复建议应要求作者：

1. 将 `libs/lerobot` 指针恢复到主仓库记录的基线。
2. 在 LeRobot authoring branch 中整理源码提交。
3. 使用 `ibrobot-lerobot-patch` 工作流导出 mailbox patch。
4. 更新 `series*.txt`、`manifest.yaml` 和 `scripts/setup/tests/test_lerobot_filter.sh`。

### 审查输出要求

- 完成三步法的前两步并展示产出后，在第 3 步先查看 `.pr.mandatory_review_checks`。出现
  `lerobot_gitlink_changed` 时必须完成人工判定，不能直接给出"无问题"。
- 对违规指针变更，issue 应定位到 `libs/lerobot`，标题明确说明"禁止直接提交 LeRobot
  submodule 指针"，并在 `fix_code` / 修复方案中给出恢复 gitlink 和导出 patch 的步骤。
- 即使 PR 同时增加了 `third_party/patches/lerobot/*.patch`，也不能自动认为 gitlink 变化
  合法；普通 patch 增量通常不应改变根仓库记录的 submodule 指针。

## 2. README / 文档联动检查（按变更内容决定）

- review 时必须判断本次提交是否改变了**用户可见**的使用方式，而不是机械地要求所有 PR 都改 README。
- 当 PR 修改了以下内容之一时，应检查对应 README / 使用文档是否需要同步更新：
  - 安装、部署、启动、构建、配置、依赖声明或运行步骤
  - 对外暴露的命令、接口、参数、launch 用法、目录约定
  - 会影响用户接入、复现、验证或排障的方法
- 如果变更只涉及内部重构、实现细节、无用户感知的代码整理，则不应为了凑要求而强行提出 README 修改意见。
- 如果判断"应该改 README / 文档但没有改"，应将其作为有效 review issue 提出。

## 3. 依赖 / setup / build 变更的 Verification 强制门禁

- 如果 PR 修改了 ROS 包的 `package.xml` 依赖声明（尤其是新增/删除/调整 `exec_depend`、`build_depend`、`depend`、`test_depend` 等），或修改了全局 setup/build 流程相关文件（如 `scripts/setup.sh`、`scripts/build.sh`、`scripts/setup/platforms/*.sh`、`scripts/setup/verify_env.sh`、`scripts/install_ros.sh`、顶层 `CMakeLists.txt`、顶层 `pyproject.toml` 等），则 **PR 描述中的 Verification 不再是可选项，而是必填项**。
- ROS 包内的 `setup.py` 普通改动（例如 console entry point、Python package metadata 或 Python-only `install_requires` 调整）不单独触发双平台 `setup.sh + build.sh` Verification 门禁；只有同一 PR 还修改了 `package.xml` 依赖声明或全局 setup/build 流程文件时才触发。
- 如果标题以标准 `[WIP]` 前缀开头，表示作者仍在初步提交阶段，暂缓本节的双平台 Docker 证据检查。WIP 不豁免其他 review、DCO、AI 披露或 CI；标题移除 `[WIP]` 后，本节门禁立即恢复。
- 该 Verification 必须体现**真实执行过的验证**，并明确写出：
  - **Scenario**：在哪类干净环境中验证
  - **Method**：如何执行 setup 和 build
  - **Result**：setup / build 是否成功、是否有关键限制或失败点
- 对此类 PR，review 过程中应检查 **PR 描述中的 Verification 是否由开发者提供**，且必须同时覆盖：
  - Ubuntu 22.04 纯净 Docker 环境中的 `setup.sh + build.sh` 完整验证
  - openEuler Embedded 纯净 Docker 环境中的 `setup.sh + build.sh` 完整验证
- PR 描述必须且只能包含一个结构化 `## Docker Verification` 块，含 `Docker verification mode`、`Verified inputs`、`Tested source tree`、`Docker environment` 四个字段。
- 必须将该字段与 PR 最新 head commit 的 tree SHA 比对。`pr_review.py` 会自动生成：
  - `docker_verification_missing`：块缺失或字段不完整。
  - `docker_verification_mismatch`：inputs 指纹不匹配当前输入，或 full 模式下 tested tree 不匹配当前 head tree。
- 两种情况都是阻塞性问题。作者 push 后只有源码 tree 改变才要求重跑 Ubuntu 与 openEuler；只改 commit message、作者或 trailer 而 tree 不变时，已有验证仍有效。
- **review 默认只检查 PR 描述中由开发者声明的验证结果，禁止自动执行双平台 Docker 验证。**
- "review / 审查 / 帮我看看 PR"本身不等于授权执行验证。禁止审查者代替开发者运行 `ibrobot-docker-verify` 或 `ibrobot-docker-verify-oee` 来补齐 PR 描述；只有当用户在当前请求中明确要求 agent 实际执行验证（例如"你来跑一下 Ubuntu/openEuler Docker 验证""帮我实际验证 setup/build"）时，才调用对应验证 skill。
- 如果 PR 描述缺少任一平台验证说明，只给出命令但没有结果，验证没有覆盖 setup/build 两个阶段，或验证 tree 与最新 head tree 不一致，都应视为**阻塞性 review 问题**，要求开发者补充或重跑。

## 4. openEuler AI 贡献元数据检查（阻塞性）

- 当 PR 声明 AI 参与，或任一 commit 包含 `Co-Authored-By` 时，必须检查 PR 正文是否完整披露：
  Agent 平台及版本、模型名称及版本、Prompt 摘要、人工审查情况、第三方材料来源和许可证情况。
- 每个 AI-assisted commit 必须包含 `Co-Authored-By: <AI 模型名称及版本>`；不同 commit 可以使用不同模型，PR
  的模型信息必须完整覆盖这些值。缺失、占位值或存在未披露模型均为阻塞性问题。人类共同作者使用
  `Co-Authored-By: Name <email>`，不参与 AI 模型集合比较。
- 模型元数据只记录模型名称及版本（如 `gpt-5.6-sol`），不得携带 `xunxing/` 等 provider 前缀。
- Agent 平台字段必须包含 coding agent 实际版本命令确认的工具名和语义版本。仓库不维护工具白名单，
  reviewer 也不应在本机重新执行作者的工具；只将 `latest`、`unknown`、缺少版本、手工拼接的危险字符或
  无法解析的格式视为阻塞性元数据问题。
- 检查贡献者是否声明已进行人工审查，并关注无法解释或维护的输出、许可证不兼容材料、商业秘密、
  个人信息、敏感数据、私有代码、内部文档和未公开漏洞信息。
- 元数据完整不代表代码自动合规；仍需按风险检查正确性、安全性、许可证和必要 Verification。
- 完整规则见 [openEuler 社区生成式AI工具使用与开源贡献策略](https://www.openeuler.openatom.cn/zh/community/ai-coding-assistants/)。

任一必填披露缺失，或 commit 使用了 PR 未披露的模型时，应提交 `severity=error` 的阻塞性 review issue。

## 5. 禁止本地重复执行 pre-commit 已覆盖的检查

- IB_Robot 的 `.pre-commit-config.yaml` 已把 `ruff --fix` 与 `ruff-format` 作为强制 pre-commit hook，且 `.git/hooks/pre-commit` 随仓库安装；开发者 `git commit` 时必然已通过 ruff 校验，**PR 上线代码不会再有 `ruff check` / `ruff format` 报错**。
- 因此 review 时**禁止**在本地做以下动作（属于重复劳动，浪费上下文且无新增信息）：
  - `git apply` / `git checkout` / `git diff` 拼接 PR diff 后跑 `ruff check` / `ruff format --check` / `pyright` / `mypy` / `py_compile` 等 lint / format / typecheck 命令
  - 切到 PR 分支跑 `colcon build` 来"验证"代码能否编译——这属于开发者侧 Verification 范畴，由 §3 门禁管理
- 替代做法：
  - **信任 PR 描述中开发者声明的 ruff / py_compile / build 结果**，除非描述缺失且 PR 触发 §2 门禁
  - 如怀疑某行有 lint / 类型 / 风格问题，**直接在 inline 评论中指出并附修复建议**，由开发者在下一次提交时让 pre-commit 自动修复，不要在本地复跑
  - 静态阅读 diff 与本仓库源码（`Read` / `Grep` / `Glob`）始终允许且推荐——这是判断架构/逻辑/命名问题的正常手段
- **例外**：用户在当前请求中明确要求"你帮我跑一下 ruff / typecheck / build 看看"时才执行相应命令；"review 这个 PR""帮我看看这个 PR"本身**不构成**授权。

## 6. 大型 PR 复用自查门禁（阻塞性）

### 触发条件与脚本信号

- **阈值**：PR 变更行数（additions + deletions，由 `pr_review.py` 按 PR 文件统计求和，统计缺失时按 patch 的 +/- 行计数兜底）**超过 2000 行**即触发。
- `[WIP]` **不豁免**本门禁：与双平台 Docker 门禁不同，复用自查是纯文档要求，且复用/重复造轮子问题在 WIP 阶段暴露价值最大——作者还没在错误方向上投入太多。
- `pr_review.py --extract-info` 会自动生成 `.pr.mandatory_review_checks`：
  - `large_pr_reuse_self_check_missing`：描述中没有 `## Reuse Self-Check` 章节。
  - `large_pr_reuse_self_check_incomplete`：章节存在，但四个字段有缺失或空值。
  - `large_pr_reuse_self_check_invalid`：章节格式歧义（多个同名标题或重复字段）。
  - 三者均为 severity=error 的阻塞性检查，`.pr.reuse_self_check` 额外给出
    `changed_lines` / `required` / `status`。

### 必需的块格式

描述中必须且只能包含一个如下结构（字段标签固定英文，正文默认中文，"无"也要显式写明）：

```markdown
## Reuse Self-Check

**Reinvented workflows:** <是否重新发明了现有流程；无重叠写"无"，有则列出>
**Reused components:** <是否沿用仓库与 libs/lerobot 已有内容；列出复用点及位置>
**Reinvention justification:** <重新发明的必要性论证；未重新发明时写"无（未重新发明现有流程）">
**Architecture conformance:** <对齐的同类功能架构及一致性说明，或偏离原因>
```

### Reviewer 审计协议（块完整时仍必须执行）

脚本只能校验"块存在且字段非空"，**无法判断声明是否属实**。看到大型 PR 时，即使
`status == complete`，也必须对照 diff 审计四项声明：

1. **验证"没有重新发明"的声明**：
   - 新增的 setup / 数据集 / 评测 benchmark / 推理 / 部署管线，先查 `libs/lerobot`
     （datasets、policies、benchmarks 等模块）是否已原生支持（参考
     [PR #309](https://atomgit.com/openeuler/IB_Robot/pull/309)：为 lerobot 已支持的
     benchmark 另建一整套 setup 流程）。
   - 再查仓库既有能力：`inference_service` 推理框架、`robot_config` SSOT、`tensormsg`
     契约、`dataset_tools`、`model_utils`、既有 `scripts/`。绕过既有框架另起平行实现
     是典型违例（参考 [PR #317](https://atomgit.com/openeuler/IB_Robot/pull/317)）。
   - 声明"无重叠"但 diff 中存在功能重复实现的，提交 severity=error 的阻塞性 issue，
     定位到重复实现的具体文件，fix 建议给出复用路径（如改用 lerobot 对应模块、接入
     inference_service；改动确实属于 `libs/lerobot` 的走 `ibrobot-lerobot-patch` 导出 patch）。
2. **验证复用清单**：`Reused components` 列出的模块/接口在 diff 中确实被调用，而不是
   装饰性罗列。
3. **评估必要性论证**：`Reinvention justification` 需给出既有内容不满足的具体原因
   （接口缺失、性能、许可证、平台限制等）及"是否评估过扩展既有实现"。空泛理由
   （"现有代码不好改""顺手重写更快"）不构成正当理由，应作为 error 提出；论证充分且
   合理的重新发明可以放行，但架构不一致处降级为 warning/suggestion 指出。
4. **验证架构一致性声明**：对照 `atomgit-pr-architecture-review` 的支柱（SSOT、契约、
   包职责、数据流）检查 `Architecture conformance` 声称对齐的同类模块是否真实对齐。

### 与其他门禁的关系

- 本门禁只依赖变更行数，与 §3 的文件触发型 Docker 门禁相互独立；一个 PR 可以同时触发两者。
- 声明不属实属于**阻塞性问题**；块缺失/不完整时不要代替作者编造声明，应要求作者补充。
