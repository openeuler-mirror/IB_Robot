# 测试规范详解（ROS 2 / ament）

本文是 [`AGENTS.md`](../AGENTS.md)「测试规范」章节的详细展开，覆盖设计动机、判定标准、排障案例与反例。核心规则以 `AGENTS.md` 为准，本文提供执行细节。

## 1. 正式验收唯一入口：`colcon test`

`src/` 下所有 ROS 包的测试**必须**能被 `colcon test` 发现并执行。这是合入前的正式验收口径：

```bash
source .shrc_local && colcon test --packages-select <pkg>
source .shrc_local && colcon test-result --all --verbose
```

**裸 `pytest` 的定位**：允许用于**定向验证**——复现某个具体缺陷、隔离排查某个行为、快速迭代单个文件。定向验证的结论要如实标注为定向场景（例："`timeout 45 pytest -q test_x.py` 复现挂死，退出码 124"），它不能替代包级验收；合入前对包的正式结论必须来自 `colcon test`。

**`ament_python` 包必须声明 pytest**，否则 colcon 静默回退到 `unittest`，收集 0 个用例并报告 `Ran 0 tests ... OK`——测试看起来"通过"，实际一个都没跑：

```python
extras_require={"test": ["pytest"]},
```

不要用 `tests_require=["pytest"]`。该参数已被 setuptools 移除，只会产生 `UserWarning: Unknown distribution option` 并被丢弃，效果等同于没写。

`ament_cmake` 包的每个 pytest 文件都要在 `CMakeLists.txt` 的 `BUILD_TESTING` 块内用 `ament_add_pytest_test(<target> test/<file>.py)` 单独注册；漏注册的文件不会被执行。

**包内 `pytest.ini` 的两个陷阱**：

1. `testpaths` 必须覆盖该包**全部**测试目录。只列一个目录会让另一个目录被静默跳过（`src/inference_service/pytest.ini` 曾因此漏掉整个 `test/` 调度面）。
2. 根 `pyproject.toml` 的 `[tool.pytest.ini_options]` 只设 `norecursedirs`，**不要设 `testpaths`**：`colcon test` 调用 `pytest` 时不传路径参数、工作目录是包目录，pytest 会向上找到根配置作为 rootdir 配置，于是用根 `testpaths` **替换掉**每个包自己的测试目录——所有包的测试会被静默换成这个列表的内容。
3. 包内 ini 会替换根配置，根 `addopts` 里关闭 `launch_testing` / `launch_ros` 插件的参数必须在包内 ini 重复（见第 5 节）。

ROS 包之外的仓库级测试（`tests/`、`scripts/`、`.agents/`）需要显式指定路径运行：

```bash
source .shrc_local && pytest tests scripts .agents
```

## 2. 跨包测试依赖

测试导入的**每个**兄弟 ROS 包都必须出现在本包 `package.xml` 的 `<test_depend>` 中。否则 `colcon test --packages-select <pkg>` 会因为对方未构建而失败，整工作区 `colcon test` 也只是碰巧依赖构建顺序才通过。

如果补上这条 `<test_depend>` 会形成依赖环，`colcon` 会直接拒绝为整个工作区排序（`Unable to order packages topologically`），**连构建都无法进行**。此时说明**测试放错了包**：被测行为属于依赖方，测试应当移到依赖方的包里，而不是在被依赖方里反向 import。不要为了让测试跑起来而声明成环的依赖。

判断移动前先区分两种情况：

- **测试放错包**：只有测试导入对方，生产代码没有。把测试移到依赖方，必要时连同其 fixture 一起复制，并在新文件的 docstring 里写明为什么在这里、复制了什么。
- **架构环路**：双方的**生产代码**互相 import（例如 A 的模块 import B，B 的模块又 import A，哪怕是函数内的延迟 import）。这不是测试问题，移动测试只会掩盖它。应当记录并单独走架构重构，不要在测试治理里顺手处理。

移动的代价要算清楚：如果搬一个测试需要复制上百行 fixture，说明缺的是共享 fixture 包，而不是这个测试放错了位置。记录下来，不要硬搬。

## 3. ROS 域隔离

`colcon test` 每个包起一个独立 pytest 进程并且并行执行。创建真实 ROS 节点的测试如果都落在默认域 0，会互相发现对方的 publisher 和 service，测试变得依赖执行顺序且随机失败。

仓库根 `conftest.py` 用 `domain_coordinator.domain_id()` 为每个 pytest 进程独占分配一个域，并设置 `ROS_DOMAIN_ID`、`IBROBOT_TEST_ROS_DOMAIN_ID` 和 `ROS_LOCALHOST_ONLY=1`。这与 `ament_cmake_ros` 的 `run_test_isolated.py` 是同一套机制，只是 `ament_python` 包没有对应的 CMake 包装。

**根 conftest 不是自动加载的**。pytest 的 conftest 搜索从 rootdir 向上直到 confcutdir 边界：包内自带 `pytest.ini` 会把 rootdir 定在包目录，向上搜索截断在包目录，根 `conftest.py` **不会加载**——该包的节点测试会在无域隔离下静默运行（本 PR 的评审意见即指出 `inference_service` 曾处在这个状态）。`confcutdir` 是仅命令行选项、不能写进 ini，因此带自有 ini 的包必须在**包内 `conftest.py`** 显式加载根隔离逻辑（导入根模块并重导出钩子，保持单一实现；范例：`src/inference_service/conftest.py`）。

新增带自有 `pytest.ini` 的包时，验证隔离生效的方式：清除 `ROS_DOMAIN_ID` / `IBROBOT_TEST_ROS_DOMAIN_ID` / `ROS_LOCALHOST_ONLY` 后，从包目录以标准入口跑一个断言 `IBROBOT_TEST_ROS_DOMAIN_ID` 已设置的探针测试，应当通过。

其余约定：

- `ROS_LOCALHOST_ONLY` 必须**强制覆盖**为 `1`，不能用 `setdefault`：ROS 环境会导出 `0`，而独立域并不能阻止多播发现跑出本机。
- 需要 ROS 图的测试应断言 `IBROBOT_TEST_ROS_DOMAIN_ID` 已设置，而不是直接读 `ROS_DOMAIN_ID`，这样"为本进程分配过域"和"开发者碰巧 export 了 `ROS_DOMAIN_ID`"可以区分开。
- 调试时用 `DISABLE_ROS_ISOLATION=1` 关闭分配；此时依赖隔离的测试会失败退出，而不是静默共享域 0。

## 4. ROS 测试必须能退出

测试"全部通过"和进程"能退出"是两件事。rclpy 把 action 的 `execute_callback` 放在 `ThreadPoolExecutor` 的 worker 线程上，而 `concurrent.futures` 注册的 atexit 钩子会**无超时地** join 这些 worker。只要有一个回调还在跑，pytest 打印完 `50 passed` 之后解释器就永远退不出去，`colcon test` 会一直等这个包，最后整条流水线超时被杀——而测试报告上一条失败都没有。

因此，测试里的 action server mock **不能只依赖取消来结束**：

```python
# 错误：只有被取消才会退出，测试忘记取消就吊死整个进程
while not handle.is_cancel_requested:
    time.sleep(0.01)

# 正确：上下文关闭或超时也要退出
deadline = time.monotonic() + 10.0
while not handle.is_cancel_requested:
    if not rclpy.ok() or time.monotonic() > deadline:
        handle.abort()
        return Action.Result()
    time.sleep(0.01)
```

同理，清理路径上的 `thread.join(timeout=...)` 超时后**不要静默继续**。线程还活着就说明 executor 没停干净，此时再 `rclpy.shutdown()` 是在一个仍被使用的上下文上做销毁。要么把超时当失败抛出，要么确保 spin 循环会响应关闭。

自查方式：单跑一个包不足以发现这类问题，因为它依赖负载。用带超时的方式重复跑，**退出码 124 表示挂死**，即使测试全部通过：

```bash
timeout 60 pytest -q   # rc=124 → 进程没退出，不是测试失败
```

## 5. pytest 插件与静默归零

ROS 的 `launch_testing` / `launch_ros` 插件是自动加载的，它们会把一个**模块级** skip（如 `pytest.importorskip("placo")` 目标不可导入）升级成整个 session 的收集中止：本该只跳过一个文件，实际变成 `collected 0 items`。`robot_moveit` 曾因此从 34 个测试变成 0 个。

更糟的是 colcon 把 pytest 的 `NO_TESTS_COLLECTED` 退出码**当作成功**（`colcon_core/task/python/test/pytest.py`），所以一个包可以在"零测试执行"的状态下通过。

因此仓库根 `pyproject.toml` 的 `addopts` 关掉了这两个插件。没有任何被收集的测试需要它们（`tests/ci_smoke/*.py` 是独立脚本，不以 `test_` 开头）。**包内自带 `pytest.ini` 会覆盖根配置，必须同样带上这两个 `-p no:` 参数**，`src/inference_service/pytest.ini` 即是例子。

新增模块级 `importorskip` 前先想清楚：它的目标一旦在某个 CI 镜像里缺失，整个包的测试会静默归零而不是失败。优先用逐个测试的 `@pytest.mark.skipif`。

自查：`pytest --collect-only -q <pkg>` 的数量应与预期一致；出现 `collected 0 items` 时，它是缺陷信号而不是"这个包没测试"。

## 6. 测试分层

| 层 | 内容 | 依赖 | 归属 |
|---|---|---|---|
| 1. 纯逻辑 | 算法、决策规则、数据变换、契约与 schema 校验 | 无 ROS、无硬件 | 各包 `test/` |
| 2. 节点接线 | 参数声明、话题/服务名与类型、生命周期迁移 | rclpy，无多节点 | 各包 `test/` |
| 3. 集成 | 多节点协作、启动顺序、端到端 | ROS graph、mock 硬件 | 集成测试包 |

绝大多数测试应在第 1 层。业务逻辑要与 rclpy 节点分离，让逻辑能用普通 Python 类型直接测试。需要 `rclpy.init()` 或真实 ROS graph 的测试属于第 2、3 层，数量应远少于第 1 层。

用 `object.__new__(Node)` 绕过构造函数、再手工赋一堆私有属性来"造"一个节点，是逻辑没有从节点里拆出来的信号。这类测试绑死私有字段布局，任何 `__init__` 重构都会成片变红。新增测试不要采用这种写法；遇到时优先把被测逻辑提取为独立函数或类。

需要真机、板端 NPU、或仓库中不存在的模型 bundle 的测试**不进默认门禁**，归入发布验收。

## 7. 什么测试该留，什么不该留

留下的每个测试都要能回答一个问题：**它失败的时候，说明了什么 bug？**答不上来就不该留。

应当保留：

- **契约与 SSOT 测试**：`robot_config` 的配置摘要与跨包策略校验、`embodied_common` 的冻结 digest、消息 wire schema 双向校验（IDL 文本与生成类型一致）。
- **有命名失败模式的回归测试**：测试名描述被防住的行为，而不是 issue 编号。
- **真实数值与边界行为**：坐标变换、插值、量化、时间戳对齐。
- **端到端 smoke**：用仓库内签入的小资产驱动完整路径。

不应保留（删除前先过第 8 节的行为存续判定）：

- **断言常量等于自身定义**，如 `assert MOTOR_COUNT == 6`。只在有意改值时失败，不提供信息。
- **把被测数据表重新抄一遍**，包括逐字复述提示语文案。改一个措辞就红，且没发现任何缺陷。
- **只断言 mock 调用序列而不断言可观察结果的测试**（change-detector）。例外：当被测行为本身就是协作契约——例如"传给子进程的参数""生成消息的字段"就是规格所要求的行为——那不是 change-detector，断言它们是断言真实行为。
- **针对不存在的 API 写的测试**。但要先走第 8 节的判定确认 API 确实不存在、被测行为确实消失了，再删；导入失败也可能只是环境缺失（见第 5 节的静默归零）。
- **断言源码文本或文档措辞**，如 `assert "def start_viewer(...)" in source` 或 `assert "某段说明" in README`。它是把 grep 伪装成测试：改个措辞就红，真坏了却可能照过。改为断言可观察行为（传给子进程的参数、生成消息的字段）。
- **不是测试的文件**。CLI、可视化脚本、真机验证脚本不要用 `test_` 前缀命名，否则会被 pytest 收集并报错；用 `debug_` / `verify_` 前缀。
- **与仓库 linter 重复或冲突的样板测试**。本仓库的 Python linter 是 Ruff（见 `AGENTS.md`「Python 风格」），不使用 `ament_flake8` / `ament_pep257` / `ament_copyright`。`ament_cmake` 包在 `BUILD_TESTING` 块内用 `set(ament_cmake_<linter>_FOUND TRUE)` 跳过，并写明原因。

**永远无法通过的 linter 等于没有 linter**，而且会训练所有人无视红色。要么让它通过，要么带理由关掉，不要留着长期爆红。关掉之前必须先用仓库自己的 linter 把真实问题修掉，否则就是掩盖：`ament_flake8` 报的问题里，行长超限是配置冲突，但未使用的 import、尾随空白是 Ruff 同样会报的真实缺陷。

删除测试时，必须在 commit message 里说明被删覆盖面由什么替代，或为什么不需要替代。历史先例见 `src/inference_service/tests/LEGACY_COVERAGE.md` 的做法：逐项记录被移除的测试面与其接替者。没有退役记录，被删的测试会被后续的人或 Agent 重新生成回来。

## 8. 测试落后于接口变更时：先判断行为还在不在

被测接口变了导致测试红，不等于测试该删。先回答：**它想验证的行为，今天还存在吗？**

- **行为还在，只是入口变了** → 修测试。字段新增（`status_origin`）、前置条件收紧（映射需 `encoded_frames > 0`）、实现搬到别的包，都属于这类。修的时候要理解**为什么**改，把语义写进注释，而不是把断言改成迎合当前输出。
- **行为本身没了** → 删测试，并在 commit message 里写清覆盖面去向。
- **生产代码里的测试专用方法被删了**（如只为测试存在的 `flush()`）→ 用公共 API 重建**等价的等待能力**，但等待条件必须是被测的完成结果本身：预期发送数量、对应的完成标识、失败状态或丢帧计数。**不要**拿"队列清空"当完成信号——工作线程把帧出队之后、编码和发送完成之前，队列深度就已经读 0，按它等待会在结果尚未产生时提前返回（`test_device_video_streams.py` 的 `_drain` 曾有这个竞态，评审指出后改为显式完成条件）。也不要要求把测试专用方法加回生产代码。
- **测试深挖已消失的私有属性**（`manager._streams`）→ 改用构造期注入等受支持的入口。如果连注入点都没有，那才是真正需要重新设计的信号。

一个反例要特别小心：**测试红也可能是生产代码的 bug**。曾出现 `test_observation_video_integration` 报协议版本不匹配，查下来是设备侧重构后漏传 `protocol_version`，退回默认值 5 而计算侧要求 6——功能在生产中是断的。这种情况必须改生产代码，把测试改绿等于把 bug 藏起来。

## 9. AI 生成的测试

AI 生成的测试按**草稿**对待，不是生成即入库。合入前必须逐个确认：

1. 它断言的是**行为**，不是当前实现的结构。
2. 它**真的在 `colcon test` 下跑过并通过**，不是只在作者本机的某个路径下跑过。
3. 它针对的 API **确实存在**。AI 会对着想象中的接口写出成百上千行看似合理的测试。
4. 它没有和既有测试重复覆盖同一条路径。

不要为了提高覆盖率数字而保留一批相互重复的小测试。宁可少而准。
