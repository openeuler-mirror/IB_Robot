# IB-Robot Tracing Web

`ibrobot_tracing_web` 在 Ubuntu 上读取本地 trace，并提供浏览器性能分析界面。当前不支持在
openEuler 上运行 Web server；openEuler 只负责采集 trace，再将 trace 复制到 Ubuntu 分析。

仓库不托管前端 `dist`。每个 Ubuntu 工作区都需要先构建 Web UI，再启动 Web server。

本包位于 `tools/ibrobot_tracing_web`，是显式启用的独立工具。默认 `setup.sh` 的 rosdep 扫描和
`build.sh` 只覆盖 `src`，不会安装 Web 专用依赖或构建本包；ROS 包名仍为 `ibrobot_tracing_web`。

## 1. 采集 Trace

### 1.1 随机器人启动采集（推荐）

在 IB-Robot 工作区根目录执行：

```bash
source .shrc_local

ros2 launch robot_config robot.launch.py \
  robot_config:=so101_single_arm \
  control_mode:=model_inference \
  enable_tracing:=true \
  trace_session_name:=my_trace
```

根据实际机器人修改 `robot_config`。Launch 退出时会停止 tracing session，trace 默认保存在：

```text
~/.ros/tracing/my_trace
```

### 1.2 手动控制采集

```bash
source .shrc_local
export IB_TRACE_ENABLED=1

bash scripts/tracing/start_trace.sh my_trace

# 启动并操作机器人

bash scripts/tracing/stop_trace.sh my_trace
```

### 1.3 从 openEuler 复制 Trace

openEuler 不构建 `ibrobot_tracing_web`。在 Ubuntu Web 主机上执行：

```bash
mkdir -p ~/.ros/tracing
scp -r root@<openeuler-board>:/root/.ros/tracing/<session> ~/.ros/tracing/
```

## 2. 构建 Web UI

Ubuntu 构建需要 Node.js 18 或更高版本。`setup.sh` 不安装 Node.js；请先确认系统已有
`node` 和 `npm`：

```bash
node --version
npm --version
```

先按仓库说明初始化普通工作区环境。只有需要 Web 的 Ubuntu 主机才额外准备以下依赖：

```bash
source .shrc_local
python3 -m pip install -r requirements/tracing-web.txt
npm ci --prefix web/ibrobot_tracing_ui
```

这些是显式安装步骤，不属于默认 Ubuntu requirements 或 Web 构建脚本。随后从仓库根目录构建
Web UI 和两个 tracing ROS 包：

```bash
source .shrc_local
bash scripts/build_tracing_web.sh
```

该脚本只检查已有 Node.js/npm、前端构建依赖和 Python Web 依赖，缺失时退出并提示准备命令，
不自动安装任何依赖。检查通过后依次执行：

```bash
npm run build --prefix web/ibrobot_tracing_ui
./scripts/build.sh -- --base-paths src tools/ibrobot_tracing_web \
  --packages-select ibrobot_tracing ibrobot_tracing_web
```

显式 `--base-paths` 覆盖统一构建脚本的默认 `src`，两个不重叠的根目录各扫描一次；无需
`COLCON_IGNORE` 或额外 setup profile。

生成的 `web/ibrobot_tracing_ui/dist` 由 Git 忽略。首次 clone 或前端源码变化后，都需要重新执行
`build_tracing_web.sh`；前端依赖变化时先显式重跑 `npm ci`。Node.js 只用于构建，Web server
运行阶段不需要 Node.js。

构建后可从仓库根目录验证（需已有 pytest 和 Web 测试依赖）：

```bash
source .shrc_local
python3 -m pytest tools/ibrobot_tracing_web/test
npm test --prefix web/ibrobot_tracing_ui
```

同步到已有部署时，必须删除旧 `src/ibrobot_tracing_web` 目录，不能只复制新的 `tools` 目录。
旧副本残留会重新进入默认 rosdep/build 扫描，并在显式 Web 构建时造成同名包冲突。迁移后重新
构建 Web 包，更新指向旧源码位置的构建/安装产物。

## 3. 启动 Web UI

### 3.1 本机访问

```bash
source .shrc_local
source install/setup.bash

ros2 run ibrobot_tracing_web ibrobot-tracing-web \
  --trace-root ~/.ros/tracing
```

浏览器访问：

```text
http://127.0.0.1:8000/
```

检查服务和 trace 目录：

```bash
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS http://127.0.0.1:8000/api/v1/sources
```

Web server 启动后又复制了新 trace 时，刷新目录：

```bash
curl -fsS -X POST http://127.0.0.1:8000/api/v1/sources/refresh
```

### 3.2 局域网访问

Web server 当前没有认证。只应在可信局域网使用，不要暴露到公网。

```bash
source .shrc_local
source install/setup.bash

ros2 run ibrobot_tracing_web ibrobot-tracing-web \
  --host 0.0.0.0 \
  --port 8767 \
  --trace-root ~/.ros/tracing \
  --allowed-host <ubuntu-host-ip> \
  --allowed-host localhost \
  --allowed-host 127.0.0.1 \
  --allow-unauthenticated-lan
```

远端浏览器访问：

```text
http://<ubuntu-host-ip>:8767/
```

进入页面后选择 trace source，等待分析完成即可查看 Summary、Requests、Timeline、Span 分析和
Critical Path。需要比较时，选择另一份 trace 作为 baseline。

### 3.3 分析输入与资源限制

服务只分析 trace root 中的普通文件和目录，不接受符号链接或特殊文件。分析和比较任务通过
`dir_fd` 与 `O_NOFOLLOW` 打开已校验对象，创建私有临时快照后再交给日志/Babeltrace reader；
不会在校验后让 reader 重新打开可被替换的原路径。扫描条目数、复制字节数及 job history
分别受配置限制，容量不足时拒绝新任务；成功、失败、取消和超时退出后清理临时快照。

这会增加一次最多为 source byte limit 的临时磁盘占用和读取成本。复制期间源发生可观察变更
会拒绝分析，需刷新 source 后重试；复制并复检完成后，源的后续变化不使已接受的快照失效。
该机制防止源目录的符号链接替换造成的越界读取，不是文件系统原子
快照，也不防御拥有同 UID 完整进程权限的攻击者。同步分析器仍需返回或达到自身超时后才能
完成取消清理。trace root 应由可信操作方管理；硬链接按普通文件处理，不保证其内容来源。

快照和 Babeltrace 文本输出使用 Python 临时目录规则（优先 `TMPDIR`）。如果 `/tmp` 是 tmpfs，
它们会占用内存/交换空间；部署时可设置 `TMPDIR` 到 trace root 之外的可写磁盘目录。复制前有
剩余空间预检，但这不是空间预留，不能防止配额或并发耗尽。

默认每源 64 MiB、20 万事件、分析结果缓存总估算预算 256 MiB；可用 `--max-source-bytes`、
`--max-events`、`--max-result-bytes` 调整，同名 snake_case 参数也可通过 ROS launch 传入。
估算预算不包含解释器/SDK开销，结果检查不能阻止解析过程的所有峰值；两个 worker 可以并行，
comparison 运行时还会保留一个 candidate。排队的 comparison 只保留 ID，若 candidate 已淘汰会
明确失败，不把旧结果长期锁在内存中。不要把这些预算当作 OS 硬内存隔离。

刷新接口合并在途扫描、完成后至少间隔 2 秒；写接口拒绝非允许 Origin 的浏览器请求。
warnings API 最多返回 100 条且提供 total/limit/truncated；未知分析错误仅公开固定文案，详情
保留服务端日志。Web 的 ROS launch 测试需要 `launch`/`launch_ros`，依赖已在 package.xml 声明。

## 常见问题

- 提示 Node.js/npm 或 Web 依赖缺失：按脚本提示显式准备依赖后重新构建。
- 服务提示 `API-only`：没有找到 `dist`，重新执行 `bash scripts/build_tracing_web.sh`。
- 页面没有新 trace：确认 trace 位于 `--trace-root` 下，并调用 sources refresh 接口。
