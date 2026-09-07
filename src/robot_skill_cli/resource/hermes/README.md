# Hermes 集成发布源

此目录是 IB-Robot Hermes 集成的唯一发布源，包含推荐 `SOUL.md`、强制策略 (`POLICY.md`)、
`pre_tool_call` / `post_llm_call` hook 与手动同步入口。同步会安装当前 `ibrobot-control` Skill，
绑定当前 workspace、robot config 和 ROS Domain，并清理已退役的 `ibrobot-robot-control` Plugin
遗留副本（即时执行改由 `robot-skill run-workflow` 的显式 `immediate_after_presentation` 模式承担）。

## 资源

| 文件 | 说明 |
|---|---|
| `SOUL.md` | Hermes persona 与完整具身任务工作流（含 `ibrobot-perceive` 使用指引） |
| `POLICY.md` | 强制策略：运动只能走 `robot-skill` + Gateway；感知读取只能走 `ibrobot-perceive`；裸 `ros2` 子命令被禁止；TTS 由 `post_llm_call` hook 自动完成 |
| `hooks/ibrobot-block-raw-ros` | Hermes `pre_tool_call` hook，用 `shlex` 分词拦截裸 `ros2`/`rclpy`/`roslaunch` 调用 |
| `hooks/ibrobot-speak` | Hermes `post_llm_call` speech hook wrapper，source `.shrc_local` 后 `exec python3 -m robot_skill_cli.hermes_tts_hook`；TTS 服务名与超时来自 `robot_config` SSOT |
| `hooks/ibrobot-lifecycle-speech` | Hermes `pre_tool_call` / `post_tool_call` hook；只投递状态检查、规划和计划授权事件，文案生成、TTS 合成和播放均在后台执行 |
| `hooks/ibrobot-interim-speech` | Hermes `on_interim_message` hook；加载 workspace `.shrc_local` 后按 turn 有界异步转交当前 profile 的 TTS hook |
| `sync_hermes.sh` | 手动同步入口（等价于 `hermes-robot-configure`） |

## `ibrobot-perceive`（感知读取唯一入口）

`ibrobot-perceive` 是 `robot_skill_cli` 提供的独立 console script，不经过 Gateway。它以硬编码
source/field allowlist 限制可读取的感知量，通过 `ros2 topic echo --once` 读取 YAML 输出并打印请求
字段的裸字面量值，供 LLM 直接读取并注入 `workflow_json`。当前 allowlist 包含
`voice_direction`（topic `/voice/speech_direction`，字段 `azimuth_rad`、`seq_id`）和
`arm_joint_position`（字段 `position`），扩展必须修改源码，不接受 config.yaml 覆盖。

`--source` 是语义别名而非 ROS topic 名：`voice_direction` 是 `voice_asr_service` 的固定契约，topic 写死
不读 robot_config；`arm_joint_position` 的 topic 在运行时从 `robot_config` 的 `moveit.joint_state_topic`
解析（so101 -> `/joint_states`，lekiwi_handeye -> `/arm_joint_state_broadcaster/joint_states`），复用
`resolve_robot_config_path()`，不维护第二套路径优先级。安全边界是硬编码的 *field* 集合，不是 topic 名。
`arm_joint_position.position` 直接返回原始弧度数组，不提供关节名映射。

`ros2 topic echo --once` 返回的是下一条已发布消息的单次点时值，不是持久快照。对
`/voice/speech_direction` 这类事件型 topic，发布方不活跃时会在超时内取不到值；取到的值在消费时
可能已经过期。该值在 `plan-workflow` 时冻结进 plan digest，后续按 frozen plan 语义审计；执行结果
以真实运动为准，不自动重试或修正。

## `ibrobot-block-raw-ros`（pre_tool_call 拦截 hook）

此 hook 是 `pre_tool_call` 防御层。它从 stdin 读取 Hermes tool payload，提取 `tool_input.command`，
用 `shlex` 分词后阻断任何裸 `ros2` 子命令和 `rclpy`/`roslaunch` 间接调用，强制 LLM 走
`ibrobot-perceive`（感知读取）或 `robot-skill`（运动控制）。

**它是 defense-in-depth，不是沙箱。** 权威边界仍然是操作员的 `authorize_motion` 授权门禁和
Gateway 的 plan validation。hook 输出 `{"action":"block","message":"..."}` 表示拦截，空输出表示放行；
exit code 不用于拦截判定。

协议：

- stdin：JSON，含 `tool_input.command`（字符串或列表）。
- stdout：拦截时输出一行 `{"action":"block","message":"..."}`；放行时不输出。
- 解析失败或缺少 `command` 字段时放行（fail-open），因为 hook 无法判断意图。

## 同步

从仓库根目录执行：

```bash
source .shrc_local
export ROS_DOMAIN_ID=56

hermes-robot-configure \
  --config-name so101_single_arm \
  --soul-mode replace \
  --accept-hooks \
  --restart-gateway
```

`.shrc_local` 是 ROS+venv+install overlay 的 SSOT 入口（见 `AGENTS.md` 环境初始化章节），
无需重复 `source install/setup.bash`。

也可以使用此目录中的 shell 入口：

```bash
bash src/robot_skill_cli/resource/hermes/sync_hermes.sh \
  --config-name so101_single_arm \
  --soul-mode replace \
  --accept-hooks \
  --restart-gateway
```

先预览而不写文件：

```bash
hermes-robot-configure --config-name so101_single_arm --dry-run
```

`--soul-mode replace` 会先备份现有 `SOUL.md`，再安装仓库中的完整推荐版本。使用
`--soul-mode merge` 会在现有文件中维护一个 `IBROBOT-MANAGED` 区块；`--soul-mode skip` 不修改 SOUL。

## 同步产物

同步在当前 Hermes profile 下生成以下受管文件（带 managed 标记，重跑同步即覆盖，不要手改）：

- `skills/ibrobot-control/SKILL.md`：当前 `ibrobot-control` Agent Skill 副本。
- `ibrobot/bin/robot-skill`：绑定 `ROBOT_CONFIG`、`ROS_DOMAIN_ID` 并 source workspace
  `.shrc_local` 的 wrapper；进入 Hermes 会话后用它而非裸 `robot-skill`，且不得再传
  `--config-name` / `--config-path`。
- `ibrobot/ibrobot-env.sh`：写入 `terminal.shell_init_files` 的环境文件，并把
  `auto_source_bashrc` 置 `false`，使受管环境优先于用户 bashrc；它同时把 `ibrobot/bin`
  加入 `PATH`。排查「Hermes 不再 source 我的 bashrc」时先看此文件与该开关。
- `hooks/ibrobot-speak`：`post_llm_call` speech hook wrapper，source `.shrc_local` 后
  `exec python3 -m robot_skill_cli.hermes_tts_hook`；TTS 服务名与超时来自 `robot_config` SSOT。
- `hooks/ibrobot-lifecycle-speech`：机器人任务生命周期 speech hook wrapper，使用当前 IB-Robot
  workspace 中的 `robot_skill_cli` 和 `embodied_agent` 异步生成文案，并投递状态检查、规划和计划授权成功三类语音事件。
- `hooks/ibrobot-interim-speech`：由配置器绑定当前 workspace 和 `ibrobot-speak` 绝对路径，使用完整
  `.shrc_local` 环境启动；每个 `(session_id, turn_id)` 最多投递一次，失败不阻塞 Hermes。

`--accept-hooks` 先 `hermes hooks revoke` 清理旧 mtime，再用
`hermes --accept-hooks hooks doctor` 重新批准；首次安装无既有审批时，revoke 的非零退出经
`hermes hooks list` 确认未注册后跳过，不会阻断审批。

## 前置条件

- 已安装 Hermes Agent 0.16.0 或更新版本。
- 已构建 `robot_skill_cli`、`robot_config`、`ibrobot_msgs` 和 TTS 相关包。
- 当前终端已 source 目标 IB-Robot workspace（`source .shrc_local`）。
- 启用语音时，目标 robot config 的 `voice_tts.enabled` 必须为 `true`。
- 真机运动仍必须由操作员在启动 pipeline 时显式设置 `authorize_motion:=true`。

### 生命周期文案模型环境

生命周期 Hook 使用 IB-Robot 已提供的 `robot_skill_cli`、`embodied_agent`、`embodied_common` 和
`rclpy`，不会从 Hermes 或其他目录加载 Python 包。Hook 只需要访问文案模型的 API key。

Hook 固定使用 `gpt-5.6-sol` 路由，该路由在 `embodied_common/config/vlm_models.yaml` 中声明：

```yaml
api_key_env: XUNXING_API_KEY
```

推荐在启动 Hermes 的环境中直接设置：

```bash
export XUNXING_API_KEY='你的实际密钥'
```

如果现有 Hermes 已经把密钥保存在自己的 `.env` 中，可以指定该文件：

```bash
export HERMES_ENV_FILE=/path/to/hermes/data/.env
```

文件中可以直接配置 `XUNXING_API_KEY`；对于当前 Hermes 的自定义 provider，也支持：

```dotenv
HERMES_CUSTOM_AZ_GPTPLUS5_COM_API_KEY=你的实际密钥
```

Hook 会在没有 `XUNXING_API_KEY` 时将该变量映射为 `XUNXING_API_KEY`。310P 的现有路径是
`/root/claw/hermes/data/.env`，其他机器应改成自己的 Hermes `.env` 路径。真实 API key 不得提交到仓库。

## 生效与验证

本地 CLI：

```bash
hermes-robot --config-name so101_single_arm -- --cli
```

飞书使用长期运行的 Hermes Gateway。同步配置或 Skill 后必须重启，并在飞书中发送 `/new` 创建新会话：

```bash
hermes gateway restart
hermes gateway status
hermes hooks list
hermes hooks doctor
```

动作请求的预期流程为：展示并 flush 冻结计划，内部调用 `confirm-plan` 绑定 exact tuple，然后立即
`execute-plan`，不等待用户再次回复“确认”。这不会绕过 `authorize_motion`、Gateway 校验或停止能力。

最终自然语言回复通过 `post_llm_call` 自动调用 `/voice_tts/synthesize` 和 `/voice_tts/play`，本地扬声器
播放结果。诊断日志位于 `/tmp/hermes-speak.log`。

生命周期语音不会阻塞 `robot-skill`：`pre_llm_call` 只暂存用户原话，确认该回合调用 `status` 后才在
后台启动文案模型，普通对话不会产生额外调用。每个机器人任务最多调用一次 `LLMClientService` 生成三条文案，
TTS 合成在后台执行，音频播放通过单设备锁串行化，避免多个进程争用同一声卡；TTS 或文案模型失败
只使用 fallback 或记录日志，不改变规划、执行和安全门禁。

## 升级

每次更新并重新构建 IB-Robot 后再次执行同一同步命令。同步是幂等的，不会重复添加 hook、Skill 或
SOUL 托管区块。配置变化后必须重启 Gateway；仅使用本地 CLI 时重新启动 CLI 会话即可。
