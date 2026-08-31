## IB-Robot 控制、感知与语音策略

- 对真实机器人请求，必须使用 `ibrobot-control` Skill 和 Capability Gateway，不得调用裸 ROS 运动类接口
  （`ros2 topic pub`、`ros2 service call`、`ros2 action`、MoveIt、controller、primitive 或硬件接口）。
- 只读感知数据只能通过 `ibrobot-perceive --source <s> --field <f> [--config-name NAME]` 包装命令读取；
  不得直接调用 `ros2 topic echo`、`ros2 topic list`、`ros2 param get` 或任何其他 `ros2` 子命令。
  `ibrobot-perceive` 的 source/field 白名单硬编码在源码中，不得通过修改 config.yaml 扩展。`--source`
  是语义别名；config-backed source 的实际 topic 从 `robot_config` 解析（SSOT），安全边界是 field 集合。
- `ibrobot-perceive` 通过 `ros2 topic echo --once` 读取下一条消息，返回的是单次点时值，不是持久快照。
  对 `voice_direction` 这类事件型 source，只有在发布方活跃时才能在超时内取到值；返回值在消费时
  可能已经过期。`ibrobot-perceive` 返回的字面量值可以作为 `workflow_json` 的参数注入；该值在
  `plan-workflow` 时冻结进 plan digest，后续 validate/confirm/execute 链路按 frozen plan 语义审计。
  感知数据可能过期，执行结果以真实运动为准，不自动重试或修正。
- 用户询问当前电机或关节角度时，只能运行
  `ibrobot-perceive --source arm_joint_position --field position`（可用 `--config-name` 指定机器人）。
  该接口返回原始弧度数组且不返回 `name` 字段，不得编造关节名称、重排数值或把数组索引解释为具体关节。
- `run-workflow` 请求使用显式 `execution_mode=immediate_after_presentation`：生成并验证冻结计划后，先展示并
  flush exact ordered steps、参数、plan digest、registry identity 和 task ID，随后立即调用内部 `confirm-plan`
  绑定 exact tuple 并执行；不得询问“确认执行吗”，不得等待用户再次回复。历史分阶段入口默认使用
  `interactive_confirmation`，不会被该模式改变。
- `confirm-plan` 是 Gateway 技术绑定，不是用户确认门禁。物理运动仍只能由操作员启动 pipeline 时设置的
  `authorize_motion` 授权；Hermes 不得启动或重启 pipeline、修改 ROS 参数或开启运动授权。
- Gateway 不可用、未授权、计划校验失败、执行超时或状态未知时必须停止，不得自动重试或发起新运动。
- 用户发送“别动”“停止”或“取消”时立即通过受控取消入口处理；取消请求不等于机器人已停止，必须等待
  stop-confirmed terminal result。
- TTS 是系统自动功能，不是机器人 Skill。最终自然语言回复由 `post_llm_call` hook 合成并播放，不得在
  workflow 中加入“播报”“语音”或“发声”步骤。
- 机器人任务生命周期语音由受管 hook 自动触发：状态检查前、规划前和计划确认成功后各最多一次。
  文案生成、语音合成和播放必须在后台执行，不得等待、修改或阻断 status/plan/confirm/execute 主链路；
  规划、校验或确认失败时不得播报“开始执行”。
