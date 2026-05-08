# VoiceASRNode 节点说明

`voice_asr_node.py` 是 `voice_asr_service` 包中的运行时 ROS 2 语音识别节点。
它负责把麦克风音频或音频文件转换成文本，并统一管理音频采集、VAD、sherpa-onnx 模型加载、ROS 接口以及内部状态机。

这份 README 说明的是**节点本身**，不是仅针对包级入口的简单介绍。

## 1. 节点职责

`VoiceASRNode` 支持两类输入路径：

| 输入路径 | 作用 | 模型要求 |
| --- | --- | --- |
| 麦克风实时识别 | 从音频输入设备持续监听并输出识别文本 | **必须使用流式模型** |
| 音频文件识别 | 解码文件并返回/发布识别结果 | 可使用流式或离线模型 |

核心职责包括：

1. 读取 ROS 参数并初始化各个运行模块。
2. 在模型文件缺失时自动解析并下载默认 ASR bundle。
3. 从麦克风采集音频或从文件加载音频。
4. 使用 VAD 判断语音起止边界。
5. 调用 sherpa-onnx 执行解码，并发布中间/最终结果。
6. 通过 ROS topic 和 service 暴露控制与文件识别能力。

## 2. 文件位置与启动入口

| 项目 | 路径 |
| --- | --- |
| 节点实现 | `src/voice_asr_service/voice_asr_service/voice_asr_node.py` |
| 控制台入口 | `voice_asr_node = voice_asr_service.voice_asr_node:main` |
| 包级 README | `src/voice_asr_service/README.md` |

直接调试节点时可这样运行：

```bash
cd /path/to/IB_Robot
source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 run voice_asr_service voice_asr_node --ros-args \
  -p model_path:=models/voice_asr/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23 \
  -p model_type:=streaming
```

生产或完整系统场景仍建议通过 `robot_config` 启动，因为机器人级参数的单一事实来源仍然是 `robot_config`：

```bash
cd /path/to/IB_Robot
source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm
```

如果希望在统一 launch 下**临时启用并自动开始监听**，可直接增加：

```bash
cd /path/to/IB_Robot
source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 launch robot_config robot.launch.py \
  robot_config:=so101_single_arm \
  voice_asr_auto_start:=true
```

这里的 `voice_asr_auto_start` 是 **launch 参数**，不是 YAML 字段。它会在启动时临时覆盖为：

- `voice_asr.enabled=true`

`active_mode` 默认已经是 `continuous`，因此启用后会自动开始监听。

## 3. 运行时结构

`VoiceASRNode` 本身更像一个编排节点，具体功能主要分发给内部模块：

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| `AudioCaptureModule` | `audio_capture_module.py` | 麦克风设备选择、缓冲、pre-roll、分块采集 |
| `FileInputModule` | `file_input_module.py` | 文件加载、解码、重采样、进度回调 |
| `VADModule` | `vad_module.py` | 语音活动检测与语音/静音分段 |
| `ASRInferenceModule` | `asr_inference_module.py` | sherpa-onnx 模型初始化与解码 |
| `StateMachine` | `state_machine.py` | 节点模式与状态切换 |
| `model_manager` | `model_manager.py` | 在配置模型缺失时解析/下载默认 ASR bundle |

整体数据流：

```text
麦克风或音频文件
  -> 音频归一化 / 缓冲
  -> VAD 分段
  -> sherpa-onnx 解码
  -> 中间 / 最终文本
  -> ROS topic / service 响应
```

## 4. 识别模式

节点内部通过 `StateMachine` 维护 `active_mode`，当前支持：

| 值 | 含义 |
| --- | --- |
| `manual` | 默认空闲，由 service 或 `/voice_control` 触发识别 |
| `continuous` | 节点启动后自动进入监听 |
| `wake_word` | 状态机预留值；当前节点里还没有独立的唤醒词流水线 |

关键行为约束：

- **麦克风实时识别必须使用流式模型。**
- **离线模型仍可用于 `~/recognize_file` 和 `/voice_file_input`。**
- 如果当前加载的是离线模型，而外部请求实时识别，节点会明确拒绝并记录错误，而不是崩溃。

## 5. 模型加载与自动下载

节点主要读取这些参数：

- `model_path`
- `tokens_path`
- `model_type`
- `language`
- `provider`
- `auto_download_model`

初始化流程如下：

1. `resolve_model_assets()` 先检查 `model_path` 是否为空或已存在。
2. 如果 `model_path` 为空，或配置的模型缺失，且 `auto_download_model=true`，
   节点会按当前意图选择默认 bundle。
3. 下载后的 bundle 路径会回填到节点实际使用的运行参数里。
4. `ASRInferenceModule.initialize()` 根据模型类型创建流式或离线 recognizer。

当前默认 bundle：

| Profile | Bundle | 用途 |
| --- | --- | --- |
| `streaming_zh` | `sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23` | 默认中文实时 ASR |
| `offline_zh` | `sherpa-onnx-paraformer-zh-int8-2025-10-07` | 默认中文离线文件识别 |

模型目录：

```text
models/voice_asr/
```

### 流式与离线模型的判定

运行时的区分方式是：

- 流式模型：目录中存在 `encoder*.onnx`、`decoder*.onnx`、`joiner*.onnx`
- 离线模型：通常是 paraformer 这种单模型 ONNX 布局，例如 `model.int8.onnx`

## 6. 实时麦克风识别流程

实时识别由控制循环定时器和 `_process_audio()` 驱动：

1. 从 `AudioCaptureModule` 读取一个音频块。
2. 调用 `VADModule.process()` 判断当前音频状态。
3. 检测到开始讲话后，创建一个流式 ASR 会话。
4. 先补喂一小段 pre-roll，避免句首被截断；默认通过 `realtime_pre_roll_seconds=0.5` 保留实时缓存，实际一次性喂给流式 ASR 的音频会被限制在最近 0.5 秒内，避免启动识别时阻塞控制循环。
5. 在语音活动期间持续向 ASR 喂入音频块。
6. 如果 `publish_partial=true`，就发布中间结果。
7. 在静音或超时后结束识别，并发布最终结果。

实时链路的几个细节：

- VAD 进入 `STARTING`、`SPEAKING` 或 `ENDING` 都会被视为语音活动并喂给 ASR，避免截断句首或句尾。
- `realtime_pre_roll_seconds` 会保留 VAD 判定前的实时音频，减少句首丢失；当前帧会从 pre-roll 中裁掉，避免重复喂入。为保证实时性，流式 ASR 启动时最多一次性补喂最近 0.5 秒。
- 如果检测到讲话时当前模型是离线模型，节点会停止采集并记录明确错误。

## 7. 文件识别流程

即使麦克风实时识别不可用，文件识别仍然可以工作。

当前有两个入口：

| 入口 | 类型 | 行为 |
| --- | --- | --- |
| `~/recognize_file` | Service | 同步请求 / 响应 |
| `/voice_file_input` | Topic | 异步后台线程处理 |

### `ibrobot_msgs/srv/RecognizeFile`

**请求字段**

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `file_path` | `string` | 待识别文件路径 |
| `enable_vad` | `bool` | 是否先做 VAD 分段 |

**响应字段**

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `success` | `bool` | 是否识别成功 |
| `error_message` | `string` | 失败原因 |
| `results` | `string[]` | 每段识别文本 |
| `timestamps` | `float32[]` | 每段起始时间 |
| `durations` | `float32[]` | 每段时长 |

## 8. ROS 接口

### 发布的话题

| 话题 | 类型 | 含义 |
| --- | --- | --- |
| `output_topic`（默认 `/voice_command`） | `std_msgs/String` | 最终识别文本 |
| `/voice_partial` | `std_msgs/String` | 中间识别结果 |
| `/voice_status` | `std_msgs/String` | 当前节点状态 |
| `/voice_confidence` | `std_msgs/Float32` | 最终结果置信度 |
| `/voice_file_progress` | `std_msgs/Float32` | 文件处理进度 |

### 订阅的话题

| 话题 | 类型 | 含义 |
| --- | --- | --- |
| `/voice_control` | `std_msgs/String` | 通过文本命令控制开始/停止识别 |
| `/voice_file_input` | `std_msgs/String` | 提交待异步识别的文件路径 |

当前可识别的 `/voice_control` 命令包括：

- `start`
- `开始`
- `开始监听`
- `stop`
- `停止`
- `停止监听`

### 服务

| 服务 | 类型 | 含义 |
| --- | --- | --- |
| `~/start_recognition` | `std_srvs/srv/Empty` | 开始一次实时监听 |
| `~/stop_recognition` | `std_srvs/srv/Empty` | 停止当前实时监听 |
| `~/set_hotwords` | `ibrobot_msgs/srv/SetHotwords` | 更新热词增强配置 |
| `~/recognize_file` | `ibrobot_msgs/srv/RecognizeFile` | 识别一个音频文件 |

### `ibrobot_msgs/srv/SetHotwords`

**请求字段**

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `hotwords` | `string[]` | 需要增强的热词 |
| `boost_scores` | `float32[]` | 每个热词对应的增强分数 |

**响应字段**

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `success` | `bool` | 是否设置成功 |
| `error_message` | `string` | 失败原因 |

## 9. 参数说明

### ASR 行为参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `active_mode` | `continuous` | 节点激活模式 |
| `language` | `zh` | 传给 ASR 初始化的语言提示 |
| `model_path` | `""` | 模型文件或目录路径；具体机器人配置可在 `robot_config` YAML 中覆盖 |
| `tokens_path` | `""` | 可选的显式 tokens 路径 |
| `provider` | `cpu` | sherpa-onnx 推理 provider |
| `model_type` | `auto` | `auto`、`streaming` 或 `offline` |
| `auto_download_model` | `true` | 配置模型缺失时是否自动下载默认 bundle |
| `max_recording_duration` | `10.0` | 实时识别最长录音时长，超时后强制收尾 |
| `publish_partial` | `true` | 是否发布中间解码结果 |
| `output_topic` | `/voice_command` | 最终命令输出 topic |
| `exit_on_init_failure` | `true` | 初始化失败时是否直接抛错退出 |

### 音频 / VAD 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `vad_sensitivity` | `0.6` | VAD 灵敏度 |
| `realtime_pre_roll_seconds` | `0.5` | 识别启动时补回的实时缓存时长，用于减少句首丢字 |
| `sample_rate` | `16000` | 运行时音频采样率 |
| `chunk_size` | `512` | 每个音频块的帧数 |
| `buffer_seconds` | `5.0` | 音频环形缓冲区时长 |
| `device_index` | `-1` | 显式音频设备索引；`-1` 表示默认设备 |
| `device_name` | `""` | 优先按设备名匹配，失败后回退到索引 |

## 10. 状态机

节点状态包括：

| 状态 | 含义 |
| --- | --- |
| `idle` | 空闲，等待触发 |
| `listening` | 正在监听并等待语音开始 |
| `recognizing` | 已检测到语音，ASR 流正在运行 |
| `hold` | 预留中间状态 |
| `error` | 运行时错误状态 |

典型的实时路径如下：

```text
idle -> listening -> recognizing -> listening -> idle
```

节点会把状态变化发布到 `/voice_status`。

## 11. 失败处理

节点已经对以下常见失败情况做了显式保护：

- `model_path` 缺失
- ASR 初始化失败
- 使用离线模型请求实时识别
- 文件解码失败
- 初始化失败后继续收到识别请求

需要注意：

- `VoiceASRNode initialized` **并不代表** ASR 已经可用。
- 真正的成功信号通常是后续日志里的 `ASR model loaded: ...`。
- 如果 `exit_on_init_failure=true`，初始化失败会直接导致启动失败。
- 如果 `exit_on_init_failure=false`，节点会继续存活，但在 ASR 初始化成功之前会拒绝相关请求。

## 12. 推荐配置方式

机器人级别的 SSOT 位于：

```text
src/robot_config/config/robots/so101_single_arm.yaml
```

典型的 ASR 配置片段如下：

```yaml
robot:
  voice_asr:
    enabled: false
    active_mode: continuous
    language: zh
    model_path: models/voice_asr/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23
    tokens_path: ""
    provider: cpu
    model_type: streaming
    auto_download_model: true
    max_recording_duration: 10.0
    vad_sensitivity: 0.6
    realtime_pre_roll_seconds: 0.5
    publish_partial: true
    output_topic: /voice_command
    sample_rate: 16000
    chunk_size: 512
    buffer_seconds: 5.0
    device_index: -1
    device_name: ""
    exit_on_init_failure: true
```

默认建议把 `enabled` 保持为 `false`，只在需要时通过 `voice_asr_auto_start:=true` 临时启用；如果你的机器人就是要长期带语音入口，再把 YAML 改成 `enabled: true` 即可。

如果只想做离线文件识别，可以切换到离线 bundle，并继续使用 `~/recognize_file`。

## 13. 排障

| 现象 | 常见原因 | 检查点 |
| --- | --- | --- |
| 节点能启动，但实时识别始终不可用 | 加载的是离线模型 | 查看日志里是否出现 `Offline ASR model loaded` |
| `start_recognition` 被拒绝 | ASR 未就绪，或当前模型是离线模型 | 查看 `_asr_init_error` 相关日志和模型类型 |
| 文件识别立即失败 | 文件路径错误或解码失败 | 确认文件存在且格式受支持 |
| 麦克风没有音频输入 | 设备选择不对 | 检查启动时的设备日志，使用 `device_name` 或 `device_index` 指定 |
| 模型路径缺失 | bundle 尚未下载完成 | 开启 `auto_download_model` 或在 setup 阶段预拉取模型 |

## 14. 当前已验证行为

当前实现已经验证过以下能力：

- 流式模型初始化
- 离线模型下的实时识别保护逻辑
- 配置模型缺失时的自动解析与下载
- 使用自带 streaming 样例音频进行真实解码
- 保持离线文件识别可用

这意味着节点当前支持的预期分工是：

- **流式模型负责麦克风实时识别**
- **离线或流式模型都可以用于文件识别**
