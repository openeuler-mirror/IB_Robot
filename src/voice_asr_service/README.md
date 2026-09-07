# VoiceASRNode 节点说明

`voice_asr_node.py` 是 `voice_asr_service` 包中的运行时 ROS 2 语音识别节点。
它负责把 ROS 音频或音频文件转换成文本，并统一管理音频缓冲、VAD、sherpa-onnx 模型加载、ROS 接口以及内部状态机。

这份 README 同时说明两个独立节点：本文第 1～14 节描述 `VoiceASRNode`；下方“Speech Direction 独立节点”一节描述 `SpeechDirectionNode`。两者共享由 `robot_config` 直接编排的 `audio_common` ROS 音频链路。

## Speech Direction 独立节点

`speech_direction_node` 属于 `voice_asr_service`，与上文所述的 `VoiceASRNode` 是两个独立节点。它负责多通道人声音频增强、语音门控和方向估计，不执行 ASR，也不控制底盘。

阵列、模型和运行时效参数由以下包内配置管理；音频设备和 Topic 由
`robot_config.audio_io` 统一管理：

```text
src/voice_asr_service/config/speech_direction.yaml
```

可独立启动方向节点：

```bash
ros2 launch voice_asr_service speech_direction.launch.py
```

该 launch 默认读取上述 YAML；也可通过 `config_file` 指向同结构的配置文件。模型相对路径在 launch 边界相对 `models_root`（默认 `<标准工作区>/models`）解析，绝对路径保持原样。自定义 colcon install base 时应显式指定模型根，例如：

```bash
ros2 launch voice_asr_service speech_direction.launch.py models_root:=/path/to/models
```

YAML 的模型相对路径以 models 目录为根，不再包含额外的 `models/` 前缀。`speech_direction` 不属于 `robot_config` 的机器人级配置，也不由 `sound_follow` 管理。

### 平台 profile

公共算法、阵列、门控、时序和诊断参数只维护在 `speech_direction.yaml`，平台差异由独立的 `speech_direction_profiles.yaml` 覆盖，launch 边界通过 `profile` 参数选择：

| profile | 默认 | 适用平台 | 后端组合 |
| --- | --- | --- | --- |
| `ascend_310p` | 是 | Atlas 310P 推理盒 | Silero `ascend` + FullSubNet `ascend` |
| `ubuntu_cuda` | 否 | x86 + NVIDIA CUDA | Silero `onnx` + FullSubNet `stateful_torch_cuda` |
| `custom` | 否 | 自定义 | 用户自行填写，仍受半组合校验约束 |

```bash
ros2 launch voice_asr_service speech_direction.launch.py profile:=ubuntu_cuda
```

profile 只允许覆盖 `silero_vad_backend`、`silero_vad_deployment`、`fullsubnet_backend`、`fullsubnet_deployment` 四个平台字段；写入公共算法字段会被 launch 拒绝。`ascend_310p` 与 `ubuntu_cuda` 还会被校验后端组合一致性，launch 还会用 manifest 校验 deployment 的 backend 与声明的平台后端一致，避免后端与 deployment 错配的半切换配置。仅 `ascend` 后端会在节点环境注入 CANN 库路径，`ubuntu_cuda` 不注入无关环境。

### 模型资产下载

Ubuntu 依赖（Silero ONNX、FullSubNet cumulative checkpoint）通过 `./scripts/download_speech_direction_models.sh` 显式预取，分别落入 `models/silero-vad/assets/` 与 `models/fullsubnet/assets/`；310P OM 资产需从 NAS/HuggingFace 手动获取后放入对应 bundle 的 `artifacts/ascend/` 目录。执行 `python3 scripts/verify_speech_direction_assets.py` 会遍历 `models/silero-vad` 与 `models/fullsubnet` 两个独立 bundle 的全部 deployment（`ascend_310p`/`torch_cpu`/`torch_cuda`），先用 `load_inference_manifest_metadata` 校验 bundle 结构与 bindings，再逐资产校验文件存在性与 SHA-256。脚本只校验不下载；缺失的资产会打印来源提示并跳过，已存在的资产校验不通过则报错。FullSubNet 两平台共用同一 cumulative 218epochs checkpoint 权重：310P 预导出为 FB/SB 拆分 OM，Ubuntu 由 Torch 直接加载同一 checkpoint。

### 配置所有权

| 类别 | YAML 参数 | 当前配置 / 含义 |
| --- | --- | --- |
| 麦克风与音频 | `audio_topic` | `/audio/capture_stamped`，由 `audio_capture_node` 发布的多通道 PCM |
| 麦克风与音频 | `audio_channels` | `6`，由 `robot_config` 的 microphone peripheral 覆盖 |
| 麦克风与音频 | `sample_rate` | `16000` Hz；当前 speech-direction 完整算法链仅支持此采样率，其他值会在参数校验阶段被拒绝 |
| 麦克风与音频 | `channel_indices` | `[1, 2, 3, 4]`，参与处理的输入通道 |
| 阵列 | `mount_yaw_deg` | `0.0`，阵列安装偏角（度），逆时针为正。把阵列坐标系角度对齐到小车坐标系，详见下方[坐标系与安装偏角](#坐标系与安装偏角) |
| 阵列 | `angle_step_degree` | `5`，SRP-PHAT 扫描角度步长（度），DOA 输出只能为该步长的整数倍；必须为 360 的正整数约数。详见下方[SRP 角度精度](#srp-角度精度) |
| 阵列 | `mic_positions` | 四麦二维坐标的一维展开数组，长度必须为通道数的两倍 |
| 模型 | `speech_direction_inference_bundle` | FullSubNet 独立 bundle 目录（`models/fullsubnet`），相对 `models/` |
| 模型 | `fullsubnet_deployment` | FullSubNet deployment 名；`ascend_310p`（310P 拆分 OM）或 `torch_cpu`/`torch_cuda`（Torch checkpoint） |
| 模型 | `fullsubnet_backend` | `ascend`（310P 拆分 OM）；Ubuntu profile 覆盖为 `stateful_torch_cuda`，必须与 deployment 的 manifest backend 一致 |
| 模型 | `silero_vad_inference_bundle` | Silero VAD 独立 bundle 目录（`models/silero-vad`），相对 `models/` |
| 模型 | `silero_vad_deployment` | Silero deployment 名；`ascend_310p`（OM）或 `torch_cpu`（ONNX） |
| 模型 | `silero_vad_backend` | `ascend`（310P OM）；Ubuntu profile 覆盖为 `onnx`，必须与 deployment 的 manifest backend 一致 |
| 运行时效 | `speech_direction_max_age_ms` | `1300` ms，方向结果最大保鲜时间 |

这些参数均由 `voice_asr_service` 独占管理。`sound_follow` 只维护底盘最小集和跟随行为参数，其完整 launch 通过无参数 include 复用本包的 `speech_direction.launch.py`，不读取或转发任何音频参数。


### 高通量离线维测

`speech_direction.yaml` 还包含以下 9 个高通量维测参数：

| YAML 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `diagnostics_high_throughput_enabled` | `false` | 总开关；默认关闭，现场诊断/测试时显式开启 |
| `diagnostics_rollover_seconds` | `300` | raw、enhanced 和 metrics 的固定分卷时长（秒） |
| `diagnostics_save_raw6ch` | `true` | 总开关开启时保存完整原始 6 通道 PCM WAV；总开关关闭时不写盘 |
| `diagnostics_save_enh4ch` | `true` | 总开关开启时保存 FullSubNet 输出的 4 通道 PCM WAV |
| `diagnostics_save_frame_metrics` | `true` | 总开关开启时保存逐 hop 的 VAD、RMS、DOA 和耗时 JSONL |
| `diagnostics_save_gray_events` | `true` | 总开关开启时保存灰区事件摘要；运行时不截取灰区 WAV |
| `diagnostics_queue_size` | `128` | 后台 writer 有界队列容量 |
| `diagnostics_drop_when_full` | `true` | 接口兼容项；实时链路始终非阻塞，队列满时丢弃并计数 |
| `fullsubnet_timing_enabled` | `false` | FullSubNet STFT/FB/SB/postprocess 分阶段计时；仅性能分析时开启，生产默认关闭 |

默认关闭高通量维测（总开关 `false`），避免部署即写盘到磁盘满。需要现场诊断时，
将 `diagnostics_high_throughput_enabled` 改为 `true`；四个 `save_*` 子开关在总开关
开启时表示默认保存哪些流，可按定位需求独立关闭。启用后，每次启动会创建独立会话：

```text
~/.ros/speech_direction/runs/run_<YYYYmmdd-HHMMSS>/
├── manifest.json
├── audio/full/raw6ch_*.wav
├── audio/full/enh4ch_*.wav
├── metrics/frames_*.jsonl
└── events/gray_events.jsonl
```

节点停止后 `manifest.json` 才进入 `completed` 终态。writer 写盘失败只会永久停用本次
维测旁路，方向 pipeline 继续运行，并在基础 `/diagnostics` 中报告 `WARN`；不会在线生成
报告，也不会加载 Plotly 或任何绘图 fallback。

离线报告依赖 Plotly 与 Matplotlib，仅 `speech_direction_report` CLI 需要，`speech_direction_node`
运行时不加载。首次使用前通过可选开关安装（不会污染板端/CI 的 base 安装链）：

```bash
./scripts/setup.sh --with-diagnostics      # 新装工作区时一并带上
# 或:
python3 -m pip install -r requirements/diagnostics.txt
```

停止节点后可离线生成报告：

```bash
source .shrc_local && speech_direction_report \
  ~/.ros/speech_direction/runs/run_<YYYYmmdd-HHMMSS>
```

默认严格同时生成 `reports/doa_curves.html` 和 `reports/doa_curves.png`，不会生成灰区音频。
需要导出灰区时显式增加 `--extract-gray-audio`；事件跨分卷时会按统一 sample 轴拼接
`raw6ch` 和 `enh4ch`：

```bash
source .shrc_local && speech_direction_report \
  ~/.ros/speech_direction/runs/run_<YYYYmmdd-HHMMSS> \
  --extract-gray-audio
```

输出已存在时命令默认拒绝覆盖；确认重跑可加 `--overwrite`。HTML 与 PNG 是严格双依赖：
必须同时安装 Plotly 和 Matplotlib，缺任一项即失败，不提供降级或 fallback。同一会话不要
并发生成报告。输出提交边界只是先生成临时 HTML/PNG，再按 HTML、PNG 顺序分别替换；
它不是事务锁或 journal，第二次替换失败时可能留下半套新输出，请检查后带
`--overwrite` 顺序重跑。

### 链路分叉：stateful 生产链路与显式对照链路

本 PR 同时保留两套门控 + 两套 pipeline，`node.py` 按后端是否使用 stateful 执行器选择哪一套，二者不静默回退：

| 链路 | 后端 | 门控 | pipeline | 时序参数 |
| --- | --- | --- | --- | --- |
| **生产链路** | `ascend` / `stateful_torch_*` | `TemporalSpeechGate`（状态机帧级） | `StreamingSpeechDirectionPipeline` | tick=256、model_batch=512、SRP frame=4096/hop=512 |
| 显式对照 | `torch` | `SpeechGate`（Top-2 hop 级） | `SpeechDirectionPipeline` | hop=2048、enh_block=8192 |

- 生产部署用 `ascend` 或 `stateful_torch_*`，走 `StreamingSpeechDirectionPipeline` + `TemporalSpeechGate`；`torch` 仅作显式对照保留，启动时按后端名严格选择，**不会从 stateful 静默回退到对照链路**，后端与路径错配会在 `_build_and_start` 抛 `ValueError`。
- 两套 pipeline 都消费同一组 `VadState` / `DoaState`，但调用节奏不同：streaming 每 256 样本 tick 一次 `vad_state.update`（两个 tick 合并为一次 T=2 模型推理），legacy 每 2048 样本 hop 一次。两者都向 `DoaState.update` 写段级 DOA，`meta.type` 取值相同（`mid_long_seg` / `seg_end`），但 `mid_long_seg` 的触发条件不同——streaming 按 `max_accum_samples`（样本计数），legacy 按 `max_accum_dur_s`（墙钟时长）。`node._poll_and_publish` 按 `result["type"]` 区分段末与中间方向，上游消费者无需感知链路分叉。
- `SpeechGate` 的 Top-2 选择对 hop_size 敏感（旧值 2048 vs 新值 512 会改变 Top-2 候选），故 legacy 链路固定用 2048 hop，不沿用 stateful 的 512；两套链路的时序参数各自独立，不共享。

### 发布契约、坐标与故障语义

- 方向发布到 `/voice/speech_direction`，消息类型为 `ibrobot_msgs/msg/SpeechDirection`，QoS 为 `RELIABLE + KEEP_LAST(1)`。
- `header.frame_id` 为 `base_link`；`azimuth_rad` 遵循 REP-103：`0` 为前、`+π/2` 为左、`-π/2` 为右，左转为正。
- `header.stamp` 按方向类型分流构造，承载方向的"真年龄"信息，而非固定取发布时刻：
  - 段末方向（`seg_end`）：`stamp = 发布时刻的 ROS 时钟 − age`，还原段结束时刻，使消费者按 `now − stamp` 判过期时得到真实年龄，executor 积压/DDS 延迟不会被盖掉；
  - 中间方向（`mid_long_seg`）：`stamp = 发布时刻的 ROS 时钟`，`age≈0`，符合"立即响应正在说话"的低延迟设计，由 QoS `KEEP_LAST(1)` 与 `seq_id` 去重兜底，不额外设过期上限。
  - 方向的 `age` 在 `runtime` 内部用墙钟（`time.time()`）计算；`stamp` 在 `node` 用 ROS 时钟（`get_clock().now()`）构造。**当前部署 `use_sim_time=false`，ROS 时钟与墙钟同属系统时钟域，二者差值有效**，上述分流在实车上正确。`use_sim_time=true`（仿真/Bag 回放）下 ROS 时钟与墙钟不同步，本 PR 不解决该混合时钟域；如需在仿真或回放场景消费方向，应使用 `use_sim_time=false`，或后续单独统一为单一时钟域。
- 长语音累计达到 `max_accum_dur_s` 时发布一次中间方向并清理本轮累积，避免持续讲话时等待整段结束才响应；语音段结束时再发布当前累积窗口的段末方向。两类输出都是有效方向事件。
- 每次中间方向或段末方向都有独立递增的 `seq_id`，消费者按输出事件去重；`seq_id` 不表示“一段语音只对应一个序号”。
- 节点只在取得新的有效方向事件时发布；无人声、结果过期或降级时不发布方向。
- 参数缺失、参数非法或配置的模型资产不存在时，节点启动失败。其中模型资产缺失会提示运行 `python3 scripts/verify_speech_direction_assets.py` 校验资产清单（脚本只校验不下载，资产需从 NAS 手动获取）。
- 模型资产已存在但模型加载、音频设备打开或运行时推理失败时，节点保持运行、不发布方向，并通过 `/diagnostics` 持续报告降级状态。

#### 坐标系与安装偏角

阵列坐标系（SRP-PHAT 算法内部约定）：`0°=右(+x)`，`90°=前(+y)`，`180°=左`，`270°=后`，逆时针为正。发布到 `/voice/speech_direction` 的 `azimuth_rad` 遵循 REP-103：`0=前`，`+π/2=左`，`-π/2=右`，左转为正。

`mount_yaw_deg` 就是把阵列坐标系对齐到小车坐标系的安装偏角（度），转换式为：

```
ros_azimuth = radians(阵列角度) - π/2 + radians(mount_yaw_deg)
```

- `mount_yaw_deg=0`：阵列正前方（90°）对齐小车正前方（0 rad）。
- 阵列相对小车正前方**逆时针**偏 α°：`mount_yaw_deg=α`（偏角为正）。
- 阵列相对小车正前方**顺时针**偏 α°：`mount_yaw_deg=-α`（偏角为负）。

按阵列实际安装姿态填入该夹角，输出的 `azimuth_rad` 即在车体坐标系下。默认 `0.0`，通过 `speech_direction.yaml` 配置。

**当前默认情况**：默认值 `0.0` 表示假设阵列正前方（90°）对齐小车正前方、无安装偏角——这是默认假设，**并非物理标定结果**。离线回归基线（`test/speech_direction/audio/` 下 6 文件 12 段，均 ≤15° 通过）即在此假设下录制与校验，因此基线对偏角敏感的标定数据不反映实车安装姿态。实车上若阵列安装存在夹角，必须按实际测量据实填入，否则输出方位会带一个等于该夹角的系统偏差；保持默认 `0.0` 等价于"忽略安装夹角，按无偏角处理"。

#### SRP 角度精度

`angle_step_degree` 是 SRP-PHAT 扫描角度步长（度），节点启动时生成 `0, step, 2·step, …, 360-step` 的候选角度序列并预计算导向相位矩阵；最终 DOA 只能为该步长的整数倍。必须为 360 的正整数约数（如 `1、2、3、4、5、6、8、9、10`），否则候选角度无法均匀覆盖整圈，节点启动时校验失败。

默认 `5`（对应回归基线：5 文件 8 段 max err ~15°）。

**关于调成更小步长（如 1°）**：程序能正常计算并按 1° 步长输出角度值，**但 4 麦小阵列的物理角度分辨能力有限**——空间谱主瓣较宽，相邻多个角度的 score 接近，`argmax` 在主瓣顶部易被噪声和相位误差在相邻角度间随机推动。步长小于阵列物理分辨能力后，输出精度不再随之提升，仅增加计算量（候选角度数线性增长，导向相位矩阵与 einsum 投票代价随之上升）与噪声敏感度。

真正限制角度精度的是阵列孔径与频段（参见 `doa/srp_phat.py` 的几何与频段参数），而非扫描步长。要追求更高角度精度，需换更大孔径或更多麦克风的阵列，不是单靠调小 `angle_step_degree`。默认 `5` 是与当前基线匹配的务实值。

### 实时音频采集

Ubuntu 和 openEuler 均由 `robot_config` 启动唯一的 `audio_capture_node`。Voice ASR 和
Speech Direction 固定订阅 `/audio/capture_stamped`，不直接打开 ALSA，因此可以同时使用
同一个 ReSpeaker。音频设备参数仅在 `robot_config` 的 microphone peripheral 中配置，
生产运行时不提供其他设备后端。

离线回归测试可直接向 runtime 注入 PCM/WAV 数据；这是测试入口，不是生产平台 fallback。

## 1. VoiceASRNode 节点职责

`VoiceASRNode` 支持两类输入路径：

| 输入路径 | 作用 | 模型要求 |
| --- | --- | --- |
| 麦克风实时识别 | 从音频输入设备持续监听并输出识别文本 | **必须使用流式模型** |
| 音频文件识别 | 解码文件并返回/发布识别结果 | 可使用流式或离线模型 |

核心职责包括：

1. 读取 ROS 参数并初始化各个运行模块。
2. 从选定的 schema-v3 bundle deployment 解析 ASR 与 Silero VAD 模型资产（不做启动期下载或目录名推断）。
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
  -p bundle_path:=models/voice_asr/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23 \
  -p deployment:=torch_cpu
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
| `ASRInferenceModule` | `asr_inference_module.py` | 从 bundle deployment 加载 sherpa-onnx recognizer 并解码 |
| `ManifestVadRuntime` | `vad_runtime.py` | Silero VAD 的 manifest-backed 统一 runtime 会话 |
| `StateMachine` | `state_machine.py` | 节点模式与状态切换 |

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

## 5. 模型 bundle 与 deployment

节点主要读取这些参数：

- `bundle_path`（ASR bundle 目录）
- `deployment`（bundle manifest 中的命名 deployment）
- `vad_bundle_path`（Silero VAD 独立 bundle 目录）
- `vad_deployment`（Silero deployment 名）
- `language`

初始化流程如下：

1. `ASRInferenceModule.initialize()` 用 `load_inference_manifest()` 校验 bundle，
   并按 deployment 声明的 artifact 角色创建 recognizer：
   - 流式 transducer：`encoder` + `decoder` + `joiner` + `tokens`
   - 流式 Paraformer：`encoder` + `decoder` + `tokens`
   - 离线：`model` + `tokens`
2. 采样率从 recognizer 配置读取，并与音频输入的 `sample_rate` 交叉校验。
3. `ManifestVadRuntime` 加载 Silero VAD 的 `torch_cpu`（ONNX Runtime）deployment，
   并校验其 `audio_contract`（采样率、帧长、mono float32）与节点音频输入一致。
4. 任何 manifest/artifact 缺失或契约不匹配都会 fail-closed：节点不会下载模型、
   不会按目录名推断模型类型，也不会静默切换到其他后端。

历史遗留的 `model_path`、`tokens_path`、`provider`、`model_type`、`auto_download_model`
字段已退役；在启用的 Voice ASR 配置里出现会被 `robot_config` 与 launch builder 直接拒绝。

### Bundle 制作与预取（离线/气隙环境）

ASR bundle 通过打包器显式生成，产物是 schema-v3 manifest + 声明式 artifacts：

```bash
# 流式 transducer（默认）
python3 -m voice_asr_service.package_sherpa_asr_bundle \
  --bundle-root models/voice_asr/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23 \
  --tokens <tokens.txt> --encoder <encoder.onnx> --decoder <decoder.onnx> --joiner <joiner.onnx>

# 流式 Paraformer：把 --joiner 换成 --paraformer
# 离线：--offline --model <model.onnx>
```

打包器会同时生成 `torch_cpu` 与 `torch_cuda` 两个 deployment（`--skip-cuda` 可只保留 CPU），
并对每个 deployment 执行 `load_inference_manifest()` 自校验。

Silero VAD 与 FullSubNet 的 Ubuntu 依赖由 `./scripts/download_speech_direction_models.sh`
下载并重新打包对应独立 bundle；部署完整性用 `python3 scripts/verify_speech_direction_assets.py`
校验。气隙环境可先在有网机器执行上述步骤，再把 `models/silero-vad`、`models/fullsubnet`
与 ASR bundle 目录整体拷贝到目标机器。

### 从旧版 raw 模型目录迁移

旧布局（`models/voice_asr/<bundle>/` 下平铺 onnx + tokens.txt，配置 `model_path` +
`auto_download_model`）迁移步骤：

1. 用 `package_sherpa_asr_bundle` 把现有 onnx/tokens 打成 schema-v3 bundle（见上）。
2. 把 robot YAML 的 `voice_asr` 段从 `model_path`/`tokens_path`/`provider`/`model_type`/
   `auto_download_model` 改为 `bundle_path` + `deployment`（流式实时识别用含
   `encoder`/`decoder` 的 deployment；离线文件识别用 `model` + `tokens` 的 deployment）。
3. 保留旧模型目录直到新 bundle 验证通过；回滚时只需把 YAML 恢复为旧字段前先确认
   目标部署仍包含旧布局——新版本节点**不再读取** raw 字段，回滚需要同时回滚
   `voice_asr_service` 包。

### 流式与离线模型的判定

运行时的区分方式是 deployment 声明的 artifact 角色：

- 流式 transducer：`encoder`、`decoder`、`joiner`、`tokens`
- 流式 Paraformer：`encoder`、`decoder`、`tokens`
- 离线：`model`、`tokens`

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
| `bundle_path` | `models/voice_asr` | schema-v3 ASR bundle 目录；具体机器人配置可在 `robot_config` YAML 中覆盖 |
| `deployment` | `torch_cpu` | bundle manifest 中的命名 deployment；模型类型/后端/artifact 均由 deployment 派生 |
| `vad_bundle_path` | `models/silero-vad` | Silero VAD 独立 bundle 目录 |
| `vad_deployment` | `torch_cpu` | Silero deployment 名（ONNX Runtime 后端） |
| `max_recording_duration` | `10.0` | 实时识别最长录音时长，超时后强制收尾 |
| `publish_partial` | `true` | 是否发布中间解码结果 |
| `output_topic` | `/voice_command` | 最终命令输出 topic |
| `exit_on_init_failure` | `true` | 初始化失败时是否直接抛错退出 |

### 音频 / VAD 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `vad_sensitivity` | `0.6` | VAD 灵敏度 |
| `realtime_pre_roll_seconds` | `0.5` | 识别启动时补回的实时缓存时长，用于减少句首丢字 |
| `sample_rate` | `16000` | 当前完整 Voice ASR 链路仅支持 16000 Hz；其他值属于无效配置，节点会拒绝初始化 |
| `chunk_size` | `512` | 当前完整 Voice ASR 链路仅支持 512 样本帧；其他值属于无效配置，节点会拒绝初始化 |
| `buffer_seconds` | `5.0` | 音频环形缓冲区时长 |
| `audio_topic` | `/audio/capture_stamped` | `audio_common_msgs/AudioDataStamped` 输入 Topic |
| `audio_channels` | `6` | ROS PCM 的交错通道数 |
| `audio_input_channel` | `1` | Voice ASR 从多通道 PCM 中选取的单通道索引 |

当前 16kHz/512 是实时麦克风、文件识别、VAD 后端和 ASR 模型共同遵守的系统级硬限制；
Silero ONNX 后端会在 512 样本音频帧前额外拼接 64 个内部 context 样本，该 context 由 VAD 内部跨帧维护，不应配置到 `chunk_size` 中。

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

- ASR bundle/deployment 缺失或校验失败
- ASR 初始化失败
- 使用离线模型请求实时识别
- 文件解码失败
- 初始化失败后继续收到识别请求

需要注意：

- `VoiceASRNode initialized` **并不代表** ASR 已经可用。
- 真正的成功信号通常是后续日志里的 `ASR deployment loaded: ...`。
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
    bundle_path: models/voice_asr/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23
    deployment: torch_cpu
    max_recording_duration: 10.0
    vad_sensitivity: 0.6
    vad_bundle_path: models/silero-vad
    vad_deployment: torch_cpu
    realtime_pre_roll_seconds: 0.5
    publish_partial: true
    output_topic: /voice_command
    sample_rate: 16000
    chunk_size: 512
    buffer_seconds: 5.0
    exit_on_init_failure: true

  audio_io:
    enabled: true
    microphone: respeaker
    capture_stamped_topic: /audio/capture_stamped

  peripherals:
    - type: microphone
      name: respeaker
      driver: alsa
      params:
        device: "hw:0,0"
        channels: 6
        sample_rate: 16000
        sample_format: S16LE
```

默认建议把 `enabled` 保持为 `false`，只在需要时通过 `voice_asr_auto_start:=true` 临时启用；如果你的机器人就是要长期带语音入口，再把 YAML 改成 `enabled: true` 即可。

如果只想做离线文件识别，可以切换到离线 bundle，并继续使用 `~/recognize_file`。

## 13. 排障

| 现象 | 常见原因 | 检查点 |
| --- | --- | --- |
| 节点能启动，但实时识别始终不可用 | 加载的是离线模型 | 查看日志里是否出现 `Offline ASR model loaded` |
| `start_recognition` 被拒绝 | ASR 未就绪，或当前模型是离线模型 | 查看 `_asr_init_error` 相关日志和模型类型 |
| 文件识别立即失败 | 文件路径错误或解码失败 | 确认文件存在且格式受支持 |
| 麦克风没有音频输入 | `audio_capture_node` 未就绪或 microphone peripheral 配置不对 | 检查 `/audio/capture_stamped` 及 `audio_io.microphone` 引用的 `device`/`channels` |
| 启动报 bundle/deployment 无效 | bundle 尚未打包或 deployment 名不匹配 | 用 `package_sherpa_asr_bundle` 生成 bundle，并确认 YAML 的 `deployment` 与 manifest 中的名字一致 |

## 14. 当前已验证行为

当前实现已经验证过以下能力：

- 流式模型初始化
- 离线模型下的实时识别保护逻辑
- 从 bundle deployment 解析 ASR 与 Silero VAD 资产并校验音频契约
- 使用自带 streaming 样例音频进行真实解码
- 保持离线文件识别可用

这意味着节点当前支持的预期分工是：

- **流式模型负责麦克风实时识别**
- **离线或流式模型都可以用于文件识别**
