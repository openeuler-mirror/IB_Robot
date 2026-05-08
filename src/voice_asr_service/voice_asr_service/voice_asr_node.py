#!/usr/bin/env python3
"""
VoiceASRNode - 语音识别 ROS2 节点

整合所有模块，提供完整的语音识别服务
支持麦克风实时采集和音频文件输入
"""

import threading
import time
from dataclasses import dataclass

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float32, String
from std_srvs.srv import Empty

from ibrobot_msgs.srv import RecognizeFile, SetHotwords

from .asr_inference_module import ASRInferenceModule
from .audio_capture_module import AudioCaptureModule, AudioConfig
from .defaults import VOICE_ASR_DEFAULTS
from .file_input_module import FileInputModule
from .model_manager import resolve_model_assets
from .state_machine import ActiveMode, NodeState, StateMachine
from .vad_module import VADConfig, VADModule, VADState

_AUDIO_STALL_TIMEOUT_SECONDS = 2.0
_AUDIO_RESTART_MIN_INTERVAL_SECONDS = 5.0
_STATUS_HEARTBEAT_INTERVAL_SECONDS = 1.0
_MAX_STREAMING_PREROLL_SECONDS = 0.5


@dataclass
class RecognizeFileRequest:
    file_path: str
    enable_vad: bool


@dataclass
class RecognizeFileResponse:
    success: bool
    error_message: str
    results: list
    timestamps: list
    durations: list


class VoiceASRNode(Node):
    """
    VoiceASRNode - 语音识别节点

    核心职责：
    - 音频采集管理
    - 实时语音识别
    - 指令发布与状态同步
    """

    def _transition_state(self, new_state: NodeState, trigger: str) -> bool:
        """执行状态转换并记录失败日志"""
        old_state = self._state_machine.state
        if self._state_machine.transition(new_state, trigger):
            return True

        self.get_logger().error(
            f"Invalid state transition: {old_state.value} -> {new_state.value} (trigger: {trigger})"
        )
        return False

    def _asr_not_ready_message(self) -> str:
        """Build a stable error message for ASR calls made before init succeeded."""
        if self._asr_init_error:
            return f"ASR is not ready: {self._asr_init_error}"
        return f"ASR is not ready (state={self._asr.state.value})"

    def _fail_response(self, response, message: str):
        """Populate common failure fields for ROS service responses."""
        if hasattr(response, "success"):
            response.success = False
        if hasattr(response, "error_message"):
            response.error_message = message
        if hasattr(response, "results"):
            response.results = []
        if hasattr(response, "timestamps"):
            response.timestamps = []
        if hasattr(response, "durations"):
            response.durations = []
        return response

    def _ensure_asr_ready(self, response=None, operation: str = "ASR request") -> bool:
        """Guard ASR entrypoints so failed initialization cannot crash the node."""
        if self._asr.is_ready:
            return True

        message = f"{operation} rejected. {self._asr_not_ready_message()}"
        self.get_logger().error(message)
        if response is not None:
            self._fail_response(response, message)
        return False

    def _ensure_realtime_asr_supported(self, response=None, operation: str = "Realtime ASR request") -> bool:
        """Realtime microphone recognition requires a streaming-capable model."""
        if not self._ensure_asr_ready(response=response, operation=operation):
            return False
        if self._asr.is_streaming():
            return True

        message = (
            f"{operation} requires a streaming ASR model. "
            "Current model is offline, so microphone realtime recognition is unavailable. "
            "Offline file recognition remains available via ~/recognize_file."
        )
        self.get_logger().error(message)
        if response is not None:
            self._fail_response(response, message)
        return False

    def _stop_realtime_capture(self, trigger: str, reason: str) -> None:
        """Stop microphone capture when realtime ASR cannot proceed safely."""
        self.get_logger().error(reason)
        self._audio_capture.stop_capture()
        self._recording_start_time = None
        self._last_partial_text = ""
        self._transition_state(NodeState.IDLE, trigger)

    def __init__(self):
        super().__init__("voice_asr_node")

        self._declare_parameters()
        self._init_modules()
        self._init_ros_interface()
        self._init_control_loop()

        self.get_logger().info("VoiceASRNode initialized")

    def _declare_parameters(self):
        """声明 ROS 参数"""
        self.declare_parameter("active_mode", VOICE_ASR_DEFAULTS["active_mode"])
        self.declare_parameter("language", VOICE_ASR_DEFAULTS["language"])
        self.declare_parameter("model_path", VOICE_ASR_DEFAULTS["model_path"])
        self.declare_parameter("tokens_path", VOICE_ASR_DEFAULTS["tokens_path"])
        self.declare_parameter("provider", VOICE_ASR_DEFAULTS["provider"])
        self.declare_parameter("model_type", VOICE_ASR_DEFAULTS["model_type"])
        self.declare_parameter("auto_download_model", VOICE_ASR_DEFAULTS["auto_download_model"])
        self.declare_parameter("max_recording_duration", VOICE_ASR_DEFAULTS["max_recording_duration"])
        self.declare_parameter("vad_sensitivity", VOICE_ASR_DEFAULTS["vad_sensitivity"])
        self.declare_parameter("publish_partial", VOICE_ASR_DEFAULTS["publish_partial"])
        self.declare_parameter("output_topic", VOICE_ASR_DEFAULTS["output_topic"])
        self.declare_parameter("sample_rate", VOICE_ASR_DEFAULTS["sample_rate"])
        self.declare_parameter("chunk_size", VOICE_ASR_DEFAULTS["chunk_size"])
        self.declare_parameter("buffer_seconds", VOICE_ASR_DEFAULTS["buffer_seconds"])
        self.declare_parameter("realtime_pre_roll_seconds", VOICE_ASR_DEFAULTS["realtime_pre_roll_seconds"])
        self.declare_parameter("device_index", VOICE_ASR_DEFAULTS["device_index"])
        self.declare_parameter("device_name", VOICE_ASR_DEFAULTS["device_name"])
        self.declare_parameter("exit_on_init_failure", VOICE_ASR_DEFAULTS["exit_on_init_failure"])

        self._active_mode = self.get_parameter("active_mode").value
        self._language = self.get_parameter("language").value
        self._model_path = self.get_parameter("model_path").value
        self._tokens_path = self.get_parameter("tokens_path").value
        self._provider = self.get_parameter("provider").value
        self._model_type = self.get_parameter("model_type").value
        self._auto_download_model = self.get_parameter("auto_download_model").value
        self._max_recording_duration = self.get_parameter("max_recording_duration").value
        self._vad_sensitivity = self.get_parameter("vad_sensitivity").value
        self._publish_partial = self.get_parameter("publish_partial").value
        self._output_topic = self.get_parameter("output_topic").value
        self._sample_rate = self.get_parameter("sample_rate").value
        self._chunk_size = self.get_parameter("chunk_size").value
        self._buffer_seconds = self.get_parameter("buffer_seconds").value
        self._realtime_pre_roll_seconds = max(0.0, float(self.get_parameter("realtime_pre_roll_seconds").value))
        device_index = self.get_parameter("device_index").value
        self._device_index = device_index if device_index >= 0 else None
        self._device_name = self.get_parameter("device_name").value
        self._exit_on_init_failure = self.get_parameter("exit_on_init_failure").value

    def _init_modules(self):
        """初始化各模块"""
        self._state_machine = StateMachine()
        self._state_machine.set_mode_str(self._active_mode)

        audio_config = AudioConfig(
            sample_rate=self._sample_rate, chunk_size=self._chunk_size, buffer_seconds=self._buffer_seconds
        )
        self._audio_capture = AudioCaptureModule(audio_config)
        self._audio_capture.set_error_callback(self._on_audio_error)

        # 打印可用音频设备
        devices = AudioCaptureModule.list_audio_devices()
        if devices:
            self.get_logger().info("Available audio input devices:")
            for dev in devices:
                self.get_logger().info(
                    f"  [{dev['index']}] {dev['name']} "
                    f"(channels={dev['channels']}, sr={dev['sample_rate']}, backend={dev['backend']})"
                )
        else:
            self.get_logger().warn("No audio input devices found")

        # 优先按名称匹配，否则按 index
        if self._device_name:
            if self._audio_capture.set_device_by_name(self._device_name):
                self.get_logger().info(
                    f"Audio device selected by name '{self._device_name}' -> index {self._audio_capture._device_index}"
                )
            else:
                self.get_logger().error(
                    f"Failed to find audio device matching '{self._device_name}'. "
                    f"Falling back to device_index={self._device_index or 'default'}"
                )
                self._audio_capture.set_device(self._device_index)
        else:
            self._audio_capture.set_device(self._device_index)

        self._file_input = FileInputModule()
        self._file_input.set_progress_callback(self._on_file_progress)

        vad_config = VADConfig(sample_rate=self._sample_rate, frame_size=self._chunk_size)
        self._vad = VADModule(vad_config)
        self._vad.set_logger(self.get_logger())
        self._vad.set_sensitivity(self._vad_sensitivity)

        self._asr = ASRInferenceModule()
        self._asr_init_error: str | None = None

        try:
            resolved_assets = resolve_model_assets(
                model_path=self._model_path,
                tokens_path=self._tokens_path,
                model_type=self._model_type,
                active_mode=self._active_mode,
                language=self._language,
                auto_download_model=self._auto_download_model,
                logger=self.get_logger(),
            )
            self._model_path = resolved_assets.model_path
            self._tokens_path = resolved_assets.tokens_path

            if not self._model_path:
                raise ValueError(
                    "ASR model_path is empty. Configure robot.voice_asr.model_path, "
                    "pass the model_path parameter, or enable auto_download_model."
                )

            self._asr.initialize(
                model_path=self._model_path,
                tokens_path=self._tokens_path if self._tokens_path else None,
                provider=self._provider,
                language=self._language,
                model_type=self._model_type,
            )
            self._vad.initialize()
            model_type_str = "streaming" if self._asr.is_streaming() else "offline"
            self.get_logger().info(f"ASR model loaded: {self._model_path} (type: {model_type_str})")
            if resolved_assets.downloaded:
                self.get_logger().info(f"ASR model bundle '{resolved_assets.profile}' was downloaded automatically.")
            if not self._asr.is_streaming():
                self.get_logger().info(
                    "Offline ASR model loaded. File recognition is available; "
                    "microphone realtime recognition still requires a streaming model."
                )
        except Exception as e:
            self._asr_init_error = str(e)
            self.get_logger().error(f"Failed to load ASR model: {e}")
            if self._exit_on_init_failure:
                raise RuntimeError(f"ASR initialization failed and exit_on_init_failure is true: {e}") from e

        self._recognition_lock = threading.Lock()
        self._recording_start_time: float | None = None
        self._last_partial_text: str = ""
        self._last_audio_chunk_time = time.monotonic()
        self._last_audio_restart_time = 0.0
        self._last_status_publish_time = 0.0

    def _create_vad_module(self) -> VADModule:
        vad_config = VADConfig(sample_rate=self._sample_rate, frame_size=self._chunk_size)
        vad = VADModule(vad_config)
        vad.set_logger(self.get_logger())
        vad.set_sensitivity(self._vad_sensitivity)
        vad.initialize()
        return vad

    def _reset_recognition_runtime(self, clear_audio: bool = False):
        self._recording_start_time = None
        self._last_partial_text = ""
        self._vad.reset()
        self._asr.reset()
        if clear_audio:
            self._audio_capture.clear_buffer()

    def _init_ros_interface(self):
        """初始化 ROS 接口"""
        self._pub_command = self.create_publisher(String, self._output_topic, 10)
        self._pub_partial = self.create_publisher(String, "/voice_partial", 10)
        self._pub_status = self.create_publisher(String, "/voice_status", 10)
        self._pub_confidence = self.create_publisher(Float32, "/voice_confidence", 10)
        self._pub_file_progress = self.create_publisher(Float32, "/voice_file_progress", 10)

        self._sub_control = self.create_subscription(String, "/voice_control", self._on_voice_control, 10)
        self._sub_file_input = self.create_subscription(String, "/voice_file_input", self._on_file_input, 10)

        self._cb_group = MutuallyExclusiveCallbackGroup()
        self._srv_start = self.create_service(
            Empty, "~/start_recognition", self._on_start_recognition, callback_group=self._cb_group
        )
        self._srv_stop = self.create_service(
            Empty, "~/stop_recognition", self._on_stop_recognition, callback_group=self._cb_group
        )
        self._srv_set_hotwords = self.create_service(
            SetHotwords, "~/set_hotwords", self._on_set_hotwords, callback_group=self._cb_group
        )

        self._srv_recognize_file = self.create_service(
            RecognizeFile, "~/recognize_file", self._on_recognize_file, callback_group=ReentrantCallbackGroup()
        )

        self._state_machine.register_callback(NodeState.IDLE, self._on_state_change)
        self._state_machine.register_callback(NodeState.LISTENING, self._on_state_change)
        self._state_machine.register_callback(NodeState.RECOGNIZING, self._on_state_change)
        self._state_machine.register_callback(NodeState.ERROR, self._on_state_change)

    def _init_control_loop(self):
        """初始化控制循环"""
        self._control_timer = self.create_timer(0.01, self._control_loop, callback_group=self._cb_group)

        if self._state_machine.mode == ActiveMode.CONTINUOUS:
            self._start_continuous_listening()

    def _control_loop(self):
        """主控制循环"""
        if self._state_machine.is_error():
            return

        self._publish_status_heartbeat()

        if self._state_machine.is_recognizing() and self._check_recognition_timeout():
            return

        if self._state_machine.is_listening() or self._state_machine.is_recognizing():
            self._process_audio()

    def _process_audio(self):
        """处理音频数据"""
        audio_chunk = self._audio_capture.get_audio_chunk(timeout=0.01)

        if audio_chunk is None:
            self._check_audio_capture_stall()
            return

        self._last_audio_chunk_time = time.monotonic()
        vad_result = self._vad.process(audio_chunk)
        # 语音活动期间（含开始、持续、结束缓冲）都向 ASR 喂数据
        if vad_result.state in (VADState.STARTING, VADState.SPEAKING, VADState.ENDING):
            if not self._state_machine.is_recognizing():
                if not self._asr.is_streaming():
                    self._stop_realtime_capture(
                        "realtime_asr_unavailable",
                        "Detected speech while using an offline ASR model. "
                        "Realtime microphone recognition requires a streaming model.",
                    )
                    return
                if not self._transition_state(NodeState.RECOGNIZING, "vad_speech_detected"):
                    return
                self._recording_start_time = time.time()
                try:
                    if not self._asr.start_streaming():
                        self._stop_realtime_capture(
                            "realtime_asr_unavailable",
                            "Failed to start the streaming ASR session.",
                        )
                        return
                except Exception as e:
                    self._stop_realtime_capture(
                        "realtime_asr_unavailable",
                        f"Failed to start streaming ASR: {e}",
                    )
                    return

                pre_roll = self._audio_capture.get_pre_roll_audio(self._realtime_pre_roll_seconds)
                if len(pre_roll) > len(audio_chunk):
                    pre_roll = pre_roll[: -len(audio_chunk)]
                max_pre_roll_samples = int(self._sample_rate * _MAX_STREAMING_PREROLL_SECONDS)
                if len(pre_roll) > max_pre_roll_samples:
                    pre_roll = pre_roll[-max_pre_roll_samples:]

                if len(pre_roll) > 0:
                    self._asr.accept_waveform(pre_roll)

            result = self._asr.accept_waveform(audio_chunk)

            if result and self._publish_partial and result.text != self._last_partial_text:
                self._publish_partial_result(result.text)
                self._last_partial_text = result.text

        elif vad_result.state == VADState.SILENCE:
            if self._state_machine.is_recognizing():
                final_result = self._asr.end_streaming()

                if final_result.text:
                    self._publish_command(final_result.text, final_result.confidence)
                else:
                    self.get_logger().warn("Final ASR result is empty after VAD silence")

                if not self._transition_state(NodeState.LISTENING, "vad_silence_detected"):
                    return
                self._reset_recognition_runtime(clear_audio=False)

    def _check_recognition_timeout(self) -> bool:
        """检查识别超时"""
        if self._recording_start_time is None:
            return False

        elapsed = time.time() - self._recording_start_time
        if elapsed >= self._max_recording_duration:
            self.get_logger().warn("Recognition timeout, forcing final result")

            latency = time.time() - self._recording_start_time
            final_result = self._asr.end_streaming()

            if final_result.text:
                self._publish_command(final_result.text, final_result.confidence)
                self.get_logger().info(f"Recognition latency: {latency * 1000:.1f} ms")
            else:
                self.get_logger().warn("Final ASR result is empty after recognition timeout")

            if not self._transition_state(NodeState.LISTENING, "timeout"):
                return True
            self._reset_recognition_runtime(clear_audio=False)
            return True

        return False

    def _check_audio_capture_stall(self):
        now = time.monotonic()
        if now - self._last_audio_chunk_time < _AUDIO_STALL_TIMEOUT_SECONDS:
            return
        if now - self._last_audio_restart_time < _AUDIO_RESTART_MIN_INTERVAL_SECONDS:
            return

        self._last_audio_restart_time = now
        self.get_logger().warn(
            "No microphone audio chunks received; restarting audio capture "
            f"(idle_for={now - self._last_audio_chunk_time:.1f}s)"
        )

        if self._state_machine.is_recognizing():
            final_result = self._asr.end_streaming()
            if final_result.text:
                self._publish_command(final_result.text, final_result.confidence)
            else:
                self.get_logger().warn("Final ASR result is empty before audio capture restart")
            if not self._transition_state(NodeState.LISTENING, "audio_capture_stalled"):
                return

        self._audio_capture.stop_capture()
        if not self._audio_capture.start_capture():
            self._state_machine.set_error("Failed to restart audio capture after input stall")
            return

        self._last_audio_chunk_time = time.monotonic()
        self._reset_recognition_runtime(clear_audio=True)

    def _start_continuous_listening(self):
        """开始持续监听"""
        if not self._ensure_realtime_asr_supported(operation="continuous_listening"):
            return
        if not self._audio_capture.initialize():
            self._state_machine.set_error("Failed to initialize audio capture")
            return

        if not self._audio_capture.start_capture():
            self._state_machine.set_error("Failed to start audio capture")
            return

        self._last_audio_chunk_time = time.monotonic()
        self._reset_recognition_runtime(clear_audio=True)
        self._transition_state(NodeState.LISTENING, "continuous_mode_start")

    def _on_start_recognition(self, request, response):
        """开始识别服务回调"""
        if not self._ensure_realtime_asr_supported(response=response, operation="start_recognition"):
            return response

        if not self._state_machine.is_idle():
            self.get_logger().warn("Recognition already in progress")
            return response

        if not self._audio_capture.initialize():
            self._state_machine.set_error("Failed to initialize audio capture")
            return response

        if not self._audio_capture.start_capture():
            self._state_machine.set_error("Failed to start audio capture")
            return response

        self._last_audio_chunk_time = time.monotonic()
        self._reset_recognition_runtime(clear_audio=True)
        self._transition_state(NodeState.LISTENING, "service_request")

        return response

    def _on_stop_recognition(self, request, response):
        """停止识别服务回调"""
        if self._state_machine.is_recognizing():
            final_result = self._asr.end_streaming()
            if final_result.text:
                self._publish_command(final_result.text, final_result.confidence)

        self._audio_capture.stop_capture()
        self._transition_state(NodeState.IDLE, "service_request")
        self._reset_recognition_runtime(clear_audio=False)

        return response

    def _on_set_hotwords(self, request, response):
        """设置热词服务回调"""
        try:
            hotwords = {}
            if request.hotwords:
                for i, word in enumerate(request.hotwords):
                    boost = request.boost_scores[i] if i < len(request.boost_scores) else 1.5
                    hotwords[word] = boost

            self._asr.set_hotwords(hotwords)
            response.success = True
            response.error_message = ""
        except Exception as e:
            response.success = False
            response.error_message = str(e)

        return response

    def _on_recognize_file(self, request, response):
        """识别文件服务回调"""
        if not self._ensure_asr_ready(response, operation="recognize_file"):
            return response

        file_path = request.file_path
        enable_vad = request.enable_vad

        result = self._file_input.load_file(file_path)

        if not result.success:
            return self._fail_response(response, result.error_message)

        try:
            vad_module = self._create_vad_module() if enable_vad else None
            asr_results = self._asr.recognize_file(
                result.audio_data,
                enable_vad=enable_vad,
                vad_module=vad_module,
            )
        except Exception as e:
            message = f"Failed to recognize file '{file_path}': {e}"
            self.get_logger().error(message)
            return self._fail_response(response, message)

        response.success = True
        response.error_message = ""
        response.results = [r.text for r in asr_results]
        response.timestamps = [r.start_time if r.start_time is not None else 0.0 for r in asr_results]
        response.durations = [r.duration if r.duration is not None else 0.0 for r in asr_results]

        for asr_result in asr_results:
            self._publish_command(asr_result.text, asr_result.confidence)

        return response

    def _on_voice_control(self, msg: String):
        """语音控制话题回调"""
        command = msg.data.lower()

        if command in ["start", "开始", "开始监听"]:
            if self._state_machine.is_idle():
                if not self._ensure_realtime_asr_supported(operation="voice_control.start"):
                    return
                if not self._audio_capture.initialize():
                    return
                if self._audio_capture.start_capture():
                    self._last_audio_chunk_time = time.monotonic()
                    self._reset_recognition_runtime(clear_audio=True)
                    self._transition_state(NodeState.LISTENING, "topic_command")

        elif command in ["stop", "停止", "停止监听"] and (
            self._state_machine.is_listening() or self._state_machine.is_recognizing()
        ):
            if self._state_machine.is_recognizing():
                final_result = self._asr.end_streaming()
                if final_result.text:
                    self._publish_command(final_result.text, final_result.confidence)
            self._audio_capture.stop_capture()
            self._transition_state(NodeState.IDLE, "topic_command")
            self._reset_recognition_runtime(clear_audio=False)

    def _on_file_input(self, msg: String):
        """文件输入话题回调（异步处理）"""
        if not self._ensure_asr_ready(operation="voice_file_input"):
            return

        file_path = msg.data

        def process_file():
            result = self._file_input.load_file(file_path)

            if not result.success:
                self.get_logger().error(f"Failed to load file: {result.error_message}")
                return

            try:
                asr_results = self._asr.recognize_file(
                    result.audio_data,
                    enable_vad=True,
                    vad_module=self._create_vad_module(),
                )
            except Exception as e:
                self.get_logger().error(f"Failed to recognize file '{file_path}': {e}")
                return

            for asr_result in asr_results:
                self._publish_command(asr_result.text, asr_result.confidence)

        thread = threading.Thread(target=process_file, daemon=True)
        thread.start()

    def _on_audio_error(self, error_message: str):
        """音频错误回调"""
        self.get_logger().error(f"Audio error: {error_message}")
        self._state_machine.set_error(error_message)

    def _on_file_progress(self, progress: float):
        """文件处理进度回调"""
        msg = Float32()
        msg.data = progress
        self._pub_file_progress.publish(msg)

    def _on_state_change(self, old_state: NodeState, new_state: NodeState):
        """状态变化回调"""
        self._publish_status(new_state)

        self.get_logger().debug(f"State changed: {old_state.value} -> {new_state.value}")

    def _publish_status(self, state: NodeState):
        msg = String()
        msg.data = state.value
        self._pub_status.publish(msg)
        self._last_status_publish_time = time.monotonic()

    def _publish_status_heartbeat(self):
        now = time.monotonic()
        if now - self._last_status_publish_time >= _STATUS_HEARTBEAT_INTERVAL_SECONDS:
            self._publish_status(self._state_machine.state)

    def _publish_command(self, text: str, confidence: float = 1.0):
        """发布最终识别结果"""
        msg = String()
        msg.data = text
        self._pub_command.publish(msg)

        conf_msg = Float32()
        conf_msg.data = confidence
        self._pub_confidence.publish(conf_msg)

        self.get_logger().info(f"Command: {text}")

    def _publish_partial_result(self, text: str):
        """发布中间识别结果"""
        msg = String()
        msg.data = text
        self._pub_partial.publish(msg)

    def destroy_node(self):
        """销毁节点"""
        if hasattr(self, "_control_timer"):
            self._control_timer.cancel()
        self._audio_capture.cleanup()
        self._asr.cleanup()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    try:
        node = VoiceASRNode()
    except Exception as e:
        import logging

        logging.getLogger("voice_asr_node").error(f"Node initialization failed: {e}")
        if rclpy.ok():
            rclpy.shutdown()
        return 1

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        executor.shutdown(timeout_sec=0.0)
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    main()
