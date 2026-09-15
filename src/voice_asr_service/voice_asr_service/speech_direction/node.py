"""speech_direction_node ROS 适配层。

~250 行,职责:
    - ROS 参数:sample_rate, mount_yaw_deg, audio_topic, 模型路径
    - 构造 FullSubNetEnhancer + SpeechGate + StftSrpPhat + SpeechDirectionPipeline + Runtime
    - 定时器轮询 get_speech_direction(pull 模型,~10Hz)
    - 坐标转换:阵列坐标系(度) → REP-103(弧度)(统一角度单位,显式 deg2rad)
    - 发布 SpeechDirection(header.stamp=音频样本时间, seq_id=段序号)
    - 降级处理:设备打开失败 / FullSubNet 加载失败 → 不发布, diagnostic_msgs 报告, 不 crash

发布契约(段级事件,无人声不发布):
    - 人声开始后发布稳定初始方向,并在人声结束时发布最终方向
    - 无人声时不发布任何方向消息
"""

from __future__ import annotations

import logging
import math
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy

from ibrobot_msgs.msg import SpeechDirection
from inference_manifest import load_inference_manifest
from inference_service.backends import RuntimeContext
from inference_service.runtime_composition import (
    build_model_service_runtime_dependencies,
    require_runtime_dependencies,
)
from inference_service.unified_runtime import (
    ExecutionContract,
    ModelRuntimeHandle,
    OwnedComponent,
    RegistrySet,
    RuntimeAssembly,
    RuntimeProviders,
)

from .config import SpeechDirectionConfig
from .diagnostics import DiagnosticsRecorder, RecorderStatus
from .doa.srp_phat import StftSrpPhat
from .enhancement.factory import build_stateful_fullsubnet
from .enhancement.fullsubnet import FullSubNetEnhancer
from .model_sessions import SpeechDirectionRoleRunner, SpeechDirectionSessionResources
from .pipeline import DoaState, PipelineParams, SpeechDirectionPipeline, VadState
from .pipeline_streaming import StreamingPipelineParams, StreamingSpeechDirectionPipeline
from .runtime import SpeechDirectionRuntime
from .speech_gate import SileroVadEngine, SpeechGate
from .streaming_runtime import SpeechDirectionStreamingRuntime

logger = logging.getLogger(__name__)

# 发布节拍(轮询 get_speech_direction 的频率,~10Hz)
POLL_PERIOD_SEC = 0.1

_PARAMETER_TYPES = {
    "audio_channels": Parameter.Type.INTEGER,
    "sample_rate": Parameter.Type.INTEGER,
    "srp_update_interval_hops": Parameter.Type.INTEGER,
    "early_direction_min_scores": Parameter.Type.INTEGER,
    "raw_activity_gate_enabled": Parameter.Type.BOOL,
    "raw_activity_start_rms": Parameter.Type.DOUBLE,
    "raw_activity_stop_rms": Parameter.Type.DOUBLE,
    "raw_activity_tail_sec": Parameter.Type.DOUBLE,
    "mount_yaw_deg": Parameter.Type.DOUBLE,
    "angle_step_degree": Parameter.Type.INTEGER,
    "audio_topic": Parameter.Type.STRING,
    "speech_direction_inference_bundle": Parameter.Type.STRING,
    "silero_vad_inference_bundle": Parameter.Type.STRING,
    "silero_vad_backend": Parameter.Type.STRING,
    "silero_vad_deployment": Parameter.Type.STRING,
    "fullsubnet_backend": Parameter.Type.STRING,
    "fullsubnet_deployment": Parameter.Type.STRING,
    "speech_direction_max_age_ms": Parameter.Type.INTEGER,
    "channel_indices": Parameter.Type.INTEGER_ARRAY,
    "mic_positions": Parameter.Type.DOUBLE_ARRAY,
    "diagnostics_high_throughput_enabled": Parameter.Type.BOOL,
    "diagnostics_rollover_seconds": Parameter.Type.INTEGER,
    "diagnostics_save_raw6ch": Parameter.Type.BOOL,
    "diagnostics_save_enh4ch": Parameter.Type.BOOL,
    "diagnostics_save_frame_metrics": Parameter.Type.BOOL,
    "diagnostics_save_gray_events": Parameter.Type.BOOL,
    "diagnostics_queue_size": Parameter.Type.INTEGER,
    "diagnostics_drop_when_full": Parameter.Type.BOOL,
    "fullsubnet_timing_enabled": Parameter.Type.BOOL,
}
_PARAMETER_NAMES = tuple(_PARAMETER_TYPES)


def _require_string(values: Mapping[str, Any], name: str, *, allow_empty: bool = False) -> str:
    """严格读取字符串参数，不接受隐式字符串化。"""
    raw_value = values[name]
    if not isinstance(raw_value, str):
        raise ValueError(f"参数 {name} 必须是字符串")
    value = raw_value.strip()
    if not allow_empty and not value:
        raise ValueError(f"参数 {name} 不能为空")
    return value


def _require_non_empty_string(values: Mapping[str, Any], name: str) -> str:
    """读取并校验必填字符串参数。"""
    return _require_string(values, name)


def _convert_int(values: Mapping[str, Any], name: str) -> int:
    """严格读取整数参数，bool、浮点和字符串均不接受。"""
    value = values[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"参数 {name} 必须是整数")
    return value


def _require_bool(values: Mapping[str, Any], name: str) -> bool:
    """严格读取布尔参数，不接受 0/1 或字符串等隐式转换。"""
    value = values[name]
    if not isinstance(value, bool):
        raise ValueError(f"参数 {name} 必须是布尔值")
    return value


def _require_positive_int(values: Mapping[str, Any], name: str) -> int:
    """严格读取正整数参数。"""
    value = _convert_int(values, name)
    if value <= 0:
        raise ValueError(f"参数 {name} 必须是正整数")
    return value


def _convert_float(values: Mapping[str, Any], name: str) -> float:
    """严格读取 DOUBLE 参数，不接受整数或字符串代替。"""
    value = values[name]
    if not isinstance(value, float):
        raise ValueError(f"参数 {name} 必须是浮点数")
    return value


def _convert_int_list(values: Mapping[str, Any], name: str) -> list[int]:
    """严格读取整数数组及其元素类型。"""
    raw_values = values[name]
    if not isinstance(raw_values, list | tuple) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in raw_values
    ):
        raise ValueError(f"参数 {name} 必须是整数数组")
    return list(raw_values)


def _convert_float_list(values: Mapping[str, Any], name: str) -> list[float]:
    """严格读取 DOUBLE 数组及其元素类型。"""
    raw_values = values[name]
    if not isinstance(raw_values, list | tuple) or any(not isinstance(value, float) for value in raw_values):
        raise ValueError(f"参数 {name} 必须是浮点数数组")
    return list(raw_values)


def build_config_from_parameter_values(values: Mapping[str, Any]) -> SpeechDirectionConfig:
    """将已读取的 ROS 参数校验并映射为算法配置。"""
    audio_channels = _require_positive_int(values, "audio_channels")
    sample_rate = _convert_int(values, "sample_rate")
    if sample_rate != 16000:
        raise ValueError("参数 sample_rate 当前仅支持 16000 Hz")
    srp_update_interval_hops = _require_positive_int(values, "srp_update_interval_hops")
    early_direction_min_scores = _require_positive_int(values, "early_direction_min_scores")

    channel_indices = _convert_int_list(values, "channel_indices")
    # 当前 FullSubNet、缓冲区和 SRP 阵列均固定处理四路麦克风信号。
    if len(channel_indices) != 4 or any(value < 0 for value in channel_indices):
        raise ValueError("参数 channel_indices 必须包含 4 个非负整数通道")
    if len(set(channel_indices)) != len(channel_indices):
        raise ValueError("参数 channel_indices 不能包含重复通道")

    flat_positions = _convert_float_list(values, "mic_positions")
    if len(flat_positions) != 2 * len(channel_indices):
        raise ValueError("参数 mic_positions 长度必须等于 channel_indices 数量的两倍")
    if not all(math.isfinite(value) for value in flat_positions):
        raise ValueError("参数 mic_positions 必须全部为有限浮点数")
    mic_positions = [flat_positions[index : index + 2] for index in range(0, len(flat_positions), 2)]

    silero_backend_raw = _require_string(values, "silero_vad_backend")
    if silero_backend_raw not in {"ascend", "onnx"}:
        raise ValueError("参数 silero_vad_backend 只能为 ascend 或 onnx")
    silero_backend = silero_backend_raw
    silero_deployment = _require_non_empty_string(values, "silero_vad_deployment")

    fullsubnet_backend = _require_string(values, "fullsubnet_backend")
    if fullsubnet_backend not in {"ascend", "stateful_torch_cuda", "stateful_torch_cpu", "torch"}:
        raise ValueError("参数 fullsubnet_backend 不是受支持的 Ascend/Torch 后端")
    bundle = _require_non_empty_string(values, "speech_direction_inference_bundle")
    fullsubnet_deployment = _require_non_empty_string(values, "fullsubnet_deployment")
    silero_bundle = _require_string(values, "silero_vad_inference_bundle", allow_empty=True)
    # Torch stateful 后端需要 checkpoint + manifest；Model 类由 ibrobot-fullsubnet wheel 提供。
    max_age_ms = _convert_int(values, "speech_direction_max_age_ms")
    if max_age_ms <= 0:
        raise ValueError("参数 speech_direction_max_age_ms 必须大于 0")
    mount_yaw_deg = _convert_float(values, "mount_yaw_deg")
    if not math.isfinite(mount_yaw_deg):
        raise ValueError("参数 mount_yaw_deg 必须是有限浮点数")
    raw_activity_gate_enabled = _require_bool(values, "raw_activity_gate_enabled")
    raw_activity_start_rms = _convert_float(values, "raw_activity_start_rms")
    raw_activity_stop_rms = _convert_float(values, "raw_activity_stop_rms")
    raw_activity_tail_sec = _convert_float(values, "raw_activity_tail_sec")
    if (
        not math.isfinite(raw_activity_start_rms)
        or raw_activity_start_rms < 0.0
        or not math.isfinite(raw_activity_stop_rms)
        or raw_activity_stop_rms < 0.0
        or raw_activity_stop_rms > raw_activity_start_rms
        or not math.isfinite(raw_activity_tail_sec)
        or raw_activity_tail_sec < 0.0
    ):
        raise ValueError("raw activity gate parameters are invalid")

    # SRP-PHAT 扫描角度步长(度)，360 必须能被其整除，否则候选角度无法均匀覆盖整圈。
    angle_step_degree = _convert_int(values, "angle_step_degree")
    if angle_step_degree <= 0 or 360 % angle_step_degree != 0:
        raise ValueError("参数 angle_step_degree 必须为 360 的正整数约数（如 1、2、3、4、5、6、8、9、10...）")

    # 高通量维测契约拒绝 Python 的隐式类型兼容，避免 ROS 参数被静默改写。
    diagnostics_high_throughput_enabled = _require_bool(values, "diagnostics_high_throughput_enabled")
    diagnostics_rollover_seconds = _require_positive_int(values, "diagnostics_rollover_seconds")
    diagnostics_save_raw6ch = _require_bool(values, "diagnostics_save_raw6ch")
    diagnostics_save_enh4ch = _require_bool(values, "diagnostics_save_enh4ch")
    diagnostics_save_frame_metrics = _require_bool(values, "diagnostics_save_frame_metrics")
    diagnostics_save_gray_events = _require_bool(values, "diagnostics_save_gray_events")
    diagnostics_queue_size = _require_positive_int(values, "diagnostics_queue_size")
    diagnostics_drop_when_full = _require_bool(values, "diagnostics_drop_when_full")
    fullsubnet_timing_enabled = _require_bool(values, "fullsubnet_timing_enabled")

    # 仅使用上述归一化变量构造配置，避免未校验输入进入算法链。
    cfg = SpeechDirectionConfig()
    cfg.audio.channels = audio_channels
    cfg.audio.sample_rate = sample_rate
    cfg.pipeline.sample_rate = sample_rate
    cfg.pipeline.srp_update_interval_hops = srp_update_interval_hops
    cfg.pipeline.early_direction_min_scores = early_direction_min_scores
    cfg.pipeline.raw_activity_gate_enabled = raw_activity_gate_enabled
    cfg.pipeline.raw_activity_start_rms = raw_activity_start_rms
    cfg.pipeline.raw_activity_stop_rms = raw_activity_stop_rms
    cfg.pipeline.raw_activity_tail_sec = raw_activity_tail_sec
    cfg.vad.sample_rate = sample_rate
    cfg.doa.sample_rate = sample_rate
    if any(value >= cfg.audio.channels for value in channel_indices):
        raise ValueError(f"参数 channel_indices 必须小于音频通道数 {cfg.audio.channels}")
    cfg.audio.channel_indices = channel_indices
    cfg.pipeline.input_channels = list(channel_indices)
    cfg.doa.input_channels = list(channel_indices)
    cfg.doa.mic_positions = mic_positions
    cfg.fullnet.inference_bundle = bundle
    cfg.fullnet.deployment = fullsubnet_deployment
    if silero_bundle:
        cfg.vad.inference_bundle = silero_bundle
    cfg.fullnet.device = {"stateful_torch_cuda": "cuda", "stateful_torch_cpu": "cpu", "torch": "cpu"}.get(
        fullsubnet_backend, "cuda"
    )
    cfg.fullnet.backend = fullsubnet_backend
    cfg.vad.backend = silero_backend
    cfg.vad.deployment = silero_deployment
    cfg.audio_topic = _require_non_empty_string(values, "audio_topic")
    cfg.mount_yaw_deg = mount_yaw_deg
    cfg.doa.angle_step_degree = angle_step_degree
    cfg.speech_direction_max_age_ms = max_age_ms
    cfg.diagnostics.high_throughput_enabled = diagnostics_high_throughput_enabled
    cfg.diagnostics.rollover_seconds = diagnostics_rollover_seconds
    cfg.diagnostics.save_raw6ch = diagnostics_save_raw6ch
    cfg.diagnostics.save_enh4ch = diagnostics_save_enh4ch
    cfg.diagnostics.save_frame_metrics = diagnostics_save_frame_metrics
    cfg.diagnostics.save_gray_events = diagnostics_save_gray_events
    cfg.diagnostics.queue_size = diagnostics_queue_size
    cfg.diagnostics.drop_when_full = diagnostics_drop_when_full
    cfg.diagnostics.fullsubnet_timing_enabled = fullsubnet_timing_enabled
    return cfg


def _validate_silero_deployment_contract(manifest: Any, cfg: Any) -> None:
    """Fail closed when the selected Silero deployment contradicts the host pipeline.

    The node feeds fixed 16 kHz mono float32 audio through the Silero frame
    engine (frame_size sub-frames); a deployment declaring a different rate,
    offline execution, or an incompatible frame would silently mislabel
    samples or fail deep inside the session.
    """
    contract = getattr(manifest.deployment, "audio_contract", None)
    if contract is None:
        raise ValueError(
            f"Silero deployment {cfg.vad.deployment!r} declares no audio contract; "
            "the speech_direction pipeline cannot verify its audio interface"
        )
    mismatches = []
    if contract.sample_rate_hz != cfg.vad.sample_rate:
        mismatches.append(
            f"sample_rate_hz={contract.sample_rate_hz} but the pipeline processes {cfg.vad.sample_rate} Hz"
        )
    if contract.channels != 1 or contract.sample_dtype != "float32":
        mismatches.append(
            f"channels={contract.channels} sample_dtype={contract.sample_dtype} but the pipeline feeds mono float32"
        )
    if contract.frame_size is not None and contract.frame_size != cfg.vad.frame_size:
        mismatches.append(
            f"frame_size={contract.frame_size} but the Silero engine assembles {cfg.vad.frame_size}-sample sub-frames"
        )
    # The engine feeds frame_size + context samples per model call; a declared
    # chunk_size that disagrees with that model input length is a contract lie.
    context_size = 64 if contract.sample_rate_hz == 16000 else 32
    expected_model_input = (contract.frame_size or 0) + context_size
    if contract.chunk_size is not None and contract.chunk_size != expected_model_input:
        mismatches.append(
            f"chunk_size={contract.chunk_size} but the engine feeds {expected_model_input} "
            "samples (frame + context) per model input"
        )
    if contract.execution_mode not in {None, "streaming", "both"}:
        mismatches.append(f"execution_mode={contract.execution_mode} but speech_direction streams audio")
    if mismatches:
        raise ValueError(
            f"Silero deployment {cfg.vad.deployment!r} audio contract mismatches the host pipeline: "
            + "; ".join(mismatches)
        )


class SpeechDirectionNode(Node):
    """speech_direction ROS 节点。"""

    def __init__(
        self,
        *,
        registry_set: RegistrySet | None = None,
        providers: RuntimeProviders | None = None,
    ) -> None:
        super().__init__("speech_direction_node")
        registry_set, providers = require_runtime_dependencies(
            registry_set,
            providers,
            owner=type(self).__name__,
        )
        self._registry_set = registry_set
        self._providers = providers

        # ============================ 声明并加载 ROS 参数 ============================
        # 部署参数必须由 launch/YAML 注入，节点代码不提供业务默认值。
        self._declare_parameters()
        cfg = self._load_config_from_parameters()
        self._apply_bundle_artifacts(cfg)
        self._config = cfg
        self._mount_yaw_deg = cfg.mount_yaw_deg

        # ============================ 发布者 ============================
        # 方向 topic 用 KEEP_LAST(1),只保留最新方向
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self._direction_pub = self.create_publisher(SpeechDirection, "/voice/speech_direction", qos)
        self._audio_subscription = None
        # 在线状态(诊断)
        self._diag_pub = self.create_publisher(DiagnosticArray, "/diagnostics", 10)

        # ============================ 降级状态 ============================
        self._degraded = False
        self._degrade_reason = ""
        self._degraded_lock = threading.Lock()

        # ============================ 构建算法链(可能失败 → 降级)============================
        self._runtime: SpeechDirectionRuntime | None = None
        self._runtime_handle: ModelRuntimeHandle | None = None
        self._session_resources: SpeechDirectionSessionResources | None = None
        self._pending_backend_resources: tuple[object, ...] = ()
        self._diagnostics_recorder: DiagnosticsRecorder | None = None
        self._diagnostics_status = RecorderStatus(False, "disabled", None, None, 0)
        self._diagnostics_session_dir = ""
        # destroy_node 可能被 finally 与外部生命周期重复或并发调用。
        self._destroy_condition = threading.Condition(threading.RLock())
        self._destroy_started = False
        self._destroy_completed = False
        self._destroy_owner_id: int | None = None
        self._destroy_result = False
        self._last_published_seq_id: int | None = None  # 已发布的段序号,避免重复发布

        # 模型校验:缺失直接 raise FileNotFoundError 退出(对齐 perception_service 风格,
        # 不降级保持运行)。模型文件"没下载"是部署问题,应由用户跑下载脚本,不是运行时故障。
        from .model_downloader import require_configured_models

        require_configured_models(
            cfg.vad.model_path,
            cfg.fullnet.ckpt,
            silero_backend=cfg.vad.backend,
            fullsubnet_backend=cfg.fullnet.backend,
            fullsubnet_stateful_fb_om_path=cfg.fullnet.stateful_fb_om_path,
            fullsubnet_stateful_sb_om_path=cfg.fullnet.stateful_sb_om_path,
            fullsubnet_stateful_manifest_path=cfg.fullnet.stateful_manifest_path,
        )

        # 算法链构建:模型文件已就绪,此后的加载/推理失败才走降级(运行时故障降级策略)
        try:
            self._build_and_start()
        except Exception as e:
            self._enter_degraded(f"算法链构建/启动失败: {e}")
            logger.error("speech_direction 降级: %s", e, exc_info=True)

        # ============================ 轮询定时器 ============================
        self._poll_timer = self.create_timer(POLL_PERIOD_SEC, self._poll_and_publish)
        # 诊断定时器(1Hz)
        self._diag_timer = self.create_timer(1.0, self._publish_diagnostics)

        self.get_logger().info(
            f"speech_direction_node 已启动: audio_topic={cfg.audio_topic}, "
            f"mount_yaw_deg={cfg.mount_yaw_deg}, degraded={self._degraded}"
        )

    def _declare_parameters(self) -> None:
        """声明必须由 YAML 提供的 ROS 参数，不在节点代码中设置部署默认值。"""
        for name, parameter_type in _PARAMETER_TYPES.items():
            self.declare_parameter(name, parameter_type)

    def _load_config_from_parameters(self) -> SpeechDirectionConfig:
        """读取 ROS 参数并通过纯函数完成集中校验与配置构造。"""
        values = {name: self.get_parameter(name).value for name in _PARAMETER_NAMES}
        return build_config_from_parameter_values(values)

    @staticmethod
    def _apply_bundle_artifacts(cfg: SpeechDirectionConfig) -> None:
        """Derive artifact paths from the standalone FullSubNet/Silero bundles."""
        bundle = Path(cfg.fullnet.inference_bundle)
        fullsubnet = load_inference_manifest(bundle, cfg.fullnet.deployment)
        resolved = fullsubnet.resolved_artifacts
        if cfg.fullnet.backend == "ascend":
            cfg.fullnet.stateful_fb_om_path = str(resolved["fullsubnet_fb"])
            cfg.fullnet.stateful_sb_om_path = str(resolved["fullsubnet_sb"])
            cfg.fullnet.stateful_manifest_path = str(resolved["state_manifest"])
            profile = getattr(fullsubnet.runtime_profile, "profile", None)
            cfg.fullnet.device_id = int(getattr(profile, "device_id", 0) if profile is not None else 0)
        else:
            cfg.fullnet.ckpt = str(resolved["fullsubnet_fb"])
            cfg.fullnet.stateful_manifest_path = str(resolved["fullsubnet_sb"])

        # Silero VAD always resolves through the standalone reusable bundle.
        silero_bundle = Path(cfg.vad.inference_bundle)
        silero = load_inference_manifest(silero_bundle, cfg.vad.deployment)
        cfg.vad.model_path = str(silero.resolved_artifacts["silero_vad"])

    # ------------------------------------------------------------------ 算法链构建
    def _make_session_dir(self) -> str:
        """创建本次运行的 session 目录路径(runs/run_<timestamp>)。"""
        import datetime

        # 目录由 recorder 在 start() 的原子会话初始化中创建；此处只分配路径。
        base = os.path.expanduser("~/.ros/speech_direction/runs")
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return os.path.join(base, f"run_{ts}")

    def _build_and_start(self) -> None:
        """构建 FullSubNet + SpeechGate + SRP + Pipeline + Runtime 并启动。

        模型文件已由 __init__ 调用 require_configured_models() 校验过(缺失直接 raise 退出)。
        本方法只负责加载与线程启动;此处的加载/推理失败走降级(运行时故障降级策略)。
        """
        cfg = self._config
        session_entries: dict[str, tuple[object, RuntimeContext]] = {}
        backend_resources: list[object] = []

        # 两个平台仅在 executor 选择上分叉，随后共用 cumulative Host 增强器。
        stateful_backend = cfg.fullnet.backend in {"ascend", "stateful_torch_cuda", "stateful_torch_cpu"}
        if stateful_backend and cfg.fullnet.backend == "ascend":
            bundle = Path(cfg.fullnet.inference_bundle)
            fullsubnet_manifest = load_inference_manifest(bundle, cfg.fullnet.deployment)
            fullsubnet_role = "fullsubnet_fb"
            fullsubnet_context = RuntimeContext(
                fullsubnet_manifest,
                {"device_id": cfg.fullnet.device_id},
                role=fullsubnet_role,
            )
            fullsubnet_session = self._registry_set.session_builder_registry.create(
                fullsubnet_context,
                backend_registry=self._registry_set.backend_registry,
                providers=self._providers,
            )
            session_entries["fullsubnet"] = (fullsubnet_session, fullsubnet_context)
            self._session_resources = SpeechDirectionSessionResources(
                {"fullsubnet": (fullsubnet_session, fullsubnet_context)}
            )
            fullnet = build_stateful_fullsubnet(
                backend=cfg.fullnet.backend,
                manifest_path=cfg.fullnet.stateful_manifest_path,
                timing_enabled=cfg.diagnostics.fullsubnet_timing_enabled,
                initialize_backend=False,
                executor=SpeechDirectionRoleRunner(fullsubnet_session, fullsubnet_context, owns_session=False),
            )
        elif stateful_backend:
            bundle = Path(cfg.fullnet.inference_bundle)
            fullsubnet_manifest = load_inference_manifest(bundle, cfg.fullnet.deployment)
            fullsubnet_context = RuntimeContext(
                fullsubnet_manifest,
                {"timing_enabled": cfg.diagnostics.fullsubnet_timing_enabled},
                role="fullsubnet_fb",
            )
            fullsubnet_session = self._registry_set.session_builder_registry.create(
                fullsubnet_context,
                backend_registry=self._registry_set.backend_registry,
                providers=self._providers,
            )
            session_entries["fullsubnet"] = (fullsubnet_session, fullsubnet_context)
            self._session_resources = SpeechDirectionSessionResources(
                {"fullsubnet": (fullsubnet_session, fullsubnet_context)}
            )
            fullnet = build_stateful_fullsubnet(
                backend=cfg.fullnet.backend,
                manifest_path=cfg.fullnet.stateful_manifest_path,
                timing_enabled=cfg.diagnostics.fullsubnet_timing_enabled,
                initialize_backend=False,
                executor=SpeechDirectionRoleRunner(fullsubnet_session, fullsubnet_context, owns_session=False),
            )
        else:
            # The non-stateful Torch path is an explicit comparison mode; it is never
            # selected as a fallback after a stateful runtime failure.
            fullnet = FullSubNetEnhancer(
                ckpt=cfg.fullnet.ckpt,
                device=cfg.fullnet.device,
            )
            backend_resources.append(fullnet)

        vad_runner = None
        if cfg.vad.backend in {"ascend", "onnx"}:
            vad_manifest = load_inference_manifest(Path(cfg.vad.inference_bundle), cfg.vad.deployment)
            _validate_silero_deployment_contract(vad_manifest, cfg)
            vad_context = RuntimeContext(
                vad_manifest,
                {"device_id": cfg.fullnet.device_id},
                role="silero_vad",
            )
            vad_session = self._registry_set.session_builder_registry.create(
                vad_context,
                backend_registry=self._registry_set.backend_registry,
                providers=self._providers,
            )
            session_entries["silero_vad"] = (vad_session, vad_context)
            if self._session_resources is None:
                self._session_resources = SpeechDirectionSessionResources({"silero_vad": (vad_session, vad_context)})
            else:
                self._session_resources.add("silero_vad", vad_session, vad_context)
            vad_runner = SpeechDirectionRoleRunner(vad_session, vad_context, owns_session=False)

        # 人声门控(复用 common/vad/silero)
        # vad_runner(manifest 驱动)只做裸推理转发，不含 SileroVadEngine 的帧间 context 拼接
        # (Silero 要求 [上一帧末尾64样本|本帧512样本] 组成 (1,576) 输入)；
        # 用 SileroVadEngine(inference_runner=vad_runner) 包一层，复用其 context 拼接逻辑，
        # 同时避免重复加载 OM。
        silero_engine = (
            SileroVadEngine(
                model_path=cfg.vad.model_path,
                sample_rate=cfg.vad.sample_rate,
                backend=cfg.vad.backend,
                inference_runner=vad_runner,
            )
            if vad_runner is not None
            else None
        )
        speech_gate = SpeechGate(
            model_path=cfg.vad.model_path,
            sample_rate=cfg.vad.sample_rate,
            vad_threshold=cfg.gray_region.vad_threshold,
            rms_threshold=cfg.gray_region.rms_threshold,
            backend=cfg.vad.backend,
            silero_engine=silero_engine,
        )
        if silero_engine is None and speech_gate.silero is not None:
            backend_resources.append(speech_gate.silero)

        # SRP-PHAT(阵列几何与声学参数从配置传入,配置驱动)
        angles = np.arange(0, 360, cfg.doa.angle_step_degree, dtype=np.float32)
        srp = StftSrpPhat(
            frame_size=cfg.doa.frame_size,
            hop_size=cfg.doa.hop_size,
            angles_deg=angles,
            sample_rate=cfg.doa.sample_rate,
            mic_positions=np.array(cfg.doa.mic_positions, dtype=np.float64),
            sound_speed=cfg.doa.sound_speed,
            freq_lo=cfg.doa.freq_band_hz[0],
            freq_hi=cfg.doa.freq_band_hz[1],
            diag_freq_hi=cfg.doa.diag_pair_freq_max_hz,
        )

        vad_state = VadState()
        doa_state = DoaState()

        if stateful_backend:
            # 两个平台统一使用 256 tick、512 T=2 模型批次和 4096/512 SRP。
            streaming_params = StreamingPipelineParams(
                sample_rate=cfg.pipeline.sample_rate,
                processing_samples=cfg.pipeline.processing_hop_samples,
                model_batch_samples=cfg.pipeline.model_batch_samples,
                srp_update_interval_hops=cfg.pipeline.srp_update_interval_hops,
                early_direction_min_scores=cfg.pipeline.early_direction_min_scores,
                raw_activity_gate_enabled=cfg.pipeline.raw_activity_gate_enabled,
                raw_activity_start_rms=cfg.pipeline.raw_activity_start_rms,
                raw_activity_stop_rms=cfg.pipeline.raw_activity_stop_rms,
                raw_activity_tail_samples=round(cfg.pipeline.raw_activity_tail_sec * cfg.pipeline.sample_rate),
                input_channels=tuple(cfg.pipeline.input_channels),
                srp_frame_samples=cfg.doa.frame_size,
                srp_hop_samples=cfg.doa.hop_size,
                candidate_window_samples=round(0.064 * cfg.pipeline.sample_rate),
                segment_end_gap_samples=math.ceil(cfg.gray_region.seg_end_gap_s * cfg.pipeline.sample_rate),
                min_segment_samples=math.ceil(cfg.gray_region.min_seg_dur_s * cfg.pipeline.sample_rate),
                min_accum_samples=cfg.gray_region.min_accum_frames * cfg.doa.hop_size,
                max_accum_samples=math.ceil(cfg.gray_region.max_accum_dur_s * cfg.pipeline.sample_rate),
                segment_max_rms_threshold=cfg.gray_region.seg_max_rms_threshold,
            )
            params = None
        else:
            # 旧 offline 链路只允许显式 legacy backend，不与 stateful 状态混用。
            params = PipelineParams(
                sample_rate=cfg.pipeline.sample_rate,
                frame_size=cfg.pipeline.frame_size,
                hop_size=cfg.pipeline.hop_size,
                input_channels=cfg.pipeline.input_channels,
                seg_end_gap_s=cfg.gray_region.seg_end_gap_s,
                min_seg_dur_s=cfg.gray_region.min_seg_dur_s,
                min_accum_frames=cfg.gray_region.min_accum_frames,
                max_accum_dur_s=cfg.gray_region.max_accum_dur_s,
                seg_max_rms_threshold=cfg.gray_region.seg_max_rms_threshold,
            )
            streaming_params = None
        pipeline = None  # diagnostics 完成初始化后统一注入 pipeline

        # 高通量维测是旁路：关闭或初始化失败均不改变定位 pipeline 的健康状态。
        diagnostics = None
        if cfg.diagnostics.high_throughput_enabled:
            session_dir = ""
            try:
                session_dir = self._make_session_dir()
                session_id = os.path.basename(session_dir)
                diagnostics = DiagnosticsRecorder(
                    session_dir,
                    session_id=session_id,
                    sample_rate=cfg.audio.sample_rate,
                    rollover_seconds=cfg.diagnostics.rollover_seconds,
                    save_raw6ch=cfg.diagnostics.save_raw6ch,
                    save_enh4ch=cfg.diagnostics.save_enh4ch,
                    save_frame_metrics=cfg.diagnostics.save_frame_metrics,
                    save_gray_events=cfg.diagnostics.save_gray_events,
                    queue_size=cfg.diagnostics.queue_size,
                    drop_when_full=cfg.diagnostics.drop_when_full,
                    gray_merge_gap_samples=round(cfg.gray_region.seg_end_gap_s * cfg.audio.sample_rate),
                )
                diagnostics.start()
                self._diagnostics_recorder = diagnostics
                self._diagnostics_status = diagnostics.status
                self._diagnostics_session_dir = session_dir
                if diagnostics.status.enabled:
                    self.get_logger().info(f"维测已启用: session_dir={session_dir}")
                else:
                    self.get_logger().warning(
                        f"维测初始化失败,关闭维测: {diagnostics.status.disabled_reason or '未知原因'}"
                    )
            except Exception as e:
                self._diagnostics_recorder = None
                # 空异常消息仍须暴露可定位的异常类型，避免 WARN reason 为空。
                reason = str(e) or type(e).__name__
                # 构造失败时没有 recorder 可查询，节点保存等价的持久状态快照。
                self._diagnostics_status = RecorderStatus(False, "diagnostics_disabled", reason, None, 0)
                self._diagnostics_session_dir = session_dir
                self.get_logger().warning(f"维测初始化失败,关闭维测: {reason}")
                diagnostics = None

        if stateful_backend:
            assert streaming_params is not None
            pipeline = StreamingSpeechDirectionPipeline(
                fullnet,
                speech_gate.silero,
                srp,
                streaming_params,
                vad_state,
                doa_state,
                vad_threshold=cfg.gray_region.vad_threshold,
                rms_threshold=cfg.gray_region.rms_threshold,
                diagnostics=diagnostics,
                initialize_backend=False,
            )
        else:
            assert params is not None
            pipeline = SpeechDirectionPipeline(
                fullnet, speech_gate, srp, params, vad_state, doa_state, diagnostics=diagnostics
            )

        self._pending_backend_resources = tuple(backend_resources)

        streaming_runtime = SpeechDirectionStreamingRuntime(pipeline, close_backends=False)
        owned_components: list[OwnedComponent] = []
        if self._session_resources is not None:
            for role, (session, context) in session_entries.items():
                owned_components.append(
                    OwnedComponent(session, f"speech_direction_session:{role}", load_context=context)
                )
        for index, resource in enumerate(backend_resources):
            owned_components.append(OwnedComponent(resource, f"speech_direction_backend:{index}"))
        owned_components.append(OwnedComponent(streaming_runtime, "speech_direction_streaming_runtime"))

        state_links = [
            {
                "role": "__runtime__",
                "state_name": "host.stft_ola",
                "owner": "streaming_runtime",
                "source": "host.stft",
                "target": "host.ola",
                "scope": "runtime",
                "state_bank": "speech_direction.host",
            },
            {
                "role": "__runtime__",
                "state_name": "host.segment",
                "owner": "streaming_runtime",
                "source": "host.gate",
                "target": "host.segment",
                "scope": "runtime",
                "state_bank": "speech_direction.host",
            },
        ]
        state_links.extend(
            {
                "role": role,
                "state_name": "recurrent",
                "owner": "session",
                "source": "state.in",
                "target": "state.out",
                "scope": "runtime",
                "state_bank": f"{role}.bank",
            }
            for role in sorted(session_entries)
        )
        execution_contract = ExecutionContract(
            state_scope="stream",
            execution_structure="direct",
            cancellation_granularity="checkpoint",
            state_bank_mode="runtime_exclusive",
            max_open_streams=1,
            state_links=tuple(state_links),
        )
        assembly = RuntimeAssembly(
            runtime_executor=streaming_runtime,
            streaming_runtime=streaming_runtime,
            session=next((session for session, _context in session_entries.values()), None),
            role_assemblies={role: session for role, (session, _context) in session_entries.items()},
            owned_components=tuple(owned_components),
            stateful=True,
            resettable=True,
            state_scope="stream",
            state_bank_mode="runtime_exclusive",
            max_open_streams=1,
            cancellation_granularity="checkpoint",
            execution_contract=execution_contract,
            identity=("tensor_model", "speech_direction", "enhance_and_vad"),
            declared_capabilities={
                "state_owner": "streaming_runtime",
                "state_bank_mode": "runtime_exclusive",
                "max_open_streams": 1,
                "host_state": ("stft", "ola", "gate", "segment"),
            },
            runtime_id="speech-direction",
        )
        self._runtime_handle = ModelRuntimeHandle(assembly)
        # Session ownership has transferred to the handle's concrete owned
        # components; the construction-failure cleanup wrapper is no longer
        # used by the node.
        self._session_resources = None
        self._pending_backend_resources = ()

        self._runtime = SpeechDirectionRuntime(
            cfg,
            pipeline,
            on_fatal_error=self._enter_degraded,
            model_runtime_handle=self._runtime_handle,
        )
        self._runtime.start()

        from audio_common_msgs.msg import AudioDataStamped

        self._audio_subscription = self.create_subscription(
            AudioDataStamped, cfg.audio_topic, self._on_audio_message, 50
        )

    def _on_audio_message(self, message) -> None:
        """Feed interleaved shared PCM into the existing multi-channel ring."""
        if self._runtime is None:
            return
        self._runtime.feed_audio(bytes(message.audio.data))

    # ------------------------------------------------------------------ 轮询发布
    def _poll_and_publish(self) -> None:
        """轮询 get_speech_direction,有新段级方向时发布。"""
        if self._degraded or self._runtime is None:
            return

        try:
            result = self._runtime.get_speech_direction()
        except Exception as e:
            self.get_logger().error(f"get_speech_direction 异常: {e}")
            return

        angle = result["angle"]
        seq_id = result["seq_id"]

        # 无可靠方向(冷启动/过期/无人声)→ 不发布
        if angle is None:
            return

        # 段级去重:同一 seq_id 只发布一次。
        # pipeline 每次段级输出(中间方向 / 段末方向)前 _output_seq += 1,
        # 中间方向与段末方向 seq_id 各自独立,都会被发布;此处去重只防"同一条方向被重复拉取发布"。
        if self._last_published_seq_id == seq_id:
            return
        self._last_published_seq_id = seq_id

        # 坐标转换:阵列坐标系(度) → REP-103(弧度)(统一角度单位,显式 deg2rad)
        # angle: 阵列坐标系角度(度),0°=右,90°=前,180°=左,逆时针为正
        # mount_yaw_deg: 阵列安装偏角(度),逆时针为正
        # ros_azimuth: REP-103 平面角度(弧度),0=前,+π/2=左,左为正
        ros_azimuth = self._normalize_radians(
            math.radians(float(angle)) - math.pi / 2 + math.radians(self._mount_yaw_deg)
        )

        msg = SpeechDirection()
        # header.stamp 承载方向的"真实产生时间",而非发布时刻,使 sound_follow
        # 算 age = ros_now - stamp 时得到方向真年龄,executor 积压/DDS 延迟不会被盖掉。
        # 按方向类型分流(对齐 SpeechDirection.msg 契约:stamp=最后音频样本时间):
        #   - mid_long_seg(长语音中间方向):用发布时刻,age≈0,低延迟响应正在说话的方向
        #   - seg_end(段末方向):用来源真实 age 还原段结束时刻,sound_follow 据此判过期
        ros_now = self.get_clock().now()
        direction_type = result.get("type")
        if direction_type == "seg_end":
            age_sec = float(result.get("age_ms", 0.0)) / 1000.0
            if not math.isfinite(age_sec) or age_sec < 0:
                age_sec = 0.0  # 异常 age 安全钳制为 0(视为新鲜)
            msg.header.stamp = (ros_now - rclpy.duration.Duration(seconds=age_sec)).to_msg()
        else:
            # mid_long_seg 或缺失 type:用发布时刻,age≈0
            msg.header.stamp = ros_now.to_msg()
        msg.header.frame_id = "base_link"
        msg.azimuth_rad = float(ros_azimuth)
        msg.seq_id = int(seq_id)
        msg.segment_id = int(result.get("segment_id", 0))
        msg.direction_type = str(direction_type or "")
        self._direction_pub.publish(msg)
        # 段级 DOA 上报打印：类型/seq_id/阵列角/ROS方位角/age，便于与 sound_follow 侧对照
        self.get_logger().info(
            f"[段级DOA] type={direction_type} seq_id={seq_id} "
            f"angle={float(angle):.1f}deg ros_azimuth={float(ros_azimuth):.3f}rad "
            f"age_ms={result.get('age_ms', 0.0):.1f}"
        )

    @staticmethod
    def _normalize_radians(angle: float) -> float:
        """归一化到 [-π, π]。"""
        return math.atan2(math.sin(angle), math.cos(angle))

    # ------------------------------------------------------------------ 降级处理
    def _enter_degraded(self, reason: str) -> None:
        """进入降级状态:不发布方向,通过 diagnostic_msgs 报告,不 crash。"""
        # runtime worker 可能从后台线程回调；锁保证首个原因稳定且不重复刷日志。
        with self._degraded_lock:
            if self._degraded:
                return
            self._degraded = True
            self._degrade_reason = reason
        self.get_logger().error(f"speech_direction 进入降级状态: {reason}")

    def _recorder_diagnostics(self) -> tuple[RecorderStatus, str]:
        """获取基础 diagnostics 使用的高通量状态，不触碰 recorder 内部并发状态。"""
        recorder = self._diagnostics_recorder
        if recorder is None:
            return self._diagnostics_status, self._diagnostics_session_dir
        session_dir = self._diagnostics_session_dir or str(getattr(recorder, "session_dir", ""))
        return recorder.status, session_dir

    def _publish_diagnostics(self) -> None:
        """发布诊断状态(1Hz)，基础状态不依赖高通量总开关。"""
        recorder_status, session_dir = self._recorder_diagnostics()
        # 状态位与原因必须来自同一临界区，避免发布混合的新旧降级信息。
        with self._degraded_lock:
            degraded = self._degraded
            degrade_reason = self._degrade_reason
        if degraded:
            # 定位链降级优先级最高；高通量状态仍作为附加字段持续暴露。
            status = DiagnosticStatus()
            status.level = DiagnosticStatus.ERROR
            status.name = "speech_direction"
            status.message = f"降级: {degrade_reason}"
            status.hardware_id = "respeaker_4mic"
            values = [KeyValue(key="degraded", value="true")]
        elif self._runtime is not None:
            status = DiagnosticStatus()
            status.name = "speech_direction"
            status.hardware_id = "respeaker_4mic"
            full = self._runtime.doa_state.get_full()
            values = [
                KeyValue(key="latest_angle", value=str(full.get("angle"))),
                KeyValue(key="seq_id", value=str(full.get("seq_id"))),
                KeyValue(
                    key="audio_subscription",
                    value=str(self._audio_subscription is not None),
                ),
            ]
            if recorder_status.state == "diagnostics_disabled":
                status.level = DiagnosticStatus.WARN
                status.message = f"高通量维测已停用: {recorder_status.disabled_reason or '未知原因'}"
            else:
                status.level = DiagnosticStatus.OK
                status.message = "运行正常"
        else:
            return

        values.extend(
            [
                KeyValue(key="enabled", value=str(recorder_status.enabled).lower()),
                KeyValue(key="state", value=recorder_status.state),
                KeyValue(key="reason", value=recorder_status.disabled_reason or ""),
                KeyValue(key="dropped", value=str(recorder_status.dropped_count)),
                KeyValue(key="session_dir", value=session_dir),
            ]
        )
        status.values = values

        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.status = [status]
        self._diag_pub.publish(msg)

    # ------------------------------------------------------------------ 资源清理
    def destroy_node(self) -> bool:
        """关闭时完成全部资源收尾，并让并发调用共享同一结果。"""
        condition = getattr(self, "_destroy_condition", None)
        if condition is None:
            # 兼容 object.__new__ 构造的生命周期单测，同时保持生产节点同一语义。
            condition = threading.Condition(threading.RLock())
            self._destroy_condition = condition
            self._destroy_started = False
            self._destroy_completed = False
            self._destroy_owner_id = None
            self._destroy_result = False

        current_id = threading.get_ident()
        with condition:
            if self._destroy_completed:
                return self._destroy_result
            if self._destroy_started:
                if self._destroy_owner_id == current_id:
                    # 同线程重入：外层清理尚未完成，无法等待自己。
                    # 不伪造成功（不返回 True），保守返回 False 并记 warning。
                    self.get_logger().warning("destroy_node 同线程重入，外层清理尚未完成")
                    return False
                while not self._destroy_completed:
                    condition.wait()
                return self._destroy_result
            self._destroy_started = True
            self._destroy_owner_id = current_id

        cleanup_error = None
        result = False
        try:
            if self._runtime is not None:
                try:
                    self._runtime.stop()
                except Exception as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
            # Model Sessions are owned by the unified runtime handle.  If
            # assembly failed before a SpeechDirectionRuntime was installed,
            # release the pending Session owner here as a construction-failure
            # fallback; never close Sessions a second time after handle stop.
            if self._runtime is None and self._session_resources is not None:
                try:
                    self._session_resources.close()
                except Exception as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
            if self._runtime is None:
                for resource in reversed(self._pending_backend_resources):
                    try:
                        close = getattr(resource, "close", None)
                        if callable(close):
                            close()
                    except Exception as exc:
                        if cleanup_error is None:
                            cleanup_error = exc
            # 在线节点只关闭高通量记录会话；报告生成严格留给离线 CLI。
            if self._diagnostics_recorder is not None:
                try:
                    self._diagnostics_recorder.stop()
                except Exception as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
            try:
                result = super().destroy_node()
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
            if cleanup_error is not None:
                self.get_logger().error(f"清理异常: {cleanup_error}")
            return result
        finally:
            # 结果发布与唤醒必须同处临界区，等待者不会观察到半完成状态。
            with condition:
                self._destroy_result = result
                self._destroy_completed = True
                self._destroy_owner_id = None
                condition.notify_all()


def main(args=None) -> None:
    """speech_direction_node 入口。"""
    rclpy.init(args=args)
    dependencies = build_model_service_runtime_dependencies()
    node = SpeechDirectionNode(
        registry_set=dependencies.registry_set,
        providers=dependencies.providers,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("用户中断,正在停止...")
    finally:
        node.destroy_node()
        dependencies.providers.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
