"""Configuration dataclasses for unified robot configuration."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from robot_config.observation_transport import (
    ObservationTransportSpec,
    require_valid_observation_transports,
    resolve_observation_transport,
)

if TYPE_CHECKING:
    from robot_config.contract_utils import Contract
    from robot_config.perception_runtime_config import PerceptionRuntimeConfig


@dataclass
class Ros2ControlConfig:
    """ros2_control hardware configuration."""

    hardware_plugin: str
    params: dict[str, Any] = field(default_factory=dict)
    urdf_path: str | None = None


@dataclass
class CameraConfig:
    """Camera peripheral configuration.

    This configuration is used to generate launch parameters for
    existing ROS2 camera drivers:
    - usb_cam for USB cameras (driver: opencv)
    - realsense2_camera for RealSense D400 series (driver: realsense)
    """

    name: str
    driver: str  # opencv, realsense, etc.
    index_or_port: str | int  # USB index for opencv, serial/port for realsense
    width: int
    height: int
    fps: int
    frame_id: str
    optical_frame_id: str | None = None
    camera_info_url: str | None = None  # Path to calibration file
    pixel_format: str = "bgr8"  # bgr8, rgb8, etc.

    # Realsense-specific parameters
    depth_width: int | None = None
    depth_height: int | None = None
    depth_fps: int | None = None
    enable_pointcloud: bool = False
    enable_sync: bool = True
    align_depth: bool = False
    direct_topic_remap: bool = False

    # USB camera specific parameters
    brightness: int | None = None
    contrast: int | None = None
    saturation: int | None = None
    sharpness: int | None = None

    # Transform (parent to camera frame)
    transform: dict[str, float] | None = None  # {x, y, z, roll, pitch, yaw}


@dataclass
class PeripheralConfig:
    """Generic peripheral device configuration.

    Use CameraConfig for camera-specific configuration.
    """

    type: str  # camera, microphone, etc.
    name: str
    driver: str
    params: dict[str, Any] = field(default_factory=dict)
    frame_id: str | None = None


@dataclass
class ContractObservation:
    """Contract observation reference."""

    key: str
    topic: str
    type: str | None = None  # Explicit ROS message type (e.g. sensor_msgs/msg/PointCloud2)
    peripheral: str | None = None  # References peripheral by name
    selector: dict[str, Any] | None = None
    image: dict[str, Any] | None = None
    align: dict[str, Any] | None = None
    qos: dict[str, Any] | None = None
    transport: ObservationTransportSpec | None = None


@dataclass
class ContractAction:
    """Contract action definition."""

    key: str
    publish: dict[str, Any]
    selector: dict[str, Any] | None = None
    from_tensor: dict[str, Any] | None = None
    safety_behavior: str = "zeros"


@dataclass
class ContractExtensionConfig:
    """Rosetta contract extension configuration."""

    base_contract: str | None = None
    observations: list[ContractObservation] = field(default_factory=list)
    actions: list[ContractAction] = field(default_factory=list)
    rate_hz: float = 20.0
    max_duration_s: float = 30.0


@dataclass
class SkillGatewayRuntimeConfig:
    """Typed runtime settings consumed by the skill Gateway."""

    status_service: str = "/embodied/get_skill_gateway_status"
    required_control_mode: str = ""
    # Retain the YAML-declared control_modes keys so validate_config can fully
    # SSOT-check required_control_mode membership instead of rebuilding the set
    # from the module-level _SUPPORTED_CONTROL_MODES constant.
    control_modes: tuple[str, ...] = ()
    default_skill_timeout_sec: float = 120.0
    robot_state_freshness_sec: float = 0.5
    task_budget_sec: float = 180.0
    rpc_timeout_sec: float = 5.0


@dataclass
class SoundOrientationConfig:
    """Fixed-trigger idle sound-orientation behavior."""

    enabled: bool = False
    trigger_phrases: tuple[str, ...] = ("转向我",)
    direction_topic: str = "/voice/speech_direction"
    command_topic: str = "/voice_command"
    skill_name: str = "nav_turn"
    direction_frame: str = "base_link"
    deadband_deg: float = 15.0
    max_direction_age_sec: float = 1.3
    direction_wait_sec: float = 0.5
    cooldown_sec: float = 1.5
    max_turn_deg: float = 180.0
    turn_timeout_sec: float = 10.0
    action_acceptance_timeout_sec: float = 2.0
    status_retry_sec: float = 0.5
    reset_status_max_age_sec: float = 2.0


@dataclass
class EmbodiedConfig:
    """Minimum embodied claw closure configuration."""

    enabled: bool = False
    entry_mode: str = "hermes"
    agent: dict[str, Any] = field(default_factory=dict)
    debug_tracing: bool = True
    task_input_topic: str = "/voice_command"
    task_command_topic: str = "/embodied/task_command"
    planned_task_topic: str = "/embodied/planned_task"
    status_topic: str = "/embodied/task_status"
    skill_action_name: str = "/embodied/execute_skill"
    primitive_action_name: str = "/embodied/execute_primitive"
    validate_skill_service: str = "/embodied/validate_skill"
    validate_primitive_service: str = "/embodied/validate_primitive"
    skill_gateway_status_service: str = "/embodied/get_skill_gateway_status"
    skill_catalog_source_mode: str = "installed"
    skill_catalog_source_root: str = ""
    skill_catalog_profile: str = ""
    start_visual_game_service: str = "/embodied/start_visual_game"
    get_visual_game_result_service: str = "/embodied/get_visual_game_result"
    visual_game_event_topic: str = "/embodied/visual_game_events"
    visual_game_result_capacity: int = 128
    default_target_name: str = "demo_object"
    default_place_name: str = "tray_right"
    skill_timeout_sec: float = 120.0
    primitive_timeout_sec: float = 5.0
    primitive_wait_sec: float = 1.0
    timeouts: dict[str, Any] = field(default_factory=dict)
    relative_motion_step_m: float = 0.03
    relative_motion_reference_frame: str = "base"
    relative_motion_direction_mapping: dict[str, Any] = field(default_factory=dict)
    perception: dict[str, Any] = field(default_factory=dict)
    visual_games: dict[str, Any] = field(default_factory=dict)
    imitate_human_motion: dict[str, Any] = field(default_factory=dict)
    idle_behaviors: dict[str, Any] = field(default_factory=dict)
    sound_orientation: SoundOrientationConfig = field(default_factory=SoundOrientationConfig)
    gripper_open_position: float = 1.0
    gripper_closed_position: float = 0.0
    skill_templates: dict[str, Any] = field(default_factory=dict)
    named_poses: dict[str, Any] = field(default_factory=dict)
    named_targets: dict[str, Any] = field(default_factory=dict)
    workspace: dict[str, Any] = field(default_factory=dict)


@dataclass
class VoiceASRConfig:
    """Voice ASR node configuration managed by robot_config."""

    enabled: bool = False
    active_mode: str = "continuous"
    language: str = "zh"
    bundle_path: str = "models/voice_asr/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23"
    deployment: str = "torch_cpu"
    max_recording_duration: float = 10.0
    vad_sensitivity: float = 0.6
    vad_bundle_path: str = "models/silero-vad"
    vad_deployment: str = "torch_cpu"
    realtime_pre_roll_seconds: float = 0.5
    publish_partial: bool = True
    output_topic: str = "/voice_command"
    sample_rate: int = 16000
    chunk_size: int = 512
    buffer_seconds: float = 5.0
    audio_input_channel: int = 1
    exit_on_init_failure: bool = True


@dataclass
class AudioIOConfig:
    """Cross-platform audio_common device-owner configuration."""

    enabled: bool = False
    microphone: str = ""
    capture_topic: str = "/audio/capture"
    capture_stamped_topic: str = "/audio/capture_stamped"
    audio_info_topic: str = "/audio/info"
    playback_topic: str = "/audio/play"
    playback_device: str = ""
    playback_channels: int = 1
    playback_sample_rate: int = 24000
    playback_sample_format: str = "S16LE"


@dataclass
class SpeechDirectionConfig:
    """Typed speech-direction configuration managed by robot_config."""

    enabled: bool = False
    profile: str = "ascend_310p"
    microphone: str = ""
    config_file: str = ""
    profiles_file: str = ""
    models_root: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class VoiceTTSConfig:
    """Typed Voice TTS service configuration managed by robot_config."""

    enabled: bool = False
    bundle_path: str = "models/zipvoice"
    deployment: str = ""
    service_name: str = "/voice_tts/synthesize"
    playback_service_name: str = "/voice_tts/play"
    playback_timeout_sec: float = 300.0
    synthesis_timeout_sec: float = 90.0
    prompt_profile: str = "default"
    segment_max_chars: int = 200
    segment_pause_ms: int = 150
    max_request_chars: int = 4000
    max_prompt_audio_bytes: int = 10 * 1024 * 1024
    max_prompt_duration_sec: float = 30.0
    max_segments: int = 32
    max_response_audio_bytes: int = 64 * 1024 * 1024
    tts_timeout_sec: float = 15.0
    device_id: int = 0
    exit_on_init_failure: bool = True


@dataclass
class SemanticMappingConfig:
    """Standalone semantic mapping configuration managed by robot_config."""

    enabled: bool = False
    camera: dict[str, Any] = field(default_factory=dict)
    slam: dict[str, Any] = field(default_factory=dict)
    perception: dict[str, Any] = field(default_factory=dict)
    persistence: dict[str, Any] = field(default_factory=dict)
    filtering: dict[str, Any] = field(default_factory=dict)
    queue: dict[str, Any] = field(default_factory=dict)
    lifecycle: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)
    label_refinement: dict[str, Any] = field(default_factory=dict)
    target_watch: dict[str, Any] = field(default_factory=dict)
    interfaces: dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotConfig:
    """Unified robot configuration.

    This is the main configuration class that defines:
    - Robot metadata (name, type, robot_type for LeRobot)
    - ros2_control configuration (joints/motors)
    - Peripheral devices (cameras, etc.)
    - Contract extensions (ML I/O mapping)
    """

    name: str
    type: str
    robot_type: str  # For LeRobot dataset metadata (e.g., so_101)
    ros2_control: Ros2ControlConfig
    peripherals: list[CameraConfig | PeripheralConfig] = field(default_factory=list)
    contract: ContractExtensionConfig = field(default_factory=ContractExtensionConfig)
    voice_asr: VoiceASRConfig = field(default_factory=VoiceASRConfig)
    audio_io: AudioIOConfig = field(default_factory=AudioIOConfig)
    speech_direction: SpeechDirectionConfig = field(default_factory=SpeechDirectionConfig)
    voice_tts: VoiceTTSConfig = field(default_factory=VoiceTTSConfig)
    embodied: EmbodiedConfig = field(default_factory=EmbodiedConfig)
    skill_gateway: SkillGatewayRuntimeConfig = field(default_factory=SkillGatewayRuntimeConfig)
    semantic_mapping: SemanticMappingConfig = field(default_factory=SemanticMappingConfig)
    perception_services: "PerceptionRuntimeConfig | None" = None
    placement_execution: dict[str, Any] = field(default_factory=dict)

    def get_camera(self, name: str) -> CameraConfig | None:
        """Get camera configuration by name."""
        for cam in self.peripherals:
            if isinstance(cam, CameraConfig) and cam.name == name:
                return cam
        return None

    def get_all_cameras(self) -> list[CameraConfig]:
        """Get all camera configurations."""
        return [p for p in self.peripherals if isinstance(p, CameraConfig)]

    def to_contract(self) -> "Contract":
        """Generate a Contract dataclass directly from this robot_config.

        This establishes RobotConfig as the Single Source of Truth for I/O mappings.
        """
        from robot_config.contract_utils import ActionSpec, Contract, ObservationSpec, _as_align

        obs_specs = []
        for obs in self.contract.observations:
            # Resolve peripheral if referenced
            image_meta = obs.image
            cam = None
            # Prefer explicit type from YAML; fall back to inference
            topic_type = obs.type or "sensor_msgs/msg/JointState"

            if obs.peripheral:
                cam = self.get_camera(obs.peripheral)
                if cam:
                    topic_type = "sensor_msgs/msg/Image"
                    if not image_meta:
                        image_meta = {"resize": [cam.height, cam.width], "encoding": cam.pixel_format}
                else:
                    topic_type = "sensor_msgs/msg/Image"

            transport = resolve_observation_transport(
                obs.transport,
                image=image_meta,
                camera_width=cam.width if cam else None,
                camera_height=cam.height if cam else None,
                camera_fps=cam.fps if cam else None,
            )

            obs_specs.append(
                ObservationSpec(
                    key=obs.key,
                    topic=obs.topic,
                    type=topic_type,
                    selector=obs.selector,
                    image=image_meta,
                    align=_as_align(obs.align),
                    qos=obs.qos,
                    transport=transport,
                )
            )

        act_specs = []
        for act in self.contract.actions:
            pub = act.publish
            sb = str(act.safety_behavior).lower().strip()
            if sb not in ("zeros", "hold"):
                sb = "zeros"
            act_specs.append(
                ActionSpec(
                    key=act.key,
                    publish_topic=pub.get("topic", ""),
                    type=pub.get("type", ""),
                    selector=act.selector,
                    from_tensor=act.from_tensor,
                    publish_qos=pub.get("qos"),
                    publish_strategy=pub.get("strategy"),
                    safety_behavior=sb,
                )
            )

        # Assuming no tasks for now as they are not explicitly typed in ContractExtensionConfig
        task_specs = []

        contract = Contract(
            name=self.name,
            version=1,
            rate_hz=float(self.contract.rate_hz),
            max_duration_s=float(self.contract.max_duration_s),
            observations=obs_specs,
            actions=act_specs,
            tasks=task_specs,
            recording={"storage": "mcap"},
            robot_type=self.robot_type,
            timestamp_source="receive",
            process={},
        )
        require_valid_observation_transports(contract.observations)
        return contract
