"""Configuration dataclasses for unified robot configuration."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robot_config.contract_utils import Contract


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
class VoiceASRConfig:
    """Voice ASR node configuration managed by robot_config."""

    enabled: bool = False
    auto_download_model: bool = True
    active_mode: str = "continuous"
    language: str = "zh"
    model_path: str = ""
    tokens_path: str = ""
    provider: str = "cpu"
    model_type: str = "auto"
    max_recording_duration: float = 10.0
    vad_sensitivity: float = 0.6
    realtime_pre_roll_seconds: float = 0.5
    publish_partial: bool = True
    output_topic: str = "/voice_command"
    sample_rate: int = 16000
    chunk_size: int = 512
    buffer_seconds: float = 5.0
    device_index: int = -1
    device_name: str = ""
    exit_on_init_failure: bool = True


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

    def get_camera(self, name: str) -> CameraConfig | None:
        """Get camera configuration by name."""
        for cam in self.peripherals:
            if cam.name == name:
                return cam
        return None

    def get_all_cameras(self) -> list[CameraConfig]:
        """Get all camera configurations."""
        return [p for p in self.peripherals if isinstance(p, CameraConfig)]

    def to_contract(self) -> "Contract":
        """Generate a Contract dataclass directly from this robot_config.

        This establishes RobotConfig as the Single Source of Truth for I/O mappings.
        """
        from robot_config.contract_utils import (
            ActionSpec,
            AlignSpec,
            Contract,
            ObservationSpec,
        )

        def _as_align(d):
            if not d:
                return None
            return AlignSpec(
                strategy=str(d.get("strategy", "hold")).lower(),
                tol_ms=int(d.get("tol_ms", 0)),
                stamp=str(d.get("stamp", "receive")).lower(),
            )

        obs_specs = []
        for obs in self.contract.observations:
            # Resolve peripheral if referenced
            image_meta = obs.image
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

            obs_specs.append(
                ObservationSpec(
                    key=obs.key,
                    topic=obs.topic,
                    type=topic_type,
                    selector=obs.selector,
                    image=image_meta,
                    align=_as_align(obs.align),
                    qos=obs.qos,
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

        return Contract(
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
