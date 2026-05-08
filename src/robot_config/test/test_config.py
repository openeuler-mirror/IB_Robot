"""Tests for robot_config package."""

from pathlib import Path

import pytest

from robot_config.config import (
    CameraConfig,
    ContractExtensionConfig,
    PeripheralConfig,
    RobotConfig,
    Ros2ControlConfig,
    VoiceASRConfig,
)
from robot_config.launch_builders.recording import get_recording_topics
from robot_config.launch_builders.voice_asr import (
    default_voice_asr_model_path,
    resolve_voice_asr_path,
)
from robot_config.loader import (
    build_contract_from_robot_config_dict,
    load_robot_config,
    load_robot_config_dict,
    load_voice_asr_config,
    validate_config,
)
from voice_asr_service.defaults import VOICE_ASR_DEFAULTS
from voice_asr_service.model_manager import (
    STREAMING_ZH_BUNDLE,
    infer_model_bundle_from_path_hint,
    resolve_model_assets,
)


def test_load_single_arm_config():
    """Test loading SO-101 single arm configuration."""
    # This test assumes the example config exists
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"

    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")

    config = load_robot_config(config_path)

    assert config.name == "so101_single_arm"
    assert config.robot_type == "so_101"
    assert config.ros2_control.hardware_plugin == "so101_hardware/SO101SystemHardware"
    assert len(config.peripherals) == 3
    assert config.voice_asr.enabled is False
    assert config.voice_asr.output_topic == "/voice_command"
    assert config.voice_asr.model_path.endswith("models/voice_asr/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23")
    assert config.voice_asr.realtime_pre_roll_seconds == 0.5
    assert not Path(config.voice_asr.model_path).is_absolute()
    assert config.voice_asr.device_name == ""
    assert config.voice_asr.device_index == 13
    assert config.voice_asr.exit_on_init_failure is True

    # Check cameras
    top_cam = config.get_camera("top")
    assert top_cam is not None
    assert top_cam.driver == "opencv"
    assert top_cam.width == 640
    assert top_cam.height == 480
    assert top_cam.fps == 30

    wrist_cam = config.get_camera("wrist")
    assert wrist_cam is not None
    assert wrist_cam.fps == 60  # Higher FPS for wrist camera


def test_load_single_arm_config_dict_preserves_launch_schema():
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"

    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")

    config = load_robot_config_dict(config_path)

    assert config["name"] == "so101_single_arm"
    assert "models" in config
    assert "control_modes" in config
    assert "joints" in config
    assert "simulation" in config
    assert config["_config_path"] == str(config_path.resolve())


def test_dict_contract_builder_matches_typed_contract_shape():
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"

    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")

    typed_config = load_robot_config(config_path)
    dict_config = load_robot_config_dict(config_path)

    typed_contract = typed_config.to_contract()
    dict_contract = build_contract_from_robot_config_dict(dict_config)

    assert dict_contract.name == typed_contract.name
    assert dict_contract.robot_type == typed_contract.robot_type
    assert len(dict_contract.observations) == len(typed_contract.observations)
    assert len(dict_contract.actions) == len(typed_contract.actions)
    assert [obs.key for obs in dict_contract.observations] == [obs.key for obs in typed_contract.observations]
    assert [act.key for act in dict_contract.actions] == [act.key for act in typed_contract.actions]
    assert dict_contract.tasks == typed_contract.tasks


def test_dict_contract_builder_uses_camera_defaults_for_missing_resize():
    contract = build_contract_from_robot_config_dict(
        {
            "name": "test_robot",
            "robot_type": "so_101",
            "peripherals": [
                {
                    "type": "camera",
                    "name": "top",
                    "height": 0,
                    "width": None,
                }
            ],
            "contract": {
                "observations": [
                    {
                        "key": "observation.images.top",
                        "topic": "/camera/top/image_raw",
                        "peripheral": "top",
                    }
                ]
            },
        }
    )

    assert contract.observations[0].image == {
        "resize": [480, 640],
        "encoding": "bgr8",
    }


def test_dict_contract_builder_warns_when_camera_lookup_fails(capsys):
    contract = build_contract_from_robot_config_dict(
        {
            "name": "test_robot",
            "robot_type": "so_101",
            "peripherals": [],
            "contract": {
                "observations": [
                    {
                        "key": "observation.images.top",
                        "topic": "/camera/top/image_raw",
                        "peripheral": "missing_camera",
                    }
                ]
            },
        }
    )
    stderr = capsys.readouterr().err

    assert contract.observations[0].type == "sensor_msgs/msg/Image"
    assert contract.observations[0].image is None
    assert "Observation 'observation.images.top' references peripheral 'missing_camera' but no camera found" in stderr


def test_dict_contract_builder_ignores_tasks_to_match_typed_contract():
    contract = build_contract_from_robot_config_dict(
        {
            "name": "test_robot",
            "robot_type": "so_101",
            "contract": {
                "tasks": [
                    {
                        "key": "task.command",
                        "topic": "/task",
                        "type": "std_msgs/msg/String",
                    }
                ]
            },
        }
    )

    assert contract.tasks == []


def test_dict_contract_builder_requires_topic_without_peripheral():
    with pytest.raises(
        ValueError,
        match="Observation 'observation.state' must specify a topic when no peripheral is set",
    ):
        build_contract_from_robot_config_dict(
            {
                "name": "test_robot",
                "robot_type": "so_101",
                "contract": {
                    "observations": [
                        {
                            "key": "observation.state",
                        }
                    ]
                },
            }
        )


def test_dict_contract_builder_allows_empty_topic_for_peripheral_observation():
    contract = build_contract_from_robot_config_dict(
        {
            "name": "test_robot",
            "robot_type": "so_101",
            "peripherals": [
                {
                    "type": "camera",
                    "name": "top",
                    "height": 480,
                    "width": 640,
                }
            ],
            "contract": {
                "observations": [
                    {
                        "key": "observation.images.top",
                        "peripheral": "top",
                    }
                ]
            },
        }
    )

    assert contract.observations[0].topic == ""
    assert contract.observations[0].type == "sensor_msgs/msg/Image"


def test_load_lekiwi_config_dict():
    config_path = Path(__file__).parent.parent / "config" / "robots" / "lekiwi.yaml"
    robot_config = load_robot_config_dict(config_path)

    assert robot_config["name"] == "lekiwi"
    assert robot_config["ros2_control"]["urdf_path"] == "$(find lekiwi_description)/urdf/base.urdf.xacro"
    assert robot_config["control_modes"]["teleop"]["controllers"] == [
        "joint_state_broadcaster",
        "imu_sensor_broadcaster",
        "base_controller",
    ]
    assert robot_config["navigation"]["default_mode"] == "full"


def test_recording_topics_follow_contract_and_peripherals():
    config_path = Path(__file__).parent.parent / "config" / "robots" / "lekiwi.yaml"
    robot_config = load_robot_config_dict(config_path)
    topics = get_recording_topics(robot_config)

    assert "/joint_states" in topics
    assert "/camera/front/image_raw" in topics
    assert "/camera/front/camera_info" in topics
    assert "/base_controller/odom" in topics
    assert "/base_controller/cmd_vel" in topics
    assert "/scan" in topics


def test_validate_valid_config():
    """Test validation of valid configuration."""
    config = RobotConfig(
        name="test_robot",
        type="so101",
        robot_type="so_101",
        ros2_control=Ros2ControlConfig(
            hardware_plugin="so101_hardware/SO101SystemHardware",
            params={"port": "/dev/ttyACM0"},
        ),
        peripherals=[
            CameraConfig(
                name="test_cam",
                driver="opencv",
                index_or_port=0,
                width=640,
                height=480,
                fps=30,
                frame_id="camera_test_frame",
            )
        ],
        contract=ContractExtensionConfig(
            observations=[],
            actions=[],
        ),
    )

    errors = validate_config(config)
    assert len(errors) == 0


def test_validate_generic_peripherals_do_not_break_camera_validation():
    config = RobotConfig(
        name="lekiwi",
        type="lekiwi",
        robot_type="lekiwi",
        ros2_control=Ros2ControlConfig(
            hardware_plugin="sts_hardware_interface/STSHardwareInterface",
            params={"port": "/dev/ttySERVO"},
        ),
        peripherals=[
            CameraConfig(
                name="front",
                driver="camera_ros",
                index_or_port=0,
                width=640,
                height=480,
                fps=15,
                frame_id="camera",
            ),
            PeripheralConfig(
                type="lidar",
                name="laser",
                driver="ldlidar",
                params={"laser_scan_topic_name": "scan"},
                frame_id="laser_frame",
            ),
        ],
        contract=ContractExtensionConfig(
            observations=[],
            actions=[],
        ),
    )

    errors = validate_config(config)
    assert len(errors) == 0


def test_validate_duplicate_camera_names():
    """Test validation catches duplicate camera names."""
    config = RobotConfig(
        name="test_robot",
        type="so101",
        robot_type="so_101",
        ros2_control=Ros2ControlConfig(
            hardware_plugin="so101_hardware/SO101SystemHardware",
            params={},
        ),
        peripherals=[
            CameraConfig(
                name="test_cam",
                driver="opencv",
                index_or_port=0,
                width=640,
                height=480,
                fps=30,
                frame_id="camera_test_frame",
            ),
            CameraConfig(
                name="test_cam",  # Duplicate name
                driver="opencv",
                index_or_port=1,
                width=640,
                height=480,
                fps=30,
                frame_id="camera_test_frame2",
            ),
        ],
        contract=ContractExtensionConfig(
            observations=[],
            actions=[],
        ),
    )

    errors = validate_config(config)
    assert len(errors) > 0
    assert any("Duplicate camera name" in error for error in errors)


def test_validate_invalid_camera_dimensions():
    """Test validation catches invalid camera dimensions."""
    config = RobotConfig(
        name="test_robot",
        type="so101",
        robot_type="so_101",
        ros2_control=Ros2ControlConfig(
            hardware_plugin="so101_hardware/SO101SystemHardware",
            params={},
        ),
        peripherals=[
            CameraConfig(
                name="test_cam",
                driver="opencv",
                index_or_port=0,
                width=0,  # Invalid
                height=480,
                fps=30,
                frame_id="camera_test_frame",
            )
        ],
        contract=ContractExtensionConfig(
            observations=[],
            actions=[],
        ),
    )

    errors = validate_config(config)
    assert len(errors) > 0
    assert any("Invalid camera dimensions" in error for error in errors)


def test_get_all_cameras():
    """Test getting all cameras from configuration."""
    config = RobotConfig(
        name="test_robot",
        type="so101",
        robot_type="so_101",
        ros2_control=Ros2ControlConfig(
            hardware_plugin="so101_hardware/SO101SystemHardware",
            params={},
        ),
        peripherals=[
            CameraConfig(
                name="cam1",
                driver="opencv",
                index_or_port=0,
                width=640,
                height=480,
                fps=30,
                frame_id="camera_cam1_frame",
            ),
            CameraConfig(
                name="cam2",
                driver="realsense",
                index_or_port="12345678",
                width=640,
                height=480,
                fps=30,
                frame_id="camera_cam2_frame",
            ),
        ],
        contract=ContractExtensionConfig(
            observations=[],
            actions=[],
        ),
    )

    cameras = config.get_all_cameras()
    assert len(cameras) == 2
    assert cameras[0].name == "cam1"
    assert cameras[1].name == "cam2"


def test_validate_voice_asr_requires_model_path_when_auto_download_is_disabled():
    """Test validation catches enabled voice ASR without a model path when auto-download is off."""
    config = RobotConfig(
        name="test_robot",
        type="so101",
        robot_type="so_101",
        ros2_control=Ros2ControlConfig(
            hardware_plugin="so101_hardware/SO101SystemHardware",
            params={},
        ),
        contract=ContractExtensionConfig(
            observations=[],
            actions=[],
        ),
    )
    config.voice_asr.enabled = True
    config.voice_asr.auto_download_model = False
    config.voice_asr.model_path = ""

    errors = validate_config(config)
    assert any("voice_asr.model_path" in error for error in errors)


def test_load_voice_asr_config_preserves_empty_model_path_for_launch_builder():
    """Test voice ASR loader remains a pure field mapper for launch-time defaulting."""
    config = load_voice_asr_config(
        {
            "enabled": True,
            "auto_download_model": True,
            "active_mode": "continuous",
            "model_type": "streaming",
        }
    )

    assert config.model_path == ""
    assert config.device_name == ""
    assert config.exit_on_init_failure is True


def test_voice_asr_launch_builder_infers_shared_default_model_path():
    """Test launch builder resolves the shared default Voice ASR model path."""
    resolved = default_voice_asr_model_path("streaming", "continuous")

    assert resolved.endswith("models/voice_asr/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23")
    assert Path(resolved).is_absolute()


def test_voice_asr_empty_model_path_auto_resolves_streaming_bundle(tmp_path):
    """Test standalone ASR keeps generic defaults while resolving runtime assets."""
    bundle_dir = tmp_path / STREAMING_ZH_BUNDLE.directory
    bundle_dir.mkdir()
    for file_name in ("tokens.txt", "encoder.onnx", "decoder.onnx", "joiner.onnx"):
        (bundle_dir / file_name).write_text("placeholder")

    resolved = resolve_model_assets(
        model_path="",
        model_type="auto",
        active_mode="continuous",
        model_root=tmp_path,
        auto_download_model=True,
    )

    assert resolved.model_path == str(bundle_dir)
    assert resolved.tokens_path == str(bundle_dir / "tokens.txt")
    assert resolved.profile == STREAMING_ZH_BUNDLE.profile


def test_resolve_voice_asr_path_uses_workspace_root_for_relative_paths():
    """Test voice ASR relative paths resolve from the workspace root."""
    resolved = resolve_voice_asr_path("models/voice_asr/demo-bundle")

    assert Path(resolved).is_absolute()
    assert resolved.endswith("models/voice_asr/demo-bundle")


def test_model_hint_inference_uses_model_manager_ssot():
    """Test path-hint inference delegates to model_manager."""
    bundle = infer_model_bundle_from_path_hint("models/voice_asr/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23")

    assert bundle is not None
    assert bundle.profile == STREAMING_ZH_BUNDLE.profile


def test_voice_asr_runtime_defaults_match_robot_config_defaults():
    """Test robot_config defaults stay aligned with runtime Voice ASR defaults."""
    config_defaults = VoiceASRConfig()

    assert config_defaults.enabled == VOICE_ASR_DEFAULTS["enabled"]
    assert config_defaults.auto_download_model == VOICE_ASR_DEFAULTS["auto_download_model"]
    assert config_defaults.active_mode == VOICE_ASR_DEFAULTS["active_mode"]
    assert config_defaults.language == VOICE_ASR_DEFAULTS["language"]
    assert config_defaults.model_path == VOICE_ASR_DEFAULTS["model_path"]
    assert config_defaults.tokens_path == VOICE_ASR_DEFAULTS["tokens_path"]
    assert config_defaults.provider == VOICE_ASR_DEFAULTS["provider"]
    assert config_defaults.model_type == VOICE_ASR_DEFAULTS["model_type"]
    assert config_defaults.max_recording_duration == VOICE_ASR_DEFAULTS["max_recording_duration"]
    assert config_defaults.vad_sensitivity == VOICE_ASR_DEFAULTS["vad_sensitivity"]
    assert config_defaults.realtime_pre_roll_seconds == VOICE_ASR_DEFAULTS["realtime_pre_roll_seconds"]
    assert config_defaults.publish_partial == VOICE_ASR_DEFAULTS["publish_partial"]
    assert config_defaults.output_topic == VOICE_ASR_DEFAULTS["output_topic"]
    assert config_defaults.sample_rate == VOICE_ASR_DEFAULTS["sample_rate"]
    assert config_defaults.chunk_size == VOICE_ASR_DEFAULTS["chunk_size"]
    assert config_defaults.buffer_seconds == VOICE_ASR_DEFAULTS["buffer_seconds"]
    assert config_defaults.device_index == VOICE_ASR_DEFAULTS["device_index"]
    assert config_defaults.device_name == VOICE_ASR_DEFAULTS["device_name"]
    assert config_defaults.exit_on_init_failure == VOICE_ASR_DEFAULTS["exit_on_init_failure"]
