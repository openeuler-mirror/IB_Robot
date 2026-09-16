"""Tests for robot_config package."""

import logging
from pathlib import Path

import pytest
import yaml

from robot_config.config import (
    CameraConfig,
    ContractExtensionConfig,
    EmbodiedConfig,
    PeripheralConfig,
    RobotConfig,
    Ros2ControlConfig,
    SkillGatewayRuntimeConfig,
    VoiceASRConfig,
    VoiceTTSConfig,
)
from robot_config.contract_utils import contract_fingerprint, iter_specs
from robot_config.launch_builders.recording import get_recording_topics
from robot_config.launch_builders.voice_asr import (
    resolve_voice_asr_path,
)
from robot_config.loader import (
    _warn_unbound_serial_cameras,
    build_contract_from_robot_config_dict,
    load_embodied_config,
    load_robot_config,
    load_robot_config_dict,
    load_voice_asr_config,
    load_voice_tts_config,
    validate_config,
)
from robot_config.timeout_policy import DEFAULT_EMBODIED_TIMEOUT_POLICY, resolve_embodied_timeout_policy
from robot_config.utils import resolve_lerobot_norm_mode
from voice_asr_service.defaults import VOICE_ASR_DEFAULTS
from voice_tts_service.defaults import VOICE_TTS_DEFAULTS

HUMBLE_FLOAT32_MAX = 3.402823466e38
IEEE_FLOAT32_MAX = 3.4028234663852886e38
HUMBLE_FLOAT32_OVERFLOW = 3.4028234661e38


def _valid_capability_parameters(properties=None, required=None):
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {} if properties is None else properties,
        "required": [] if required is None else required,
    }


def _valid_skill_capability(parameters=None, required_control_mode="moveit_planning"):
    return {
        "schema_version": 1,
        "summary": "Open the gripper.",
        "domain": "manipulation",
        "moves_robot": True,
        "required_control_mode": required_control_mode,
        "parameters": _valid_capability_parameters() if parameters is None else parameters,
        "recovery_policy": "never_retry",
    }


def _typed_config_with_skill_gateway_mode(required_control_mode):
    return RobotConfig(
        name="test_robot",
        type="so101",
        robot_type="so_101",
        ros2_control=Ros2ControlConfig(
            hardware_plugin="so101_hardware/SO101SystemHardware",
            params={},
        ),
        contract=ContractExtensionConfig(observations=[], actions=[]),
        embodied=EmbodiedConfig(
            enabled=True,
            default_place_name="home",
            skill_templates={
                "open_gripper": {
                    "description": {
                        "summary": "Open the gripper.",
                        "category": "gripper",
                        "when_to_use": ["release an object"],
                    },
                    "capability": _valid_skill_capability(required_control_mode="moveit_planning"),
                    "primitive_sequence": [{"primitive_name": "open_gripper"}],
                }
            },
            named_poses={"home": {}, "observe_table": {}, "zero": {}},
        ),
        skill_gateway=SkillGatewayRuntimeConfig(required_control_mode=required_control_mode),
    )


def _write_skill_capability_config(tmp_path, capability, *, global_control_mode="moveit_planning"):
    config_path = tmp_path / "robot.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "robot": {
                    "name": "skill_robot",
                    "control_modes": {"moveit_planning": {}, "teleop": {}},
                    "skill_required_control_mode": global_control_mode,
                    "embodied": {
                        "skill_templates": {
                            "open_gripper": {
                                "description": {
                                    "summary": "Open the gripper.",
                                    "category": "motion",
                                    "when_to_use": ["release an object"],
                                },
                                "capability": capability,
                                "primitive_sequence": [{"primitive_name": "open_gripper"}],
                            }
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_default_skill_timeout_is_two_minutes():
    assert EmbodiedConfig().skill_timeout_sec == 120.0
    assert load_embodied_config({}).skill_timeout_sec == 120.0


def test_loader_rejects_removed_inline_skill_templates(tmp_path):
    config_path = tmp_path / "robot.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "robot": {
                    "name": "skill_robot",
                    "control_modes": {"moveit_planning": {}},
                    "embodied": {
                        "skill_templates": {
                            "open_gripper": {
                                "description": {
                                    "summary": "Open the gripper.",
                                    "category": "motion",
                                    "when_to_use": ["release an object"],
                                },
                                "primitive_sequence": [{"primitive_name": "open_gripper"}],
                            }
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="skill_templates is removed"):
        load_robot_config_dict(config_path)


@pytest.mark.parametrize(
    ("capability", "expected_error"),
    [
        (None, "embodied.skill_templates.open_gripper.capability is required"),
        (
            {
                "schema_version": 2,
                "summary": "Open the gripper.",
                "domain": "manipulation",
                "moves_robot": True,
                "required_control_mode": "moveit_planning",
                "parameters": {},
                "recovery_policy": "never_retry",
            },
            "embodied.skill_templates.open_gripper.capability.schema_version must equal 1",
        ),
        (
            {
                "schema_version": 1,
                "domain": "manipulation",
                "moves_robot": True,
                "required_control_mode": "moveit_planning",
                "parameters": {},
                "recovery_policy": "never_retry",
            },
            "embodied.skill_templates.open_gripper.capability.summary is required",
        ),
        (
            {
                "schema_version": 1,
                "summary": "Open the gripper.",
                "domain": "",
                "moves_robot": True,
                "required_control_mode": "moveit_planning",
                "parameters": {},
                "recovery_policy": "never_retry",
            },
            "embodied.skill_templates.open_gripper.capability.domain must be a non-empty string",
        ),
        (
            {
                "schema_version": 1,
                "summary": "Open the gripper.",
                "domain": "manipulation",
                "moves_robot": "true",
                "required_control_mode": "moveit_planning",
                "parameters": {},
                "recovery_policy": "never_retry",
            },
            "embodied.skill_templates.open_gripper.capability.moves_robot must be a boolean",
        ),
        (
            {
                "schema_version": 1,
                "summary": "Open the gripper.",
                "domain": "manipulation",
                "moves_robot": True,
                "required_control_mode": "unsupported",
                "parameters": {},
                "recovery_policy": "never_retry",
            },
            "embodied.skill_templates.open_gripper.capability.required_control_mode must be one of",
        ),
        (
            {
                "schema_version": 1,
                "summary": "Open the gripper.",
                "domain": "manipulation",
                "moves_robot": True,
                "required_control_mode": "moveit_planning",
                "parameters": [],
                "recovery_policy": "never_retry",
            },
            "embodied.skill_templates.open_gripper.capability.parameters must be a mapping",
        ),
        (
            {
                "schema_version": 1,
                "summary": "Open the gripper.",
                "domain": "manipulation",
                "moves_robot": True,
                "required_control_mode": "moveit_planning",
                "parameters": {},
                "recovery_policy": "retry_automatically",
            },
            "embodied.skill_templates.open_gripper.capability.recovery_policy must be one of",
        ),
        (
            {
                "schema_version": 1,
                "summary": "Open the gripper.",
                "domain": "manipulation",
                "moves_robot": True,
                "required_control_mode": "moveit_planning",
                "parameters": {},
                "recovery_policy": [],
            },
            "embodied.skill_templates.open_gripper.capability.recovery_policy must be a string",
        ),
    ],
)
def test_loader_validates_enabled_skill_capability_metadata(tmp_path, capability, expected_error):
    template = {
        "description": {
            "summary": "Open the gripper.",
            "category": "motion",
            "when_to_use": ["release an object"],
        },
        "primitive_sequence": [{"primitive_name": "open_gripper"}],
    }
    if capability is not None:
        template["capability"] = capability
    config_path = tmp_path / "robot.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "robot": {
                    "name": "skill_robot",
                    "control_modes": {"moveit_planning": {}},
                    "skill_required_control_mode": "moveit_planning",
                    "embodied": {"skill_templates": {"open_gripper": template}},
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_robot_config_dict(config_path)

    assert "embodied.skill_templates is removed" in str(exc_info.value)


def test_loader_allows_robot_without_embodied_skill_templates_to_omit_gateway_control_mode(tmp_path):
    config_path = tmp_path / "robot.yaml"
    config_path.write_text(yaml.safe_dump({"robot": {"name": "plain_robot"}}), encoding="utf-8")

    config = load_robot_config_dict(config_path)

    assert config["name"] == "plain_robot"
    assert "skill_required_control_mode" not in config


@pytest.mark.parametrize(
    ("parameters", "expected_error"),
    [
        (
            {"type": "object", "additionalProperties": False, "required": []},
            "embodied.skill_templates.open_gripper.capability.parameters.properties must be a mapping",
        ),
        (
            _valid_capability_parameters() | {"properties": []},
            "embodied.skill_templates.open_gripper.capability.parameters.properties must be a mapping",
        ),
        (
            _valid_capability_parameters() | {"private": "value"},
            "embodied.skill_templates.open_gripper.capability.parameters contains unsupported key 'private'",
        ),
        (
            _valid_capability_parameters(properties={"private": {"type": "string"}}),
            "embodied.skill_templates.open_gripper.capability.parameters.properties contains unsupported property 'private'",
        ),
        (
            _valid_capability_parameters(properties={"target_name": "string"}),
            "embodied.skill_templates.open_gripper.capability.parameters.properties.target_name must be a mapping",
        ),
        (
            _valid_capability_parameters(required=["target_name"]),
            "embodied.skill_templates.open_gripper.capability.parameters.required[0] must reference a property",
        ),
        (
            _valid_capability_parameters(
                properties={"target_name": {"type": "string"}}, required=["target_name", "target_name"]
            ),
            "embodied.skill_templates.open_gripper.capability.parameters.required entries must be unique",
        ),
        (
            _valid_capability_parameters(properties={"target_name": {"type": "number"}}),
            "embodied.skill_templates.open_gripper.capability.parameters.properties.target_name.type must be 'string'",
        ),
        (
            _valid_capability_parameters(properties={"target_name": {"type": "string"}}),
            "embodied.skill_templates.open_gripper.capability.parameters.properties.target_name.enum must be a non-empty list",
        ),
        (
            _valid_capability_parameters(properties={"target_name": {"type": "string", "private": "value"}}),
            "embodied.skill_templates.open_gripper.capability.parameters.properties.target_name contains unsupported key 'private'",
        ),
        (
            _valid_capability_parameters(properties={"target_name": {"type": "string", "enum": []}}),
            "embodied.skill_templates.open_gripper.capability.parameters.properties.target_name.enum must be a non-empty list",
        ),
        (
            _valid_capability_parameters(properties={"motion_direction": {"type": "string", "enum": ["north"]}}),
            "embodied.skill_templates.open_gripper.capability.parameters.properties.motion_direction.enum contains unsupported direction",
        ),
        (
            _valid_capability_parameters(
                properties={"motion_distance": {"type": "string", "exclusiveMinimum": 0, "unit": "meters"}}
            ),
            "embodied.skill_templates.open_gripper.capability.parameters.properties.motion_distance.type must be 'number'",
        ),
        (
            _valid_capability_parameters(
                properties={"motion_distance": {"type": "number", "exclusiveMinimum": -1, "unit": "meters"}}
            ),
            "embodied.skill_templates.open_gripper.capability.parameters.properties.motion_distance.exclusiveMinimum must equal 0",
        ),
        (
            _valid_capability_parameters(
                properties={"motion_distance": {"type": "number", "exclusiveMinimum": 0, "unit": "radians"}}
            ),
            "embodied.skill_templates.open_gripper.capability.parameters.properties.motion_distance.unit must be one of",
        ),
    ],
)
def test_loader_validates_capability_parameter_schema(tmp_path, parameters, expected_error):
    config_path = _write_skill_capability_config(tmp_path, _valid_skill_capability(parameters))

    with pytest.raises(ValueError) as exc_info:
        load_robot_config_dict(config_path)

    assert "embodied.skill_templates is removed" in str(exc_info.value)


def test_loader_requires_capability_control_mode_to_match_gateway_mode(tmp_path):
    config_path = _write_skill_capability_config(
        tmp_path,
        _valid_skill_capability(required_control_mode="teleop"),
    )

    with pytest.raises(ValueError) as exc_info:
        load_robot_config_dict(config_path)

    assert "embodied.skill_templates is removed" in str(exc_info.value)


def test_skill_gateway_interfaces_are_registered_with_rosidl():
    workspace_root = Path(__file__).parents[3]
    message_path = workspace_root / "src" / "ibrobot_msgs" / "msg" / "SkillCapabilityStatus.msg"
    service_path = workspace_root / "src" / "ibrobot_msgs" / "srv" / "GetSkillGatewayStatus.srv"
    cmake_path = workspace_root / "src" / "ibrobot_msgs" / "CMakeLists.txt"

    assert message_path.is_file()
    assert service_path.is_file()
    message_contents = message_path.read_text(encoding="utf-8")
    service_contents = service_path.read_text(encoding="utf-8")
    for field in (
        "uint32 schema_version",
        "string name",
        "string semantic_level",
        "bool planner_visible",
        "bool ready",
    ):
        assert field in message_contents
    for field in (
        "uint32 schema_version",
        "string task_id",
        "string capability_digest",
        "string registry_epoch",
        "string registry_digest",
        "bool control_plane_ready",
        "SkillCapabilityStatus[] capabilities",
    ):
        assert field in service_contents

    cmake_contents = cmake_path.read_text(encoding="utf-8")
    assert '"msg/SkillCapabilityStatus.msg"' in cmake_contents
    assert '"srv/GetSkillGatewayStatus.srv"' in cmake_contents


@pytest.mark.parametrize("invalid_value", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_timeout_policy_rejects_non_positive_and_non_finite_values(invalid_value):
    timeout_names = (*DEFAULT_EMBODIED_TIMEOUT_POLICY, "default_skill_timeout_sec", "robot_state_freshness_sec")

    for timeout_name in timeout_names:
        with pytest.raises(ValueError, match=timeout_name):
            resolve_embodied_timeout_policy({"timeouts": {timeout_name: invalid_value}})


def test_timeout_policy_uses_legacy_execution_values_and_rejects_default_over_budget():
    policy = resolve_embodied_timeout_policy(
        {
            "execution": {"skill_timeout_sec": 12.0, "primitive_wait_sec": 2.0},
        }
    )

    assert policy["default_skill_timeout_sec"] == 12.0
    assert policy["gripper_settle_sec"] == 2.0

    with pytest.raises(ValueError, match="default_skill_timeout_sec.*task_budget_sec"):
        resolve_embodied_timeout_policy(
            {
                "timeouts": {
                    "default_skill_timeout_sec": 11.0,
                    "task_budget_sec": 10.0,
                }
            }
        )


@pytest.mark.parametrize(
    "timeout_name",
    [
        "default_skill_timeout_sec",
        "task_budget_sec",
        "rpc_timeout_sec",
        "robot_state_freshness_sec",
        "visual_game_timeout_sec",
    ],
)
def test_timeout_policy_rejects_gateway_timeout_values_outside_float32_range(timeout_name):
    timeouts = {timeout_name: 1e39}
    if timeout_name == "default_skill_timeout_sec":
        timeouts["task_budget_sec"] = HUMBLE_FLOAT32_MAX

    with pytest.raises(ValueError, match=f"{timeout_name}.*float32"):
        resolve_embodied_timeout_policy({"timeouts": timeouts})


def test_timeout_policy_accepts_humble_float32_maximum_for_gateway_runtime_timeouts():
    policy = resolve_embodied_timeout_policy(
        {
            "timeouts": {
                "default_skill_timeout_sec": HUMBLE_FLOAT32_MAX,
                "task_budget_sec": HUMBLE_FLOAT32_MAX,
                "rpc_timeout_sec": HUMBLE_FLOAT32_MAX,
                "robot_state_freshness_sec": HUMBLE_FLOAT32_MAX,
                "visual_game_timeout_sec": HUMBLE_FLOAT32_MAX,
            }
        }
    )

    assert policy["default_skill_timeout_sec"] == HUMBLE_FLOAT32_MAX
    assert policy["task_budget_sec"] == HUMBLE_FLOAT32_MAX
    assert policy["rpc_timeout_sec"] == HUMBLE_FLOAT32_MAX
    assert policy["robot_state_freshness_sec"] == HUMBLE_FLOAT32_MAX
    assert policy["visual_game_timeout_sec"] == HUMBLE_FLOAT32_MAX


def test_visual_game_timeout_boundary_matches_scene_request_setter():
    from ibrobot_msgs.msg import SceneAnalysisRequest

    request = SceneAnalysisRequest()
    policy = resolve_embodied_timeout_policy({"timeouts": {"visual_game_timeout_sec": HUMBLE_FLOAT32_MAX}})
    request.timeout_sec = policy["visual_game_timeout_sec"]

    with pytest.raises(AssertionError):
        request.timeout_sec = IEEE_FLOAT32_MAX
    with pytest.raises(ValueError, match="visual_game_timeout_sec.*float32"):
        resolve_embodied_timeout_policy({"timeouts": {"visual_game_timeout_sec": HUMBLE_FLOAT32_OVERFLOW}})


@pytest.mark.parametrize("timeout_name", ["default_skill_timeout_sec", "task_budget_sec", "rpc_timeout_sec"])
def test_gateway_timeout_policy_boundary_matches_humble_response_setters(timeout_name):
    from ibrobot_msgs.srv import GetSkillGatewayStatus

    policy = resolve_embodied_timeout_policy(
        {
            "timeouts": {
                "default_skill_timeout_sec": HUMBLE_FLOAT32_MAX,
                "task_budget_sec": HUMBLE_FLOAT32_MAX,
                "rpc_timeout_sec": HUMBLE_FLOAT32_MAX,
            }
        }
    )
    response = GetSkillGatewayStatus.Response()

    for field_name in ("default_skill_timeout_sec", "task_budget_sec", "rpc_timeout_sec"):
        setattr(response, field_name, policy[field_name])

    with pytest.raises(AssertionError):
        setattr(response, timeout_name, IEEE_FLOAT32_MAX)
    with pytest.raises(ValueError, match=f"{timeout_name}.*float32"):
        resolve_embodied_timeout_policy({"timeouts": {timeout_name: HUMBLE_FLOAT32_OVERFLOW}})


def test_robot_state_freshness_defaults_independently_from_perception_scene_freshness():
    policy = resolve_embodied_timeout_policy({"perception": {"scene_sources": {"max_scene_age_sec": 10.0}}})

    assert policy["scene_freshness_sec"] == 10.0
    assert policy["robot_state_freshness_sec"] == 0.5


def test_load_single_arm_config():
    """Test loading SO-101 single arm configuration."""
    # This test assumes the example config exists
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm_legacy.yaml"

    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")

    config = load_robot_config(config_path)

    assert config.name == "so101_single_arm"
    assert config.robot_type == "so_101"
    assert config.ros2_control.hardware_plugin == "so101_hardware/SO101SystemHardware"
    assert len(config.peripherals) == 3
    assert config.voice_asr.enabled is False
    assert config.voice_asr.output_topic == "/voice_command"
    assert config.voice_asr.bundle_path == "models/voice_asr/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23"
    assert config.voice_asr.realtime_pre_roll_seconds == 0.5
    assert not Path(config.voice_asr.bundle_path).is_absolute()
    assert config.voice_asr.exit_on_init_failure is True
    assert config.voice_tts.enabled is True
    assert config.voice_tts.bundle_path == "models/zipvoice"
    assert config.voice_tts.deployment == "ubuntu_onnx"
    assert config.voice_tts.service_name == "/voice_tts/synthesize"
    assert config.voice_tts.playback_service_name == "/voice_tts/play"
    assert config.voice_tts.playback_timeout_sec == 300.0
    assert config.voice_tts.synthesis_timeout_sec == 90.0
    assert config.skill_gateway.status_service == "/embodied/get_skill_gateway_status"
    assert config.skill_gateway.required_control_mode == "moveit_planning"
    assert config.skill_gateway.default_skill_timeout_sec == 120.0
    assert config.skill_gateway.robot_state_freshness_sec == 0.5
    assert config.skill_gateway.task_budget_sec == 180.0
    assert config.skill_gateway.rpc_timeout_sec == 5.0
    assert config.embodied.timeouts["default_skill_timeout_sec"] == 120.0
    assert config.embodied.timeouts["robot_state_freshness_sec"] == 0.5

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
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm_legacy.yaml"

    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")

    config = load_robot_config_dict(config_path)

    assert config["name"] == "so101_single_arm"
    assert "control_modes" in config
    assert "policy" in config["control_modes"]["model_inference"]["inference"]["pipelines"]
    assert "joints" in config
    assert "simulation" in config
    assert config["_config_path"] == str(config_path.resolve())
    assert "semantic_mapping" in config
    assert "semantic_mapping" not in config["embodied"]["perception"]


def test_so101_single_arm_uses_degrees_for_lerobot_joint_conversion():
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm_legacy.yaml"

    config = load_robot_config_dict(config_path)

    assert resolve_lerobot_norm_mode(config) == "degrees"


def test_so101_single_arm_policy_inputs_require_fresh_live_observations():
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm_legacy.yaml"

    config = load_robot_config_dict(config_path)
    policy_keys = {"observation.state", "observation.images.top", "observation.images.wrist"}
    policy_observations = [
        observation for observation in config["contract"]["observations"] if observation["key"] in policy_keys
    ]

    assert len(policy_observations) == len(policy_keys)
    assert {observation["align"]["max_age_ms"] for observation in policy_observations} == {500}


def test_dict_contract_builder_matches_typed_contract_shape():
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm_legacy.yaml"

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
    assert [obs.align for obs in dict_contract.observations] == [obs.align for obs in typed_contract.observations]
    assert [obs.image for obs in dict_contract.observations] == [obs.image for obs in typed_contract.observations]
    assert [obs.qos for obs in dict_contract.observations] == [obs.qos for obs in typed_contract.observations]
    assert [obs.transport for obs in dict_contract.observations] == [
        obs.transport for obs in typed_contract.observations
    ]


def test_align_max_age_is_normalized_and_changes_contract_fingerprint():
    base = {
        "name": "test",
        "contract": {
            "rate_hz": 10,
            "observations": [
                {
                    "key": "observation.state",
                    "topic": "/joint_states",
                    "type": "sensor_msgs/msg/JointState",
                    "align": {"strategy": "hold", "max_age_ms": 500},
                }
            ],
            "actions": [],
        },
    }
    contract = build_contract_from_robot_config_dict(base)
    changed = build_contract_from_robot_config_dict(
        {
            **base,
            "contract": {
                **base["contract"],
                "observations": [
                    {
                        **base["contract"]["observations"][0],
                        "align": {"strategy": "hold", "max_age_ms": 1000},
                    }
                ],
            },
        }
    )

    spec = next(iter(iter_specs(contract)))
    assert contract.observations[0].align.max_age_ms == 500
    assert spec.max_age_ms == 500
    assert contract_fingerprint(contract) != contract_fingerprint(changed)


def test_align_rejects_negative_max_age():
    config = {
        "name": "test",
        "contract": {
            "rate_hz": 10,
            "observations": [
                {
                    "key": "observation.state",
                    "topic": "/joint_states",
                    "type": "sensor_msgs/msg/JointState",
                    "align": {"max_age_ms": -1},
                }
            ],
            "actions": [],
        },
    }

    with pytest.raises(ValueError, match="max_age_ms must be non-negative"):
        build_contract_from_robot_config_dict(config)


def test_so101_single_arm_contract_includes_motor_current_observation():
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm_legacy.yaml"

    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")

    contract = build_contract_from_robot_config_dict(load_robot_config_dict(config_path))
    current_obs = next(obs for obs in contract.observations if obs.key == "observation.current")

    assert current_obs.topic == "/so101_follower/joint_currents"
    assert current_obs.type == "ibrobot_msgs/msg/JointCurrent"
    assert current_obs.selector == {
        "names": ["current.1", "current.2", "current.3", "current.4", "current.5", "current.6"]
    }


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


def test_dict_contract_builder_warns_when_camera_lookup_fails(caplog):
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

    assert contract.observations[0].type == "sensor_msgs/msg/Image"
    assert contract.observations[0].image is None
    assert (
        "Observation 'observation.images.top' references peripheral 'missing_camera' but no camera found" in caplog.text
    )


def test_unbound_serial_cameras_warn_for_shared_driver_without_serials(caplog):
    with caplog.at_level(logging.WARNING, logger="robot_config.loader"):
        _warn_unbound_serial_cameras(
            {
                "peripherals": [
                    {"type": "camera", "name": "wrist", "driver": "realsense", "serial_number": ""},
                    {"type": "camera", "name": "front", "driver": "realsense", "serial_number": ""},
                ]
            }
        )

    assert "Multiple realsense cameras (wrist, front) have empty serial_number" in caplog.text


def test_unbound_serial_cameras_stay_silent_when_any_serial_is_pinned(caplog):
    with caplog.at_level(logging.WARNING, logger="robot_config.loader"):
        _warn_unbound_serial_cameras(
            {
                "peripherals": [
                    {"type": "camera", "name": "wrist", "driver": "realsense", "serial_number": "12345"},
                    {"type": "camera", "name": "front", "driver": "realsense", "serial_number": ""},
                ]
            }
        )
        _warn_unbound_serial_cameras(
            {
                "peripherals": [
                    {"type": "camera", "name": "wrist", "driver": "realsense", "serial_number": ""},
                ]
            }
        )
        _warn_unbound_serial_cameras(
            {
                "peripherals": [
                    {"type": "camera", "name": "left", "driver": "opencv", "serial_number": ""},
                    {"type": "camera", "name": "right", "driver": "opencv", "serial_number": ""},
                ]
            }
        )

    assert "empty serial_number" not in caplog.text


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


def test_validate_voice_asr_requires_bundle_and_deployment():
    """Test validation catches enabled voice ASR without explicit bundle identity."""
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
    config.voice_asr.bundle_path = ""
    config.voice_asr.deployment = ""

    errors = validate_config(config)
    assert any("voice_asr.bundle_path" in error for error in errors)
    assert any("voice_asr.deployment" in error for error in errors)


def test_load_voice_asr_config_preserves_bundle_identity():
    """Test voice ASR loader remains a pure bundle/deployment field mapper."""
    config = load_voice_asr_config(
        {
            "enabled": True,
            "active_mode": "continuous",
            "bundle_path": "models/voice_asr/demo",
            "deployment": "torch_cpu",
        }
    )

    assert config.bundle_path == "models/voice_asr/demo"
    assert config.deployment == "torch_cpu"
    assert config.audio_input_channel == 1
    assert config.exit_on_init_failure is True


def test_resolve_voice_asr_path_uses_workspace_root_for_relative_paths():
    """Test voice ASR relative paths resolve from the workspace root."""
    resolved = resolve_voice_asr_path("models/voice_asr/demo-bundle")

    assert Path(resolved).is_absolute()
    assert resolved.endswith("models/voice_asr/demo-bundle")


def test_voice_asr_runtime_defaults_match_robot_config_defaults():
    """Test robot_config defaults stay aligned with runtime Voice ASR defaults."""
    config_defaults = VoiceASRConfig()

    assert config_defaults.enabled == VOICE_ASR_DEFAULTS["enabled"]
    assert config_defaults.active_mode == VOICE_ASR_DEFAULTS["active_mode"]
    assert config_defaults.language == VOICE_ASR_DEFAULTS["language"]
    assert config_defaults.bundle_path == VOICE_ASR_DEFAULTS["bundle_path"]
    assert config_defaults.deployment == VOICE_ASR_DEFAULTS["deployment"]
    assert config_defaults.max_recording_duration == VOICE_ASR_DEFAULTS["max_recording_duration"]
    assert config_defaults.vad_sensitivity == VOICE_ASR_DEFAULTS["vad_sensitivity"]
    assert config_defaults.vad_bundle_path == VOICE_ASR_DEFAULTS["vad_bundle_path"]
    assert config_defaults.vad_deployment == VOICE_ASR_DEFAULTS["vad_deployment"]
    assert config_defaults.realtime_pre_roll_seconds == VOICE_ASR_DEFAULTS["realtime_pre_roll_seconds"]
    assert config_defaults.publish_partial == VOICE_ASR_DEFAULTS["publish_partial"]
    assert config_defaults.output_topic == VOICE_ASR_DEFAULTS["output_topic"]
    assert config_defaults.sample_rate == VOICE_ASR_DEFAULTS["sample_rate"]
    assert config_defaults.chunk_size == VOICE_ASR_DEFAULTS["chunk_size"]
    assert config_defaults.buffer_seconds == VOICE_ASR_DEFAULTS["buffer_seconds"]
    assert config_defaults.exit_on_init_failure == VOICE_ASR_DEFAULTS["exit_on_init_failure"]


def test_voice_tts_loader_and_runtime_defaults_match():
    config = load_voice_tts_config(
        {
            "enabled": True,
            "bundle_path": "models/voice_tts/custom",
            "deployment": "torch_cpu",
            "max_request_chars": 1000,
        }
    )
    defaults = VoiceTTSConfig()

    assert config.enabled is True
    assert config.bundle_path == "models/voice_tts/custom"
    assert config.deployment == "torch_cpu"
    assert config.max_request_chars == 1000
    assert defaults.bundle_path == VOICE_TTS_DEFAULTS["bundle_path"]
    assert defaults.deployment == VOICE_TTS_DEFAULTS["deployment"]
    assert defaults.service_name == VOICE_TTS_DEFAULTS["service_name"]
    assert defaults.playback_service_name == VOICE_TTS_DEFAULTS["playback_service_name"]
    assert defaults.playback_timeout_sec == VOICE_TTS_DEFAULTS["playback_timeout_sec"]
    assert defaults.prompt_profile == VOICE_TTS_DEFAULTS["prompt_profile"]
    assert defaults.max_response_audio_bytes == VOICE_TTS_DEFAULTS["max_response_audio_bytes"]


def test_validate_voice_tts_requires_explicit_deployment_and_valid_limits():
    config = RobotConfig(
        name="test_robot",
        type="so101",
        robot_type="so_101",
        ros2_control=Ros2ControlConfig(hardware_plugin="so101_hardware/SO101SystemHardware", params={}),
    )
    config.voice_tts.enabled = True
    config.voice_tts.deployment = ""
    config.voice_tts.playback_service_name = "voice_tts/play"
    config.voice_tts.playback_timeout_sec = 0.0
    config.voice_tts.max_segments = 0

    errors = validate_config(config)

    assert "voice_tts.deployment is required when voice_tts.enabled is true" in errors
    assert "voice_tts.playback_service_name must be an absolute ROS service name" in errors
    assert "voice_tts.playback_timeout_sec must be positive" in errors
    assert "voice_tts.max_segments must be positive" in errors


def test_validate_embodied_ignores_disabled_named_target_poses():
    """Disabled target-grasp config must not be required for the basic embodied pipeline."""
    config = RobotConfig(
        name="test_robot",
        type="so101",
        robot_type="so_101",
        ros2_control=Ros2ControlConfig(
            hardware_plugin="so101_hardware/SO101SystemHardware",
            params={},
        ),
        contract=ContractExtensionConfig(observations=[], actions=[]),
        embodied=EmbodiedConfig(
            enabled=True,
            named_poses={
                "home": {"position": {"x": 0.2, "y": 0.0, "z": 0.2}},
                "observe_table": {"position": {"x": 0.3, "y": 0.0, "z": 0.25}},
                "zero": {"position": {"x": 0.1, "y": 0.0, "z": 0.2}},
                "tray_right": {"position": {"x": 0.2, "y": -0.15, "z": 0.18}},
            },
            named_targets={
                "demo_object": {
                    "pregrasp_pose": "missing_pregrasp",
                    "grasp_pose": "missing_grasp",
                    "lift_pose": "missing_lift",
                }
            },
            workspace={"x": [0.0, 0.5], "y": [-0.3, 0.3], "z": [0.0, 0.5]},
        ),
    )

    errors = validate_config(config)
    assert not any("missing_pregrasp" in error for error in errors)


def test_validate_embodied_relative_motion_direction_mapping():
    """Embodied relative motion mapping must fully describe base-frame directions."""
    config = RobotConfig(
        name="test_robot",
        type="so101",
        robot_type="so_101",
        ros2_control=Ros2ControlConfig(
            hardware_plugin="so101_hardware/SO101SystemHardware",
            params={},
        ),
        contract=ContractExtensionConfig(observations=[], actions=[]),
        embodied=EmbodiedConfig(
            enabled=True,
            relative_motion_reference_frame="base",
            relative_motion_direction_mapping={
                "forward": [1.0, 0.0, 0.0],
                "backward": [-1.0, 0.0, 0.0],
                "left": [0.0, 1.0, 0.0],
                "right": [0.0, -1.0, 0.0],
                "up": [0.0, 0.0, 1.0],
            },
            named_poses={
                "home": {"position": {"x": 0.2, "y": 0.0, "z": 0.2}},
                "observe_table": {"position": {"x": 0.3, "y": 0.0, "z": 0.25}},
                "tray_right": {"position": {"x": 0.2, "y": -0.15, "z": 0.18}},
            },
            named_targets={
                "demo_object": {
                    "pregrasp_pose": "home",
                    "grasp_pose": "observe_table",
                    "lift_pose": "tray_right",
                }
            },
            workspace={"x": [0.0, 0.5], "y": [-0.3, 0.3], "z": [0.0, 0.5]},
        ),
    )

    errors = validate_config(config)
    assert any("missing directions: down" in error for error in errors)


def test_validate_embodied_rejects_removed_skill_templates():
    config = RobotConfig(
        name="test_robot",
        type="so101",
        robot_type="so_101",
        ros2_control=Ros2ControlConfig(
            hardware_plugin="so101_hardware/SO101SystemHardware",
            params={},
        ),
        contract=ContractExtensionConfig(observations=[], actions=[]),
        embodied=EmbodiedConfig(
            enabled=True,
            entry_mode="hermes",
            skill_templates={
                "inspect_scene": {
                    "primitive_sequence": [{"primitive_name": "move_to_named_pose", "pose_name": "observe_table"}]
                }
            },
            named_poses={
                "home": {"position": {"x": 0.2, "y": 0.0, "z": 0.2}},
                "observe_table": {"position": {"x": 0.3, "y": 0.0, "z": 0.25}},
                "tray_right": {"position": {"x": 0.2, "y": -0.15, "z": 0.18}},
            },
            named_targets={
                "demo_object": {
                    "pregrasp_pose": "home",
                    "grasp_pose": "observe_table",
                    "lift_pose": "tray_right",
                }
            },
            workspace={"x": [0.0, 0.5], "y": [-0.3, 0.3], "z": [0.0, 0.5]},
        ),
    )

    errors = validate_config(config)
    assert "embodied.skill_templates is removed; use embodied.skill_catalog_profile" in errors


def test_validate_embodied_accepts_skill_declared_by_current_robot():
    capability = {
        "schema_version": 1,
        "summary": "Move the robot for the requested skill.",
        "domain": "manipulation",
        "moves_robot": True,
        "required_control_mode": "moveit_planning",
        "parameters": _valid_capability_parameters(),
        "recovery_policy": "never_retry",
    }
    descriptions = {
        "inspect_scene": {
            "summary": "Inspect the configured scene.",
            "category": "observation",
            "when_to_use": ["inspect"],
            "motion_scope": ["arm"],
            "intensity": "subtle",
        },
        "custom_signal": {
            "summary": "Signal using the gripper.",
            "category": "demo",
            "when_to_use": ["signal"],
            "motion_scope": ["gripper"],
            "intensity": "subtle",
        },
    }
    config = RobotConfig(
        name="test_robot",
        type="so101",
        robot_type="so_101",
        ros2_control=Ros2ControlConfig(
            hardware_plugin="so101_hardware/SO101SystemHardware",
            params={},
        ),
        contract=ContractExtensionConfig(observations=[], actions=[]),
        embodied=EmbodiedConfig(
            enabled=True,
            default_place_name="home",
            skill_templates={
                "inspect_scene": {
                    "description": descriptions["inspect_scene"],
                    "capability": capability,
                    "primitive_sequence": [{"primitive_name": "move_to_named_pose", "pose_name": "observe_table"}],
                },
                "custom_signal": {
                    "description": descriptions["custom_signal"],
                    "capability": capability,
                    "primitive_sequence": [{"primitive_name": "open_gripper"}],
                },
            },
            named_poses={
                "home": {},
                "observe_table": {},
                "zero": {},
            },
        ),
    )

    errors = validate_config(config)

    assert not any("custom_signal" in error for error in errors)


def test_validate_typed_config_requires_capability_mode_to_match_gateway_mode():
    config = _typed_config_with_skill_gateway_mode("teleop")

    errors = validate_config(config)

    assert "embodied.skill_templates is removed; use embodied.skill_catalog_profile" in errors


@pytest.mark.parametrize("required_control_mode", ["", None, 1])
def test_validate_typed_config_requires_gateway_mode_for_enabled_skills(required_control_mode):
    errors = validate_config(_typed_config_with_skill_gateway_mode(required_control_mode))

    assert "embodied.skill_templates is removed; use embodied.skill_catalog_profile" in errors


def test_validate_embodied_perception_conversation_history():
    """Embodied perception config should validate conversation history setting."""
    config = RobotConfig(
        name="test_robot",
        type="so101",
        robot_type="so_101",
        ros2_control=Ros2ControlConfig(
            hardware_plugin="so101_hardware/SO101SystemHardware",
            params={},
        ),
        contract=ContractExtensionConfig(observations=[], actions=[]),
        embodied=EmbodiedConfig(
            enabled=True,
            perception={
                "enabled": True,
                "request_topic": "/embodied/perception_request",
                "result_topic": "/embodied/perception_result",
                "scene_sources": {"primary_camera_topic": "/camera/top/image_raw"},
                "vlm_api": {
                    "provider": "kimicode",
                    "base_url": "https://api.kimi.com/coding/v1",
                    "api_key_env": "KIMICODE_API_KEY",
                    "model": "kimi-for-coding",
                },
                "conversation": {"max_history_turns": -1},
            },
            skill_templates={
                "inspect_scene": {
                    "primitive_sequence": [{"primitive_name": "move_to_named_pose", "pose_name": "observe_table"}]
                }
            },
            named_poses={
                "home": {"position": {"x": 0.2, "y": 0.0, "z": 0.2}},
                "observe_table": {"position": {"x": 0.3, "y": 0.0, "z": 0.25}},
                "tray_right": {"position": {"x": 0.2, "y": -0.15, "z": 0.18}},
            },
            named_targets={
                "demo_object": {
                    "pregrasp_pose": "home",
                    "grasp_pose": "observe_table",
                    "lift_pose": "tray_right",
                }
            },
            workspace={"x": [0.0, 0.5], "y": [-0.3, 0.3], "z": [0.0, 0.5]},
        ),
    )

    errors = validate_config(config)
    assert any("max_history_turns must be >= 0" in error for error in errors)


def test_validate_embodied_openai_compatible_allows_empty_api_key_env():
    config = RobotConfig(
        name="test_robot",
        type="so101",
        robot_type="so_101",
        ros2_control=Ros2ControlConfig(
            hardware_plugin="so101_hardware/SO101SystemHardware",
            params={},
        ),
        contract=ContractExtensionConfig(observations=[], actions=[]),
        embodied=EmbodiedConfig(
            enabled=True,
            skill_templates={
                "inspect_scene": {
                    "primitive_sequence": [{"primitive_name": "move_to_named_pose", "pose_name": "observe_table"}]
                }
            },
            named_poses={
                "home": {"position": {"x": 0.2, "y": 0.0, "z": 0.2}},
                "observe_table": {"position": {"x": 0.3, "y": 0.0, "z": 0.25}},
                "tray_right": {"position": {"x": 0.2, "y": -0.15, "z": 0.18}},
            },
            named_targets={
                "demo_object": {
                    "observe_pose": "observe_table",
                    "pregrasp_pose": "home",
                    "hover_pose": "observe_table",
                    "grasp_pose": "observe_table",
                    "lift_pose": "tray_right",
                    "retreat_pose": "home",
                }
            },
            workspace={"x": [0.0, 0.5], "y": [-0.3, 0.3], "z": [0.0, 0.5]},
        ),
    )

    errors = validate_config(config)
    assert not any("api_key_env is required" in error for error in errors)


def test_validate_embodied_does_not_validate_removed_planner_depth_policy():
    config = RobotConfig(
        name="test_robot",
        type="so101",
        robot_type="so_101",
        ros2_control=Ros2ControlConfig(
            hardware_plugin="so101_hardware/SO101SystemHardware",
            params={},
        ),
        contract=ContractExtensionConfig(observations=[], actions=[]),
        embodied=EmbodiedConfig(
            enabled=True,
            skill_templates={
                "inspect_scene": {
                    "primitive_sequence": [{"primitive_name": "move_to_named_pose", "pose_name": "observe_table"}]
                }
            },
            named_poses={
                "home": {"position": {"x": 0.2, "y": 0.0, "z": 0.2}},
                "observe_table": {"position": {"x": 0.3, "y": 0.0, "z": 0.25}},
                "tray_right": {"position": {"x": 0.2, "y": -0.15, "z": 0.18}},
            },
            named_targets={
                "demo_object": {
                    "observe_pose": "observe_table",
                    "pregrasp_pose": "home",
                    "hover_pose": "observe_table",
                    "grasp_pose": "observe_table",
                    "lift_pose": "tray_right",
                    "retreat_pose": "home",
                }
            },
            workspace={"x": [0.0, 0.5], "y": [-0.3, 0.3], "z": [0.0, 0.5]},
        ),
    )

    errors = validate_config(config)
    assert not any("require_depth=true requires at least one aligned depth topic" in error for error in errors)


@pytest.mark.parametrize(
    "pose_source",
    [
        {"target_pose_key": "hover_pose"},
        {"place_name_from_request": True},
    ],
)
def test_validate_embodied_skill_template_accepts_dynamic_pose_sources(pose_source):
    config = RobotConfig(
        name="test_robot",
        type="so101",
        robot_type="so_101",
        ros2_control=Ros2ControlConfig(
            hardware_plugin="so101_hardware/SO101SystemHardware",
            params={},
        ),
        contract=ContractExtensionConfig(observations=[], actions=[]),
        embodied=EmbodiedConfig(
            enabled=True,
            skill_templates={
                "dynamic_named_pose": {
                    "description": {
                        "summary": "Resolve a named pose at runtime.",
                        "category": "motion",
                        "when_to_use": ["move to a runtime pose"],
                    },
                    "primitive_sequence": [{"primitive_name": "move_to_named_pose", **pose_source}],
                }
            },
            named_poses={
                "home": {"position": {"x": 0.2, "y": 0.0, "z": 0.2}},
                "observe_table": {"position": {"x": 0.3, "y": 0.0, "z": 0.25}},
                "tray_right": {"position": {"x": 0.2, "y": -0.15, "z": 0.18}},
            },
            named_targets={
                "demo_object": {
                    "pregrasp_pose": "home",
                    "grasp_pose": "observe_table",
                    "lift_pose": "tray_right",
                }
            },
            workspace={"x": [0.0, 0.5], "y": [-0.3, 0.3], "z": [0.0, 0.5]},
        ),
    )

    errors = validate_config(config)

    assert not any("move_to_named_pose step must define" in error for error in errors)


def test_validate_embodied_skill_template_requires_pose_source():
    config = RobotConfig(
        name="test_robot",
        type="so101",
        robot_type="so_101",
        ros2_control=Ros2ControlConfig(
            hardware_plugin="so101_hardware/SO101SystemHardware",
            params={},
        ),
        contract=ContractExtensionConfig(observations=[], actions=[]),
        embodied=EmbodiedConfig(
            enabled=True,
            skill_templates={
                "missing_named_pose": {
                    "description": {
                        "summary": "Invalid pose source.",
                        "category": "motion",
                        "when_to_use": ["never"],
                    },
                    "primitive_sequence": [{"primitive_name": "move_to_named_pose"}],
                }
            },
            named_poses={"home": {}},
        ),
    )

    errors = validate_config(config)

    assert "embodied.skill_templates is removed; use embodied.skill_catalog_profile" in errors
