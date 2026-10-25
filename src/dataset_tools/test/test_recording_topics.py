"""Topic-discovery tests for the recording builder (dataset_tools' contribution).

Migrated from robot_config's config / aero-hand / interface-binding suites
(openspec change ``migrate-business-launch-builders``, task A.3/A.8): the
assertions exercise ``get_recording_topics``, which moved with the recording
module; the robot_config files keep their config-level tests. Robot configs
load through the installed robot_config share directory; assertions are
verbatim.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from dataset_tools.launch_builders.recording import get_recording_topics
from robot_config.interface_binding import bind_robot_interfaces
from robot_config.loader import _load_robot_section, load_robot_config_dict
from robot_runtime import interface_description as description_api

ROBOT_CONFIG_SHARE = Path(get_package_share_directory("robot_config"))


def test_recording_topics_follow_contract_and_peripherals():
    """Migrated from robot_config's config suite: topics come from the
    contract, the peripherals and the ros2_control joint states.

    Generic behaviour (D11): moved from the retired ``lekiwi.yaml`` onto
    robot_config's minimal fixture."""
    config_path = ROBOT_CONFIG_SHARE / "test" / "config" / "minimal_sim_controllers.yaml"
    robot_config = load_robot_config_dict(config_path)
    topics = get_recording_topics(robot_config)

    assert "/joint_states" in topics
    assert "/camera/front/image_raw" in topics
    assert "/camera/front/camera_info" in topics
    assert "/base_controller/odom" in topics
    assert "/base_controller/cmd_vel" in topics
    assert "/scan" in topics


def test_aero_hand_topics_are_in_continuous_recording_set():
    """Migrated from robot_config's aero-hand config suite: the aero hand
    contract actions join the continuous recording set; the raw hand-source
    frames do not."""
    config_path = ROBOT_CONFIG_SHARE / "config" / "robots" / "so101_arm_aero_hand.yaml"
    topics = get_recording_topics(_load_robot_section(config_path)[1])

    assert "/aero_hand_right/commands" in topics
    assert "/aero_hand_right/joint_states" in topics
    assert "/hand_sources/mhandpro/right/frame" not in topics


def test_recording_uses_public_topics():
    """Migrated from robot_config's interface-binding suite: after live
    binding, the recording set uses the bound public endpoints instead of
    the peripheral-local names."""
    qos = {"reliability": "best_effort", "durability": "volatile", "history": "keep_last", "depth": 5}
    profile = {"width": 640, "height": 480, "fps": 30.0, "encoding": "bgr8"}
    descriptor = {
        "schema_version": 1,
        "robot": {"id": "unit-1", "type": "test", "runtime_name": "test_robot", "runtime_version": "1.0.0"},
        "execution": "simulated",
        "interfaces": {
            "camera.front.color": {
                "capability": "perception.camera",
                "kind": "topic",
                "direction": "publish",
                "endpoint": "/unit/front/image",
                "message_type": "sensor_msgs/msg/Image",
                "qos": qos,
                "frame_id": "front_optical",
                "camera_info_topic": "/unit/front/info",
                "configured_profile": profile,
                "supported_profiles": None,
            },
            "camera.front.color_info": {
                "capability": "perception.camera",
                "kind": "topic",
                "direction": "publish",
                "endpoint": "/unit/front/info",
                "message_type": "sensor_msgs/msg/CameraInfo",
                "qos": qos,
                "frame_id": "front_optical",
            },
            "joint.state": {
                "capability": "joint.state",
                "kind": "topic",
                "direction": "publish",
                "endpoint": "/unit/joints",
                "message_type": "sensor_msgs/msg/JointState",
                "qos": qos,
            },
            "base.cmd_vel": {
                "capability": "base.cmd_vel",
                "kind": "topic",
                "direction": "subscribe",
                "endpoint": "/unit/velocity",
                "message_type": "geometry_msgs/msg/Twist",
                "qos": {**qos, "reliability": "reliable"},
            },
        },
        "states": {
            "camera.front.color": {
                "state": "ready",
                "observed_profile": dict(profile),
                "observed_frame_id": "front_optical",
                "last_seen": 1_800_000_000.0,
                "detail": "",
            },
            "joint.state": {
                "state": "ready",
                "observed_profile": None,
                "observed_frame_id": None,
                "last_seen": 1_800_000_000.0,
                "detail": "",
            },
        },
    }
    descriptor["digest"] = description_api.description_digest(descriptor)
    description_api.validate_description(descriptor)
    config = {
        "name": "consumer",
        "type": "test",
        "runtime": {"provider": "test_robot", "profile": "test", "instance_id": "unit-1", "target": "simulation"},
        "default_control_mode": "teleop",
        "contract": {
            "observations": [
                {
                    "key": "observation.images.front",
                    "interface": "camera.front.color",
                    "image": {"resize": [224, 224], "encoding": "rgb8"},
                    "align": {"strategy": "asof", "stamp": "header", "tol_ms": 25},
                    "requires": {"width": 640, "height": 480, "encoding": "bgr8", "min_fps": 20},
                },
                {"key": "observation.state", "interface": "joint.state", "selector": {"names": ["one"]}},
            ],
            "actions": [
                {
                    "key": "action",
                    "publish": {"interface": "base.cmd_vel", "strategy": {"type": "immediate"}},
                    "from_tensor": {"field": "linear.x"},
                    "safety_behavior": "hold",
                }
            ],
        },
        "peripherals": [{"type": "camera", "name": "not_public"}],
    }

    topics = get_recording_topics(bind_robot_interfaces(config, descriptor))
    assert "/unit/front/image" in topics and "/unit/front/info" in topics and "/unit/joints" in topics
    assert "/unit/velocity" in topics
    assert "/joint_states" not in topics and "/camera/not_public/image_raw" not in topics
