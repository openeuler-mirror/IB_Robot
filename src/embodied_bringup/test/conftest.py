"""Portable public snapshots for consumer tests, without robot-private assets."""

import pytest

from robot_runtime.interface_description import description_digest, validate_description
from robot_runtime.model_metadata import build_model_metadata


@pytest.fixture
def public_description():
    names = [str(index) for index in range(1, 7)]
    profile = {
        "joints": names,
        "arm_joints": names[:-1],
        "gripper_joints": names[-1:],
        "home_positions": {name: 0.0 for name in names},
        "motion": {"base_link": "base", "ee_link": "gripper"},
    }
    urdf = '<robot name="fixture"><link name="base"/><link name="gripper"/>'
    for name in names:
        lower, upper = (0, 1) if name == "6" else (-3.14, 3.14)
        urdf += f'<joint name="{name}" type="revolute"><limit lower="{lower}" upper="{upper}"/></joint>'
    urdf += "</robot>"
    qos = {"reliability": "reliable", "durability": "volatile", "history": "keep_last", "depth": 10}
    interfaces = {}
    for key, message_type, endpoint in (
        ("runtime.status", "ibrobot_msgs/msg/RuntimeStatus", "/test/runtime_status"),
        ("joint.state", "sensor_msgs/msg/JointState", "/test/joint_states"),
        ("joint.current", "ibrobot_msgs/msg/JointCurrent", "/test/joint_currents"),
        ("motion.ee_pose", "geometry_msgs/msg/PoseStamped", "/test/ee_pose"),
        ("joint.arm_stream", "std_msgs/msg/Float64MultiArray", "/test/arm_commands"),
        ("joint.gripper_stream", "std_msgs/msg/Float64MultiArray", "/test/gripper_commands"),
    ):
        interfaces[key] = {
            "capability": "runtime.status" if key == "runtime.status" else "joint.state",
            "kind": "topic",
            "direction": "subscribe" if key.endswith("stream") else "publish",
            "endpoint": endpoint,
            "message_type": message_type,
            "qos": dict(qos),
        }
    for camera in ("top", "front", "wrist"):
        interfaces[f"camera.{camera}.color"] = {
            "capability": "perception.camera",
            "kind": "topic",
            "direction": "publish",
            "endpoint": f"/test/{camera}/image",
            "message_type": "sensor_msgs/msg/Image",
            "qos": dict(qos),
            "configured_profile": {"width": 640, "height": 480, "fps": 30.0, "encoding": "rgb8"},
            "supported_profiles": None,
            "camera_info_topic": f"/test/{camera}/info",
            "frame_id": f"{camera}_optical",
        }
        interfaces[f"camera.{camera}.color_info"] = {
            "capability": "perception.camera",
            "kind": "topic",
            "direction": "publish",
            "endpoint": f"/test/{camera}/info",
            "message_type": "sensor_msgs/msg/CameraInfo",
            "qos": dict(qos),
            "frame_id": f"{camera}_optical",
        }
    for key, service in (("runtime.set_mode", "SetRuntimeMode"), ("motion.move_to_joint", "MoveToConfiguration")):
        interfaces[key] = {
            "capability": "runtime.status" if key.startswith("runtime") else key,
            "kind": "service",
            "direction": "serve",
            "endpoint": "/test/" + key.replace(".", "/"),
            "message_type": f"ibrobot_msgs/srv/{service}",
        }
    interfaces["joint.arm_trajectory"] = {
        "capability": "joint.trajectory",
        "kind": "action",
        "direction": "serve",
        "endpoint": "/test/arm/follow_joint_trajectory",
        "message_type": "control_msgs/action/FollowJointTrajectory",
        "target_group": "arm",
        "joint_names": names[:-1],
    }
    description = {
        "schema_version": 1,
        "robot": {"id": "so101_robot", "type": "so101", "runtime_name": "so101_robot", "runtime_version": "0.1.0"},
        "execution": "simulated",
        "interfaces": interfaces,
        "states": {},
        "model": build_model_metadata(profile, simulated=True, robot_description=urdf),
    }
    description["digest"] = description_digest(description)
    validate_description(description)
    return description
