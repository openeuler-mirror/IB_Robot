"""Unit tests for sim_backend adapter registry and sim_peripheral_bridge.

Tests cover both Gazebo and MuJoCo backends without requiring
a ROS or Gazebo runtime environment.
"""

from pathlib import Path

import pytest
import yaml

from robot_config.launch_builders.sim_backend import (
    SimBackendAdapter,
    get_backend_caps,
    get_sim_backend,
)
from robot_config.launch_builders.sim_backend.gazebo_adapter import GazeboAdapter
from robot_config.launch_builders.sim_backend.mock_adapter import MockAdapter
from robot_config.launch_builders.sim_backend.mujoco_adapter import MujocoAdapter

# ──────────────────────────────────────────────────────────────────────────────
# Registry tests (unchanged from T2)
# ──────────────────────────────────────────────────────────────────────────────


def test_registry_gazebo():
    """get_sim_backend('gazebo') returns a GazeboAdapter instance."""
    adapter = get_sim_backend("gazebo")
    assert isinstance(adapter, GazeboAdapter)
    assert isinstance(adapter, SimBackendAdapter)


def test_registry_mujoco():
    """get_sim_backend('mujoco') returns a MujocoAdapter instance."""
    adapter = get_sim_backend("mujoco")
    assert isinstance(adapter, MujocoAdapter)
    assert isinstance(adapter, SimBackendAdapter)


def test_registry_mock():
    """get_sim_backend('mock') returns a MockAdapter instance."""
    adapter = get_sim_backend("mock")
    assert isinstance(adapter, MockAdapter)
    assert isinstance(adapter, SimBackendAdapter)


def test_registry_unknown_raises():
    """get_sim_backend with an unknown name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown sim platform"):
        get_sim_backend("webots")


def test_backend_caps_mock():
    """Mock declares that it has no /clock and no ros2_control manager."""
    assert get_backend_caps("mock") == {"provides_clock": False, "needs_ros2_control": False}
    assert MockAdapter.provides_clock is False
    assert MockAdapter.needs_ros2_control is False


# ──────────────────────────────────────────────────────────────────────────────
# MuJoCo no-op tests
# ──────────────────────────────────────────────────────────────────────────────


def test_mujoco_noop_methods_return_empty():
    """MuJoCo no-op adapter methods return empty launch action lists."""
    adapter = get_sim_backend("mujoco")

    assert adapter.load_scene("fake.xml") == []
    assert adapter.ensure_controller_manager({}) == []
    assert adapter.update_object_pose("box", None) is None


def test_mujoco_spawn_peripheral_bridges_returns_empty():
    """MujocoAdapter.spawn_peripheral_bridges returns [] (no bridge nodes needed).

    MuJoCo publishes ROS2 topics directly. T6 will configure topic names
    to match the contract /camera/{name}/image_raw without extra bridge nodes.
    """
    adapter = get_sim_backend("mujoco")
    result = adapter.spawn_peripheral_bridges([])
    assert result == []
    # Also works with non-empty peripherals list
    result = adapter.spawn_peripheral_bridges(
        [
            {"type": "camera", "name": "top", "frame_id": "camera_top_frame"},
        ]
    )
    assert result == []


# ──────────────────────────────────────────────────────────────────────────────
# sim_peripheral_bridge tests (T3)
# ──────────────────────────────────────────────────────────────────────────────


def test_sim_peripheral_bridge_camera_count():
    """generate_peripheral_sim_bridges returns 1 bridge_node for any camera config."""
    from launch_ros.actions import Node

    from robot_config.launch_builders.sim_peripheral_bridge import (
        generate_peripheral_sim_bridges,
    )

    peripherals = [
        {"type": "camera", "name": "top", "frame_id": "camera_top_frame"},
    ]
    nodes = generate_peripheral_sim_bridges(peripherals, model_name="test_robot")
    assert len(nodes) == 1  # single bridge_node covers all cameras via YAML config
    assert isinstance(nodes[0], Node)


def test_sim_peripheral_bridge_multi_camera():
    """generate_peripheral_sim_bridges returns 1 bridge_node for multiple cameras."""
    from robot_config.launch_builders.sim_peripheral_bridge import (
        generate_peripheral_sim_bridges,
    )

    peripherals = [
        {"type": "camera", "name": "top", "frame_id": "camera_top_frame"},
        {"type": "camera", "name": "wrist", "frame_id": "camera_wrist_frame"},
        {"type": "camera", "name": "front", "frame_id": "camera_front_link"},
    ]
    nodes = generate_peripheral_sim_bridges(peripherals, model_name="test_robot")
    assert len(nodes) == 1  # single bridge_node handles all 3 cameras (6 topics in YAML)


def test_sim_peripheral_bridge_depth_camera_entries():
    """Depth-enabled cameras add an aligned-depth bridge entry."""
    from robot_config.launch_builders.sim_peripheral_bridge import (
        generate_peripheral_sim_bridges,
    )

    peripherals = [
        {
            "type": "camera",
            "name": "front",
            "frame_id": "camera_front_link",
            "align_depth": True,
        },
    ]
    nodes = generate_peripheral_sim_bridges(peripherals, model_name="test_robot")
    assert len(nodes) == 1

    bridge_config = yaml.safe_load(Path("/tmp/ros_gz_camera_bridge.yaml").read_text())
    ros_topics = {entry["ros_topic_name"] for entry in bridge_config}
    gz_topics = {entry["gz_topic_name"] for entry in bridge_config}

    assert "/camera/front/image_raw" in ros_topics
    assert "/camera/front/camera_info" in ros_topics
    assert "/camera/front/aligned_depth_to_color/image_raw" in ros_topics
    assert "/front_camera/depth_image" in gz_topics


def test_sim_peripheral_bridge_lidar_and_imu_supported():
    """LiDAR + IMU peripherals share one bridge_node in Gazebo."""
    from launch_ros.actions import Node

    from robot_config.launch_builders.sim_peripheral_bridge import (
        generate_peripheral_sim_bridges,
    )

    peripherals = [
        {"type": "lidar", "name": "main_lidar", "params": {"laser_scan_topic_name": "/scan"}},
        {"type": "imu", "name": "imu_sensor", "topic": "/imu_sensor_broadcaster/imu"},
    ]
    nodes = generate_peripheral_sim_bridges(peripherals, model_name="test_robot")
    assert len(nodes) == 1
    assert isinstance(nodes[0], Node)


def test_mock_start_backend_reuses_contract_mock():
    """Mock backend starts hardware_mock directly and has no spawn node."""
    from launch_ros.actions import Node

    adapter = get_sim_backend("mock")
    actions, create_node = adapter.start_backend({"_config_path": "/tmp/robot.yaml"})

    assert create_node is None
    assert len(actions) == 1
    assert isinstance(actions[0], Node)
    assert actions[0].node_package == "hardware_mock"


def test_gazebo_start_backend_returns_create_node():
    """GazeboAdapter.start_backend returns (actions, ros_gz_sim create Node)."""
    from launch_ros.actions import Node

    adapter = get_sim_backend("gazebo")
    actions, create_node = adapter.start_backend({"name": "test_robot", "gazebo_world_name": "demo"})
    assert isinstance(actions, list)
    assert len(actions) >= 1
    assert isinstance(create_node, Node)


def test_gazebo_adapter_bridge_delegates():
    """GazeboAdapter.spawn_peripheral_bridges returns 1 bridge_node for camera peripherals."""
    adapter = get_sim_backend("gazebo")
    adapter._model_name = "test_robot"
    peripherals = [
        {"type": "camera", "name": "top", "frame_id": "camera_top_frame"},
    ]
    nodes = adapter.spawn_peripheral_bridges(peripherals)
    assert len(nodes) == 1


# ──────────────────────────────────────────────────────────────────────────────
# World-only mode (runtime-simulation-transport D4, task 3.2)
# ──────────────────────────────────────────────────────────────────────────────


def test_gazebo_world_only_emits_server_and_clock_but_no_embodiment():
    from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
    from launch_ros.actions import Node

    adapter = get_sim_backend("gazebo")
    actions, world_name, spawn_pose = adapter.start_world_only(
        {
            "name": "test_robot",
            "gazebo_world_name": "demo",
            "simulation": {"platform": "gazebo"},
        }
    )
    assert world_name == "demo"
    assert spawn_pose == (0.0, 0.0, 0.0, 0.0)
    executables = {node._Node__node_executable for node in actions if isinstance(node, Node)}
    assert "parameter_bridge" in executables, "the world owner bridges /clock"
    assert "create" not in executables, "the provider spawns its own embodiment"
    assert "bridge_node" not in executables, "camera bridges are embodiment"
    assert "wait_for_clock" not in executables, "the provider gates its own create"
    assert sum(isinstance(action, IncludeLaunchDescription) for action in actions) == 1
    # no description-derived resource paths: the provider's description carries
    # absolute file:// mesh URIs and the scene world resolves its own assets
    assert not any(isinstance(action, SetEnvironmentVariable) for action in actions)


def test_gazebo_world_only_resolves_scene_world_and_full_spawn_pose():
    adapter = get_sim_backend("gazebo")
    config = {
        "name": "test_robot",
        "gazebo_world_name": "demo",
        "simulation": {
            "platform": "gazebo",
            "scene": "pick_banana",
            "robot_spawn": {"x": 0.25, "y": -0.1, "z": -0.03, "yaw": 0.7},
        },
    }
    actions, world_name, spawn_pose = adapter.start_world_only(config)
    assert world_name == "demo"
    assert spawn_pose == (0.25, -0.1, -0.03, 0.7)
    # the robot_spawn precedence beats the scene layout (pick_banana: 0.0/0.30/-0.03)
    from launch.actions import IncludeLaunchDescription

    include = next(action for action in actions if isinstance(action, IncludeLaunchDescription))
    (gz_args,) = [
        value
        for key, value in include.launch_arguments
        if str(key[0].text if hasattr(key[0], "text") else key) == "gz_args"
    ]
    assert "sim_models_pick_banana.world" in str(gz_args[0].text if hasattr(gz_args[0], "text") else gz_args)


def test_gazebo_world_only_falls_back_to_the_scene_layout_spawn():
    adapter = get_sim_backend("gazebo")
    config = {
        "name": "test_robot",
        "gazebo_world_name": "demo",
        "simulation": {"platform": "gazebo", "scene": "pick_banana"},
    }
    _actions, _world, spawn_pose = adapter.start_world_only(config)
    assert spawn_pose == (0.0, 0.30, -0.03, 0.0)


def _gz_server_args(actions) -> str:
    """The gz_args string of the single ros_gz_sim server include."""
    from launch.actions import IncludeLaunchDescription

    include = next(action for action in actions if isinstance(action, IncludeLaunchDescription))
    (gz_args,) = [
        value
        for key, value in include.launch_arguments
        if str(key[0].text if hasattr(key[0], "text") else key) == "gz_args"
    ]
    return str(gz_args[0].text if hasattr(gz_args[0], "text") else gz_args)


def test_gazebo_world_only_without_scene_uses_the_empty_scene_world():
    """No scene configured: the generic fallback is the sim_models empty scene.

    ``simulation.robot_spawn`` is ignored (spawn only applies with a scene)
    and no ``initial_pose_*`` is written back to the config.
    """
    adapter = get_sim_backend("gazebo")
    config = {
        "name": "test_robot",
        "gazebo_world_name": "demo",
        "simulation": {
            "platform": "gazebo",
            "robot_spawn": {"x": 0.25, "y": -0.1, "z": -0.03, "yaw": 0.7},
        },
    }
    actions, world_name, spawn_pose = adapter.start_world_only(config)
    assert world_name == "demo"
    assert spawn_pose == (0.0, 0.0, 0.0, 0.0)
    for key in ("initial_pose_x", "initial_pose_y", "initial_pose_z"):
        assert key not in config
    assert "sim_models_empty.world" in _gz_server_args(actions)


def test_gazebo_world_only_unknown_scene_falls_back_to_the_empty_scene_world():
    """A configured scene that fails to load falls back to the empty scene world."""
    adapter = get_sim_backend("gazebo")
    config = {
        "name": "test_robot",
        "gazebo_world_name": "demo",
        "simulation": {"platform": "gazebo", "scene": "no_such_scene"},
    }
    actions, world_name, spawn_pose = adapter.start_world_only(config)
    assert world_name == "demo"
    assert spawn_pose == (0.0, 0.0, 0.0, 0.0)
    assert "sim_models_empty.world" in _gz_server_args(actions)


def test_provider_gazebo_simulation_startup_carries_world_and_pose():
    from robot_config.launch_builders.simulation import build_simulation
    from robot_config.launch_registry.registry import LaunchBuildContext
    from robot_config.runtime_target import RuntimeTarget

    context = LaunchBuildContext(
        robot_config={
            "name": "test_robot",
            "runtime": {"provider": "test_robot", "profile": "p"},
            "simulation": {
                "platform": "gazebo",
                "scene": "pick_banana",
                "robot_spawn": {"x": 0.1, "y": 0.2, "z": 0.3, "yaw": 0.5},
            },
        },
        runtime_target=RuntimeTarget.SIMULATION,
        sim_platform="gazebo",
        use_sim=True,
    )
    result = build_simulation(context)
    assert result.simulation_startup is not None
    assert result.simulation_startup.entity_action is None
    assert result.simulation_startup.world_name == "demo"
    assert result.simulation_startup.spawn_pose == (0.1, 0.2, 0.3, 0.5)
    # world-only: no create, no joint-state bridge, no camera bridges
    from launch_ros.actions import Node

    executables = {node._Node__node_executable for node in result.actions if isinstance(node, Node)}
    assert "parameter_bridge" in executables
    assert "create" not in executables and "bridge_node" not in executables


def test_preseeded_simulation_startup_is_reused_without_relaunching():
    from robot_config.launch_builders.simulation import build_simulation
    from robot_config.launch_registry.registry import LaunchBuildContext, SimulationStartup
    from robot_config.runtime_target import RuntimeTarget

    preseed = SimulationStartup(entity_action=None, world_name="demo", spawn_pose=(0.0, 0.3, -0.03, 0.0))
    context = LaunchBuildContext(
        robot_config={"name": "test_robot", "runtime": {"provider": "test_robot", "profile": "p"}},
        runtime_target=RuntimeTarget.SIMULATION,
        sim_platform="gazebo",
        use_sim=True,
        simulation_startup=preseed,
    )
    result = build_simulation(context)
    assert result.actions == [], "the second plan pass must not relaunch the world"
    assert result.simulation_startup is preseed


def test_provider_sdk_simulation_still_yields_the_empty_startup():
    from robot_config.launch_builders.simulation import build_simulation
    from robot_config.launch_registry.registry import LaunchBuildContext
    from robot_config.runtime_target import RuntimeTarget

    context = LaunchBuildContext(
        robot_config={"name": "test_robot", "runtime": {"provider": "test_robot", "profile": "p"}},
        runtime_target=RuntimeTarget.SIMULATION,
        sim_platform="sdk",
        use_sim=True,
    )
    result = build_simulation(context)
    assert result.actions == []
    assert result.simulation_startup.entity_action is None
    assert result.simulation_startup.world_name == ""
