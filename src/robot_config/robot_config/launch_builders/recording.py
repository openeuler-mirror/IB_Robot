"""Recording node generation for robot_config.

This module provides utilities to generate recording nodes
for integration with the robot_config launch system.

Supports two recording modes:
1. Continuous: Traditional ros2 bag record (all-in-one file)
2. Episodic: Triggered episode-by-episode recording via episode_recorder Action Server
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

from launch.actions import EmitEvent, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node

from robot_config.interface_binding import required_interface_ids, resolve_robot_interfaces
from robot_config.logger_utils import get_colored_logger
from robot_config.utils import (
    resolve_gripper_joints_from_config,
    resolve_joint_names_from_config,
    resolve_lerobot_norm_mode,
    resolve_ros_path,
)


def _sanitize_dataset_name(value: str) -> str:
    """Normalize dataset names so launch logs match recorder output paths."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    normalized = normalized.strip("._-")
    return normalized or "dataset"


logger = get_colored_logger("robot_config.recording")


def build_semantic_preview_command(robot_config: dict | None = None) -> list[str]:
    """Build the proven calibration preview command without launch-injected ROS arguments.

    The preview provider package comes from configuration
    (``recording.semantic_preview_package``); it defaults to the LeKiwi
    sensor-calibration suite package, the only preview implementation today.
    """
    package = str(((robot_config or {}).get("recording") or {}).get("semantic_preview_package", "")).strip()
    if not package:
        package = "lekiwi_calibration"
    return [
        "ros2",
        "run",
        package,
        "calib_capture_preview",
        "--output-image-topic",
        "/semantic_mapping/preview/image/compressed",
        "--output-cloud-topic",
        "/semantic_mapping/preview/cloud",
        "--max-fps",
        "8.0",
        "--jpeg-quality",
        "70",
        "--max-points",
        "6000",
    ]


def resolve_recording_launch(robot_config: dict, *, requested: bool, mode: str) -> tuple[bool, str]:
    """Apply recording parameters without allowing semantic dataset profiles to run incomplete."""
    if robot_config.get("recording", {}).get("semantic_dataset", False):
        if mode != "continuous":
            raise ValueError("semantic datasets require continuous recording")
        return True, mode
    return requested, mode


def _record_cli_command(active_control_mode: str, *, scheduler_enabled: bool = False) -> str:
    command = f"ros2 run dataset_tools record_cli --ros-args -p control_mode:={active_control_mode}"
    if scheduler_enabled:
        command += " -p restart_session_service:=/action_dispatcher/restart_session"
    return command


def generate_recording_nodes(
    robot_config: dict,
    active_control_mode: str,
    record_mode: str = "continuous",
    *,
    scheduler_enabled: bool = False,
) -> list[Node | ExecuteProcess | RegisterEventHandler]:
    """
    Generate recording nodes based on robot configuration and recording mode.

    This function creates ROS 2 nodes/actions for data recording based on the
    recording mode. It integrates with the robot_config launch system.

    Args:
        robot_config: Robot configuration dictionary loaded from YAML
        active_control_mode: The control mode currently active
        record_mode: Recording mode - 'continuous' or 'episodic'
                     - continuous: Uses ros2 bag record for all-in-one recording
                     - episodic: Uses episode_recorder Action Server for triggered recording

    Returns:
        List of Node or ExecuteProcess actions for recording

    Example:
        >>> from robot_config.launch_builders.recording import generate_recording_nodes
        >>> config = load_robot_config('so101_single_arm')
        >>> nodes = generate_recording_nodes(config, record_mode='episodic')
        >>> ld.add_action(nodes[0])

    Usage in launch file:
        # Continuous recording (default)
        ros2 launch robot_config robot.launch.py record:=true

        # Episodic recording (requires manual record_cli in separate terminal)
        ros2 launch robot_config robot.launch.py record:=true record_mode:=episodic
        # Then use the record_cli command printed by the launch process.
    """
    if record_mode == "episodic":
        return generate_episodic_recording_node(
            robot_config,
            active_control_mode,
            scheduler_enabled=scheduler_enabled,
        )
    else:
        return generate_continuous_recording_action(robot_config)


def generate_continuous_recording_action(robot_config: dict) -> list[Node | ExecuteProcess | RegisterEventHandler]:
    """
    Generate continuous recording action using ros2 bag record.

    This creates a single rosbag file that records everything continuously
    from launch until shutdown.

    Args:
        robot_config: Robot configuration dictionary

    Returns:
        List containing ExecuteProcess action for ros2 bag record

    Behavior:
        - Auto-discovers topics from robot config (joints, cameras, controllers)
        - Generates filename: ~/rosbag/<robot_name>_<timestamp>.mcap
        - Records continuously until node shutdown
    """
    logger.info("Using CONTINUOUS recording (ros2 bag record)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rosbag_command = build_continuous_recording_command(robot_config, timestamp)
    output_file = rosbag_command[rosbag_command.index("-o") + 1]
    recording = robot_config.get("recording", {})
    command = (
        build_supervised_recording_command(robot_config, timestamp, rosbag_command)
        if recording.get("semantic_dataset", False)
        else rosbag_command
    )
    topics = get_recording_topics(robot_config)

    logger.info(f"Recording {len(topics)} topics to: {output_file}")
    logger.info(f"Topics: {topics}")

    # Create recording action
    recording_action = ExecuteProcess(cmd=command, output="screen")
    if recording.get("semantic_dataset", False):
        return [
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=recording_action,
                    on_exit=lambda event, _context: semantic_recording_exit_actions(event.returncode),
                )
            ),
            recording_action,
            ExecuteProcess(cmd=build_semantic_preview_command(robot_config), output="screen"),
        ]

    logger.info("✓ Continuous recording action created")
    return [recording_action]


def semantic_recording_exit_actions(returncode: int | None) -> list[EmitEvent]:
    """Stop a semantic launch when its supervised recorder exits unexpectedly."""
    if returncode in (None, 0):
        return []
    return [
        EmitEvent(
            event=Shutdown(reason=f"semantic dataset recorder exited with return code {returncode}"),
        )
    ]


def build_continuous_recording_command(robot_config: dict, timestamp: str) -> list[str]:
    """Build the profile-specific rosbag2 command without changing generic defaults."""
    recording = robot_config.get("recording", {})
    robot_name = robot_config.get("name", "robot")
    if recording.get("semantic_dataset", False):
        session_root = Path(recording.get("session_base_dir", "~/rosbag/semantic_mapping")).expanduser()
        output_path = session_root / f"{robot_name}_{timestamp}" / "bag"
    else:
        output_path = Path(f"~/rosbag/{robot_name}_{timestamp}.mcap").expanduser()
        return ["ros2", "bag", "record", "-o", str(output_path), *get_recording_topics(robot_config)]

    command = ["ros2", "bag", "record", "-o", str(output_path)]
    options = (
        ("storage", "-s"),
        ("max_bag_size", "--max-bag-size"),
        ("max_bag_duration", "--max-bag-duration"),
        ("max_cache_size", "--max-cache-size"),
        ("compression_mode", "--compression-mode"),
        ("compression_format", "--compression-format"),
        ("qos_profile_overrides_path", "--qos-profile-overrides-path"),
    )
    for config_key, cli_flag in options:
        value = recording.get(config_key)
        if value in (None, ""):
            continue
        if config_key == "qos_profile_overrides_path":
            value = resolve_ros_path(str(value))
        command.extend([cli_flag, str(value)])
    command.extend(get_recording_topics(robot_config))
    return command


def build_supervised_recording_command(
    robot_config: dict,
    timestamp: str,
    rosbag_command: list[str] | None = None,
) -> list[str]:
    """Wrap semantic dataset recording with the minimal session supervisor."""
    recording = robot_config.get("recording", {})
    rosbag_command = rosbag_command or build_continuous_recording_command(robot_config, timestamp)
    bag_path = Path(rosbag_command[rosbag_command.index("-o") + 1])
    session_root = bag_path.parent
    config_path = robot_config.get("_config_path")
    if not config_path:
        raise ValueError("semantic dataset recording requires robot_config['_config_path']")

    mount_file = robot_config.get("mid360_mount_file")
    if not mount_file:
        raise ValueError("semantic dataset recording requires mid360_mount_file")
    camera_file = robot_config.get("sensor_calibration", {}).get("artifacts", {}).get("base_to_front_camera", "")
    state_path = Path(
        recording.get("mapping_session_state_path", "~/.ros/ibrobot/semantic_mapping/current.json")
    ).expanduser()
    command = [
        "ros2",
        "run",
        "robot_config",
        "semantic_mapping_recorder",
        "--session-id",
        timestamp,
        "--profile",
        str(robot_config.get("name", "robot")),
        "--robot-config",
        str(Path(config_path).expanduser()),
        "--session-root",
        str(session_root),
        "--state-file",
        str(state_path),
        "--mount-file",
        resolve_ros_path(str(mount_file)),
        "--camera-info-topic",
        str(recording.get("camera_info_topic", "/camera/front/camera_info")),
    ]
    if camera_file:
        command.extend(["--camera-file", resolve_ros_path(str(camera_file))])
    for topic in get_recording_topics(robot_config):
        command.extend(["--topic", topic])
    command.extend(["--", *rosbag_command])
    return command


def generate_episodic_recording_node(
    robot_config: dict,
    active_control_mode: str,
    *,
    scheduler_enabled: bool = False,
) -> list[Node]:
    """
    Generate episodic recording node using episode_recorder Action Server.

    This creates an Action Server that waits for trigger commands to start
    recording individual episodes. Each episode is saved as a dataset-scoped
    bag directory with semantic metadata (operator prompt).

    **IMPORTANT**: The episode_recorder Action Server runs in the background.
    You MUST manually run `record_cli` in a separate terminal to trigger recordings.

    Args:
        robot_config: Robot configuration dictionary
        active_control_mode: The active control mode string

    Returns:
        List containing Node action for episode_recorder

    Behavior:
        - Uses contract section directly from robot_config.yaml (Single Source of Truth)
        - Starts episode_recorder Action Server (background service)
        - Each episode saved as: <bag_base_dir>/<dataset_name>/episodes/episode_XXXXXX
        - Operator prompt embedded in bag metadata
    """
    logger.info("Using EPISODIC recording (episode_recorder Action Server)")

    # Check if contract section exists in robot_config
    contract_config = robot_config.get("contract")
    if not contract_config:
        logger.error("No 'contract' section found in robot configuration.")
        logger.info("  Please add 'contract' section with observations and actions.")
        return []
    if any(
        isinstance(obs, dict) and isinstance(obs.get("transport"), dict) and obs["transport"].get("mode") == "rtp"
        for obs in contract_config.get("observations", [])
    ):
        logger.info("RTP episodic recording is owned by the cloud recording_node")
        return []

    # Determine bag output directory
    recording_config = robot_config.get("recording", {})
    custom_dir = recording_config.get("bag_base_dir", "~/rosbag_demos/episodes")
    bag_base_dir = os.path.expanduser(custom_dir)

    # Get robot_config file path (passed via launch argument)
    # The launch file should pass the robot_config path as a parameter
    robot_config_path = robot_config.get("_config_path", "")

    if not robot_config_path:
        raise ValueError("robot_config dict is missing '_config_path'. Cannot launch episodic recording without it.")

    dataset_name = _sanitize_dataset_name(
        str(recording_config.get("dataset_name") or robot_config.get("name") or Path(robot_config_path).stem)
    )
    dataset_root = Path(bag_base_dir).expanduser() / str(dataset_name)
    default_task = str(recording_config.get("default_task", "") or "")
    task_family = str(recording_config.get("task_family", "") or "")
    lerobot_norm_mode = resolve_lerobot_norm_mode(robot_config, preferred_control_mode=active_control_mode)
    joint_names = resolve_joint_names_from_config(robot_config)
    gripper_joints = resolve_gripper_joints_from_config(robot_config)
    max_cache_size = int(recording_config.get("max_cache_size", 100 * 1024 * 1024) or 0)
    storage_preset_profile = str(recording_config.get("storage_preset_profile", "") or "")
    storage_config_uri = str(recording_config.get("storage_config_uri", "") or "")

    # Create episode_recorder node (Action Server)
    episode_recorder_node = Node(
        package="dataset_tools",
        executable="episode_recorder",
        name="episode_recorder",
        output="screen",
        parameters=[
            {"robot_config_path": robot_config_path},
            {"bag_base_dir": bag_base_dir},
            {"dataset_name": str(dataset_name)},
            {"control_mode": active_control_mode},
            {"default_task": default_task},
            {"task_family": task_family},
            {"lerobot_norm_mode": lerobot_norm_mode},
            {"joint_names": joint_names},
            {"gripper_joints": gripper_joints},
            {
                "robot_model_json": json.dumps(robot_config.get("robot_model"), separators=(",", ":"))
                if robot_config.get("robot_model")
                else ""
            },
            {
                "interface_description_json": json.dumps(
                    (robot_config.get("runtime") or {}).get("interface_description"), separators=(",", ":")
                )
                if (robot_config.get("runtime") or {}).get("interface_description")
                else ""
            },
            {"max_cache_size": max_cache_size},
            {"storage_preset_profile": storage_preset_profile},
            {"storage_config_uri": storage_config_uri},
        ],
    )

    logger.info("✓ Episode recorder node created")
    logger.info(f"Dataset root: {dataset_root}")
    logger.info(f"LeRobot norm mode: {lerobot_norm_mode}")
    logger.info("")
    logger.info("=" * 70)
    logger.warning("IMPORTANT: Use SEPARATE TERMINAL to trigger recordings:")
    logger.info(f"    {_record_cli_command(active_control_mode, scheduler_enabled=scheduler_enabled)}")
    logger.info("Convert later with:")
    logger.info(
        f"    ros2 run dataset_tools bag_to_lerobot --bags-dir {dataset_root} "
        f"--robot-config {robot_config_path} --out /path/to/output_dataset"
    )
    logger.info("=" * 70)

    return [episode_recorder_node]


def generate_rerun_viewer_node(robot_config: dict) -> list[Node]:
    """Generate a Rerun visualization sidecar node for recording observation.

    The node loads the same contract as ``episode_recorder`` and subscribes to
    all observation/action topics, forwarding data to a Rerun viewer in
    real-time.  It is a **pure observer** — it never writes to the bag.

    Args:
        robot_config: Robot configuration dictionary loaded from YAML.
                      Must contain ``_config_path`` (set by the launch system).

    Returns:
        List containing a single Node action, or empty if config path is missing.
    """
    robot_config_path = robot_config.get("_config_path", "")
    if not robot_config_path:
        logger.error("robot_config dict is missing '_config_path'. Cannot launch rerun_viewer without it.")
        return []

    rerun_node = Node(
        package="dataset_tools",
        executable="rerun_viewer",
        name="rerun_viewer",
        output="screen",
        additional_env={"PYTHONNOUSERSITE": "1"},
        parameters=[
            {"robot_config_path": robot_config_path},
        ],
    )

    logger.info("✓ Rerun viewer node created (sidecar)")
    return [rerun_node]


def find_workspace_root() -> str | None:
    """
    Find IB_Robot workspace root directory.

    Returns:
        Workspace root path or None if not found
    """
    # Try to find from current file location
    current_path = Path(__file__).resolve()

    # Walk up the directory tree looking for install/setup.bash
    for parent in current_path.parents:
        if (parent / "install" / "setup.bash").exists():
            return str(parent)

    return None


def get_recording_topics(robot_config: dict) -> list[str]:
    """
    Get list of topics to record based on robot configuration.

    Args:
        robot_config: Robot configuration dictionary

    Returns:
        List of topic names for rosbag recording

    Example:
        >>> topics = get_recording_topics(config)
        >>> print(topics)
        ['/joint_states', '/arm_position_controller/commands', '/camera/cam0/image_raw', ...]
    """
    robot_config = resolve_robot_interfaces(robot_config)
    logical_bindings = bool(required_interface_ids(robot_config))
    recording = robot_config.get("recording", {})
    topics = []

    def _append(topic: str):
        if not topic:
            return
        normalized = topic if topic.startswith("/") else f"/{topic}"
        if normalized not in topics:
            topics.append(normalized)

    explicit_topics = recording.get("topics")
    if explicit_topics is not None:
        for topic in explicit_topics:
            _append(topic)
        return topics

    # Always record joint states for ros2_control-backed robots.
    if not logical_bindings:
        _append("/joint_states")

    # Record contract-defined observations/actions first.
    contract = robot_config.get("contract", {})
    for obs in contract.get("observations", []):
        _append(obs.get("topic", ""))
    for action in contract.get("actions", []):
        _append((action.get("publish") or {}).get("topic", ""))

    if logical_bindings:
        description = robot_config["runtime"]["interface_description"]
        for interface in description["interfaces"].values():
            if interface["kind"] == "topic" and interface["direction"] == "publish":
                _append(interface["endpoint"])
                _append(interface.get("camera_info_topic", ""))

    # Legacy configs retain peripheral-specific auxiliary topic discovery.
    for peripheral in [] if logical_bindings else robot_config.get("peripherals", []):
        ptype = peripheral.get("type")
        name = peripheral.get("name", "peripheral")
        if ptype == "camera":
            _append(f"/camera/{name}/image_raw")
            _append(f"/camera/{name}/camera_info")
        elif ptype == "lidar":
            params = peripheral.get("params", {})
            _append(params.get("laser_scan_topic_name", peripheral.get("topic", "/scan")))
            _append(params.get("point_cloud_2d_topic_name", ""))
        elif ptype == "imu":
            _append(peripheral.get("topic", ""))

    # Diagnostics / navigation extras.
    if recording.get("include_diagnostics", True):
        _append("/diagnostics")
    for extra_topic in recording.get("extra_topics", []):
        _append(extra_topic)

    return topics
