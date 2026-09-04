"""Embodied minimal-closure launch builder."""

import json
from pathlib import Path
from typing import Any

from launch_ros.actions import Node

from embodied_common.visual_game_contracts import normalize_visual_game_policies
from robot_config.loader import (
    navigation_endpoint_projection,
    robot_config_digest,
    robot_context_schema_version,
    robot_supported_control_modes,
    validate_navigation_endpoint_contract,
)
from robot_config.logger_utils import get_colored_logger
from robot_config.timeout_policy import resolve_embodied_timeout_policy
from robot_config.utils import resolve_ros_path

logger = get_colored_logger("embodied_bringup")


def _resolve_development_source_root(config_path: Path, configured_root: str) -> Path | None:
    module_path = Path(__file__).resolve()
    anchors = [config_path]
    repository_root = next((parent.parent for parent in module_path.parents if parent.name == "src"), None)
    if repository_root is not None:
        install_root = repository_root / "install"
        try:
            config_path.absolute().relative_to(install_root)
        except ValueError:
            pass
        else:
            anchors.append(module_path)
    for anchor in anchors:
        try:
            repository_root = next(parent for parent in anchor.parents if parent.name == "src").parent
        except StopIteration:
            continue
        candidate = (repository_root / configured_root).resolve()
        if candidate.is_dir():
            return candidate
    return None


def _node_runtime_settings(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    node_params = dict(params)
    host_runtime = node_params.pop("host_runtime", {})
    environment = {}
    if "omp_threads" in host_runtime:
        environment.update(
            {
                "OMP_NUM_THREADS": str(host_runtime["omp_threads"]),
                "OMP_DYNAMIC": "FALSE",
                "OMP_WAIT_POLICY": "PASSIVE",
                "GOMP_SPINCOUNT": "0",
            }
        )
    if "blas_threads" in host_runtime:
        environment["OPENBLAS_NUM_THREADS"] = str(host_runtime["blas_threads"])
    return node_params, environment


def generate_embodied_nodes(
    robot_config: dict[str, Any],
    active_control_mode: str,
    *,
    motion_authorized: bool = False,
    include_motion: bool = True,
    include_visual_games: bool = True,
    include_perception: bool = True,
) -> list[Node]:
    """Generate embodied minimum-closure nodes from robot_config YAML."""
    embodied_config = robot_config.get("embodied", {})
    if not embodied_config.get("enabled", False):
        logger.info("Embodied minimal closure disabled, skipping")
        return []

    required_control_mode = str(robot_config.get("skill_required_control_mode", "moveit_planning")).strip()
    supported_control_modes = robot_supported_control_modes(robot_config)
    moveit_compatible = "moveit" in required_control_mode.lower() and "moveit" in active_control_mode.lower()
    motion_mode_compatible = (
        active_control_mode == required_control_mode
        or moveit_compatible
        or (active_control_mode in supported_control_modes and required_control_mode in supported_control_modes)
    )
    required_mode_error = (
        "embodied minimal closure requires a MoveIt-compatible control mode"
        if "moveit" in required_control_mode.lower()
        else f"embodied minimal closure requires control_mode:={required_control_mode}"
    )

    execution = embodied_config.get("execution", {})
    hri_runtime = embodied_config.get("imitate_human_motion", {})
    if not isinstance(hri_runtime, dict):
        raise ValueError("embodied.imitate_human_motion must be a mapping")
    entry_mode = str(embodied_config.get("entry_mode", "hermes")).lower()
    if entry_mode != "hermes":
        raise ValueError("embodied.entry_mode must be hermes")
    endpoint_errors = validate_navigation_endpoint_contract(robot_config)
    if endpoint_errors:
        raise ValueError("; ".join(endpoint_errors))
    named_poses = embodied_config.get("named_poses", {})
    named_targets = embodied_config.get("named_targets", {})
    safety = embodied_config.get("safety", {})
    joint_config = robot_config.get("joints", {})
    teleoperation = robot_config.get("teleoperation", {})
    perception = embodied_config.get("perception", {})
    grasp_execution = robot_config.get("grasp_execution", {})
    placement_execution = robot_config.get("placement_execution", {})
    semantic_mapping = robot_config.get("semantic_mapping", {})
    if isinstance(grasp_execution, dict) and isinstance(placement_execution, dict):
        motion = placement_execution.get("motion", {})
        if isinstance(motion, dict):
            grasp_execution = dict(grasp_execution)
            grasp_execution["post_grasp_motion"] = {
                "pose_name": motion.get("place_pose", "place_container"),
                "joint_names": motion.get("place_joint_names", []),
                "joint_positions": motion.get("place_joint_positions", {}),
                "duration_sec": motion.get("place_duration_sec", 5.0),
                "velocity_scaling": motion.get("place_velocity_scaling", 0.08),
            }
    perception_scene_sources = perception.get("scene_sources", {})
    perception_vlm_api = perception.get("vlm_api", {})
    perception_conversation = perception.get("conversation", {})
    visual_games = normalize_visual_game_policies(embodied_config.get("visual_games", {}))
    timeout_policy = resolve_embodied_timeout_policy(embodied_config)
    skill_catalog_source_mode = embodied_config.get("skill_catalog_source_mode", "installed")
    skill_catalog_source_root = embodied_config.get("skill_catalog_source_root", "")
    if skill_catalog_source_root and not Path(skill_catalog_source_root).is_absolute():
        config_path = Path(robot_config.get("_config_path", ""))
        resolved_source_root = _resolve_development_source_root(config_path, skill_catalog_source_root)
        if resolved_source_root is None:
            skill_catalog_source_mode = "installed"
            skill_catalog_source_root = ""
        else:
            skill_catalog_source_root = str(resolved_source_root)
    enabled_visual_games = {name: policy for name, policy in visual_games.items() if policy["enabled"]}
    announcing_visual_games = {name: policy for name, policy in enabled_visual_games.items() if policy["announce"]}
    voice_tts = robot_config.get("voice_tts", {})
    if not isinstance(voice_tts, dict):
        voice_tts = {}
    has_tts_runtime = (
        voice_tts.get("enabled") is True
        and bool(str(voice_tts.get("bundle_path", "")).strip())
        and bool(str(voice_tts.get("deployment", "")).strip())
        and str(voice_tts.get("service_name", "")).startswith("/")
        and str(voice_tts.get("playback_service_name", "")).startswith("/")
    )
    start_visual_game_service = embodied_config.get("start_visual_game_service", "/embodied/start_visual_game")
    get_visual_game_result_service = embodied_config.get(
        "get_visual_game_result_service", "/embodied/get_visual_game_result"
    )
    visual_game_event_topic = embodied_config.get("visual_game_event_topic", "/embodied/visual_game_events")
    common_params = {
        "debug_tracing": embodied_config.get("debug_tracing", True),
        "named_poses_json": json.dumps(named_poses),
        "named_targets_json": json.dumps(named_targets),
        "workspace_json": json.dumps(safety.get("workspace", {})),
        "arm_joint_names_json": json.dumps(joint_config.get("arm", [])),
        "joint_limits_json": json.dumps(teleoperation.get("safety", {}).get("joint_limits", {})),
        "default_target_name": embodied_config.get("default_target_name", "demo_object"),
        "default_place_name": embodied_config.get("default_place_name", "home"),
        "skill_action_name": embodied_config.get("skill_action_name", "/embodied/execute_skill"),
        "primitive_action_name": embodied_config.get("primitive_action_name", "/embodied/execute_primitive"),
        "validate_skill_service": embodied_config.get("validate_skill_service", "/embodied/validate_skill"),
        "validate_primitive_service": embodied_config.get("validate_primitive_service", "/embodied/validate_primitive"),
        "status_topic": embodied_config.get("status_topic", "/embodied/task_status"),
        "motion_authorized": motion_authorized,
        "active_control_mode": active_control_mode,
        "skill_required_control_mode": robot_config.get("skill_required_control_mode", ""),
        "supported_control_modes_json": json.dumps(list(supported_control_modes)),
        "motion_mode_service": str(
            robot_config.get("motion_mode", {}).get(
                "set_navigation_enabled_service", "motion_mode/set_navigation_enabled"
            )
        ),
        "skill_gateway_status_service": embodied_config.get(
            "skill_gateway_status_service", "/embodied/get_skill_gateway_status"
        ),
        "skill_catalog_source_mode": skill_catalog_source_mode,
        "skill_catalog_source_root": skill_catalog_source_root,
        "skill_catalog_profile": embodied_config.get("skill_catalog_profile", robot_config.get("name", "")),
        "skill_catalog_snapshot_service": embodied_config.get(
            "skill_catalog_snapshot_service", "/embodied/get_skill_snapshot"
        ),
        "skill_registry_event_topic": embodied_config.get(
            "skill_registry_event_topic", "/embodied/skill_registry_events"
        ),
        "begin_workflow_service": embodied_config.get("begin_workflow_service", "/embodied/begin_workflow_execution"),
        "finalize_workflow_service": embodied_config.get(
            "finalize_workflow_service", "/embodied/finalize_workflow_execution"
        ),
        "robot_name": robot_config.get("name", "unknown"),
        "context_schema_version": robot_context_schema_version(robot_config),
        "config_digest": robot_config_digest(robot_config),
        "default_skill_timeout_sec": timeout_policy["default_skill_timeout_sec"],
        "task_budget_sec": timeout_policy["task_budget_sec"],
        "robot_state_freshness_sec": timeout_policy["robot_state_freshness_sec"],
        "scene_freshness_sec": timeout_policy["scene_freshness_sec"],
        "model_idle_timeout_sec": timeout_policy["model_idle_timeout_sec"],
        "rpc_timeout_sec": timeout_policy["rpc_timeout_sec"],
        "gripper_settle_sec": timeout_policy["gripper_settle_sec"],
        "relative_motion_step_m": execution.get("relative_motion_step_m", 0.03),
        "relative_motion_reference_frame": execution.get("relative_motion_reference_frame", "base"),
        "relative_motion_direction_mapping_json": json.dumps(execution.get("relative_motion_direction_mapping", {})),
        "gripper_open_position": execution.get("gripper_open_position", 1.0),
        "gripper_closed_position": execution.get("gripper_closed_position", 0.0),
        "task_executor_action_name": execution.get("task_executor_action_name", "/task_executor/execute_task_plan"),
        "pick_action_name": grasp_execution.get("action_name", "/manipulation/execute_pick"),
        "grasp_execution_json": json.dumps(grasp_execution),
        "place_action_name": placement_execution.get("action_name", "/manipulation/execute_place"),
        "placement_execution_json": json.dumps(placement_execution),
        "imitate_human_motion_action_name": hri_runtime.get("action_name", "/hri/imitate_human_motion"),
        "imitate_human_motion_enabled": hri_runtime.get("enabled", False),
        "move_configuration_service": execution.get(
            "move_configuration_service", "/moveit_gateway/move_to_configuration"
        ),
    }
    navigation_action_name = navigation_endpoint_projection(robot_config)
    if navigation_action_name is not None:
        common_params["navigation_action_name"] = navigation_action_name
    if isinstance(semantic_mapping, dict) and semantic_mapping.get("enabled", False):
        common_params["semantic_map_target_service"] = semantic_mapping.get("interfaces", {}).get(
            "query_service", "/semantic_mapping/get_objects"
        )
        common_params["semantic_map_stand_off_distance_m"] = float(
            semantic_mapping.get("target_watch", {}).get("stand_off_distance_m", 0.3)
        )
    perception_node = None
    if perception.get("enabled", False):
        perception_node = Node(
            package="perception_service",
            executable="perception_service_node",
            name="perception_service_node",
            output="screen",
            parameters=[
                {
                    "debug_tracing": embodied_config.get("debug_tracing", True),
                    "request_topic": perception.get("request_topic", "/embodied/perception_request"),
                    "text_input_topic": perception.get("text_input_topic", "/embodied/perception_text"),
                    "result_topic": perception.get("result_topic", "/embodied/perception_result"),
                    "summary_topic": perception.get("summary_topic", "/embodied/perception_summary"),
                    "observation_topic": perception.get("observation_topic", "/embodied/perception_observation"),
                    "default_session_id": perception.get("default_session_id", "default"),
                    "primary_camera_topic": perception_scene_sources.get(
                        "primary_camera_topic", "/camera/top/image_raw"
                    ),
                    "wrist_camera_topic": perception_scene_sources.get("wrist_camera_topic", "/camera/wrist/image_raw"),
                    "primary_camera_info_topic": perception_scene_sources.get("primary_camera_info_topic", ""),
                    "primary_aligned_depth_topic": perception_scene_sources.get("primary_aligned_depth_topic", ""),
                    "primary_pointcloud_topic": perception_scene_sources.get("primary_pointcloud_topic", ""),
                    "wrist_camera_info_topic": perception_scene_sources.get("wrist_camera_info_topic", ""),
                    "wrist_aligned_depth_topic": perception_scene_sources.get("wrist_aligned_depth_topic", ""),
                    "wrist_pointcloud_topic": perception_scene_sources.get("wrist_pointcloud_topic", ""),
                    "ee_pose_topic": perception_scene_sources.get("ee_pose_topic", "/robot_status/ee_pose"),
                    "joint_state_topic": perception_scene_sources.get("joint_state_topic", "/joint_states"),
                    "max_scene_age_sec": timeout_policy["scene_freshness_sec"],
                    "require_depth": perception_scene_sources.get("require_depth", False),
                    "require_pointcloud": perception_scene_sources.get("require_pointcloud", False),
                    "api_provider": perception_vlm_api.get("provider", "openai_compatible"),
                    "api_base_url": perception_vlm_api.get("base_url", "http://localhost:8000/v1"),
                    "api_key_env": perception_vlm_api.get("api_key_env", ""),
                    "api_model": perception_vlm_api.get("model", "Qwen3.5-9B"),
                    "api_timeout_sec": timeout_policy["model_idle_timeout_sec"],
                    "api_max_image_width": perception_vlm_api.get("max_image_width", 320),
                    "api_jpeg_quality": perception_vlm_api.get("jpeg_quality", 70),
                    "max_history_turns": perception_conversation.get("max_history_turns", 4),
                    "max_concurrent_requests": perception.get("max_concurrent_requests", 1),
                    "min_object_confidence": perception.get("min_object_confidence", 0.0),
                }
            ],
        )

    visual_game_gateway_node = None
    visual_game_announcer_node = None
    if enabled_visual_games:
        visual_game_gateway_node = Node(
            package="embodied_agent",
            executable="visual_game_gateway_node",
            name="visual_game_gateway_node",
            output="screen",
            parameters=[
                {
                    "debug_tracing": embodied_config.get("debug_tracing", True),
                    "robot_name": robot_config.get("name", "unknown"),
                    "perception_enabled": perception.get("enabled", False),
                    "visual_games_json": json.dumps(visual_games),
                    "perception_request_topic": perception.get("request_topic", "/embodied/perception_request"),
                    "perception_result_topic": perception.get("result_topic", "/embodied/perception_result"),
                    "start_service": start_visual_game_service,
                    "result_service": get_visual_game_result_service,
                    "event_topic": visual_game_event_topic,
                    "model_idle_timeout_sec": timeout_policy["model_idle_timeout_sec"],
                    "visual_game_timeout_sec": timeout_policy["visual_game_timeout_sec"],
                    "result_retention_sec": timeout_policy["visual_game_result_retention_sec"],
                    "result_capacity": embodied_config.get("visual_game_result_capacity", 128),
                }
            ],
        )
        if announcing_visual_games and has_tts_runtime:
            visual_game_announcer_node = Node(
                package="embodied_agent",
                executable="visual_game_announcer_node",
                name="visual_game_announcer_node",
                output="screen",
                parameters=[
                    {
                        "event_topic": visual_game_event_topic,
                        "tts_service": voice_tts["service_name"],
                        "playback_service": voice_tts["playback_service_name"],
                        "tts_timeout_sec": voice_tts.get("tts_timeout_sec", 15.0),
                        "playback_timeout_sec": voice_tts.get("playback_timeout_sec", 300.0),
                        "debug_tracing": embodied_config.get("debug_tracing", True),
                    }
                ],
            )

    visual_nodes = []
    if include_visual_games:
        if visual_game_gateway_node is not None:
            visual_nodes.append(visual_game_gateway_node)
        if visual_game_announcer_node is not None:
            visual_nodes.append(visual_game_announcer_node)
        if perception_node is not None and include_perception:
            visual_nodes.append(perception_node)

    has_visual_closure = visual_game_gateway_node is not None

    if not include_motion:
        if not motion_mode_compatible and not has_visual_closure:
            raise ValueError(required_mode_error)
        return visual_nodes

    if not motion_mode_compatible:
        if not include_visual_games:
            return []
        if not has_visual_closure:
            raise ValueError(required_mode_error)
        logger.info("Embodied visual-game closure enabled without motion nodes")
        return visual_nodes

    logger.info("Embodied minimal closure enabled, launching task/safety/skill nodes")

    idle_behaviors = embodied_config.get("idle_behaviors", {})
    sound_orientation = idle_behaviors.get("sound_orientation", {}) if isinstance(idle_behaviors, dict) else {}
    if not isinstance(sound_orientation, dict):
        sound_orientation = {}

    nodes = [
        Node(
            package="safety_guard",
            executable="safety_guard_node",
            name="safety_guard_node",
            output="screen",
            parameters=[common_params],
        ),
        Node(
            package="skill_library",
            executable="skill_executor_node",
            name="skill_executor_node",
            output="screen",
            parameters=[common_params],
        ),
        Node(
            package="embodied_agent",
            executable="agent_plan_node",
            name="agent_plan_node",
            output="screen",
            parameters=[
                {
                    "gateway_status_service": common_params["skill_gateway_status_service"],
                    "validate_skill_service": common_params["validate_skill_service"],
                    "skill_catalog_snapshot_service": common_params["skill_catalog_snapshot_service"],
                    "skill_action_name": common_params["skill_action_name"],
                    "begin_workflow_service": common_params["begin_workflow_service"],
                    "finalize_workflow_service": common_params["finalize_workflow_service"],
                    "rpc_timeout_sec": timeout_policy["rpc_timeout_sec"],
                    "plan_service": embodied_config.get("plan_service", "/embodied/plan_agent_command"),
                    "validate_plan_service": embodied_config.get(
                        "validate_plan_service", "/embodied/validate_agent_plan"
                    ),
                    "confirm_plan_service": embodied_config.get("confirm_plan_service", "/embodied/confirm_agent_plan"),
                    "execute_plan_action": embodied_config.get("execute_plan_action", "/embodied/execute_agent_plan"),
                }
            ],
        ),
    ]
    if hri_runtime.get("enabled", False):
        nodes.append(
            Node(
                package="manipulation_execution",
                executable="imitate_human_motion_executor_node",
                name="imitate_human_motion_executor_node",
                output="screen",
                parameters=[
                    {
                        "action_name": common_params["imitate_human_motion_action_name"],
                        "primitive_action_name": common_params["primitive_action_name"],
                        "rpc_timeout_sec": timeout_policy["rpc_timeout_sec"],
                        "startup_warmup": hri_runtime.get("startup_warmup", True),
                        "arm_joint_names_json": json.dumps(joint_config.get("arm", [])),
                        "reset_positions_json": json.dumps(
                            robot_config.get("ros2_control", {}).get("reset_positions", {})
                        ),
                        "joint_limits_json": json.dumps(teleoperation.get("safety", {}).get("joint_limits", {})),
                    }
                ],
            )
        )
    if sound_orientation.get("enabled", False):
        nodes.append(
            Node(
                package="embodied_agent",
                executable="sound_orientation_node",
                name="sound_orientation_node",
                output="screen",
                parameters=[
                    {
                        "trigger_phrases": sound_orientation.get("trigger_phrases", ["转向我"]),
                        "direction_topic": sound_orientation.get("direction_topic", "/voice/speech_direction"),
                        "command_topic": sound_orientation.get("command_topic", "/voice_command"),
                        "gateway_status_service": common_params["skill_gateway_status_service"],
                        "skill_action_name": common_params["skill_action_name"],
                        "skill_name": sound_orientation.get("skill_name", "nav_turn"),
                        "direction_frame": sound_orientation.get("direction_frame", "base_link"),
                        "deadband_deg": sound_orientation.get("deadband_deg", 15.0),
                        "max_direction_age_sec": sound_orientation.get("max_direction_age_sec", 1.3),
                        "direction_wait_sec": sound_orientation.get("direction_wait_sec", 0.5),
                        "cooldown_sec": sound_orientation.get("cooldown_sec", 1.5),
                        "max_turn_deg": sound_orientation.get("max_turn_deg", 180.0),
                        "turn_timeout_sec": sound_orientation.get("turn_timeout_sec", 10.0),
                        "action_acceptance_timeout_sec": sound_orientation.get("action_acceptance_timeout_sec", 2.0),
                        "status_retry_sec": sound_orientation.get("status_retry_sec", 0.5),
                        "reset_status_max_age_sec": sound_orientation.get("reset_status_max_age_sec", 2.0),
                        "debug_tracing": embodied_config.get("debug_tracing", True),
                    }
                ],
            )
        )
    if include_visual_games:
        nodes.extend(visual_nodes)
    elif include_perception and perception_node is not None:
        # Preserve the legacy perception placement in the controller-gated motion closure.
        nodes.append(perception_node)
    if grasp_execution.get("enabled", False):
        camera = grasp_execution.get("camera", {})
        planner_params, planner_env = _node_runtime_settings(grasp_execution.get("planner_node", {}))
        for field in ("local_manifest_path", "ascend_local_manifest_path"):
            if planner_params.get(field):
                planner_params[field] = resolve_ros_path(planner_params[field])
        verifier_params = grasp_execution.get("verifier_node", {})
        if grasp_execution.get("auto_start_dependencies", True):
            nodes.extend(
                [
                    Node(
                        package="manipulation_service",
                        executable="grasp_planner_node",
                        name="grasp_planner",
                        output="screen",
                        additional_env=planner_env,
                        parameters=[
                            {
                                "rgb_topic": camera.get("rgb_topic", "/camera/wrist/image_raw"),
                                "depth_topic": camera.get(
                                    "depth_topic", "/camera/wrist/aligned_depth_to_color/image_raw"
                                ),
                                "camera_info_topic": camera.get(
                                    "camera_info_topic", "/camera/wrist/aligned_depth_to_color/camera_info"
                                ),
                                "detect_service": grasp_execution.get("detect_service", "/perception/grounding_detect"),
                                "segment_service": grasp_execution.get("segment_service", ""),
                                "legacy_detect_service": grasp_execution.get(
                                    "fallback_detect_service", "/grasp_planner/detect_and_segment"
                                ),
                                "model_dir": resolve_ros_path(
                                    grasp_execution.get(
                                        "planner_model_dir", grasp_execution.get("model_bundle_path", "")
                                    )
                                ),
                                **planner_params,
                            }
                        ],
                    ),
                    Node(
                        package="manipulation_service",
                        executable="grasp_verifier_node",
                        name="grasp_verifier",
                        output="screen",
                        parameters=[
                            {
                                "joint_state_topic": verifier_params.get("joint_state_topic", "/joint_states"),
                                "joint_current_topic": verifier_params.get(
                                    "joint_current_topic", "/so101_follower/joint_currents"
                                ),
                                "wrist_depth_topic": verifier_params.get(
                                    "wrist_depth_topic",
                                    camera.get("depth_topic", "/camera/wrist/aligned_depth_to_color/image_raw"),
                                ),
                                **verifier_params,
                            }
                        ],
                    ),
                ]
            )
        nodes.append(
            Node(
                package="manipulation_execution",
                executable="pick_executor_node",
                name="pick_executor_node",
                output="screen",
                parameters=[
                    {
                        "action_name": grasp_execution.get("action_name", "/manipulation/execute_pick"),
                        "primitive_action_name": embodied_config.get(
                            "primitive_action_name", "/embodied/execute_primitive"
                        ),
                        "grasp_execution_json": json.dumps(grasp_execution),
                        "workspace_json": json.dumps(safety.get("workspace", {})),
                        "home_joint_positions_json": json.dumps(
                            robot_config.get("ros2_control", {}).get("reset_positions", {})
                        ),
                        "arm_joint_names_json": json.dumps(joint_config.get("arm", [])),
                        "gripper_open_position": execution.get("gripper_open_position", 1.0),
                        "gripper_closed_position": execution.get("gripper_closed_position", 0.0),
                        "rpc_timeout_sec": timeout_policy["rpc_timeout_sec"],
                    }
                ],
            )
        )
    if placement_execution.get("enabled", False):
        moveit_executor = robot_config.get("control_modes", {}).get("moveit_planning", {}).get("executor", {})
        nodes.append(
            Node(
                package="manipulation_execution",
                executable="place_executor_node",
                name="placement_executor_node",
                output="screen",
                parameters=[
                    {
                        "action_name": placement_execution.get("action_name", "/manipulation/execute_place"),
                        "primitive_action_name": embodied_config.get(
                            "primitive_action_name", "/embodied/execute_primitive"
                        ),
                        "placement_execution_json": json.dumps(placement_execution),
                        "gripper_joint_name": str(
                            next(iter(robot_config.get("joints", {}).get("gripper", ["6"])), "6")
                        ),
                        "gripper_open_position": execution.get("gripper_open_position", 1.0),
                        "gripper_closed_position": execution.get("gripper_closed_position", 0.0),
                        "gripper_position_tolerance": moveit_executor.get("gripper_position_tolerance", 0.05),
                        "rpc_timeout_sec": timeout_policy["rpc_timeout_sec"],
                    }
                ],
            )
        )
    return nodes
