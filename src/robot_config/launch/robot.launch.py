"""Main robot launch file for robot_config.

This launch file loads robot configuration from YAML and dynamically generates:
- ros2_control hardware interface and controllers
- Robot state publisher
- Camera drivers (usb_cam, realsense2_camera)
- Static TF publishers for camera frames
- Voice ASR node (optional, configured from robot.voice_asr)
- Speech direction node (optional, configured from robot.speech_direction)
- Voice TTS service (optional, configured from robot.voice_tts)
- Inference service and action dispatcher (optional, auto-detected)
- MoveIt motion planning (optional, auto-detected)

Controllers are automatically spawned in both simulation and real hardware modes:
- Simulation mode: Uses Gazebo's gz_ros2_control plugin for controller_manager
- Hardware mode: Starts ros2_control_node for controller_manager

Expected ROS interfaces (depends on ``control_mode`` and options):
- ``control_mode:=moveit_planning`` (and MoveIt enabled): planning/move_group topics such as ``/planning_scene``; not started for ``model_inference`` or ``teleop`` alone.
- Gazebo sim + cameras: bridged topics ``/camera/{top,wrist,front}/image_raw`` and ``.../camera_info`` (names from YAML ``peripherals[].name``), not raw Ignition link paths. Cameras with ``align_depth``/depth streams also bridge ``.../aligned_depth_to_color/image_raw``.
- After controller spawners succeed: ``/arm_position_controller/commands``, ``/gripper_position_controller/commands``, ``/joint_states``, etc.

**CRITICAL**: This workspace uses ROS_DOMAIN_ID=<ID> to avoid conflicts with other ROS 2 systems.
Always set this before launching:
```bash
export ROS_DOMAIN_ID=<ID>
```

Usage:
    # Basic simulation
    ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm use_sim:=true

    # Contract mock simulation
    ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm use_sim:=true sim_platform:=mock control_mode:=model_inference

    # Model inference mode (auto-detected)
    ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm use_sim:=true control_mode:=model_inference

    # Teleop mode (human teleoperation)
    ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=teleop record:=true

    # Teleop mode with episodic recording (episode-by-episode)
    ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=teleop record:=true record_mode:=episodic

    # Episodic recording with Rerun live visualization (cameras, joints, actions)
    ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=teleop record:=true record_mode:=episodic record_visualizer:=rerun

    # MoveIt planning mode (auto-detected, with RViz)
    ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=moveit_planning use_sim:=true

    # MoveIt mode without RViz (headless)
    ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=moveit_planning use_sim:=true moveit_display:=false

    # Real hardware
    ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm use_sim:=false

    # Override auto-detection
    ros2 launch robot_config robot.launch.py control_mode:=model_inference with_inference:=true use_sim:=true

**Cleanup**: If you encounter "Controller already loaded" errors, run:
```bash
./scripts/cleanup_ros.sh
```

Launch Arguments:
    robot_config: Robot configuration name (default: test_cam)
    config_path: Optional full path to robot config file
    use_sim: Use simulation mode (default: false). Backend comes from robot YAML simulation.platform (gazebo/mujoco/mock)
    sim_platform: Optional CLI override for robot YAML simulation.platform
    auto_start_controllers: Automatically spawn controllers (default: true, set to false for debugging)
    control_mode: Override control mode from YAML (teleop, model_inference, moveit_planning, etc.). If empty, uses default_control_mode from config file
    nav_stage: Select a workflow stage declared by the robot config
    with_inference: Enable inference pipeline. If empty, auto-detects from control mode config
    inference_pipeline: Pipeline ID targeted by inference launch overrides
    inference_execution_mode: Override the targeted pipeline mode (monolithic or distributed)
    with_moveit: Enable MoveIt motion planning. If empty, auto-detects from control mode name
    moveit_display: Launch RViz for MoveIt visualization (default: true, only used if MoveIt is enabled)
    with_navigation: Enable navigation pipeline. If empty, uses robot.navigation.enabled from config
    navigation_mode: Override robot.navigation.default_mode when navigation is enabled
    record: Enable automatic rosbag recording (default: false, auto-discovers topics from config)
    record_mode: Recording mode - 'continuous' (default, all-in-one bag) or 'episodic' (triggered episode-by-episode, requires manual record_cli in separate terminal)
    record_visualizer: Recording visualizer - 'none' (default) or 'rerun' (launch a Rerun sidecar for live cameras, joints, and action curves)
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node

from robot_config.benchmark_observation_transport import materialize_benchmark_observation_transport
from robot_config.inference_config import (
    InferenceConfigError,
    parse_inference_config,
    scheduler_enabled_from_raw_config,
)

# Import node generators from launch_builders modules
from robot_config.launch_builders.benchmark import generate_benchmark_nodes
from robot_config.launch_builders.control import (
    generate_auxiliary_actuator_nodes,
    generate_ros2_control_nodes,
    validate_runtime_resources,
)
from robot_config.launch_builders.execution import generate_execution_nodes
from robot_config.launch_builders.hand_sources import (
    apply_hand_profile,
    confirm_interactive_startup_p_pose,
    generate_hand_source_nodes,
)
from robot_config.launch_builders.hardware_mock import (
    mock_mode_skips_subsystem,
    validate_mock_control_mode,
)
from robot_config.launch_builders.navigation import generate_navigation_nodes
from robot_config.launch_builders.perception import (
    generate_camera_nodes,
    generate_lidar_nodes,
    generate_tf_nodes,
)
from robot_config.launch_builders.perception_models import generate_perception_model_nodes
from robot_config.launch_builders.recording import (
    generate_recording_nodes,
    generate_rerun_viewer_node,
    resolve_recording_launch,
)
from robot_config.launch_builders.sim_backend import get_backend_caps, get_sim_backend
from robot_config.launch_builders.teleop import generate_teleop_nodes
from robot_config.launch_builders.tracing import (
    DEFAULT_TRACE_SESSION_NAME,
    generate_tracing_actions,
)
from robot_config.loader import load_robot_config_dict
from robot_config.logger_utils import get_colored_logger
from robot_config.runtime_target import RuntimeTarget, resolve_runtime_target

# Import utility functions
from robot_config.utils import parse_bool

logger = get_colored_logger("robot_config.launch")


def load_robot_config(robot_config_name, config_path_override=None, nav_stage=""):
    """Load robot configuration from YAML file.

    Args:
        robot_config_name: Robot configuration name
        config_path_override: Optional full path to config file

    Returns:
        Robot configuration dict
    """
    # Get package share directory
    try:
        robot_config_share = get_package_share_directory("robot_config")
    except Exception:
        robot_config_share = str(Path(__file__).parent.parent)

    # Determine config file path
    if config_path_override:
        config_path = Path(config_path_override)
    else:
        config_path = Path(robot_config_share) / "config" / "robots" / f"{robot_config_name}.yaml"

    logger.info(f"Loading config from: {config_path}")
    logger.info(f"Config exists: {config_path.exists()}")

    robot_config = load_robot_config_dict(
        config_path,
        nav_stage=nav_stage,
        materialize_benchmark_transport=False,
    )
    logger.info(f"Loaded robot: {robot_config.get('name', 'UNKNOWN')}")
    logger.info(f"Peripherals: {len(robot_config.get('peripherals', []))}")

    return robot_config


def _apply_voice_asr_cli_overrides(context, robot_config: dict) -> None:
    """Apply optional voice ASR launch overrides onto the loaded robot config."""
    voice_asr_config = dict(robot_config.get("voice_asr", {}))

    enabled_override = context.launch_configurations.get("voice_asr_auto_start", "")
    if enabled_override != "":
        voice_asr_config["enabled"] = parse_bool(enabled_override, default=False)

    pre_roll_override = context.launch_configurations.get(
        "voice_asr_realtime_pre_roll_seconds",
        "",
    )
    if pre_roll_override != "":
        voice_asr_config["realtime_pre_roll_seconds"] = float(pre_roll_override)

    robot_config["voice_asr"] = voice_asr_config


def _validate_benchmark_inference_closure(
    robot_config: dict,
    *,
    control_mode: str,
    with_inference: bool,
) -> None:
    """Fail fast when Benchmark evaluation cannot reach the policy/action path."""
    benchmark = robot_config.get("benchmark", {})
    evaluation = benchmark.get("evaluation", {}) if isinstance(benchmark, dict) else {}
    if not isinstance(evaluation, dict) or not bool(evaluation.get("enabled", False)):
        return
    if not with_inference:
        raise ValueError("benchmark evaluation requires with_inference:=true")

    mode_config = robot_config.get("control_modes", {}).get(control_mode, {})
    if not isinstance(mode_config, dict):
        raise ValueError(f"benchmark evaluation control mode {control_mode!r} is not configured")
    executor = mode_config.get("executor", {})
    if not isinstance(executor, dict) or executor.get("type") != "benchmark":
        raise ValueError("benchmark evaluation requires control_modes.<mode>.executor.type=benchmark")
    dispatch = mode_config.get("dispatch", {}) or {}
    if not isinstance(dispatch, dict) or dispatch.get("scheduler", "continuous") != "wait_for_feedback":
        raise ValueError("benchmark evaluation requires dispatch.scheduler=wait_for_feedback")

    try:
        inference = parse_inference_config(robot_config, control_mode)
    except InferenceConfigError as exc:
        raise ValueError(f"benchmark evaluation inference configuration is invalid: {exc}") from exc
    if not inference.enabled or not inference.pipelines:
        raise ValueError("benchmark evaluation requires an enabled inference pipeline")
    selected = executor.get("inference_pipeline")
    if selected is None:
        if len(inference.pipelines) != 1:
            raise ValueError("benchmark evaluation requires executor.inference_pipeline when multiple pipelines exist")
        selected = next(iter(inference.pipelines))
    if not isinstance(selected, str) or selected not in inference.pipelines:
        raise ValueError(f"benchmark evaluation selects unknown inference pipeline {selected!r}")
    pipeline = inference.pipelines[selected]
    if pipeline.execution_mode not in {"monolithic", "distributed"}:
        raise ValueError(f"benchmark evaluation selected pipeline {selected!r} has invalid execution mode")
    transport = pipeline.transport
    required_endpoints = {
        "action_server": transport.action_server,
        "reset_service": transport.reset_service,
        "health_topic": transport.health_topic,
    }
    if pipeline.execution_mode == "distributed":
        required_endpoints.update(
            {
                "request_topic": transport.request_topic,
                "result_topic": transport.result_topic,
                "heartbeat_topic": transport.heartbeat_topic,
                "video_descriptor_topic": transport.video_descriptor_topic,
                "video_status_topic": transport.video_status_topic,
            }
        )
    missing = sorted(name for name, value in required_endpoints.items() if not isinstance(value, str) or not value)
    if missing:
        raise ValueError(
            f"benchmark evaluation selected pipeline {selected!r} is missing required endpoints: {missing}"
        )


def _apply_inference_cli_overrides(context, robot_config: dict, control_mode: str) -> None:
    """Apply an explicit execution-mode override to one named inference pipeline."""
    execution_mode = context.launch_configurations.get("inference_execution_mode", "").strip().lower()
    pipeline_id = context.launch_configurations.get("inference_pipeline", "").strip()
    if not execution_mode:
        if pipeline_id:
            raise ValueError("inference_pipeline requires inference_execution_mode")
        return
    if execution_mode not in {"monolithic", "distributed"}:
        raise ValueError("inference_execution_mode must be 'monolithic' or 'distributed'")
    if not pipeline_id:
        raise ValueError("inference_execution_mode requires inference_pipeline")

    inference = robot_config.get("control_modes", {}).get(control_mode, {}).get("inference", {})
    pipelines = inference.get("pipelines", {})
    if pipeline_id not in pipelines:
        raise ValueError(
            f"inference_pipeline selects unknown pipeline {pipeline_id!r}; available pipelines: {list(pipelines)}"
        )
    previous = pipelines[pipeline_id].get("execution_mode")
    pipelines[pipeline_id]["execution_mode"] = execution_mode
    logger.info(
        f"CLI override: control_modes.{control_mode}.inference.pipelines.{pipeline_id}."
        f"execution_mode={execution_mode} (was {previous})"
    )


def _start_actions_on_success(start_actions, success_message: str, failure_reason: str):
    """Run launch actions only when the target process exits successfully."""
    frozen_actions = tuple(start_actions)

    def _handler(event, _context):
        if event.returncode == 0:
            logger.info(success_message)
            return list(frozen_actions)

        logger.error(f"{failure_reason} (returncode={event.returncode})")
        return [EmitEvent(event=Shutdown(reason=failure_reason))]

    return _handler


def _serialize_process_startup(processes, sequence_name: str):
    """Start each process only after the previous process exits successfully."""
    frozen_processes = tuple(processes)
    if not frozen_processes:
        return []

    actions = []
    for index in range(len(frozen_processes) - 1):
        current_process = frozen_processes[index]
        next_process = frozen_processes[index + 1]
        actions.append(
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=current_process,
                    on_exit=_start_actions_on_success(
                        [next_process],
                        success_message=f"{sequence_name} step {index + 1} completed; starting next step.",
                        failure_reason=f"{sequence_name} step {index + 1} failed; aborting launch.",
                    ),
                )
            )
        )
    actions.append(frozen_processes[0])
    return actions


def _controller_readiness_barrier(deferred_spawners, readiness_waiter):
    """Use the single strict group spawner as the owned-startup barrier."""
    if len(deferred_spawners) > 1:
        raise ValueError("Controller startup must use a single controller group spawner.")
    return deferred_spawners[0] if deferred_spawners else readiness_waiter


def _resolve_controller_startup_timeout(robot_config: dict, use_sim: bool) -> float:
    """Resolve controller startup timeout from robot YAML."""
    configured_timeout = robot_config.get("controller_startup_timeout")
    if configured_timeout is None:
        return 120.0 if use_sim else 30.0

    timeout_value = configured_timeout
    if isinstance(configured_timeout, dict):
        profile_key = "sim" if use_sim else "hardware"
        timeout_value = configured_timeout.get(profile_key)
        if timeout_value is None:
            timeout_value = configured_timeout.get("default")

    if timeout_value is None:
        raise ValueError("robot.controller_startup_timeout must define a value for the active launch profile.")

    try:
        timeout = float(timeout_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("robot.controller_startup_timeout must be a number or a mapping of numbers.") from exc

    if timeout <= 0.0:
        raise ValueError("robot.controller_startup_timeout must be greater than zero.")

    return timeout


def _create_controller_ready_waiter(robot_config: dict, controller_names, use_sim: bool):
    """Create a probe that exits when the required controllers are active."""
    timeout = _resolve_controller_startup_timeout(robot_config, use_sim)
    return Node(
        package="robot_config",
        executable="wait_for_controllers",
        name="wait_for_active_controllers",
        parameters=[{"use_sim_time": use_sim}],
        arguments=[
            *controller_names,
            "--controller-manager",
            "controller_manager",
            "--timeout",
            str(timeout),
        ],
        output="screen",
    )


def launch_setup(context, *args, **kwargs):
    """Launch setup function that generates all nodes.

    This is the "orchestrator" that:
    1. Loads and normalizes all parameters
    2. Calls each builder module to generate nodes
    3. Returns the combined actions list

    Args:
        context: Launch context

    Returns:
        List of launch actions
    """
    actions = []
    controller_dependent_actions = []

    # ========== 1. Get and normalize launch parameters ==========
    robot_config_name = context.launch_configurations.get("robot_config", "test_cam")
    config_path_override = context.launch_configurations.get("config_path", "")
    use_sim_str = context.launch_configurations.get("use_sim", "")
    sim_platform_override = context.launch_configurations.get("sim_platform", "").strip().lower()
    auto_start_controllers = context.launch_configurations.get("auto_start_controllers", "true")
    control_mode_override = context.launch_configurations.get("control_mode", "")
    nav_stage = context.launch_configurations.get("nav_stage", "").strip()
    hand_profile_override = context.launch_configurations.get("hand_profile", "").strip().lower()
    voice_asr_auto_start_str = context.launch_configurations.get("voice_asr_auto_start", "")
    with_embodied_str = context.launch_configurations.get("with_embodied", "")
    with_perception_str = context.launch_configurations.get("with_perception", "")
    runtime_target_override = context.launch_configurations.get("runtime_target", "")

    logger.info("========== Launch Parameters ==========")
    logger.info(f"robot_config: {robot_config_name}")
    logger.info(f"config_path: {config_path_override if config_path_override else '(none)'}")
    logger.info(f"use_sim: {use_sim_str if use_sim_str else '(infer from runtime target)'}")
    logger.info(f"sim_platform: {sim_platform_override if sim_platform_override else '(from config)'}")
    logger.info(f"auto_start_controllers: {auto_start_controllers}")
    logger.info(f"control_mode: {control_mode_override if control_mode_override else '(from config)'}")
    logger.info(f"nav_stage: {nav_stage if nav_stage else '(from config)'}")
    logger.info(f"hand_profile: {hand_profile_override if hand_profile_override else '(from config)'}")
    logger.info(f"voice_asr_auto_start: {voice_asr_auto_start_str}")
    logger.info(f"with_embodied: {with_embodied_str if with_embodied_str else '(from config)'}")
    logger.info(f"with_perception: {with_perception_str if with_perception_str else '(from config)'}")
    logger.info(f"runtime_target: {runtime_target_override if runtime_target_override else '(from SSOT or use_sim)'}")

    # ========== 2. Load robot configuration ==========
    try:
        robot_config = load_robot_config(
            robot_config_name,
            config_path_override if config_path_override else None,
            nav_stage,
        )
    except Exception as e:
        logger.error(f"loading config: {e}")
        raise

    # Store config path for downstream modules (e.g., recording)
    if config_path_override:
        robot_config["_config_path"] = config_path_override
    else:
        try:
            robot_config_share = get_package_share_directory("robot_config")
        except Exception:
            robot_config_share = str(Path(__file__).parent.parent)
        robot_config["_config_path"] = str(Path(robot_config_share) / "config" / "robots" / f"{robot_config_name}.yaml")

    # ========== 2.5 Resolve runtime target (SSOT) ==========
    # resolve_runtime_target is the single authority for the runtime target.
    # An omitted use_sim argument is represented by None; the resolved target
    # then determines the virtual/hardware embodiment used by the launch plan.
    requested_use_sim = None if use_sim_str == "" else parse_bool(use_sim_str, default=False)
    runtime_target = resolve_runtime_target(robot_config, runtime_target_override, requested_use_sim)
    use_sim = runtime_target is not RuntimeTarget.HARDWARE
    is_benchmark = runtime_target is RuntimeTarget.BENCHMARK
    logger.info(f"Runtime target: {runtime_target}")
    logger.info(f"Effective use_sim: {use_sim}")
    if is_benchmark:
        print("[IBROBOT_BENCHMARK][RUNTIME_TARGET] benchmark")
        print("[IBROBOT_BENCHMARK][SKIP] ros2_control")
        print("[IBROBOT_BENCHMARK][SKIP] simulation_backend")
        print("[IBROBOT_BENCHMARK][SKIP] physical_perception")

    sim_platform = str(robot_config.get("simulation", {}).get("platform", "gazebo")).lower()
    if sim_platform_override:
        logger.info(f"CLI override: simulation.platform={sim_platform_override} (was {sim_platform})")
        sim_platform = sim_platform_override
        robot_config.setdefault("simulation", {})["platform"] = sim_platform
    if is_benchmark:
        backend_caps = {"provides_clock": False, "needs_ros2_control": False}
    else:
        backend_caps = (
            get_backend_caps(sim_platform)
            if use_sim
            else {
                "provides_clock": False,
                "needs_ros2_control": True,
            }
        )
    mock_backend_active = (use_sim and sim_platform == "mock") and not is_benchmark
    node_use_sim_time = use_sim and backend_caps["provides_clock"]
    sim_backend_needs_ros2_control = ((not use_sim) or backend_caps["needs_ros2_control"]) and not is_benchmark
    if use_sim and not backend_caps["needs_ros2_control"] and not is_benchmark:
        auto_start_controllers = "false"
        logger.info(f"simulation.platform={sim_platform}: backend does not use ros2_control")

    # ========== 3. Apply control mode override ==========
    if control_mode_override:
        robot_config["default_control_mode"] = control_mode_override

    active_hand_profile = apply_hand_profile(robot_config, hand_profile_override)
    if active_hand_profile is not None:
        logger.info(f"Active hand profile: {active_hand_profile}")

    voice_asr_auto_start = parse_bool(voice_asr_auto_start_str, default=False)
    if voice_asr_auto_start:
        voice_asr_cfg = robot_config.setdefault("voice_asr", {})
        voice_asr_cfg["enabled"] = True
        voice_asr_cfg["active_mode"] = "continuous"
        logger.info(
            "CLI override: voice_asr.enabled=true, voice_asr.active_mode=continuous; "
            "bundle_path and deployment must be configured explicitly"
        )

    if with_embodied_str != "":
        embodied_cfg = robot_config.setdefault("embodied", {})
        embodied_cfg["enabled"] = parse_bool(with_embodied_str, default=False)
        logger.info(f"CLI override: embodied.enabled={embodied_cfg['enabled']}")

    if with_perception_str != "":
        embodied_cfg = robot_config.setdefault("embodied", {})
        perception_cfg = embodied_cfg.setdefault("perception", {})
        perception_cfg["enabled"] = parse_bool(with_perception_str, default=False)
        logger.info(f"CLI override: embodied.perception.enabled={perception_cfg['enabled']}")

    active_control_mode = robot_config.get("default_control_mode", "model_inference")
    logger.info(f"Active control mode: {active_control_mode}")
    _apply_inference_cli_overrides(context, robot_config, active_control_mode)
    if mock_backend_active:
        validate_mock_control_mode(active_control_mode)

    _apply_voice_asr_cli_overrides(context, robot_config)
    voice_asr_config = robot_config.get("voice_asr", {})
    logger.info(
        "Voice ASR override state: "
        f"enabled={voice_asr_config.get('enabled', False)}, "
        f"realtime_pre_roll_seconds={voice_asr_config.get('realtime_pre_roll_seconds', 0.5)}"
    )

    # Determine with_inference flag globally
    with_inference_str = context.launch_configurations.get("with_inference", "")
    if with_inference_str != "":
        with_inference = parse_bool(with_inference_str, default=False)
    else:
        control_mode_config = robot_config.get("control_modes", {}).get(active_control_mode, {})
        with_inference = control_mode_config.get("inference", {}).get("enabled", False)

    # Force disable inference in teleop mode if not explicitly overridden
    if active_control_mode == "teleop" and with_inference_str == "":
        with_inference = False
        logger.info("Teleop mode: forcing with_inference=False")

    logger.info(f"Final with_inference={with_inference}")
    scheduler_enabled = with_inference and scheduler_enabled_from_raw_config(robot_config, active_control_mode)
    if scheduler_enabled and runtime_target is RuntimeTarget.SIMULATION:
        raise ValueError("simulation does not support inference.scheduler.enable=true; set scheduler.enable=false")

    # Benchmark-specific transport is derived only after all launch overlays and
    # the effective runtime target are known. The generic loader stays neutral.
    if is_benchmark:
        materialize_benchmark_observation_transport(robot_config)
        _validate_benchmark_inference_closure(
            robot_config,
            control_mode=active_control_mode,
            with_inference=with_inference,
        )

    # ========== 4. Generate Control System Nodes ==========
    logger.info("========== Generating Control Nodes ==========")
    validate_runtime_resources(robot_config, use_sim=use_sim, control_mode=active_control_mode)
    confirm_interactive_startup_p_pose(
        robot_config,
        use_sim=use_sim,
        control_mode=active_control_mode,
    )
    deferred_controller_spawners = []
    controller_names = []
    robot_description = {}
    if is_benchmark:
        # ros2_control skipped by benchmark target; the
        # [IBROBOT_BENCHMARK][SKIP] ros2_control marker is already printed at
        # runtime target resolution.
        logger.info("benchmark target: skipping ros2_control / controller spawners")
    elif not sim_backend_needs_ros2_control:
        logger.info(f"simulation.platform={sim_platform}: skipping ros2_control / controller spawners")
    else:
        try:
            control_nodes, controller_names, deferred_controller_spawners, robot_description = (
                generate_ros2_control_nodes(
                    robot_config,
                    use_sim,
                    auto_start_controllers,
                    controller_startup_timeout=_resolve_controller_startup_timeout(robot_config, use_sim),
                )
            )
            actions.extend(control_nodes)
            logger.info(f"Added {len(control_nodes)} control nodes")
        except Exception as e:
            logger.error(f"generating control nodes: {e}")
            raise

    auxiliary_actuator_nodes = generate_auxiliary_actuator_nodes(
        robot_config,
        use_sim=use_sim,
        control_mode=active_control_mode,
    )
    actions.extend(auxiliary_actuator_nodes)
    if auxiliary_actuator_nodes:
        logger.info(f"Added {len(auxiliary_actuator_nodes)} auxiliary actuator node(s)")

    hand_source_nodes = generate_hand_source_nodes(
        robot_config,
        use_sim=use_sim,
        control_mode=active_control_mode,
    )
    actions.extend(hand_source_nodes)
    if hand_source_nodes:
        logger.info(f"Added {len(hand_source_nodes)} shared hand source node(s)")

    controller_ready_waiter = None
    if parse_bool(auto_start_controllers, default=True) and controller_names:
        controller_ready_waiter = _create_controller_ready_waiter(
            robot_config,
            controller_names,
            use_sim,
        )
    controller_ready_barrier = _controller_readiness_barrier(
        deferred_controller_spawners,
        controller_ready_waiter,
    )

    # ========== 5. Generate Simulation Nodes (only in simulation mode) ==========
    gz_create_entity = None
    if use_sim and not is_benchmark:
        logger.info("========== Generating Simulation Nodes ==========")
        logger.info(f"Sim platform: {sim_platform}")
        try:
            sim_adapter = get_sim_backend(sim_platform)
            sim_nodes, gz_create_entity = sim_adapter.start_backend(robot_config)
            sim_nodes += sim_adapter.spawn_peripheral_bridges(robot_config.get("peripherals", []))
            actions.extend(sim_nodes)
            logger.info(f"Added {len(sim_nodes)} simulation nodes ({sim_platform})")

            # Scene task node for randomisation and AutoTest evaluation
            scene_name = robot_config.get("simulation", {}).get("scene", "")
            if sim_platform == "mujoco" and scene_name == "pick_banana":
                from launch_ros.actions import Node as LaunchNode  # noqa: PLC0415

                actions.append(
                    LaunchNode(
                        package="sim_models",
                        executable="pick_banana_task_node",
                        name="pick_banana_task_node",
                        parameters=[{"use_sim_time": True}],
                        output="screen",
                    )
                )
                logger.info("Added pick_banana_task_node")
        except NotImplementedError:
            logger.warning(
                f"sim platform '{sim_platform}' not implemented yet, "
                f"skipping simulation nodes (set simulation.platform: gazebo to use Gazebo)"
            )
        except Exception as e:
            logger.error(f"generating simulation nodes: {e}")
            raise
    elif is_benchmark:
        # simulation_backend skipped by benchmark target; the
        # [IBROBOT_BENCHMARK][SKIP] simulation_backend marker is already
        # printed at runtime target resolution. Benchmark must never enter the
        # SimBackendAdapter registry even though use_sim=true.
        logger.info("benchmark target: skipping simulation backend (SimBackendAdapter)")

    if deferred_controller_spawners:
        startup_processes = list(deferred_controller_spawners)
        startup_sequence = _serialize_process_startup(startup_processes, "Controller startup")
        if use_sim and gz_create_entity is not None:
            logger.info("Scheduling controller startup after ros_gz_sim create exits")
            actions.append(
                RegisterEventHandler(
                    event_handler=OnProcessExit(
                        target_action=gz_create_entity,
                        on_exit=_start_actions_on_success(
                            startup_sequence,
                            success_message="Robot entity created; starting controller startup sequence.",
                            failure_reason="Robot entity creation failed; aborting launch.",
                        ),
                    )
                )
            )
        elif use_sim:
            actions.extend(startup_sequence)
        else:
            logger.info("Starting controller startup sequence; spawners wait for controller_manager readiness")
            actions.extend(startup_sequence)
    elif controller_ready_waiter is not None:
        actions.append(controller_ready_waiter)

    # ========== 6. Generate Perception Nodes ==========
    logger.info("========== Generating Perception Nodes ==========")
    if is_benchmark:
        # physical_perception skipped by benchmark target; the
        # [IBROBOT_BENCHMARK][SKIP] physical_perception marker is already
        # printed at runtime target resolution.
        logger.info("benchmark target: skipping camera / lidar / virtual-relay / TF nodes")
    elif mock_mode_skips_subsystem(mock_backend_active, "perception"):
        logger.info("hardware_mock active: skipping camera / lidar / virtual-relay / TF nodes")
        logger.info("simulation.platform=mock: contract_mock started by simulation backend")
    else:
        try:
            perception_nodes = []
            # Camera nodes (Physical drivers)
            camera_nodes = generate_camera_nodes(robot_config, use_sim)
            perception_nodes.extend(camera_nodes)
            logger.info(f"Added {len(camera_nodes)} camera nodes")

            # LiDAR nodes (Physical drivers)
            lidar_nodes = generate_lidar_nodes(robot_config, use_sim)
            perception_nodes.extend(lidar_nodes)
            print(f"[robot_config] Added {len(lidar_nodes)} lidar nodes")

            # Virtual camera relay nodes (Topic tools)
            from robot_config.launch_builders.perception import (
                generate_virtual_camera_relays,
            )

            virtual_nodes = generate_virtual_camera_relays(robot_config)
            perception_nodes.extend(virtual_nodes)
            if virtual_nodes:
                logger.info(f"Added {len(virtual_nodes)} virtual camera relays")

            # Static TF publishers
            tf_nodes = generate_tf_nodes(robot_config, use_sim)
            perception_nodes.extend(tf_nodes)
            logger.info(f"Added {len(tf_nodes)} TF nodes")

            if controller_ready_barrier is not None:
                controller_dependent_actions.extend(perception_nodes)
                logger.info(
                    f"Deferring {len(perception_nodes)} physical perception action(s) "
                    "until required controllers are active"
                )
            else:
                actions.extend(perception_nodes)

        except Exception as e:
            logger.error(f"generating perception nodes: {e}")
            raise

    # ========== 6.5 Benchmark nodes (benchmark target only) ==========
    if is_benchmark:
        benchmark_nodes = generate_benchmark_nodes(robot_config)
        actions.extend(benchmark_nodes)
        logger.info(f"Added {len(benchmark_nodes)} benchmark nodes")

    try:
        model_service_nodes = generate_perception_model_nodes(robot_config)
        if controller_ready_barrier is not None:
            controller_dependent_actions.extend(model_service_nodes)
            if model_service_nodes:
                logger.info(
                    f"Deferring {len(model_service_nodes)} generic model service node(s) "
                    "until required controllers are active"
                )
        else:
            actions.extend(model_service_nodes)
            if model_service_nodes:
                logger.info(f"Added {len(model_service_nodes)} generic model service nodes")
    except Exception as e:
        logger.error(f"generating model service nodes: {e}")
        raise

    # ========== 6.5 Semantic Mapping Node (online updates) ==========
    try:
        from robot_config.launch_builders.semantic_mapping import generate_semantic_mapping_nodes

        semantic_mapping_nodes = generate_semantic_mapping_nodes(robot_config)
        if semantic_mapping_nodes:
            if controller_ready_barrier is not None:
                controller_dependent_actions.extend(semantic_mapping_nodes)
                logger.info("Deferring semantic mapping node until required controllers are active")
            else:
                actions.extend(semantic_mapping_nodes)
            logger.info(f"Added {len(semantic_mapping_nodes)} semantic mapping node(s)")
    except Exception as e:
        logger.error(f"generating semantic mapping nodes: {e}")
        raise

    # ========== 7. Generate Teleop Nodes (if in teleop mode) ==========
    logger.info("========== Checking Teleop Mode ==========")
    try:
        # Check if teleop mode is enabled
        _teleop_modes = ("teleop",)
        if active_control_mode in _teleop_modes:
            logger.info(f"TELEOP MODE DETECTED ({active_control_mode})")

            # Check if teleoperation is configured
            teleop_config = robot_config.get("teleoperation", {})
            if not teleop_config.get("enabled", False):
                logger.info("WARNING: Teleop mode requested but teleoperation config not found")
            else:
                # Generate teleop nodes
                teleop_nodes = generate_teleop_nodes(robot_config, robot_description)

                if controller_ready_barrier is not None:
                    logger.info("Deferring teleop nodes until required controllers are active...")
                    controller_dependent_actions.extend(teleop_nodes)
                else:
                    logger.info("No controller readiness probe active, launching teleop immediately")
                    actions.extend(teleop_nodes)

                logger.info(f"Prepared {len(teleop_nodes)} teleop nodes")
        else:
            logger.info(f"Skipping teleop nodes (mode is {active_control_mode})")
    except Exception as e:
        logger.error(f"checking teleop mode: {e}")
        raise

    # ========== 7.5 Generate shared Audio IO ==========
    logger.info("========== Checking shared Audio IO ==========")
    try:
        from robot_config.launch_builders.audio_io import generate_audio_io_actions

        audio_io_actions = generate_audio_io_actions(robot_config)
        actions.extend(audio_io_actions)
    except Exception as e:
        logger.error(f"generating shared audio IO: {e}")
        raise

    # ========== 8. Generate Voice ASR Nodes ==========
    logger.info("========== Checking Voice ASR ==========")
    if mock_mode_skips_subsystem(mock_backend_active, "voice_asr"):
        logger.info("hardware_mock active: skipping voice ASR nodes (out of mock scope)")
    else:
        try:
            from robot_config.launch_builders.voice_asr import generate_voice_asr_nodes

            voice_asr_nodes = generate_voice_asr_nodes(robot_config)
            actions.extend(voice_asr_nodes)
            if voice_asr_nodes:
                logger.info(f"Added {len(voice_asr_nodes)} voice ASR node(s)")
        except Exception as e:
            logger.error(f"generating voice ASR nodes: {e}")
            raise

    # ========== 9. Generate Voice TTS Nodes ==========
    logger.info("========== Checking Voice TTS ==========")
    try:
        from robot_config.launch_builders.voice_tts import generate_voice_tts_nodes

        voice_tts_nodes = generate_voice_tts_nodes(robot_config)
        actions.extend(voice_tts_nodes)
        if voice_tts_nodes:
            logger.info(f"Added {len(voice_tts_nodes)} voice TTS node(s)")
    except Exception as e:
        logger.error(f"generating voice TTS nodes: {e}")
        raise

    # ========== 9.5 Generate Speech Direction Node ==========
    logger.info("========== Checking Speech Direction ==========")
    if mock_mode_skips_subsystem(mock_backend_active, "speech_direction"):
        logger.info("hardware_mock active: skipping speech direction node (out of mock scope)")
    else:
        try:
            from robot_config.launch_builders.speech_direction import generate_speech_direction_actions

            speech_direction_actions = generate_speech_direction_actions(robot_config)
            actions.extend(speech_direction_actions)
            if speech_direction_actions:
                logger.info("Added speech direction launch")
        except Exception as e:
            logger.error(f"generating speech direction node: {e}")
            raise

    # ========== 10. Generate Navigation Nodes ==========
    logger.info("========== Checking Navigation ==========")
    if mock_mode_skips_subsystem(mock_backend_active, "navigation"):
        logger.info("hardware_mock active: skipping navigation nodes (out of mock scope)")
    else:
        try:
            with_navigation_str = context.launch_configurations.get("with_navigation", "")
            navigation_mode = context.launch_configurations.get("navigation_mode", "")
            navigation_config = robot_config.get("navigation", {})

            if with_navigation_str != "":
                with_navigation = parse_bool(with_navigation_str, default=False)
            else:
                with_navigation = navigation_config.get("enabled", False)

            if with_navigation:
                navigation_nodes = generate_navigation_nodes(
                    robot_config,
                    use_sim=node_use_sim_time,
                    navigation_mode=navigation_mode,
                    force_enable=True,
                )
                if controller_ready_barrier is not None:
                    logger.info("Deferring navigation nodes until required controllers are active...")
                    controller_dependent_actions.extend(navigation_nodes)
                else:
                    actions.extend(navigation_nodes)
                logger.info(f"Prepared {len(navigation_nodes)} navigation node(s)")
            else:
                logger.info("Skipping navigation nodes")
        except Exception as e:
            logger.error(f"generating navigation nodes: {e}")
            raise

    # ========== 10. Generate Execution Nodes ==========
    logger.info("========== Generating Execution Nodes ==========")
    try:
        if with_inference:
            execution_nodes = generate_execution_nodes(
                robot_config,
                active_control_mode,
                use_sim,
                use_sim_time=node_use_sim_time,
                runtime_target=runtime_target,
            )
            if controller_ready_barrier is not None:
                logger.info("Deferring execution nodes until required controllers are active...")
                controller_dependent_actions.extend(execution_nodes)
            else:
                actions.extend(execution_nodes)
            logger.info(f"Prepared {len(execution_nodes)} execution nodes")
        else:
            logger.info("Skipping execution nodes")
    except Exception as e:
        logger.error(f"generating execution nodes: {e}")
        raise

    # ========== 11. Generate MoveIt Nodes ==========
    try:
        # Determine with_moveit flag
        with_moveit_str = context.launch_configurations.get("with_moveit", "")
        moveit_display = parse_bool(context.launch_configurations.get("moveit_display", "true"), default=True)

        if with_moveit_str != "":
            with_moveit = parse_bool(with_moveit_str, default=False)
        else:
            with_moveit = "moveit" in active_control_mode.lower()

        logger.info(f"with_moveit={with_moveit}")

        if with_moveit:
            from robot_config.launch_builders.moveit import generate_moveit_nodes

            moveit_nodes = generate_moveit_nodes(
                robot_config,
                active_control_mode,
                node_use_sim_time,
                moveit_display,
                # Force MoveIt launch when the user explicitly set
                # with_moveit:=true. Without this,
                # with_moveit:=true + control_mode=teleop would call
                # generate_moveit_nodes(force=False) which tests
                # 'moveit' in 'teleop' → False → returns [], starting nothing.
                force=parse_bool(with_moveit_str, default=False),
            )

            if controller_ready_barrier is not None:
                logger.info("Deferring MoveIt nodes until required controllers are active...")
                controller_dependent_actions.extend(moveit_nodes)
            else:
                logger.info("No controller readiness probe active, launching MoveIt immediately")
                actions.extend(moveit_nodes)
        else:
            logger.info("Skipping MoveIt nodes")
    except Exception as e:
        logger.error(f"generating MoveIt nodes: {e}")
        logger.info("Continuing without MoveIt...")

    # ========== 10.5 Generate Embodied Minimal-Closure Nodes ==========
    if robot_config.get("embodied", {}).get("enabled", False):
        logger.warning(
            "robot_config no longer launches embodied runtime nodes directly. "
            "Use the dedicated embodied bringup launch entry to start "
            "the base robot launch plus embodied runtime nodes."
        )

    # ========== 11.5 Generate Task Executor Node ==========
    try:
        if with_moveit:
            from robot_config.launch_builders.task_execution import generate_task_executor_node

            task_node = generate_task_executor_node(robot_config, active_control_mode, node_use_sim_time)
            if task_node is not None:
                if controller_ready_barrier is not None:
                    controller_dependent_actions.append(task_node)
                else:
                    actions.append(task_node)
                print("[robot_config] Task executor node added")
    except Exception as e:
        logger.error(f"generating required task executor: {e}")
        raise RuntimeError("MoveIt task executor setup failed; refusing to start an incomplete motion stack") from e

    # ========== 12. Automatic Recording ==========
    try:
        record_str = context.launch_configurations.get("record", "false")
        record_mode = context.launch_configurations.get("record_mode", "continuous")
        record_enabled, record_mode = resolve_recording_launch(
            robot_config,
            requested=parse_bool(record_str, default=False),
            mode=record_mode,
        )

        if record_enabled:
            logger.info(f"========== Setting up Recording (mode: {record_mode}) ==========")

            # Generate recording nodes using the recording builder
            recording_nodes = generate_recording_nodes(
                robot_config,
                active_control_mode,
                record_mode,
                scheduler_enabled=scheduler_enabled,
            )
            actions.extend(recording_nodes)
            logger.info(f"Added {len(recording_nodes)} recording node(s)")
        else:
            logger.info(f"Recording disabled (record:={record_str})")
    except Exception as e:
        logger.error(f"setting up recording: {e}")
        if robot_config.get("recording", {}).get("semantic_dataset", False):
            actions.append(EmitEvent(event=Shutdown(reason=f"semantic dataset recording setup failed: {e}")))
        else:
            logger.info("Continuing without recording...")

    # ========== 12. Recording Visualizer (optional rerun sidecar) ==========
    try:
        record_viz = context.launch_configurations.get("record_visualizer", "none").lower()
        if record_viz == "rerun":
            logger.info("========== Setting up Rerun Visualizer ==========")
            rerun_nodes = generate_rerun_viewer_node(robot_config)
            actions.extend(rerun_nodes)
            logger.info(f"Added {len(rerun_nodes)} rerun viewer node(s)")
        elif record_viz != "none":
            logger.warning(f"Unknown record_visualizer value: '{record_viz}' (expected 'rerun' or 'none')")
    except Exception as e:
        logger.error(f"setting up rerun visualizer: {e}")
        logger.info("Continuing without recording visualizer...")

    if controller_dependent_actions:
        if controller_ready_barrier is not None:
            logger.info(
                f"Controller readiness barrier armed for "
                f"{len(controller_dependent_actions)} control-dependent action(s)"
            )
            actions.insert(
                0,
                RegisterEventHandler(
                    event_handler=OnProcessExit(
                        target_action=controller_ready_barrier,
                        on_exit=_start_actions_on_success(
                            controller_dependent_actions,
                            success_message="Required controllers are active; starting control-dependent nodes.",
                            failure_reason="Controller readiness probe failed; aborting launch.",
                        ),
                    )
                ),
            )
        else:
            actions.extend(controller_dependent_actions)

    # ========== N. Tracing (optional, ros2_tracing + LTTng) ==========
    enable_tracing = parse_bool(context.launch_configurations.get("enable_tracing", "false"), default=False)
    if enable_tracing:
        requested_trace_session = context.launch_configurations.get("trace_session_name", DEFAULT_TRACE_SESSION_NAME)
        actions[:0] = generate_tracing_actions(enable_tracing=True, requested_session_name=requested_trace_session)

    logger.info(f"========== Total nodes to launch: {len(actions)} ==========")

    return actions


def generate_launch_description():
    """Generate launch description for robot system."""
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot_config",
                default_value="so101_single_arm",
                description="Robot configuration name (without .yaml extension)",
            ),
            DeclareLaunchArgument(
                "config_path",
                default_value="",
                description="Optional: Full path to robot config file (overrides robot_config)",
            ),
            DeclareLaunchArgument(
                "use_sim",
                default_value="",
                description=(
                    "Override virtual embodiment (true/false). Empty infers true for "
                    "runtime.target benchmark/simulation and false for hardware."
                ),
            ),
            DeclareLaunchArgument(
                "sim_platform",
                default_value="",
                description="Override simulation.platform from YAML (gazebo, mujoco, or mock). Empty uses YAML.",
            ),
            DeclareLaunchArgument(
                "auto_start_controllers",
                default_value="true",
                description="Automatically spawn controllers (set to false for debugging)",
            ),
            DeclareLaunchArgument(
                "control_mode",
                default_value="",
                description="Override control mode from YAML (teleop, model_inference, or moveit_planning). If empty, uses default_control_mode from config file",
            ),
            DeclareLaunchArgument(
                "nav_stage",
                default_value="",
                description=(
                    "Select a workflow stage declared by the robot config. "
                    "Navigation profiles use mapping/navigation; combined mobile-manipulator profiles may also "
                    "declare grasp/hybrid. Empty uses default_nav_stage."
                ),
            ),
            DeclareLaunchArgument(
                "hand_profile",
                default_value="",
                description="Select a hand profile declared by the robot YAML (for example right, left, or dual).",
            ),
            DeclareLaunchArgument(
                "voice_asr_realtime_pre_roll_seconds",
                default_value="",
                description="Optional override for robot.voice_asr.realtime_pre_roll_seconds.",
            ),
            DeclareLaunchArgument(
                "with_inference",
                default_value="",
                description="Enable full execution pipeline (inference + dispatcher). If empty, auto-detects from control mode config",
            ),
            DeclareLaunchArgument(
                "inference_pipeline",
                default_value="",
                description="Pipeline ID targeted by inference launch overrides.",
            ),
            DeclareLaunchArgument(
                "inference_execution_mode",
                default_value="",
                description="Override the targeted pipeline mode: monolithic or distributed. Empty uses YAML.",
            ),
            DeclareLaunchArgument(
                "with_moveit",
                default_value="",
                description="Enable MoveIt motion planning. If empty, auto-detects from control mode config",
            ),
            DeclareLaunchArgument(
                "with_navigation",
                default_value="",
                description="Enable navigation nodes. If empty, auto-detects from robot.navigation.enabled",
            ),
            DeclareLaunchArgument(
                "navigation_mode",
                default_value="",
                description="Override robot.navigation.default_mode (for example: full, odom_only, imu_only)",
            ),
            DeclareLaunchArgument(
                "moveit_display",
                default_value="true",
                description="Launch RViz for MoveIt visualization (only used if MoveIt is enabled)",
            ),
            DeclareLaunchArgument(
                "record",
                default_value="false",
                description="Enable automatic rosbag recording (auto-discovers topics from config)",
            ),
            DeclareLaunchArgument(
                "record_mode",
                default_value="continuous",
                description="Recording mode: 'continuous' (all-in-one bag) or 'episodic' (triggered episode-by-episode via episode_recorder)",
            ),
            DeclareLaunchArgument(
                "voice_asr_auto_start",
                default_value="false",
                description="When true, enable Voice ASR and override robot.voice_asr.active_mode=continuous",
            ),
            DeclareLaunchArgument(
                "with_embodied",
                default_value="",
                description="Enable the embodied minimal closure. If empty, uses robot.embodied.enabled from YAML.",
            ),
            DeclareLaunchArgument(
                "with_perception",
                default_value="",
                description="Enable perception_service. If empty, uses robot.embodied.perception.enabled from YAML.",
            ),
            DeclareLaunchArgument(
                "record_visualizer",
                default_value="none",
                description="Recording visualizer: 'rerun' (launch Rerun sidecar for live cameras/joints/actions) or 'none' (no visualizer)",
            ),
            DeclareLaunchArgument(
                "runtime_target",
                default_value="",
                description=(
                    "Override the runtime target (hardware, simulation or benchmark). "
                    "Empty uses the SSOT runtime.target, then the historical use_sim "
                    "fallback, then the default hardware. benchmark + use_sim:=true "
                    "stays benchmark and skips ros2_control, simulation backend and "
                    "physical perception."
                ),
            ),
            DeclareLaunchArgument(
                "enable_tracing",
                default_value="false",
                description="Enable ros2_tracing / LTTng for latency analysis (requires ros2-tracing package)",
            ),
            DeclareLaunchArgument(
                "trace_session_name",
                default_value=DEFAULT_TRACE_SESSION_NAME,
                description=(
                    "Base LTTng trace session name (only used when enable_tracing:=true). "
                    "Custom names are never overwritten; the default name auto-suffixes on collision."
                ),
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
