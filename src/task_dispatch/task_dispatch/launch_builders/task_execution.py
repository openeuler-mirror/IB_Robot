"""Task execution launch builder.

This module is ``task_dispatch``'s contribution to the launch-builder
registry (openspec change ``migrate-business-launch-builders``): the
``task_executor`` entry is declared as an entry point in the
``ibrobot.launch_builders`` group (see ``setup.py``), so the node's owning
package registers its own launch wiring without any robot_config change.
The content is a verbatim migration of the task-executor half of the former
``robot_config/launch_builders/task_execution.py`` (the ``motion_planning``
validation half stays in robot_config); the ``with_moveit`` argument stays
owned by ``motion_planning``.

This module generates the task_executor_node for task-level execution
in moveit_planning-based control modes (visual_grasp, VoxPoser, etc.).

The task_executor_node provides an ExecuteTaskPlan action server.
Planners send a sequence of TaskSteps; the executor delegates arm
motion to moveit_gateway and gripper control to ros2_control.
"""

from ament_index_python.packages import PackageNotFoundError, get_package_prefix
from launch_ros.actions import Node

from robot_config.launch_registry.builder_support import resolve_with_moveit
from robot_config.launch_registry.registry import BuilderSpec, BuildResult, LaunchBuildContext
from robot_config.logger_utils import get_colored_logger
from robot_config.utils import parse_bool

logger = get_colored_logger("robot_config.task_execution")


def generate_task_executor_node(robot_config, control_mode, use_sim=False):
    """Generate task executor node for task-level control modes.

    Args:
        robot_config: Robot configuration dict
        control_mode: Active control mode
        use_sim: Simulation mode flag

    Returns:
        Node action for task_executor, or None if not applicable
    """
    is_sim = parse_bool(use_sim, default=False)

    # Only launch for modes that use task-level execution.  A hybrid robot may
    # start in navigation mode while keeping MoveIt online for an automatic
    # transition to its configured manipulation mode.  In that case the task
    # executor is still a required manipulation dependency and must be started
    # before the first skill requests the transition.
    control_modes = robot_config.get("control_modes", {})
    mode_config = control_modes.get(control_mode, {})
    motion_mode = robot_config.get("motion_mode", {})
    manipulation_control_mode = ""
    if isinstance(motion_mode, dict) and bool(motion_mode.get("enabled", False)):
        manipulation_control_mode = str(motion_mode.get("manipulation_control_mode", "")).strip()
    elif robot_config.get("nav_stage") == "hybrid":
        # Runtime-provider hybrid stages carry no motion_mode block (the mode
        # machine is the runtime's); the manipulation mode the executor must
        # stay online for is the stage's skill mode -- a hybrid stage drives by
        # default but serves manipulation skills.
        manipulation_control_mode = str(robot_config.get("skill_required_control_mode", "") or "").strip()
    manipulation_mode_config = control_modes.get(manipulation_control_mode, {})

    # Task executor is relevant for moveit-based modes with task_dispatch enabled
    executor_config = mode_config.get("executor", {})
    candidate_modes = [(control_mode, mode_config)]
    if manipulation_control_mode and control_mode != manipulation_control_mode:
        candidate_modes.append((manipulation_control_mode, manipulation_mode_config))
        executor_config = manipulation_mode_config.get("executor", {})

    # Auto-detect: launch task_executor when either the active mode or the
    # configured manipulation mode has MoveIt/task-dispatch semantics.
    needs_task_executor = any(
        "moveit" in mode_name.lower()
        or "visual_grasp" in mode_name.lower()
        or "voxposer" in mode_name.lower()
        or mode.get("executor", {}).get("task_dispatch", False)
        for mode_name, mode in candidate_modes
    )

    if not needs_task_executor:
        print(f"[robot_config] Task executor not needed for mode '{control_mode}'")
        return None

    try:
        get_package_prefix("task_dispatch")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "MoveIt task execution requires the ROS package 'task_dispatch', but it is not present in the "
            "sourced install space. Build robot_config with its runtime dependencies and source install/setup.sh."
        ) from exc

    robot_config_path = robot_config.get("_config_path", "")
    if not robot_config_path:
        raise ValueError("robot_config dict is missing '_config_path'. Ensure loader.py injects this correctly.")

    print("[robot_config] ========== Generating Task Executor ==========")
    print(f"[robot_config] Control mode: {control_mode}")

    # One owner for the gripper's open/closed positions: the mode executor
    # block, falling back to the embodied execution block the skill layer
    # commands.  The executor only receives the resolved values.
    embodied = robot_config.get("embodied")
    embodied_execution = embodied.get("execution", {}) if isinstance(embodied, dict) else {}
    if not isinstance(embodied_execution, dict):
        embodied_execution = {}

    def _gripper_position(key: str):
        value = executor_config.get(key, embodied_execution.get(key))
        return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None

    gripper_open = _gripper_position("gripper_open_position")
    gripper_closed = _gripper_position("gripper_closed_position")
    # 0.0 means "unknown": the executor then keeps the fixed full-stroke time.
    gripper_full_stroke = (
        abs(gripper_open - gripper_closed) if gripper_open is not None and gripper_closed is not None else 0.0
    )

    node = Node(
        package="task_dispatch",
        executable="task_executor_node",
        name="task_executor",
        parameters=[
            {
                "robot_config_path": str(robot_config_path),
                "use_sim_time": is_sim,
                "skip_redundant_gripper_open": bool(executor_config.get("skip_redundant_gripper_open", False)),
                "gripper_open_position": 1.0 if gripper_open is None else gripper_open,
                "gripper_full_stroke": gripper_full_stroke,
                "gripper_position_tolerance": max(float(executor_config.get("gripper_position_tolerance", 0.05)), 0.0),
                "joint_state_max_age_s": max(float(executor_config.get("joint_state_max_age_s", 0.25)), 0.0),
            }
        ],
        output="screen",
    )

    print("[robot_config] ✓ Task executor node configured")
    return node


# ---------------------------------------------------------------------------
# Registry entry (openspec change ``decentralize-launch-builder-registration``):
# the BuilderSpec constant and build function moved here from the retired
# ``adapters.py``; the body is a verbatim migration (only the shared with_moveit
# resolution now imports from ``builder_support`` instead of the former
# ``builtin_registry`` helper). Openspec change
# ``migrate-business-launch-builders`` moved this entry from
# ``robot_config/launch_builders/task_execution.py`` to its owning package.
# ---------------------------------------------------------------------------


def build_task_executor(context: LaunchBuildContext) -> BuildResult:
    """Task executor entry; runtime_ready timing.

    The registry enable predicate owns the shared with_moveit resolution;
    a false resolution never reaches this build function. The generation
    failure policy (refusing an incomplete motion stack) stays here.
    """
    from task_dispatch.launch_builders.task_execution import generate_task_executor_node

    try:
        task_node = generate_task_executor_node(context.robot_config, context.control_mode, context.node_use_sim_time)
        if task_node is not None:
            print("[robot_config] Task executor node added")
            return BuildResult(actions=[task_node])
        return BuildResult()
    except Exception as e:
        logger.error(f"generating required task executor: {e}")
        raise RuntimeError("MoveIt task executor setup failed; refusing to start an incomplete motion stack") from e


TASK_EXECUTOR_BUILDER = BuilderSpec(
    id="task_executor",
    factory=f"{__name__}:build_task_executor",
    after=("benchmark",),
    timing="runtime_ready",
    # with_moveit is declared by motion_planning; task_executor
    # only reads it (ownership decides declaration, not reads).
    # Shared with_moveit resolution (CLI switch with the
    # control-mode fallback); build-time node generation and
    # its failure policy stay in the build function.
    enabled=resolve_with_moveit,
)
