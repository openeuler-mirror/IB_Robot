"""Scene task launch builder (so101-mujoco-runtime-platform D8).

The scene declares its task (``layout.yaml`` ``task:`` section: the executable
plus the physics platforms it supports); the world owner starts that executable
once the runtime provider is live and its interfaces are bound. The builder
participates with ``timing="runtime_ready"`` and emits only on the live-binding
branch's consumer pass — the ``provider_started`` context flag is set only
inside ``construct_consumers``, so the pre-binding world bootstrap, non-binding
launches, provider-less launches and platforms the layout does not list all get
no task, and the omission is logged.

The node parameters come from the bound effective config: the trajectory
endpoints and joint names are read from the bound interface description
(``joint.trajectory_arm`` / ``joint.trajectory_gripper``), never guessed from
controller names. On Gazebo the builder also starts the single
``ros_gz_bridge`` the backend needs (set_pose service bridge plus the dynamic
pose stream remapped to ``/sim/<scene>/object_poses`` — the spike T0 choice;
the bridge is the only transport, there is no CLI fallback).
On MuJoCo the builder adds no transport: the runtime provider configures the
FreeJointStatePublisherPlugin for the scene's dynamic bodies and the backend
places through ``ResetWorld`` state overrides (issue #195).
"""

from typing import Any

from launch_ros.actions import Node

from robot_config.launch_registry.builder_support import runtime_provider_name
from robot_config.launch_registry.registry import (
    BuilderSpec,
    BuildResult,
    LaunchBuildContext,
)
from robot_config.logger_utils import get_colored_logger

logger = get_colored_logger("robot_config.launch_builders.scene_task")

_TRAJECTORY_ARM_INTERFACE = "joint.trajectory_arm"
_TRAJECTORY_GRIPPER_INTERFACE = "joint.trajectory_gripper"
_DEFAULT_TRAJECTORY_MODE = "trajectory"


def _bound_trajectory_interfaces(robot_config: dict) -> dict[str, dict] | None:
    """Read the arm/gripper trajectory interfaces from the bound config.

    Returns ``None`` when the bound description does not declare both: the
    runtime arm reset drives the controllers through these endpoints, so a
    runtime without trajectory actions cannot host the scene task.
    """
    interfaces = ((robot_config.get("runtime") or {}).get("interface_description") or {}).get("interfaces") or {}
    bound = {}
    for interface_id in (_TRAJECTORY_ARM_INTERFACE, _TRAJECTORY_GRIPPER_INTERFACE):
        entry = interfaces.get(interface_id)
        if not isinstance(entry, dict) or not entry.get("endpoint"):
            return None
        bound[interface_id] = entry
    return bound


def build_scene_task(context: LaunchBuildContext) -> BuildResult:
    """Start the scene-declared task executable on the live-binding consumer pass."""
    if not context.provider_started:
        logger.info("scene task not started: not on the live-binding consumer pass")
        return BuildResult()
    if not runtime_provider_name(context):
        logger.info("scene task not started: no runtime provider owns the transport")
        return BuildResult()
    if not context.use_sim:
        logger.info("scene task not started: not a simulation launch")
        return BuildResult()

    startup = context.simulation_startup
    robot_config = dict(context.robot_config or {})
    scene = (startup.scene_name if startup is not None else "") or str(
        (robot_config.get("simulation") or {}).get("scene", "") or ""
    )
    if not scene:
        logger.info("scene task not started: the launch has no scene")
        return BuildResult()

    try:
        from sim_models.scene_compiler import get_scene_task
    except Exception as e:  # noqa: BLE001
        logger.warning(f"scene task not started: sim_models is unavailable ({e})")
        return BuildResult()

    declaration = get_scene_task(scene)
    if declaration is None:
        logger.info(f"scene task not started: scene '{scene}' declares no task")
        return BuildResult()
    executable = str(declaration.get("executable", "") or "")
    platforms = [str(p) for p in declaration.get("platforms", [])]
    if not executable:
        logger.warning(f"scene task not started: scene '{scene}' declares no executable")
        return BuildResult()
    if context.sim_platform not in platforms:
        logger.info(
            f"scene task not started: platform '{context.sim_platform}' is not declared by "
            f"scene '{scene}' (declared: {platforms})"
        )
        return BuildResult()

    interfaces = _bound_trajectory_interfaces(robot_config)
    if interfaces is None:
        logger.warning(
            "scene task not started: the bound runtime description declares no "
            f"{_TRAJECTORY_ARM_INTERFACE}/{_TRAJECTORY_GRIPPER_INTERFACE} interfaces"
        )
        return BuildResult()

    parameters: dict[str, Any] = {
        "sim_platform": context.sim_platform,
        "use_sim_time": context.node_use_sim_time,
        "trajectory_mode": str((robot_config.get("scene_task") or {}).get("trajectory_mode", _DEFAULT_TRAJECTORY_MODE)),
        "arm_trajectory_endpoint": interfaces[_TRAJECTORY_ARM_INTERFACE]["endpoint"],
        "arm_joint_names": [str(j) for j in interfaces[_TRAJECTORY_ARM_INTERFACE].get("joint_names", [])],
        "gripper_trajectory_endpoint": interfaces[_TRAJECTORY_GRIPPER_INTERFACE]["endpoint"],
        "gripper_joint_names": [str(j) for j in interfaces[_TRAJECTORY_GRIPPER_INTERFACE].get("joint_names", [])],
    }

    actions: list[Any] = []

    if context.sim_platform == "gazebo":
        world_name = (startup.world_name if startup is not None else "") or str(
            robot_config.get("gazebo_world_name", "demo")
        )
        parameters["world_name"] = world_name
        # One bridge carries both channels (D8): the set_pose service bridge
        # the backend places objects through, and the dynamic pose stream the
        # evaluation reads, remapped to the scene-namespaced topic.
        actions.append(
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                name="scene_task_object_bridge",
                arguments=[
                    f"/world/{world_name}/set_pose@ros_gz_interfaces/srv/SetEntityPose",
                    f"/world/{world_name}/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
                ],
                remappings=[(f"/world/{world_name}/dynamic_pose/info", f"/sim/{scene}/object_poses")],
                output="screen",
            )
        )
    # On MuJoCo the task needs no extra transport: the runtime provider
    # configures the FreeJointStatePublisherPlugin for the scene's dynamic
    # bodies (issue #195) and the backend places through ResetWorld state
    # overrides — both endpoints resolve the shared topic/layout themselves.

    actions.append(
        Node(
            package="sim_models",
            executable=executable,
            name=executable,
            parameters=[parameters],
            output="screen",
        )
    )
    logger.info(
        f"Scene task '{executable}' started (scene={scene!r}, platform={context.sim_platform!r}, "
        f"trajectory_mode={parameters['trajectory_mode']!r})"
    )
    return BuildResult(actions=actions)


# ---------------------------------------------------------------------------
# Registry entry (so101-mujoco-runtime-platform D8): declared after the
# simulation entry it reads the resolved world from, and deferred behind the
# runtime readiness barrier — the task may only exist once the runtime's
# interfaces are bound.
# ---------------------------------------------------------------------------

SCENE_TASK_BUILDER = BuilderSpec(
    id="scene_task",
    factory=f"{__name__}:build_scene_task",
    requires=("simulation",),
    after=("simulation",),
    timing="runtime_ready",
)
