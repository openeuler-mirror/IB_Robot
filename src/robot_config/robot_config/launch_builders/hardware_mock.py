"""Hardware mock launch builder for robot_config.

Encapsulates the orchestration for ``use_mock:=true`` so the top-level
``robot.launch.py`` orchestrator stays focused on cross-subsystem wiring
rather than hardware_mock-specific rules.

The builder owns three pieces of logic:

1. ``validate_mock_mode`` — argument compatibility checks (mutex with
   ``use_sim``, supported ``control_mode`` values).
2. ``mock_mode_skips_subsystem`` — the small predicate used by
   subsystem sections to short-circuit when running under mock.
3. ``generate_hardware_mock_nodes`` — concrete Node construction for
   the ``contract_mock`` process.

Keeping these together makes it cheap to add new mock backends, debug
parameters, or reuse the mock node from a different launch entry point
without re-implementing the rules inside ``robot.launch.py``.
"""

from typing import Any

from launch_ros.actions import Node

from robot_config.logger_utils import get_colored_logger

logger = get_colored_logger("robot_config.hardware_mock")

# Subsystems that hardware_mock intentionally replaces or has no analogue
# for. Centralised here so subsystem sections in robot.launch.py do not
# each hard-code their own knowledge of "what mock implies".
_MOCK_SKIPPED_SUBSYSTEMS = frozenset(
    {
        "control",  # contract_mock owns /joint_states; no controller_manager
        "perception",  # contract_mock publishes synthetic camera / lidar topics
        "voice_asr",  # voice ASR is out of mock scope
        "navigation",  # navigation stack is out of mock scope
    }
)

# Control modes compatible with hardware_mock. Mock currently only models
# the model_inference observation/action loop; teleop devices and MoveIt
# action servers are not implemented behind the mock surface.
_MOCK_SUPPORTED_CONTROL_MODES = frozenset({"model_inference"})


def validate_mock_mode(
    use_mock: bool,
    use_sim: bool,
    active_control_mode: str | None = None,
) -> None:
    """Validate mock-mode argument combinations.

    Designed to be called in two phases by ``robot.launch.py``:

    1. Early, with ``use_sim`` only (``active_control_mode=None``) to enforce
       the mutex before any subsystem work runs.
    2. Later, with the resolved ``active_control_mode`` to enforce the
       supported-mode invariant.

    Raises ``RuntimeError`` with an explanation of *why* the combination is
    rejected so the launch failure points at the architectural constraint
    rather than a generic ROS error downstream.
    """
    if not use_mock:
        return

    if use_sim:
        raise RuntimeError(
            "use_mock:=true is mutually exclusive with use_sim:=true. "
            "hardware_mock owns /joint_states; running Gazebo alongside would "
            "create two publishers and corrupt the inference observation."
        )

    if active_control_mode is not None and active_control_mode not in _MOCK_SUPPORTED_CONTROL_MODES:
        supported = ", ".join(sorted(_MOCK_SUPPORTED_CONTROL_MODES))
        raise RuntimeError(
            f"use_mock:=true only supports control_mode={supported}, "
            f"got '{active_control_mode}'. hardware_mock does not implement "
            "teleop devices or MoveIt action servers."
        )


def mock_mode_skips_subsystem(use_mock: bool, subsystem: str) -> bool:
    """Return ``True`` when ``subsystem`` should be skipped under mock mode.

    The caller is expected to log the skip reason itself so each subsystem
    section retains a single, locally-readable narrative.
    """
    if not use_mock:
        return False
    return subsystem in _MOCK_SKIPPED_SUBSYSTEMS


def generate_hardware_mock_nodes(robot_config: dict[str, Any]) -> list[Node]:
    """Build the hardware_mock Node list for mock-mode launches.

    Currently returns a single ``contract_mock`` node. Future mock backends
    (different topic surfaces, debug instrumentation, multi-arm fan-out)
    can be added here without growing ``robot.launch.py``.
    """
    config_path = robot_config.get("_config_path", "")
    logger.info("Generating hardware_mock contract_mock node")
    return [
        Node(
            package="hardware_mock",
            executable="contract_mock",
            name="contract_mock",
            output="screen",
            parameters=[
                {
                    "robot_config_path": config_path,
                    "use_sim_time": False,
                }
            ],
        )
    ]
