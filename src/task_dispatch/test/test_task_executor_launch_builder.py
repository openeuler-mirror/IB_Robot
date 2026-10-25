"""Launch-builder tests for the task_executor entry (task_dispatch's contribution).

Migrated from robot_config (openspec change
``migrate-business-launch-builders``, task A.2/A.8): the tests whose subject
is ``generate_task_executor_node`` / the entry's participation predicate
followed the module to its owning package. The robot_config test files keep
the config-level assertions (moveit profile, cmd_vel bridge).

Participation is exercised through ``build_launch_registry`` with the
task_executor entry point injected (the same seam the registry tests use),
so the assertions stay hermetic — no ambient installed extensions.
"""

from importlib.metadata import EntryPoint
from pathlib import Path

import pytest
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

from robot_config.launch_registry.builtin_registry import build_launch_registry
from robot_config.launch_registry.registry import LaunchBuildContext
from robot_config.loader import load_robot_config_dict
from robot_config.runtime_target import RuntimeTarget
from task_dispatch.launch_builders.task_execution import generate_task_executor_node

CONFIG = Path(get_package_share_directory("robot_config")) / "config" / "robots" / "lekiwi_handeye_realsense_grasp.yaml"


def _launch_parameter_value(node, name: str):
    for raw_key, raw_value in node._Node__parameters[0].items():
        key = "".join(getattr(item, "text", str(item)) for item in raw_key)
        if key != name:
            continue
        if not isinstance(raw_value, tuple):
            return raw_value
        text = "".join(getattr(item, "text", str(item)) for item in raw_value).strip()
        if text.endswith("\n..."):
            text = text[:-4].rstrip()
        if text in {"True", "False"}:
            return text == "True"
        try:
            return float(text)
        except ValueError:
            return text
    raise KeyError(name)


def _registry():
    """The builtin registry plus the task_executor entry point (injected)."""
    return build_launch_registry(
        entry_points=[
            EntryPoint(
                name="task_executor",
                value="task_dispatch.launch_builders.task_execution:TASK_EXECUTOR_BUILDER",
                group="ibrobot.launch_builders",
            )
        ]
    )


def _plan_ids(**overrides) -> list[str]:
    defaults = {
        "robot_config": {"name": "test_robot"},
        "runtime_target": RuntimeTarget.HARDWARE,
        "use_sim": False,
        "sim_platform": "",
        "control_mode": "model_inference",
        "with_inference": True,
        "arguments": {},
    }
    defaults.update(overrides)
    plan = _registry().plan(LaunchBuildContext(**defaults))
    return [spec.id for spec in plan.specs]


def test_task_executor_follows_shared_with_moveit_resolution():
    """Migrated from robot_config's registry suite: CLI false beats a moveit
    control mode; an empty CLI falls back to the control mode containing
    moveit; a non-moveit mode keeps the entry out of the plan."""
    assert "task_executor" not in _plan_ids(control_mode="moveit_planning", arguments={"with_moveit": "false"})
    assert "task_executor" in _plan_ids(control_mode="moveit_planning")
    assert "task_executor" not in _plan_ids(control_mode="model_inference")


def test_task_executor_declares_after_benchmark_and_runtime_ready_timing():
    spec = _registry().snapshot()[-1]
    assert spec.id == "task_executor"
    assert spec.after == ("benchmark",)
    assert spec.timing == "runtime_ready"
    # with_moveit stays declared by robot_config's motion_planning entry;
    # task_executor only reads it (ownership decides declaration).
    assert spec.arguments == ()


# Drives a robot config whose perception services reference a gitignored model
# bundle; without scripts/download_models.py the config cannot load. A missing
# download is not a defect, so skip rather than fail. See root conftest.py.
@pytest.mark.model_bundle("grounding_dino_swint_seq8_1280x720")
def test_310p_profile_does_not_skip_the_initial_gripper_open() -> None:
    config = load_robot_config_dict(CONFIG)

    node = generate_task_executor_node(config, "moveit_planning")

    assert _launch_parameter_value(node, "skip_redundant_gripper_open") is False
    assert _launch_parameter_value(node, "gripper_open_position") == 1.0
    assert _launch_parameter_value(node, "gripper_full_stroke") == 1.0
    assert _launch_parameter_value(node, "gripper_position_tolerance") == 0.05
    assert _launch_parameter_value(node, "joint_state_max_age_s") == 0.25


# Drives a robot config whose perception services reference a gitignored model
# bundle; without scripts/download_models.py the config cannot load. A missing
# download is not a defect, so skip rather than fail. See root conftest.py.
@pytest.mark.model_bundle("grounding_dino_swint_seq8_1280x720")
def test_moveit_task_executor_fails_before_launch_when_package_is_missing(monkeypatch) -> None:
    import task_dispatch.launch_builders.task_execution as task_execution_builder

    config = load_robot_config_dict(CONFIG)

    def raise_package_not_found(_package_name: str):
        raise PackageNotFoundError("task_dispatch")

    monkeypatch.setattr(task_execution_builder, "get_package_prefix", raise_package_not_found)

    with pytest.raises(RuntimeError, match="task_dispatch.*not present"):
        generate_task_executor_node(config, "moveit_planning")


def test_task_executor_gripper_stroke_has_one_owner_and_is_unknown_without_both_positions(monkeypatch) -> None:
    import task_dispatch.launch_builders.task_execution as task_execution_builder

    # generate_task_executor_node's get_package_prefix("task_dispatch") is only an
    # install-presence check (the prefix value is unused); keep the seam so the
    # test does not depend on this package being present in the sourced space.
    monkeypatch.setattr(
        task_execution_builder, "get_package_prefix", lambda _package_name: "/opt/ros/humble/share/task_dispatch"
    )

    base = {
        "_config_path": "/tmp/robot.yaml",
        "control_modes": {"moveit_planning": {"executor": {}}},
    }

    def stroke(executor: dict, execution: dict | None) -> tuple[float, float]:
        config = {**base, "control_modes": {"moveit_planning": {"executor": executor}}}
        if execution is not None:
            config["embodied"] = {"execution": execution}
        node = generate_task_executor_node(config, "moveit_planning")
        return (
            _launch_parameter_value(node, "gripper_full_stroke"),
            _launch_parameter_value(node, "gripper_open_position"),
        )

    so101 = {"gripper_open_position": 1.0, "gripper_closed_position": 0.15}
    assert stroke({}, so101) == pytest.approx((0.85, 1.0))
    # The executor block overrides the embodied defaults position by position.
    assert stroke({"gripper_open_position": 0.9}, so101) == pytest.approx((0.75, 0.9))
    # Missing either position: unknown, the executor keeps its fixed 1 s.
    assert stroke({}, {"gripper_open_position": 1.0}) == pytest.approx((0.0, 1.0))
    assert stroke({}, None) == pytest.approx((0.0, 1.0))


# --- hybrid-derivation tests (openspec lekiwi-application-migration 3.1) --------
# Migrated from robot_config's lekiwi-application-migration suite: provider
# hybrid stages carry no motion_mode block, so the manipulation mode the
# executor must stay online for is derived from the stage's skill mode.


def _lekiwi_config(name: str, **kwargs):
    path = Path(get_package_share_directory("robot_config")) / "config" / "robots" / f"{name}.yaml"
    return load_robot_config_dict(path, defer_interface_binding=True, **kwargs)


# Drives a robot config whose perception services reference a gitignored model
# bundle; without scripts/download_models.py the config cannot load. A missing
# download is not a defect, so skip rather than fail. See root conftest.py.
@pytest.mark.model_bundle("yolox_x_640")
def test_hybrid_task_executor_derives_the_manipulation_mode_from_the_skill_mode(monkeypatch) -> None:
    import task_dispatch.launch_builders.task_execution as task_execution_builder

    hybrid = _lekiwi_config("lekiwi_nav_grasp", nav_stage="hybrid")
    monkeypatch.setattr(task_execution_builder, "get_package_prefix", lambda _name: "/opt/ros/humble")

    node = generate_task_executor_node(hybrid, hybrid["default_control_mode"])

    assert node is not None
    assert node.node_executable == "task_executor_node"


def test_navigation_stage_launches_no_task_executor_for_base_only_skills() -> None:
    navigation = _lekiwi_config("lekiwi_lidar", nav_stage="navigation")
    assert navigation["skill_required_control_mode"] == "base_navigation"

    assert generate_task_executor_node(navigation, "base_navigation") is None
