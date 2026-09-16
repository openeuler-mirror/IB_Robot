import importlib.util
from pathlib import Path

from launch import LaunchContext


def _load_launch_module():
    path = Path(__file__).parents[1] / "launch" / "ik_workers.launch.py"
    spec = importlib.util.spec_from_file_location("ik_workers_launch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_launch_only_leaves_kinematics_capability_enabled():
    module = _load_launch_module()
    disabled = set(module._DISABLED_CAPABILITIES.split())

    assert "move_group/MoveGroupKinematicsService" not in disabled
    assert "move_group/MoveGroupExecuteService" in disabled
    assert "move_group/MoveGroupExecuteTrajectoryAction" in disabled
    assert "move_group/MoveGroupMoveAction" in disabled
    assert "move_group/MoveGroupPlanService" in disabled
    assert "move_group/ApplyPlanningSceneService" in disabled
    assert "move_group/TfPublisher" in disabled


def _launch_text(value):
    if isinstance(value, tuple):
        return "".join(_launch_text(item) for item in value)
    return getattr(value, "text", str(value))


def test_worker_move_group_uses_configured_arm_only_joint_states():
    module = _load_launch_module()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "worker_count": "1",
            "namespace_prefix": "ik_worker",
            "use_sim_time": "false",
            "log_level": "warn",
            "joint_state_topic": "/arm_joint_state_broadcaster/joint_states",
        }
    )

    worker = module._launch_setup(context)[0]
    remappings = {_launch_text(source): _launch_text(target) for source, target in worker._Node__remappings}

    assert remappings["joint_states"] == "/arm_joint_state_broadcaster/joint_states"
