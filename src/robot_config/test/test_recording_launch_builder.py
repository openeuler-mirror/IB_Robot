import pytest
import yaml
from launch.substitutions import TextSubstitution

from robot_config.launch_builders.recording import (
    _record_cli_command,
    generate_episodic_recording_node,
    generate_rerun_viewer_node,
)


def _text(substitutions):
    return "".join(item.text if isinstance(item, TextSubstitution) else str(item) for item in substitutions)


def test_generate_rerun_viewer_node_forces_pythonnousesite():
    nodes = generate_rerun_viewer_node({"_config_path": "/tmp/robot.yaml"})

    assert len(nodes) == 1
    assert dict((_text(key), _text(value)) for key, value in nodes[0].additional_env) == {"PYTHONNOUSERSITE": "1"}


def test_record_cli_command_is_legacy_when_scheduler_is_disabled():
    expected = "ros2 run dataset_tools record_cli --ros-args -p control_mode:=model_inference"
    assert _record_cli_command("model_inference") == expected


def test_record_cli_command_uses_session_restart_when_scheduler_enabled():
    assert _record_cli_command("model_inference", scheduler_enabled=True) == (
        "ros2 run dataset_tools record_cli --ros-args -p control_mode:=model_inference"
        " -p restart_session_service:=/action_dispatcher/restart_session"
    )


def test_rtp_episodic_recording_is_not_launched_on_edge():
    config = {
        "_config_path": "/tmp/robot.yaml",
        "contract": {"observations": [{"transport": {"mode": "rtp"}}]},
    }

    assert generate_episodic_recording_node(config, "model_inference") == []


def test_dds_episodic_recording_still_launches_episode_recorder():
    config = {
        "_config_path": "/tmp/robot.yaml",
        "contract": {"observations": [{"transport": {"mode": "dds"}}]},
    }

    nodes = generate_episodic_recording_node(config, "teleop")

    assert len(nodes) == 1
    assert _text(nodes[0].node_executable) == "episode_recorder"


@pytest.mark.parametrize(
    "mode,kind,enabled",
    [("teleop", "leader_topic", True), ("teleop", "phone", False), ("model_inference", "leader_topic", False)],
)
def test_recording_admission_is_bound_to_managed_leader_only(tmp_path, monkeypatch, mode, kind, enabled):
    from robot_config.launch_builders import recording

    inputs = tmp_path / "inputs.yaml"
    inputs.write_text(yaml.safe_dump({"devices": [{"name": "operator", "type": kind}]}))
    config = {
        "_config_path": "/tmp/robot.yaml",
        "contract": {"observations": [{"transport": {"mode": "dds"}}]},
        "teleoperation": {"target": {"group": "arm"}, "input_config": str(inputs), "active_device": "operator"},
        "runtime": {
            "provider": "test_robot",
            "interface_description": {
                "interfaces": {
                    "runtime.set_mode": {"endpoint": "/unit/set_mode"},
                    "runtime.status": {"endpoint": "/unit/status"},
                    "motion.arm.stop": {"endpoint": "/unit/stop"},
                }
            },
        },
        "recording": {"action_stream_gap_timeout_sec": 0.75, "teleop_rearm_service": "/operator/rearm"},
    }
    monkeypatch.setattr(recording, "Node", lambda **kwargs: kwargs)
    node = generate_episodic_recording_node(config, mode)[0]
    params = {key: value for group in node["parameters"] for key, value in group.items()}
    assert params.get("require_action_stream", False) is enabled
    if enabled:
        assert params["runtime_set_mode_service"] == "/unit/set_mode"
        assert params["runtime_status_topic"] == "/unit/status"
        assert params["teleop_rearm_service"] == "/operator/rearm"
        assert params["teleop_stop_service"] == "/unit/stop"
        assert params["action_stream_gap_timeout_sec"] == 0.75
        assert params["admission_attempts"] == 3
        assert params["admission_timeout_sec"] == 6.0
    else:
        assert "runtime_set_mode_service" not in params


@pytest.mark.parametrize("admission_timeout", [5.0, 8.0, float("nan"), float("inf")])
def test_recording_rejects_timeout_not_covering_rearm(tmp_path, admission_timeout):
    inputs = tmp_path / "inputs.yaml"
    inputs.write_text(yaml.safe_dump({"devices": [{"name": "leader", "type": "leader_topic"}]}))
    config = {
        "_config_path": "/tmp/robot.yaml",
        "contract": {"observations": [{"transport": {"mode": "dds"}}]},
        "teleoperation": {
            "target": {"group": "arm"},
            "input_config": str(inputs),
            "active_device": "leader",
            "rearm_timeout_s": 8.0,
        },
        "runtime": {"provider": "test_robot", "interface_description": {"interfaces": {}}},
        "recording": {"admission_timeout_sec": admission_timeout},
    }
    with pytest.raises(ValueError, match="must exceed"):
        generate_episodic_recording_node(config, "teleop")
