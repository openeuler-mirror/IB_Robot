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
