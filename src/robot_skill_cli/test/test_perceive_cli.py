"""Unit tests for the ibrobot-perceive allowlist reader.

These tests mock ``subprocess.run`` so no ROS stack is required. The audit log
path is redirected to a tmp_path so tests do not pollute /tmp. ``_load_robot_section``
is monkeypatched for config-backed sources so no real robot_config file is read.
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from robot_skill_cli import perceive_cli


def _ok_run(stdout: str) -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stdout=stdout, stderr="")


def _patch_config(monkeypatch, robot_section: dict) -> None:
    """Redirect _load_robot_section to return a fixed robot section (no file I/O)."""
    monkeypatch.setattr(perceive_cli, "_load_robot_section", lambda *a, **k: robot_section)


def test_rejects_source_not_in_allowlist(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(perceive_cli, "LOG_PATH", tmp_path / "perceive.log")
    with pytest.raises(SystemExit) as exc:
        perceive_cli.main(["--source", "cmd_vel", "--field", "linear"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice" in err


def test_rejects_field_not_allowed(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(perceive_cli, "LOG_PATH", tmp_path / "perceive.log")
    code = perceive_cli.main(["--source", "voice_direction", "--field", "stamp"])
    assert code == 1
    err = capsys.readouterr().err
    assert "not allowed for voice_direction" in err
    assert "azimuth_rad" in err
    assert "seq_id" in err


def test_successful_read_prints_literal(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(perceive_cli, "LOG_PATH", tmp_path / "perceive.log")
    yaml_out = "---\nazimuth_rad: 0.5236\nseq_id: 42\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _ok_run(yaml_out))

    code = perceive_cli.main(["--source", "voice_direction", "--field", "azimuth_rad"])

    assert code == 0
    out = capsys.readouterr().out.strip()
    assert out == "0.5236"
    log = (tmp_path / "perceive.log").read_text(encoding="utf-8")
    assert "status=ok" in log
    assert "0.5236" in log


def test_voice_direction_uses_explicit_reliable_qos_and_message_type(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(perceive_cli, "LOG_PATH", tmp_path / "perceive.log")
    captured: dict = {}

    def _fake_run(cmd, *a, **k):
        captured["cmd"] = cmd
        captured["timeout"] = k["timeout"]
        return _ok_run("---\nazimuth_rad: 0.5236\nseq_id: 42\n")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    code = perceive_cli.main(["--source", "voice_direction", "--field", "azimuth_rad"])

    assert code == 0
    assert captured["cmd"] == [
        "ros2",
        "topic",
        "echo",
        "--once",
        "--qos-reliability",
        "reliable",
        "--qos-history",
        "keep_last",
        "--qos-depth",
        "1",
        "--qos-durability",
        "volatile",
        "/voice/speech_direction",
        "ibrobot_msgs/msg/SpeechDirection",
    ]
    assert captured["timeout"] == 60


def test_voice_direction_does_not_load_robot_config(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(perceive_cli, "LOG_PATH", tmp_path / "perceive.log")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _ok_run("---\nazimuth_rad: 0.1\nseq_id: 1\n"))

    def _fail_if_called(*a, **k):
        raise AssertionError("voice_direction must not load robot_config")

    monkeypatch.setattr(perceive_cli, "_load_robot_section", _fail_if_called)
    code = perceive_cli.main(["--source", "voice_direction", "--field", "azimuth_rad"])
    assert code == 0


def test_arm_joint_position_resolves_so101_default_topic(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(perceive_cli, "LOG_PATH", tmp_path / "perceive.log")
    captured: dict = {}

    def _fake_run(cmd, *a, **k):
        captured["cmd"] = cmd
        captured["timeout"] = k["timeout"]
        return _ok_run("---\nname: ['1', '2']\nposition: [0.12, -0.31]\n")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    _patch_config(monkeypatch, {"name": "so101_single_arm", "moveit": {"arm_group_name": "arm"}})

    code = perceive_cli.main(["--source", "arm_joint_position", "--field", "position"])

    assert code == 0
    assert captured["cmd"][-1] == "/joint_states"
    assert captured["timeout"] == 5


def test_arm_joint_position_resolves_lekiwi_arm_topic(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(perceive_cli, "LOG_PATH", tmp_path / "perceive.log")
    captured: dict = {}

    def _fake_run(cmd, *a, **k):
        captured["cmd"] = cmd
        return _ok_run("---\nname: ['1', '2']\nposition: [0.12, -0.31]\n")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    _patch_config(
        monkeypatch,
        {
            "name": "lekiwi_handeye_realsense_grasp",
            "moveit": {"joint_state_topic": "/arm_joint_state_broadcaster/joint_states"},
        },
    )

    code = perceive_cli.main(["--source", "arm_joint_position", "--field", "position"])

    assert code == 0
    assert captured["cmd"][-1] == "/arm_joint_state_broadcaster/joint_states"


def test_joint_positions_read_prints_raw_array(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(perceive_cli, "LOG_PATH", tmp_path / "perceive.log")
    yaml_out = "---\nname: [shoulder_pan, shoulder_lift]\nposition: [0.12, -0.31]\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _ok_run(yaml_out))
    _patch_config(monkeypatch, {"name": "so101_single_arm", "moveit": {}})

    code = perceive_cli.main(["--source", "arm_joint_position", "--field", "position"])

    assert code == 0
    out = capsys.readouterr().out.strip()
    assert json.loads(out) == [0.12, -0.31]
    assert "status=ok" in (tmp_path / "perceive.log").read_text(encoding="utf-8")


def test_config_error_returns_error(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(perceive_cli, "LOG_PATH", tmp_path / "perceive.log")

    def _fail(*a, **k):
        raise FileNotFoundError("missing robot_config")

    monkeypatch.setattr(perceive_cli, "_load_robot_section", _fail)
    code = perceive_cli.main(["--source", "arm_joint_position", "--field", "position"])
    assert code == 1
    assert "failed to load robot_config" in capsys.readouterr().err


def test_timeout_returns_error(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(perceive_cli, "LOG_PATH", tmp_path / "perceive.log")

    def _raise(*a, **k):
        raise subprocess.TimeoutExpired(cmd="ros2", timeout=5)

    monkeypatch.setattr(subprocess, "run", _raise)
    code = perceive_cli.main(["--source", "voice_direction", "--field", "azimuth_rad"])
    assert code == 1
    assert "timed out" in capsys.readouterr().err
    assert "timeout" in (tmp_path / "perceive.log").read_text(encoding="utf-8")


def test_ros2_error_returns_error(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(perceive_cli, "LOG_PATH", tmp_path / "perceive.log")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="topic not found\n")
    )
    code = perceive_cli.main(["--source", "voice_direction", "--field", "azimuth_rad"])
    assert code == 1
    assert "ros2 topic echo failed" in capsys.readouterr().err


def test_parse_error_returns_error(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(perceive_cli, "LOG_PATH", tmp_path / "perceive.log")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _ok_run("not: : valid: yaml: ["))
    code = perceive_cli.main(["--source", "voice_direction", "--field", "azimuth_rad"])
    assert code == 1
    assert "failed to parse" in capsys.readouterr().err


def test_missing_field_returns_error(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(perceive_cli, "LOG_PATH", tmp_path / "perceive.log")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _ok_run("---\nseq_id: 7\n"))
    code = perceive_cli.main(["--source", "voice_direction", "--field", "azimuth_rad"])
    assert code == 1
    assert "not found" in capsys.readouterr().err


def test_allowlist_fields_are_hardcoded_and_not_configurable() -> None:
    assert "voice_direction" in perceive_cli.PERCEPTION_ALLOWLIST
    assert perceive_cli.PERCEPTION_ALLOWLIST["voice_direction"]["fields"] == {"azimuth_rad", "seq_id"}
    assert perceive_cli.PERCEPTION_ALLOWLIST["arm_joint_position"]["fields"] == {"position"}
    assert "cmd_vel" not in perceive_cli.PERCEPTION_ALLOWLIST


def test_config_name_and_config_path_are_mutually_exclusive(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(perceive_cli, "LOG_PATH", tmp_path / "perceive.log")
    with pytest.raises(SystemExit) as exc:
        perceive_cli.main(
            [
                "--source",
                "voice_direction",
                "--field",
                "azimuth_rad",
                "--config-name",
                "so101_single_arm",
                "--config-path",
                "/tmp/whatever.yaml",
            ]
        )
    assert exc.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


def test_extract_field_handles_multiple_yaml_documents() -> None:
    stdout = "---\nseq_id: 1\n---\nazimuth_rad: 0.1\nseq_id: 2\n"
    assert perceive_cli._extract_field(stdout, "azimuth_rad") == 0.1


def test_extract_field_handles_ros2_echo_message_before_document_separator() -> None:
    stdout = (
        "header:\n"
        "  stamp:\n"
        "    sec: 1787553991\n"
        "    nanosec: 726186377\n"
        "  frame_id: base_link\n"
        "azimuth_rad: 0.6981316804885864\n"
        "seq_id: 1\n"
        "---\n"
    )
    assert perceive_cli._extract_field(stdout, "azimuth_rad") == 0.6981316804885864


def test_extract_field_ignores_fastdds_diagnostics_before_yaml() -> None:
    stdout = (
        "\x1b[31;1m[RTPS_TRANSPORT_SHM Error]\x1b[0m Failed init_port\n"
        "A message was lost!!!\n"
        "\ttotal count: 1---\n"
        "name: ['1', '2']\n"
        "position: [0.12, -0.31]\n"
        "---\n"
    )
    assert perceive_cli._extract_field(stdout, "position") == [0.12, -0.31]


def test_format_value_scalars_use_str() -> None:
    assert perceive_cli._format_value(0.5236) == "0.5236"
    assert perceive_cli._format_value(42) == "42"
    assert perceive_cli._format_value(0.1 + 0.2) == str(0.1 + 0.2)


def test_format_value_compound_types_use_json() -> None:
    assert perceive_cli._format_value([0.12, -0.31]) == "[0.12, -0.31]"
    assert perceive_cli._format_value({"k": 1}) == '{"k": 1}'
    assert perceive_cli._format_value([True, None, 1]) == "[true, null, 1]"


def test_format_value_compound_output_is_json_parseable() -> None:
    for value in ([0.12, -0.31], {"k": 1}, [True, None, 1], [{"a": [1, 2]}]):
        out = perceive_cli._format_value(value)
        assert json.loads(out) == value
