from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_skill_cli import hermes_launcher
from robot_skill_cli.ros_bridge import BridgeError


def test_hermes_local_command_timeout_allows_slow_embedded_targets() -> None:
    assert hermes_launcher._HERMES_LOCAL_COMMAND_TIMEOUT_SEC == 60


def test_packaged_control_skill_matches_workspace_canonical_copy() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    canonical = repository_root / ".agents" / "skills" / "ibrobot-control" / "SKILL.md"
    packaged = Path(__file__).resolve().parents[1] / "resource" / "ibrobot-control" / "SKILL.md"

    assert packaged.read_bytes() == canonical.read_bytes()


def test_launcher_preflights_and_execs_hermes_without_changing_authorization(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "robot.yaml"
    config_path.write_text("robot:\n  name: test\n", encoding="utf-8")
    skill_path = tmp_path / "installed" / "SKILL.md"
    skill_path.parent.mkdir()
    skill_path.write_text("---\nname: ibrobot-control\n---\n", encoding="utf-8")
    calls = {"binaries": [], "events": [], "exec": None, "chdir": None}

    def require_binary(name):
        calls["binaries"].append(name)
        return f"/bin/{name}"

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("PYTHONPATH", "/stale/python3.10/site-packages")
    monkeypatch.setenv("PYTHONHOME", "/stale/python")
    monkeypatch.setenv("ROS_DOMAIN_ID", "226")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "0")
    monkeypatch.setattr(hermes_launcher, "_require_binary", require_binary)
    monkeypatch.setattr(hermes_launcher, "_check_hermes_version", lambda _path: None)
    monkeypatch.setattr(
        hermes_launcher,
        "validate_public_request_wire_contracts",
        lambda: calls["events"].append("wire"),
        raising=False,
    )
    monkeypatch.setattr(
        hermes_launcher,
        "_check_robot_runtime",
        lambda _name, _path: calls["events"].append("runtime") or config_path,
    )
    monkeypatch.setattr(hermes_launcher, "_installed_skill_path", lambda: skill_path)
    monkeypatch.setattr(hermes_launcher, "_hermes_skills_directory", lambda _path: tmp_path / "hermes-skills")
    monkeypatch.setattr(hermes_launcher, "_check_hermes_skill_discovery", lambda _path: None)
    monkeypatch.setattr(os, "chdir", lambda path: calls.__setitem__("chdir", Path(path)))
    monkeypatch.setattr(
        os,
        "execvpe",
        lambda executable, arguments, environment: calls.__setitem__("exec", (executable, arguments, environment)),
    )

    assert hermes_launcher.main(["--config-name", "test"]) == 0

    assert calls["events"][:2] == ["wire", "runtime"]
    executable, arguments, environment = calls["exec"]
    assert calls["binaries"] == ["hermes", "robot-skill"]
    assert executable == "/bin/hermes"
    assert arguments == ["/bin/hermes", "--skills", "ibrobot-control"]
    assert environment["ROBOT_CONFIG"] == str(config_path)
    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    wrapper = calls["chdir"] / ".ibrobot" / "bin" / "robot-skill"
    assert wrapper.stat().st_mode & 0o111
    assert "export PYTHONPATH=/stale/python3.10/site-packages" in wrapper.read_text(encoding="utf-8")
    assert f"export ROBOT_CONFIG={config_path}" in wrapper.read_text(encoding="utf-8")
    assert "export ROS_DOMAIN_ID=226" in wrapper.read_text(encoding="utf-8")
    assert environment["PATH"].split(os.pathsep)[0] == str(wrapper.parent)
    assert "authorize_motion" not in environment
    registered_skill = tmp_path / "hermes-skills" / "ibrobot-control" / "SKILL.md"
    assert registered_skill.read_bytes() == skill_path.read_bytes()
    assert (registered_skill.parent / ".ibrobot-managed").read_text(encoding="utf-8") == (
        "robot_skill_cli:ibrobot-control\n"
    )


def test_launcher_wire_preflight_failure_blocks_gateway_agent_and_hermes(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "robot.yaml"
    config_path.write_text("robot:\n  name: test\n", encoding="utf-8")
    events = []

    monkeypatch.setattr(hermes_launcher, "_require_binary", lambda name: f"/bin/{name}")
    monkeypatch.setattr(hermes_launcher, "_check_hermes_version", lambda _path: None)
    monkeypatch.setattr(
        hermes_launcher,
        "validate_public_request_wire_contracts",
        lambda: (_ for _ in ()).throw(
            hermes_launcher.LauncherError("WIRE_CONTRACT_INVALID", "stale generated public request interface")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        hermes_launcher,
        "_check_robot_runtime",
        lambda *_args, **_kwargs: events.append("runtime") or config_path,
    )
    monkeypatch.setattr(hermes_launcher, "_prepare_hermes_workspace", lambda: tmp_path / "workspace")
    monkeypatch.setattr(hermes_launcher, "_hermes_skills_directory", lambda _path: tmp_path / "skills")
    monkeypatch.setattr(hermes_launcher, "_installed_skill_path", lambda: events.append("skill") or config_path)
    monkeypatch.setattr(hermes_launcher, "_check_hermes_skill_discovery", lambda _path: events.append("discovery"))
    monkeypatch.setattr(os, "chdir", lambda _path: events.append("chdir"))
    monkeypatch.setattr(os, "execvpe", lambda *_args: events.append("exec"))

    assert hermes_launcher.main(["--config-name", "test"]) == 4
    assert events == []


def test_runtime_check_allows_unauthorized_motion_but_requires_agent_interfaces(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "robot.yaml"
    config_path.write_text("robot:\n  name: test\n", encoding="utf-8")

    class Bridge:
        def __init__(self):
            self.closed = False
            self.timeouts = []

        def start(self):
            return True

        def get_status(self, **kwargs):
            self.timeouts.append(kwargs["timeout_sec"])
            return {
                "control_plane_ready": True,
                "control_plane_error_code": "",
                "motion_authorized": False,
            }

        def wait_for_agent_plan_interfaces(self, **kwargs):
            self.timeouts.append(kwargs["timeout_sec"])
            return True

        def close(self):
            self.closed = True

    bridge = Bridge()
    monkeypatch.setattr(hermes_launcher, "resolve_robot_config_path", lambda **_kwargs: config_path)
    monkeypatch.setattr(
        hermes_launcher,
        "load_runtime_context",
        lambda **_kwargs: (SimpleNamespace(view={"timeout_policy": {"rpc_timeout_sec": 1.0}}), object()),
    )
    monkeypatch.setattr(hermes_launcher, "_create_bridge", lambda _transport: bridge)

    assert hermes_launcher._check_robot_runtime("test", None) == config_path
    assert bridge.closed is True
    assert bridge.timeouts == [15.0, 15.0]


def test_runtime_check_visual_game_mode_does_not_require_motion_plan_interfaces(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "robot.yaml"
    config_path.write_text("robot:\n  name: test\n", encoding="utf-8")

    class Bridge:
        def __init__(self):
            self.closed = False
            self.called = False

        def start(self):
            return True

        def wait_for_visual_game_interfaces(self, **_kwargs):
            self.called = True
            return True

        def close(self):
            self.closed = True

    bridge = Bridge()
    monkeypatch.setattr(hermes_launcher, "resolve_robot_config_path", lambda **_kwargs: config_path)
    monkeypatch.setattr(
        hermes_launcher,
        "load_runtime_context",
        lambda **_kwargs: (SimpleNamespace(view={"timeout_policy": {"rpc_timeout_sec": 1.0}}), object()),
    )
    monkeypatch.setattr(
        hermes_launcher,
        "load_visual_game_runtime_context",
        lambda **_kwargs: (SimpleNamespace(view={"timeout_policy": {"rpc_timeout_sec": 1.0}}), object()),
    )
    monkeypatch.setattr(hermes_launcher, "_resolve_preflight_mode", lambda *_args: "visual-games")
    monkeypatch.setattr(hermes_launcher, "_create_bridge", lambda _transport: bridge)

    assert hermes_launcher._check_robot_runtime("test", None) == config_path
    assert bridge.called is True
    assert bridge.closed is True


def test_runtime_check_both_mode_requires_motion_and_visual_interfaces(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "robot.yaml"
    config_path.write_text("robot:\n  name: test\n", encoding="utf-8")

    class Bridge:
        def __init__(self):
            self.calls = []

        def start(self):
            return True

        def get_status(self, **_kwargs):
            self.calls.append("status")
            return {"control_plane_ready": True, "control_plane_error_code": ""}

        def wait_for_agent_plan_interfaces(self, **_kwargs):
            self.calls.append("motion")
            return True

        def wait_for_visual_game_interfaces(self, **_kwargs):
            self.calls.append("visual-games")
            return True

        def close(self):
            self.calls.append("close")

    bridge = Bridge()
    monkeypatch.setattr(hermes_launcher, "resolve_robot_config_path", lambda **_kwargs: config_path)
    monkeypatch.setattr(
        hermes_launcher,
        "load_runtime_context",
        lambda **_kwargs: (SimpleNamespace(view={"timeout_policy": {"rpc_timeout_sec": 1.0}}), object()),
    )
    monkeypatch.setattr(hermes_launcher, "_resolve_preflight_mode", lambda *_args: "both")
    monkeypatch.setattr(hermes_launcher, "_create_bridge", lambda _transport: bridge)

    assert hermes_launcher._check_robot_runtime("test", None) == config_path
    assert bridge.calls == ["visual-games", "status", "motion", "close"]


def test_auto_preflight_selects_both_for_motion_and_visual_game_config(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "robot.yaml"
    config_path.write_text(
        """robot:
  name: test
  default_control_mode: moveit_planning
  embodied:
    visual_games:
      sorting_hat:
        enabled: true
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        hermes_launcher,
        "load_robot_config_dict",
        lambda _path: {
            "skill_required_control_mode": "moveit_planning",
            "embodied": {
                "enabled": True,
                "perception": {"enabled": True},
                "skill_catalog_profile": "so101_single_arm",
                "visual_games": {"sorting_hat": {"enabled": True}},
            },
        },
    )

    assert hermes_launcher._resolve_preflight_mode("auto", config_path) == "both"


def test_auto_preflight_selects_visual_games_for_visual_only_config(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "robot.yaml"
    config_path.write_text("robot:\n  name: test\n", encoding="utf-8")
    monkeypatch.setattr(
        hermes_launcher,
        "load_robot_config_dict",
        lambda _path: {
            "embodied": {
                "enabled": True,
                "perception": {"enabled": True},
                "visual_games": {"sorting_hat": {"enabled": True}},
            },
        },
    )

    assert hermes_launcher._resolve_preflight_mode("auto", config_path) == "visual-games"


def test_auto_preflight_ignores_visual_games_when_embodied_is_disabled(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "robot.yaml"
    config_path.write_text("robot:\n  name: test\n", encoding="utf-8")
    monkeypatch.setattr(
        hermes_launcher,
        "load_robot_config_dict",
        lambda _path: {
            "embodied": {
                "enabled": False,
                "perception": {"enabled": True},
                "visual_games": {"sorting_hat": {"enabled": True}},
            },
        },
    )

    assert hermes_launcher._resolve_preflight_mode("auto", config_path) == "motion"


def test_auto_preflight_rejects_visual_game_without_perception(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "robot.yaml"
    config_path.write_text("robot:\n  name: test\n", encoding="utf-8")
    monkeypatch.setattr(
        hermes_launcher,
        "load_robot_config_dict",
        lambda _path: {
            "embodied": {
                "enabled": True,
                "perception": {"enabled": False},
                "visual_games": {"sorting_hat": {"enabled": True}},
            },
        },
    )

    with pytest.raises(ValueError, match="embodied.perception.enabled"):
        hermes_launcher._resolve_preflight_mode("auto", config_path)


def test_auto_preflight_keeps_motion_when_no_visual_game(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "robot.yaml"
    config_path.write_text("robot:\n  name: test\n", encoding="utf-8")
    monkeypatch.setattr(
        hermes_launcher,
        "load_robot_config_dict",
        lambda _path: {
            "skill_required_control_mode": "moveit_planning",
            "embodied": {"skill_catalog_profile": "so101_single_arm"},
        },
    )
    assert hermes_launcher._resolve_preflight_mode("auto", config_path) == "motion"


def test_launcher_passes_hermes_chat_arguments_after_separator(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "robot.yaml"
    config_path.write_text("robot:\n  name: test\n", encoding="utf-8")
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text("---\nname: ibrobot-control\n---\n", encoding="utf-8")
    calls = {}
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(hermes_launcher, "_require_binary", lambda name: f"/bin/{name}")
    monkeypatch.setattr(hermes_launcher, "_check_hermes_version", lambda _path: None)
    monkeypatch.setattr(hermes_launcher, "_check_robot_runtime", lambda _name, _path: config_path)
    monkeypatch.setattr(hermes_launcher, "_installed_skill_path", lambda: skill_path)
    monkeypatch.setattr(hermes_launcher, "_hermes_skills_directory", lambda _path: tmp_path / "hermes-skills")
    monkeypatch.setattr(hermes_launcher, "_check_hermes_skill_discovery", lambda _path: None)
    monkeypatch.setattr(os, "chdir", lambda _path: None)
    monkeypatch.setattr(
        os,
        "execvpe",
        lambda executable, arguments, environment: calls.update(
            executable=executable, arguments=arguments, environment=environment
        ),
    )

    assert hermes_launcher.main(["--config-path", str(config_path), "--", "chat", "-q", "check robot", "-Q"]) == 0
    assert calls["arguments"] == [
        "/bin/hermes",
        "chat",
        "-q",
        "check robot",
        "-Q",
        "--skills",
        "ibrobot-control",
    ]


def test_launcher_reports_gateway_timeout_without_traceback(monkeypatch, capsys) -> None:
    monkeypatch.setattr(hermes_launcher, "_require_binary", lambda name: f"/bin/{name}")
    monkeypatch.setattr(hermes_launcher, "_check_hermes_version", lambda _path: None)
    monkeypatch.setattr(
        hermes_launcher,
        "_check_robot_runtime",
        lambda _name, _path: (_ for _ in ()).throw(
            BridgeError("RESULT_TIMEOUT", "service response timed out", exit_code=5)
        ),
    )

    assert hermes_launcher.main(["--config-name", "test"]) == 5
    assert capsys.readouterr().err == "hermes-robot: RESULT_TIMEOUT: service response timed out\n"


def test_register_hermes_skill_updates_only_launcher_managed_copy(tmp_path) -> None:
    source = tmp_path / "source" / "SKILL.md"
    source.parent.mkdir()
    source.write_text("---\nname: ibrobot-control\n---\nold\n", encoding="utf-8")
    skills_directory = tmp_path / "hermes" / "skills"

    target = hermes_launcher.register_hermes_skill(source, skills_directory)
    source.write_text("---\nname: ibrobot-control\n---\nnew\n", encoding="utf-8")
    assert hermes_launcher.register_hermes_skill(source, skills_directory) == target

    assert target.read_bytes() == source.read_bytes()
    assert (target.parent / ".ibrobot-managed").read_text(encoding="utf-8") == ("robot_skill_cli:ibrobot-control\n")


def test_register_hermes_skill_rejects_unmanaged_conflict(tmp_path) -> None:
    source = tmp_path / "source" / "SKILL.md"
    source.parent.mkdir()
    source.write_text("---\nname: ibrobot-control\n---\nmanaged\n", encoding="utf-8")
    target = tmp_path / "hermes" / "skills" / "ibrobot-control" / "SKILL.md"
    target.parent.mkdir(parents=True)
    target.write_text("---\nname: ibrobot-control\n---\nuser copy\n", encoding="utf-8")

    with pytest.raises(hermes_launcher.LauncherError, match="unmanaged ibrobot-control") as error:
        hermes_launcher.register_hermes_skill(source, tmp_path / "hermes" / "skills")

    assert error.value.code == "AGENT_SKILL_CONFLICT"
    assert "user copy" in target.read_text(encoding="utf-8")


def test_hermes_skills_directory_uses_reported_active_profile(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "profiles" / "robot" / "config.yaml"
    monkeypatch.setattr(
        hermes_launcher.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=f"{config_path}\n"),
    )

    assert hermes_launcher._hermes_skills_directory("/bin/hermes") == config_path.parent / "skills"


def test_hermes_skill_discovery_requires_enabled_skill(monkeypatch) -> None:
    monkeypatch.setattr(
        hermes_launcher.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="ibrobot-control  local  enabled\n"),
    )
    hermes_launcher._check_hermes_skill_discovery("/bin/hermes")

    monkeypatch.setattr(
        hermes_launcher.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="another-skill  local  enabled\n"),
    )
    with pytest.raises(hermes_launcher.LauncherError) as error:
        hermes_launcher._check_hermes_skill_discovery("/bin/hermes")
    assert error.value.code == "AGENT_SKILL_UNAVAILABLE"


def test_robot_skill_wrapper_binds_config_and_ros_environment(tmp_path) -> None:
    actual = tmp_path / "actual-robot-skill"
    actual.write_text(
        "#!/bin/sh\n"
        "printf 'CONFIG=%s\\n' \"$2\"\n"
        "printf 'DOMAIN=%s\\n' \"$ROS_DOMAIN_ID\"\n"
        "printf 'LOCALHOST=%s\\n' \"$ROS_LOCALHOST_ONLY\"\n"
        "printf 'COMMAND=%s\\n' \"$3\"\n",
        encoding="utf-8",
    )
    actual.chmod(0o755)
    config_path = tmp_path / "robot.yaml"
    wrapper = (
        hermes_launcher._prepare_robot_skill_wrapper(
            tmp_path / "workspace",
            str(actual),
            "/ros/python",
            config_path,
            {"ROS_DOMAIN_ID": "226", "ROS_LOCALHOST_ONLY": "0"},
        )
        / "robot-skill"
    )

    completed = subprocess.run(
        [wrapper, "status"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "ROS_DOMAIN_ID": "174"},
    )

    assert completed.stdout == f"CONFIG={config_path}\nDOMAIN=226\nLOCALHOST=0\nCOMMAND=status\n"


def test_robot_skill_wrapper_rejects_agent_config_override(tmp_path) -> None:
    actual = tmp_path / "actual-robot-skill"
    actual.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    actual.chmod(0o755)
    wrapper = (
        hermes_launcher._prepare_robot_skill_wrapper(
            tmp_path / "workspace",
            str(actual),
            "",
            tmp_path / "robot.yaml",
            {"ROS_DOMAIN_ID": "226"},
        )
        / "robot-skill"
    )

    completed = subprocess.run([wrapper, "--config-name", "wrong", "status"], check=False, capture_output=True)

    assert completed.returncode == 2


def test_runtime_check_retries_transient_read_only_status_timeout(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "robot.yaml"
    config_path.write_text("robot:\n  name: test\n", encoding="utf-8")

    class Bridge:
        def __init__(self, fail):
            self.fail = fail
            self.closed = False

        def start(self):
            return True

        def get_status(self, **_kwargs):
            if self.fail:
                raise BridgeError("RESULT_TIMEOUT", "transient timeout", exit_code=5)
            return {"control_plane_ready": True, "control_plane_error_code": ""}

        def wait_for_agent_plan_interfaces(self, **_kwargs):
            return True

        def close(self):
            self.closed = True

    bridges = [Bridge(True), Bridge(False)]
    monkeypatch.setattr(hermes_launcher, "resolve_robot_config_path", lambda **_kwargs: config_path)
    monkeypatch.setattr(
        hermes_launcher,
        "load_runtime_context",
        lambda **_kwargs: (SimpleNamespace(view={"timeout_policy": {"rpc_timeout_sec": 1.0}}), object()),
    )
    monkeypatch.setattr(hermes_launcher, "_create_bridge", lambda _transport: bridges.pop(0))
    monkeypatch.setattr(hermes_launcher.time, "sleep", lambda _seconds: None)

    assert hermes_launcher._check_robot_runtime("test", None) == config_path
    assert bridges == []
