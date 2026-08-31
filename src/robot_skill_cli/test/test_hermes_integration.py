from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import rclpy

from robot_skill_cli import hermes_configure, hermes_tts_hook


def test_soul_managed_block_is_idempotent() -> None:
    policy = "policy text"
    first = hermes_configure._managed_soul("existing\n", policy)
    second = hermes_configure._managed_soul(first, policy)

    assert first == second
    assert first.count(hermes_configure._SOUL_BEGIN) == 1
    assert first.startswith("existing\n")


def test_soul_managed_block_rejects_partial_markers() -> None:
    with pytest.raises(hermes_configure.ConfigureError, match="incomplete"):
        hermes_configure._managed_soul(hermes_configure._SOUL_BEGIN, "policy")


def test_update_config_removes_managed_plugin_and_replaces_managed_hook(tmp_path: Path) -> None:
    environment_file = tmp_path / "ibrobot-env.sh"
    hook_path = tmp_path / "ibrobot-speak"
    config = {
        "plugins": {"enabled": ["other", "ibrobot-robot-control"], "disabled": ["ibrobot-robot-control"]},
        "toolsets": ["other", "ibrobot-robot-control"],
        "terminal": {"shell_init_files": ["/old/ibrobot-env.sh", "/keep.sh"]},
        "hooks": {
            "post_llm_call": [
                {"command": "/home/user/.hermes/hooks/hermes-speak.py", "timeout": 90},
                {"command": "/other/hook", "timeout": 5},
            ]
        },
    }

    result = hermes_configure._update_config(
        config,
        workspace=tmp_path / "workspace",
        environment_file=environment_file,
        hook_path=hook_path,
        speech_enabled=True,
    )

    assert result["plugins"] == {"enabled": ["other"], "disabled": []}
    assert result["toolsets"] == ["other"]
    assert result["terminal"]["cwd"] == str(tmp_path / "workspace")
    assert result["terminal"]["shell_init_files"] == ["/keep.sh", str(environment_file)]
    assert result["hooks"]["post_llm_call"] == [
        {"command": "/other/hook", "timeout": 5},
        {"command": str(hook_path), "timeout": hermes_configure._HOOK_TIMEOUT_SEC},
    ]
    assert "pre_tool_call" not in result["hooks"]


def test_update_config_can_disable_speech(tmp_path: Path) -> None:
    result = hermes_configure._update_config(
        {"hooks": {"post_llm_call": [{"command": "/old/.hermes/hooks/hermes-speak.py"}]}},
        workspace=tmp_path,
        environment_file=tmp_path / "env.sh",
        hook_path=tmp_path / "hook",
        speech_enabled=False,
    )

    assert "post_llm_call" not in result["hooks"]


def test_remove_stale_plugin_quarantines_managed_copy(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    target = profile / "plugins" / "ibrobot-robot-control"
    target.mkdir(parents=True)
    (target / "__init__.py").write_text(hermes_configure._MANAGED_FILE_HEADER + "\n# retired\n", encoding="utf-8")

    assert hermes_configure._remove_stale_plugin(profile) is True
    assert not target.exists()
    backups = list(profile.joinpath("plugins").glob("ibrobot-robot-control.quarantined-*"))
    assert backups, "expected a quarantine backup of the managed plugin"
    assert hermes_configure._remove_stale_plugin(profile) is False


def test_remove_stale_plugin_leaves_unmanaged_copy(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    target = profile / "plugins" / "ibrobot-robot-control"
    target.mkdir(parents=True)
    (target / "__init__.py").write_text("# user-authored plugin\n", encoding="utf-8")

    assert hermes_configure._remove_stale_plugin(profile) is False
    assert target.exists(), "unowned plugin content must not be touched"


def test_generated_wrappers_source_shrc_local_and_run_python3(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    shrc = workspace / ".shrc_local"
    shrc.parent.mkdir(parents=True, exist_ok=True)
    shrc.write_text("# stub\n", encoding="utf-8")

    hook = hermes_configure._tts_hook_wrapper(
        workspace=workspace,
        shrc=shrc,
        config_path=tmp_path / "robot.yaml",
        ros_domain_id="52",
        synthesis_service="/voice_tts/synthesize",
        playback_service="/voice_tts/play",
        playback_timeout_sec=300.0,
        synthesis_timeout_sec=90.0,
    )
    lifecycle_hook = hermes_configure._lifecycle_hook_wrapper(workspace=workspace, shrc=shrc)
    interim_hook = hermes_configure._interim_hook_wrapper(
        workspace=workspace, shrc=shrc, speaker_path=tmp_path / "profile" / "hooks" / "ibrobot-speak"
    )
    robot_skill = hermes_configure._robot_skill_wrapper(
        workspace=workspace,
        shrc=shrc,
        config_path=tmp_path / "robot.yaml",
        ros_domain_id="52",
    )
    environment = hermes_configure._shell_environment(
        workspace=workspace,
        shrc=shrc,
        config_path=tmp_path / "robot.yaml",
        ros_domain_id="52",
        profile_bin=tmp_path / "profile" / "bin",
    )

    assert "-m robot_skill_cli.hermes_tts_hook" in hook
    assert "hermes-robot-speak" not in hook
    assert "source /opt/ros/humble/setup.bash" not in hook
    assert str(shrc) in hook
    assert "exec python3 -m robot_skill_cli.hermes_tts_hook" in hook
    assert "IBROBOT_TTS_SYNTHESIS_TIMEOUT_SEC=90" in hook
    assert "IBROBOT_TTS_PLAYBACK_TIMEOUT_SEC=300" in hook
    assert "venv/bin/python3" not in hook
    assert str(shrc) not in lifecycle_hook
    assert str(shrc) in interim_hook
    assert str(tmp_path / "profile" / "hooks" / "ibrobot-speak") in interim_hook
    assert "exec python3 -m robot_skill_cli.hermes_interim_speech" in interim_hook
    assert "exec python3 -m robot_skill_cli.hermes_lifecycle_speech" in lifecycle_hook
    assert "-m robot_skill_cli.cli" in robot_skill
    assert "source /opt/ros/humble/setup.bash" not in robot_skill
    assert "exec python3 -m robot_skill_cli.cli" in robot_skill
    assert "IBROBOT_HERMES_LIFECYCLE_SPEECH=1" in robot_skill
    assert '--config-path "$ROBOT_CONFIG"' in robot_skill
    assert "configuration is bound" in robot_skill
    assert "source /opt/ros/humble/setup.bash" not in environment
    assert f"export PATH={tmp_path / 'profile' / 'bin'}:$PATH" in environment


def test_configure_dry_run_does_not_create_profile_files(tmp_path: Path, monkeypatch, capsys) -> None:
    profile = tmp_path / "profile"
    resource = tmp_path / "resource"
    install = tmp_path / "install"
    config_path = tmp_path / "robot.yaml"
    config_path.write_text("robot: test\n", encoding="utf-8")
    (resource / "hooks").mkdir(parents=True)
    (resource / "SOUL.md").write_text("soul\n", encoding="utf-8")
    (resource / "POLICY.md").write_text("policy\n", encoding="utf-8")
    (resource / "hooks" / "ibrobot-speak").write_text("hook\n", encoding="utf-8")
    (resource / "hooks" / "ibrobot-interim-speech").write_text("interim\n", encoding="utf-8")
    skill = install / "share" / "robot_skill_cli" / "skills" / "ibrobot-control" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("skill\n", encoding="utf-8")
    (tmp_path / ".shrc_local").write_text("# stub\n", encoding="utf-8")

    monkeypatch.setenv("ROS_DOMAIN_ID", "52")
    monkeypatch.setattr(hermes_configure.shutil, "which", lambda _name: "/bin/hermes")
    monkeypatch.setattr(hermes_configure, "resolve_robot_config_path", lambda **_kwargs: config_path)
    monkeypatch.setattr(
        hermes_configure,
        "load_robot_config",
        lambda _path: SimpleNamespace(
            voice_tts=SimpleNamespace(
                enabled=True,
                service_name="/voice_tts/synthesize",
                playback_service_name="/voice_tts/play",
                playback_timeout_sec=300.0,
                synthesis_timeout_sec=90.0,
            )
        ),
    )
    monkeypatch.setattr(hermes_configure, "_active_profile", lambda _hermes: profile)
    monkeypatch.setattr(hermes_configure, "_resource_root", lambda: resource)
    monkeypatch.setattr(hermes_configure, "_install_prefix", lambda: install)

    assert hermes_configure.main(["--config-path", str(config_path), "--soul-mode", "replace", "--dry-run"]) == 0
    assert "changes_required=true" in capsys.readouterr().out
    assert not profile.exists()


def test_configure_dry_run_reports_stale_managed_plugin(tmp_path: Path, monkeypatch, capsys) -> None:
    profile = tmp_path / "profile"
    resource = tmp_path / "resource"
    install = tmp_path / "install"
    config_path = tmp_path / "robot.yaml"
    config_path.write_text("robot: test\n", encoding="utf-8")
    (resource / "hooks").mkdir(parents=True)
    (resource / "SOUL.md").write_text("soul\n", encoding="utf-8")
    (resource / "POLICY.md").write_text("policy\n", encoding="utf-8")
    (resource / "hooks" / "ibrobot-speak").write_text("hook\n", encoding="utf-8")
    (resource / "hooks" / "ibrobot-interim-speech").write_text("interim\n", encoding="utf-8")
    skill = install / "share" / "robot_skill_cli" / "skills" / "ibrobot-control" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("skill\n", encoding="utf-8")
    (tmp_path / ".shrc_local").write_text("# stub\n", encoding="utf-8")
    retired_plugin = profile / "plugins" / "ibrobot-robot-control"
    retired_plugin.mkdir(parents=True)
    (retired_plugin / "__init__.py").write_text(
        hermes_configure._MANAGED_FILE_HEADER + "\n# retired\n", encoding="utf-8"
    )

    monkeypatch.setenv("ROS_DOMAIN_ID", "52")
    monkeypatch.setattr(hermes_configure.shutil, "which", lambda _name: "/bin/hermes")
    monkeypatch.setattr(hermes_configure, "resolve_robot_config_path", lambda **_kwargs: config_path)
    monkeypatch.setattr(
        hermes_configure,
        "load_robot_config",
        lambda _path: SimpleNamespace(
            voice_tts=SimpleNamespace(
                enabled=True,
                service_name="/voice_tts/synthesize",
                playback_service_name="/voice_tts/play",
                playback_timeout_sec=300.0,
                synthesis_timeout_sec=90.0,
            )
        ),
    )
    monkeypatch.setattr(hermes_configure, "_active_profile", lambda _hermes: profile)
    monkeypatch.setattr(hermes_configure, "_resource_root", lambda: resource)
    monkeypatch.setattr(hermes_configure, "_install_prefix", lambda: install)

    assert hermes_configure.main(["--config-path", str(config_path), "--soul-mode", "skip", "--dry-run"]) == 0
    assert "changes_required=true" in capsys.readouterr().out
    # dry-run must not quarantine the plugin
    assert retired_plugin.is_dir(), "dry-run must not touch the managed plugin"


def test_approve_hooks_revokes_stale_mtime_before_auto_approval(tmp_path: Path, monkeypatch) -> None:
    calls = []
    hook_path = tmp_path / "hook"
    monkeypatch.setattr(
        hermes_configure,
        "_run",
        lambda arguments, **_kwargs: (
            calls.append((arguments, None)) or SimpleNamespace(returncode=0, stderr="", stdout="")
        ),
    )
    monkeypatch.setattr(
        hermes_configure.subprocess,
        "run",
        lambda arguments, **kwargs: (
            calls.append((arguments, kwargs["env"])) or SimpleNamespace(returncode=0, stderr="", stdout="")
        ),
    )

    hermes_configure._approve_hooks("/bin/hermes", hook_path)

    assert calls[0][0] == ["/bin/hermes", "hooks", "revoke", str(hook_path)]
    assert calls[1][0] == ["/bin/hermes", "--accept-hooks", "hooks", "doctor"]
    assert calls[1][1]["HERMES_ACCEPT_HOOKS"] == "1"


def test_approve_hooks_continues_when_hook_not_yet_registered(tmp_path: Path, monkeypatch) -> None:
    """First install: revoke finds nothing to revoke, hook is unlisted, doctor still runs."""
    hook_path = tmp_path / "hook"
    run_calls: list[list[str]] = []
    subprocess_calls: list[list[str]] = []

    def fake_run(arguments: list[str], **_kwargs):
        run_calls.append(list(arguments))
        if arguments[1:] == ["hooks", "revoke", str(hook_path)]:
            return SimpleNamespace(returncode=1, stderr="no such hook", stdout="")
        if arguments[1:] == ["hooks", "list"]:
            return SimpleNamespace(returncode=0, stderr="", stdout="")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(hermes_configure, "_run", fake_run)
    monkeypatch.setattr(
        hermes_configure.subprocess,
        "run",
        lambda arguments, **_kwargs: (
            subprocess_calls.append(list(arguments)) or SimpleNamespace(returncode=0, stderr="", stdout="")
        ),
    )

    hermes_configure._approve_hooks("/bin/hermes", hook_path)

    assert run_calls[0][1:] == ["hooks", "revoke", str(hook_path)]
    assert run_calls[1][1:] == ["hooks", "list"]
    assert subprocess_calls[0][1:] == ["--accept-hooks", "hooks", "doctor"]


def test_approve_hooks_raises_when_registered_hook_revoke_fails(tmp_path: Path, monkeypatch) -> None:
    """Reupgrade: hook is registered but revoke fails -> raise before reaching doctor."""
    hook_path = tmp_path / "hook"
    run_calls: list[list[str]] = []
    subprocess_calls: list[list[str]] = []

    def fake_run(arguments: list[str], **_kwargs):
        run_calls.append(list(arguments))
        if arguments[1:] == ["hooks", "revoke", str(hook_path)]:
            return SimpleNamespace(returncode=1, stderr="revoke denied", stdout="")
        if arguments[1:] == ["hooks", "list"]:
            return SimpleNamespace(returncode=0, stderr="", stdout=f"approved hooks:\n{hook_path}\n")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(hermes_configure, "_run", fake_run)
    monkeypatch.setattr(
        hermes_configure.subprocess,
        "run",
        lambda arguments, **_kwargs: (
            subprocess_calls.append(list(arguments)) or SimpleNamespace(returncode=0, stderr="", stdout="")
        ),
    )

    with pytest.raises(hermes_configure.ConfigureError, match="approval refresh failed"):
        hermes_configure._approve_hooks("/bin/hermes", hook_path)

    assert run_calls[0][1:] == ["hooks", "revoke", str(hook_path)]
    assert run_calls[1][1:] == ["hooks", "list"]
    assert subprocess_calls == [], "doctor must not run when revoke fails for a registered hook"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"extra": {"assistant_response": "你好，nod_yes ✓！"}}, "你好，nod_yes ✓！"),
        ({"assistant_response": "打开夹爪。"}, "打开夹爪。"),
        ({"extra": {"assistant_response": "```robot-skill status```"}}, "```robot-skill status```"),
    ],
)
def test_tts_payload_extraction_preserves_text_for_voice_tts(payload: dict, expected: str) -> None:
    assert hermes_tts_hook.extract_response(payload) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("你好，我是 IB-Robot 语音助手。", "你好,我是 - 语音助手。"),
        ("TTS 真机验证成功", " 真机验证成功"),
        ("打开 SO-101 夹爪", "打开 -101 夹爪"),
        ("你好世界", "你好世界"),
        ("v1.0 发布", "1.0 发布"),
        ("```robot-skill status```", "```- ```"),
        ("任务 71faa17ea1e843f2aeb800d3184ec320 完成", "任务  完成"),
        ("完成 123456 次", "完成 123456 次"),
    ],
)
def test_sanitize_for_tts_strips_english_letters(text: str, expected: str) -> None:
    assert hermes_tts_hook.sanitize_for_tts(text) == expected


def test_write_wav_requires_complete_wav_header(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(hermes_tts_hook.tempfile, "mkstemp", lambda **_: (999, str(tmp_path / "audio.wav")))
    with pytest.raises(hermes_tts_hook.SpeechHookError, match="invalid WAV"):
        hermes_tts_hook._write_wav(b"not wav")


def test_wait_for_future_rejects_service_exception(monkeypatch) -> None:
    monkeypatch.setattr(rclpy, "spin_until_future_complete", lambda *_args, **_kwargs: None)
    future = SimpleNamespace(done=lambda: True, exception=lambda: RuntimeError("boom"), result=lambda: None)
    with pytest.raises(hermes_tts_hook.SpeechHookError, match="service failed"):
        hermes_tts_hook._wait_for_future(SimpleNamespace(), future, 1.0, "operation")
