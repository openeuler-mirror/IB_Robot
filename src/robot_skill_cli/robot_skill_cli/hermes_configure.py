"""Synchronize the installed IB-Robot speech integration into one Hermes profile."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from ament_index_python.packages import get_package_share_directory

from robot_config.config_path import resolve_robot_config_path
from robot_config.loader import load_robot_config
from robot_skill_cli.hermes_launcher import LauncherError, register_hermes_skill

_LEGACY_PLUGIN = "ibrobot-robot-control"
_SOUL_BEGIN = "<!-- IBROBOT-MANAGED-BEGIN -->"
_SOUL_END = "<!-- IBROBOT-MANAGED-END -->"
_MANAGED_FILE_HEADER = "# Managed by hermes-robot-configure. Re-run the command instead of editing this file."
_HOOK_TIMEOUT_SEC = 300


class ConfigureError(RuntimeError):
    """User-facing profile synchronization failure."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hermes-robot-configure",
        description="Synchronize the IB-Robot Skill, SOUL policy, and speech hook into the active Hermes profile.",
    )
    config = parser.add_mutually_exclusive_group()
    config.add_argument("--config-name")
    config.add_argument("--config-path")
    parser.add_argument("--soul-mode", choices=("merge", "replace", "skip"), default="merge")
    parser.add_argument("--disable-speech", action="store_true")
    parser.add_argument("--accept-hooks", action="store_true")
    parser.add_argument("--restart-gateway", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _run(arguments: list[str], *, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(arguments, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ConfigureError(f"failed to run {arguments[0]!r}: {exc}") from exc


def _active_profile(hermes: str) -> Path:
    result = _run([hermes, "config", "path"])
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or not lines:
        raise ConfigureError("Hermes did not report its active config path")
    config_path = Path(lines[-1]).expanduser()
    if not config_path.is_absolute() or config_path.name != "config.yaml":
        raise ConfigureError("Hermes returned an invalid active config path")
    return config_path.parent


def _resource_root() -> Path:
    root = Path(get_package_share_directory("robot_skill_cli"))
    resource = root / "hermes"
    required = (
        resource / "SOUL.md",
        resource / "POLICY.md",
        resource / "hooks" / "ibrobot-speak",
    )
    if not all(path.is_file() for path in required):
        raise ConfigureError("installed IB-Robot Hermes resources are incomplete")
    return resource


def _install_prefix() -> Path:
    return Path(get_package_share_directory("robot_skill_cli")).parents[1]


def _workspace_shrc(install_prefix: Path) -> Path:
    shrc = install_prefix.parent / ".shrc_local"
    if not shrc.is_file():
        raise ConfigureError(
            "workspace .shrc_local is missing; it is the SSOT for ROS+venv+install overlay "
            "(see AGENTS.md). Rebuild the workspace or restore .shrc_local."
        )
    return shrc


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.name}.ibrobot-backup-{_timestamp()}")
    counter = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}.ibrobot-backup-{_timestamp()}-{counter}")
        counter += 1
    shutil.copy2(path, backup)
    return backup


def _atomic_write(path: Path, content: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
        temporary.chmod(0o755 if executable else 0o600)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigureError(f"cannot read Hermes config: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigureError("Hermes config root must be a mapping")
    return loaded


def _managed_soul(existing: str, policy: str) -> str:
    block = f"{_SOUL_BEGIN}\n\n{policy.strip()}\n\n{_SOUL_END}"
    start = existing.find(_SOUL_BEGIN)
    end = existing.find(_SOUL_END)
    if (start < 0) != (end < 0) or (start >= 0 and end < start):
        raise ConfigureError("SOUL.md contains an incomplete IB-Robot managed block")
    if start >= 0:
        end += len(_SOUL_END)
        return f"{existing[:start].rstrip()}\n\n{block}\n{existing[end:].lstrip()}".rstrip() + "\n"
    if not existing.strip():
        return block + "\n"
    return existing.rstrip() + "\n\n" + block + "\n"


def _sync_soul(profile: Path, resource: Path, mode: str, *, dry_run: bool) -> tuple[bool, Path | None]:
    if mode == "skip":
        return False, None
    soul_path = profile / "SOUL.md"
    current = soul_path.read_text(encoding="utf-8") if soul_path.is_file() else ""
    if mode == "replace":
        desired = (resource / "SOUL.md").read_text(encoding="utf-8")
    else:
        desired = _managed_soul(current, (resource / "POLICY.md").read_text(encoding="utf-8"))
    if current == desired:
        return False, None
    if dry_run:
        return True, None
    backup = _backup(soul_path)
    _atomic_write(soul_path, desired)
    return True, backup


def _tts_hook_wrapper(
    *,
    workspace: Path,
    shrc: Path,
    config_path: Path,
    ros_domain_id: str,
    synthesis_service: str,
    playback_service: str,
    playback_timeout_sec: float,
    synthesis_timeout_sec: float,
) -> str:
    return "\n".join(
        (
            "#!/usr/bin/env bash",
            "set -eo pipefail",
            _MANAGED_FILE_HEADER,
            'hermes_env="${HERMES_ENV_FILE:-/root/claw/hermes/data/.env}"',
            'if [ -f "$hermes_env" ]; then',
            '  set -a; . "$hermes_env"; set +a',
            "fi",
            'if [ -z "${XUNXING_API_KEY:-}" ] && [ -n "${HERMES_CUSTOM_AZ_GPTPLUS5_COM_API_KEY:-}" ]; then',
            '  export XUNXING_API_KEY="$HERMES_CUSTOM_AZ_GPTPLUS5_COM_API_KEY"',
            "fi",
            f"cd {shlex.quote(str(workspace))}",
            f"source {shlex.quote(str(shrc))} >/dev/null",
            f"export ROS_DOMAIN_ID={shlex.quote(ros_domain_id)}",
            f"export ROBOT_CONFIG={shlex.quote(str(config_path))}",
            f"export IBROBOT_TTS_SYNTHESIS_SERVICE={shlex.quote(synthesis_service)}",
            f"export IBROBOT_TTS_PLAYBACK_SERVICE={shlex.quote(playback_service)}",
            f"export IBROBOT_TTS_SYNTHESIS_TIMEOUT_SEC={synthesis_timeout_sec:g}",
            f"export IBROBOT_TTS_PLAYBACK_TIMEOUT_SEC={playback_timeout_sec:g}",
            "exec python3 -m robot_skill_cli.hermes_tts_hook",
            "",
        )
    )


def _lifecycle_hook_wrapper(*, workspace: Path, shrc: Path) -> str:
    # Hermes is launched from the fully initialized workspace environment.  The
    # lifecycle hook has a two-second deadline, so sourcing ROS/CANN again here
    # can consume the entire budget before the asynchronous handoff runs.
    del shrc
    return "\n".join(
        (
            "#!/usr/bin/env bash",
            "set -eo pipefail",
            _MANAGED_FILE_HEADER,
            f"cd {shlex.quote(str(workspace))}",
            "exec python3 -m robot_skill_cli.hermes_lifecycle_speech",
            "",
        )
    )


def _interim_hook_wrapper(*, workspace: Path, shrc: Path, speaker_path: Path | None = None) -> str:
    speaker = speaker_path or workspace / "build/robot_skill_cli/hermes-speak"
    return "\n".join(
        (
            "#!/usr/bin/env bash",
            "set -eo pipefail",
            _MANAGED_FILE_HEADER,
            f"cd {shlex.quote(str(workspace))}",
            f"source {shlex.quote(str(shrc))} >/dev/null",
            f"export IBROBOT_INTERIM_SPEAKER={shlex.quote(str(speaker))}",
            "exec python3 -m robot_skill_cli.hermes_interim_speech",
            "",
        )
    )


def _robot_skill_wrapper(*, workspace: Path, shrc: Path, config_path: Path, ros_domain_id: str) -> str:
    return "\n".join(
        (
            "#!/usr/bin/env bash",
            "set -eo pipefail",
            _MANAGED_FILE_HEADER,
            'for argument in "$@"; do',
            '    case "$argument" in',
            "        --config-name|--config-name=*|--config-path|--config-path=*)",
            "            echo 'robot-skill: configuration is bound by hermes-robot-configure' >&2",
            "            exit 2",
            "            ;;",
            "    esac",
            "done",
            f"cd {shlex.quote(str(workspace))}",
            f"source {shlex.quote(str(shrc))} >/dev/null",
            f"export ROS_DOMAIN_ID={shlex.quote(ros_domain_id)}",
            f"export ROBOT_CONFIG={shlex.quote(str(config_path))}",
            "export IBROBOT_HERMES_LIFECYCLE_SPEECH=1",
            'exec python3 -m robot_skill_cli.cli --config-path "$ROBOT_CONFIG" "$@"',
            "",
        )
    )


def _shell_environment(
    *,
    workspace: Path,
    shrc: Path,
    config_path: Path,
    ros_domain_id: str,
    profile_bin: Path,
) -> str:
    return "\n".join(
        (
            "#!/usr/bin/env bash",
            _MANAGED_FILE_HEADER,
            f"cd {shlex.quote(str(workspace))}",
            f"source {shlex.quote(str(shrc))} >/dev/null",
            f"export ROS_DOMAIN_ID={shlex.quote(ros_domain_id)}",
            f"export ROBOT_CONFIG={shlex.quote(str(config_path))}",
            f"export PATH={shlex.quote(str(profile_bin))}:$PATH",
            "",
        )
    )


def _remove_stale_skill_copies(profile: Path, config: dict[str, Any]) -> None:
    """Quarantine ibrobot-control skill copies that collide with the managed profile copy.

    ``hermes-robot`` writes a cache workspace under ``external_dirs`` and may leave
    a stale ``ibrobot-control`` copy there. When both the profile and the cache
    directory expose the same skill name, Hermes refuses to load it. Rename the
    cache copy to a timestamped backup so the managed profile copy is the sole
    source, without deleting user-managed skills that may live under the same path.
    """
    external_dirs = config.get("external_dirs", "")
    if not isinstance(external_dirs, str) or not external_dirs.strip():
        return
    for raw_dir in external_dirs.split(os.pathsep):
        directory = Path(raw_dir).expanduser()
        stale = directory / "skills" / "ibrobot-control"
        if stale.is_dir():
            backup = stale.with_name(f"{stale.name}.quarantined-{_timestamp()}")
            counter = 1
            while backup.exists():
                backup = stale.with_name(f"{stale.name}.quarantined-{_timestamp()}-{counter}")
                counter += 1
            try:
                stale.rename(backup)
            except OSError:
                # rename failed (cross-device or permission); fall back to leaving it
                # in place rather than rmtree, which could delete user-managed skills.
                continue


def _is_managed_plugin(plugin_dir: Path) -> bool:
    """Return True only if ``plugin_dir`` was written by hermes-robot-configure.

    A managed plugin carries at least one file whose content carries the managed
    header. Anything else is treated as user-authored and left alone, so a
    user-owned plugin that happens to share the name is never touched.
    """
    try:
        for entry in plugin_dir.iterdir():
            if not entry.is_file():
                continue
            try:
                content = entry.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if _MANAGED_FILE_HEADER in content:
                return True
    except OSError:
        return False
    return False


def _has_managed_plugin(profile: Path) -> bool:
    """True if a managed ``ibrobot-robot-control`` plugin copy is present in the profile."""
    plugin_target = profile / "plugins" / "ibrobot-robot-control"
    return plugin_target.is_dir() and _is_managed_plugin(plugin_target)


def _remove_stale_plugin(profile: Path) -> bool:
    """Quarantine a previously-installed managed ``ibrobot-robot-control`` plugin copy.

    The immediate-execution plugin was retired from this PR (its worker subcommands
    were never implemented in robot-skill). Clean up any stale installed copy so
    Hermes does not load dead code. To avoid deleting unowned plugin content, only
    directories that carry our managed marker are touched; anything else is left in
    place for the operator to resolve. The directory is renamed to a timestamped
    backup (not rmtree) so the content stays recoverable even if the marker check
    is wrong.
    """
    if not _has_managed_plugin(profile):
        return False
    plugin_target = profile / "plugins" / "ibrobot-robot-control"
    backup = plugin_target.with_name(f"{plugin_target.name}.quarantined-{_timestamp()}")
    counter = 1
    while backup.exists():
        backup = plugin_target.with_name(f"{plugin_target.name}.quarantined-{_timestamp()}-{counter}")
        counter += 1
    try:
        plugin_target.rename(backup)
    except OSError:
        # rename failed (cross-device or permission); leave it in place rather than
        # rmtree, which could delete user-managed or unowned plugin content.
        return False
    return True


def _filter_legacy(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    return [item for item in items if item != _LEGACY_PLUGIN]


def _is_managed_hook(entry: Any, target: Path) -> bool:
    if not isinstance(entry, dict):
        return False
    command = str(entry.get("command", ""))
    return command == str(target) or command.endswith("/.hermes/hooks/hermes-speak.py")


def _update_config(
    config: dict[str, Any],
    *,
    workspace: Path,
    environment_file: Path,
    hook_path: Path,
    speech_enabled: bool,
    lifecycle_hook_path: Path | None = None,
    interim_hook_path: Path | None = None,
) -> dict[str, Any]:
    plugins = config.setdefault("plugins", {})
    if not isinstance(plugins, dict):
        raise ConfigureError("Hermes plugins config must be a mapping")
    enabled = plugins.get("enabled", [])
    disabled = plugins.get("disabled", [])
    if not isinstance(enabled, list) or not isinstance(disabled, list):
        raise ConfigureError("Hermes plugins.enabled and plugins.disabled must be lists")
    plugins["enabled"] = _filter_legacy(enabled)
    plugins["disabled"] = _filter_legacy(disabled)

    toolsets = config.get("toolsets", [])
    if not isinstance(toolsets, list):
        raise ConfigureError("Hermes toolsets must be a list")
    config["toolsets"] = _filter_legacy(toolsets)

    terminal = config.setdefault("terminal", {})
    if not isinstance(terminal, dict):
        raise ConfigureError("Hermes terminal config must be a mapping")
    terminal["cwd"] = str(workspace)
    init_files = terminal.get("shell_init_files", [])
    if not isinstance(init_files, list):
        raise ConfigureError("Hermes terminal.shell_init_files must be a list")
    managed_name = str(environment_file)
    terminal["shell_init_files"] = [item for item in init_files if "ibrobot-env.sh" not in str(item)]
    terminal["shell_init_files"].append(managed_name)
    terminal["auto_source_bashrc"] = False

    hooks = config.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ConfigureError("Hermes hooks config must be a mapping")

    post = hooks.get("post_llm_call", [])
    if not isinstance(post, list):
        raise ConfigureError("Hermes hooks.post_llm_call must be a list")
    post = [entry for entry in post if not _is_managed_hook(entry, hook_path)]
    if speech_enabled:
        post.append({"command": str(hook_path), "timeout": _HOOK_TIMEOUT_SEC})
    if post:
        hooks["post_llm_call"] = post
    else:
        hooks.pop("post_llm_call", None)
    if lifecycle_hook_path is not None:
        for event in ("pre_llm_call", "pre_tool_call", "post_tool_call"):
            entries = hooks.get(event, [])
            if not isinstance(entries, list):
                raise ConfigureError(f"Hermes hooks.{event} must be a list")
            entries = [entry for entry in entries if not _is_managed_hook(entry, lifecycle_hook_path)]
            if speech_enabled:
                entry: dict[str, Any] = {"command": str(lifecycle_hook_path), "timeout": 2}
                if event in {"pre_tool_call", "post_tool_call"}:
                    entry["matcher"] = "^terminal$"
                entries.append(entry)
            if entries:
                hooks[event] = entries
            else:
                hooks.pop(event, None)
    if interim_hook_path is not None:
        entries = hooks.get("on_interim_message", [])
        if not isinstance(entries, list):
            raise ConfigureError("Hermes hooks.on_interim_message must be a list")
        entries = [entry for entry in entries if not _is_managed_hook(entry, interim_hook_path)]
        if speech_enabled:
            entries.append({"command": str(interim_hook_path), "timeout": 2})
        if entries:
            hooks["on_interim_message"] = entries
        else:
            hooks.pop("on_interim_message", None)
    return config


def _write_yaml(path: Path, config: dict[str, Any]) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    _atomic_write(path, rendered)


def _approval_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["HERMES_ACCEPT_HOOKS"] = "1"
    return environment


def _hook_registered(hermes: str, hook_path: Path) -> bool:
    """Return True if Hermes currently tracks ``hook_path`` as an approved hook.

    ``hermes hooks revoke`` is best-effort: on first install the hook was never
    approved, so revoke has nothing to remove and may exit non-zero. Before
    treating a revoke failure as fatal, confirm the hook is actually registered
    via ``hermes hooks list``. When the listing itself fails, conservatively
    return True so a real revoke failure is not silently swallowed. The match
    is by the hook's absolute path; if Hermes lists hooks by name only the
    check reports False and the subsequent ``hooks doctor`` remains the
    authoritative approval gate.
    """
    listed = _run([hermes, "hooks", "list"])
    if listed.returncode != 0:
        return True
    return str(hook_path) in (listed.stdout or "")


def _approve_hooks(hermes: str, hook_paths: Sequence[Path] | Path) -> None:
    if isinstance(hook_paths, Path):
        hook_paths = [hook_paths]
    for hook_path in hook_paths:
        revoked = _run([hermes, "hooks", "revoke", str(hook_path)])
        if revoked.returncode != 0 and _hook_registered(hermes, hook_path):
            detail = (revoked.stderr or revoked.stdout).strip()
            raise ConfigureError(f"Hermes hook approval refresh failed: {detail}")
    result = subprocess.run(
        [hermes, "--accept-hooks", "hooks", "doctor"],
        capture_output=True,
        text=True,
        timeout=60.0,
        check=False,
        env=_approval_environment(),
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ConfigureError(f"Hermes hook approval failed: {detail}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        hermes = shutil.which("hermes")
        if not hermes:
            raise ConfigureError("required executable is not installed: hermes")
        ros_domain_id = os.environ.get("ROS_DOMAIN_ID", "").strip()
        if not ros_domain_id.isdigit() or not 0 <= int(ros_domain_id) <= 232:
            raise ConfigureError("ROS_DOMAIN_ID must be set to an integer from 0 to 232")

        config_path = resolve_robot_config_path(config_name=args.config_name, config_path=args.config_path)
        robot = load_robot_config(config_path)
        if not args.disable_speech and not robot.voice_tts.enabled:
            raise ConfigureError(
                "selected robot config has voice_tts.enabled=false; use --disable-speech or enable TTS"
            )

        profile = _active_profile(hermes)
        resource = _resource_root()
        install_prefix = _install_prefix()
        workspace = install_prefix.parent.resolve()
        shrc = _workspace_shrc(install_prefix)
        profile_config_path = profile / "config.yaml"
        hook_path = profile / "hooks" / "ibrobot-speak"
        lifecycle_hook_path = profile / "hooks" / "ibrobot-lifecycle-speech"
        interim_hook_path = profile / "hooks" / "ibrobot-interim-speech"
        environment_file = profile / "ibrobot" / "ibrobot-env.sh"
        profile_bin = profile / "ibrobot" / "bin"
        robot_skill_path = profile_bin / "robot-skill"
        desired_hook = _tts_hook_wrapper(
            workspace=workspace,
            shrc=shrc,
            config_path=config_path,
            ros_domain_id=ros_domain_id,
            synthesis_service=robot.voice_tts.service_name,
            playback_service=robot.voice_tts.playback_service_name,
            playback_timeout_sec=robot.voice_tts.playback_timeout_sec,
            synthesis_timeout_sec=robot.voice_tts.synthesis_timeout_sec,
        )
        desired_lifecycle_hook = _lifecycle_hook_wrapper(workspace=workspace, shrc=shrc)
        desired_interim_hook = _interim_hook_wrapper(workspace=workspace, shrc=shrc, speaker_path=hook_path)
        desired_environment = _shell_environment(
            workspace=workspace,
            shrc=shrc,
            config_path=config_path,
            ros_domain_id=ros_domain_id,
            profile_bin=profile_bin,
        )
        desired_robot_skill = _robot_skill_wrapper(
            workspace=workspace,
            shrc=shrc,
            config_path=config_path,
            ros_domain_id=ros_domain_id,
        )
        current_config = _yaml_config(profile_config_path)
        desired_config = _update_config(
            current_config,
            workspace=workspace,
            environment_file=environment_file,
            hook_path=hook_path,
            lifecycle_hook_path=lifecycle_hook_path,
            interim_hook_path=interim_hook_path,
            speech_enabled=not args.disable_speech,
        )
        soul_changed, soul_backup = _sync_soul(profile, resource, args.soul_mode, dry_run=args.dry_run)
        plugin_removal_needed = _has_managed_plugin(profile)
        plugin_removed = _remove_stale_plugin(profile) if not args.dry_run else False
        if not args.dry_run:
            _remove_stale_skill_copies(profile, desired_config)

        skill_target = profile / "skills" / "ibrobot-control" / "SKILL.md"
        hook_changed = (
            hook_path.is_file()
            if args.disable_speech
            else (not hook_path.is_file() or hook_path.read_text(encoding="utf-8") != desired_hook)
        )
        lifecycle_changed = (
            lifecycle_hook_path.is_file()
            if args.disable_speech
            else (
                not lifecycle_hook_path.is_file()
                or lifecycle_hook_path.read_text(encoding="utf-8") != desired_lifecycle_hook
            )
        )
        interim_changed = (
            interim_hook_path.is_file()
            if args.disable_speech
            else not interim_hook_path.is_file()
            or interim_hook_path.read_text(encoding="utf-8") != desired_interim_hook
        )
        writes_needed = (
            not skill_target.is_file()
            or skill_target.read_bytes()
            != (install_prefix / "share/robot_skill_cli/skills/ibrobot-control/SKILL.md").read_bytes()
            or hook_changed
            or lifecycle_changed
            or interim_changed
            or plugin_removal_needed
            or not environment_file.is_file()
            or environment_file.read_text(encoding="utf-8") != desired_environment
            or not robot_skill_path.is_file()
            or robot_skill_path.read_text(encoding="utf-8") != desired_robot_skill
            or _yaml_config(profile_config_path) != desired_config
            or soul_changed
        )
        if args.dry_run:
            print(f"profile={profile}")
            print(f"robot_config={config_path}")
            print(f"workspace={workspace}")
            print(f"changes_required={str(writes_needed).lower()}")
            return 0

        profile.mkdir(parents=True, exist_ok=True)
        register_hermes_skill(
            install_prefix / "share/robot_skill_cli/skills/ibrobot-control/SKILL.md",
            profile / "skills",
        )
        _atomic_write(environment_file, desired_environment, executable=True)
        _atomic_write(robot_skill_path, desired_robot_skill, executable=True)
        if args.disable_speech:
            hook_path.unlink(missing_ok=True)
            lifecycle_hook_path.unlink(missing_ok=True)
            interim_hook_path.unlink(missing_ok=True)
        else:
            _atomic_write(hook_path, desired_hook, executable=True)
            _atomic_write(lifecycle_hook_path, desired_lifecycle_hook, executable=True)
            _atomic_write(interim_hook_path, desired_interim_hook, executable=True)
        if _yaml_config(profile_config_path) != desired_config:
            config_backup = _backup(profile_config_path)
            _write_yaml(profile_config_path, desired_config)
        else:
            config_backup = None

        if args.accept_hooks and not args.disable_speech:
            _approve_hooks(hermes, [hook_path, lifecycle_hook_path, interim_hook_path])
        if args.restart_gateway:
            result = _run([hermes, "gateway", "restart"], timeout=60.0)
            if result.returncode != 0:
                raise ConfigureError((result.stderr or result.stdout).strip() or "Hermes Gateway restart failed")

        print(f"Hermes profile synchronized: {profile}")
        print(f"IB-Robot workspace: {workspace}")
        print(f"Speech hook: {'disabled' if args.disable_speech else 'enabled'}")
        if soul_backup:
            print(f"SOUL backup: {soul_backup}")
        if config_backup:
            print(f"Config backup: {config_backup}")
        if plugin_removed:
            print("Retired ibrobot-robot-control plugin quarantined")
        if not args.restart_gateway:
            print("Restart the messaging Gateway to apply changes: hermes gateway restart")
        return 0
    except (ConfigureError, FileNotFoundError, LauncherError, ValueError) as exc:
        print(f"hermes-robot-configure: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
