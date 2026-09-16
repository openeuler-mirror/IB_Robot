"""Resolve robot configuration files from explicit inputs and environment."""

from __future__ import annotations

import os
from contextlib import suppress
from pathlib import Path

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

_SOURCE_ROBOTS_DIRECTORY = Path(__file__).resolve().parents[1] / "config" / "robots"


def _resolve_explicit_path(config_path: str | Path) -> Path:
    resolved_path = Path(config_path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Robot configuration not found: {config_path}")
    return resolved_path


def _resolve_config_name(config_name: str) -> Path:
    name = config_name.strip()
    if not name:
        raise FileNotFoundError(f"Robot configuration not found: {config_name}")

    file_name = name if Path(name).suffix else f"{name}.yaml"
    candidates: list[Path] = []
    with suppress(PackageNotFoundError):
        candidates.append(Path(get_package_share_directory("robot_config")) / "config" / "robots" / file_name)
    candidates.append(_SOURCE_ROBOTS_DIRECTORY / file_name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Robot configuration not found: {config_name}")


def _resolve_environment_config(value: str) -> Path:
    expanded = Path(value).expanduser()
    if expanded.is_file():
        return expanded.resolve()
    if expanded.is_absolute() or expanded.parent != Path(".") or expanded.suffix:
        return _resolve_explicit_path(value)
    return _resolve_config_name(value)


def resolve_robot_config_path(config_name: str | None = None, config_path: str | Path | None = None) -> Path:
    """Return the selected robot configuration using the documented precedence.

    Precedence (highest first):

    1. ``config_path``: treated as an existing file path. Must exist.
    2. ``config_name``: resolved against the installed ``robot_config`` share
       directory first, then the source tree ``config/robots/`` directory.
       A ``.yaml`` suffix is appended when missing.
    3. ``ROBOT_CONFIG`` environment variable: if the value is an existing file,
       use it directly; otherwise, if it contains a path separator or a suffix,
       treat it as an explicit path (must exist); otherwise treat it as a
       config name (same resolution as case 2).
    4. ``ROBOT_NAME`` environment variable: always treated as a config name.

    The library-level API has no implicit default robot: when neither an
    explicit argument nor an environment variable provides a name/path,
    a ``ValueError`` instructs the caller to provide one. Launch entry
    points may supply one explicit, documented default value.

    This function never reads ROS runtime state; it is safe to call from
    catalog-only CLI paths that do not initialize rclpy.
    """
    if config_path is not None:
        return _resolve_explicit_path(config_path)
    if config_name is not None:
        return _resolve_config_name(config_name)

    environment_config = os.environ.get("ROBOT_CONFIG")
    if environment_config:
        return _resolve_environment_config(environment_config)
    environment_name = os.environ.get("ROBOT_NAME")
    if environment_name:
        return _resolve_config_name(environment_name)
    raise ValueError(
        "Robot configuration name or path is required: pass config_name/config_path "
        "or set the ROBOT_CONFIG/ROBOT_NAME environment variable"
    )
