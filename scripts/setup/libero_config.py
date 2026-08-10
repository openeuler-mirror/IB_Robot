#!/usr/bin/env python3
"""Workspace-owned LIBERO configuration generator.

Generates a non-interactive config.yaml for the pinned LIBERO provider
so that ``import libero`` and ``import libero.libero`` work without
prompting for user input (which fails under non-interactive shells /
fresh HOME).

The config is written to ``${WORKSPACE}/venv/ibrobot_libero/config.yaml``
and the provider reads it via the ``LIBERO_CONFIG_PATH`` environment
variable (set by ``.shrc_local`` and by the benchmark setup block).

This script does NOT modify ``libs/libero``. It only creates a
workspace-owned config file that the provider reads at import time.

Properties:
  - Idempotent: re-running produces the same file
  - Atomic: writes to a temp file then renames
  - Path-aware: regenerates correct paths if WORKSPACE changes
  - Non-interactive: never calls input()
"""

from __future__ import annotations

import os
import sys
import tempfile

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required but not installed", file=sys.stderr)
    sys.exit(1)


def _resolve_paths(workspace: str) -> dict[str, str]:
    """Resolve the five required LIBERO config paths from the workspace."""
    libero_root = os.path.join(workspace, "libs", "libero", "libero", "libero")
    datasets_dir = os.path.join(workspace, "datasets", "libero")
    return {
        "benchmark_root": libero_root,
        "bddl_files": os.path.join(libero_root, "bddl_files"),
        "init_states": os.path.join(libero_root, "init_files"),
        "assets": os.path.join(libero_root, "assets"),
        "datasets": datasets_dir,
    }


def generate_config(workspace: str, config_dir: str | None = None) -> str:
    """Generate the workspace-owned LIBERO config.yaml.

    Args:
        workspace: Absolute path to the IB-Robot workspace root.
        config_dir: Optional override for the config directory.
            Defaults to ``${workspace}/venv/ibrobot_libero``.

    Returns:
        The path to the generated config file.
    """
    if config_dir is None:
        config_dir = os.path.join(workspace, "venv", "ibrobot_libero")

    config_path = os.path.join(config_dir, "config.yaml")
    paths = _resolve_paths(workspace)

    # Create the config directory if it doesn't exist.
    os.makedirs(config_dir, exist_ok=True)

    # Create the datasets directory if it doesn't exist (workspace-owned).
    os.makedirs(paths["datasets"], exist_ok=True)

    # Atomic write: write to temp file, then rename.
    fd, tmp_path = tempfile.mkstemp(dir=config_dir, suffix=".tmp", prefix="config_")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(paths, f, default_flow_style=False, sort_keys=True)
        os.replace(tmp_path, config_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return config_path


def main() -> int:
    workspace = os.environ.get("WORKSPACE", os.getcwd())
    if not os.path.isabs(workspace):
        workspace = os.path.abspath(workspace)

    config_dir = os.path.join(workspace, "venv", "ibrobot_libero")
    try:
        config_path = generate_config(workspace, config_dir)
    except Exception as exc:
        print(f"ERROR: Failed to generate LIBERO config: {exc}", file=sys.stderr)
        return 1

    paths = _resolve_paths(workspace)
    print(f"LIBERO_CONFIG_PATH={config_dir}")
    print(f"config_file={config_path}")
    for key, value in sorted(paths.items()):
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
