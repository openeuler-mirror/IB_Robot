"""Staging support for the navigation simulation overlay (D10).

The loader resolves ``base_config`` siblings only, so the Gazebo-free
resolved-config test and the navigation end-to-end test stage the shipped
``config/robots/lekiwi_lidar.yaml`` and ``test/config/lekiwi_navi_sim.yaml``
into one directory and launch with ``config_path:=<staged overlay>``. The
same staging covers the ``NAV2_TEST_PARAMS_FILE`` parameter-override
mechanism (a temporary Nav2 params variant), which rewrites the overlay's
``params_file`` before staging.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml


def _package_dir() -> Path:
    """The robot_config package root of this checkout."""
    return Path(__file__).resolve().parents[1]


def stage_navigation_config(destination: Path, nav2_params_override: str = "") -> Path:
    """Stage base + overlay side by side; return the staged overlay path.

    ``destination`` receives a copy of the shipped ``lekiwi_lidar.yaml`` and
    the ``lekiwi_navi_sim.yaml`` overlay. With ``nav2_params_override`` the
    staged overlay's Nav2 ``params_file`` points at that file instead (the
    temporary parameter-variant mechanism of the e2e run).
    """
    destination.mkdir(parents=True, exist_ok=True)
    package = _package_dir()
    shutil.copy2(package / "config" / "robots" / "lekiwi_lidar.yaml", destination / "lekiwi_lidar.yaml")
    overlay_path = destination / "lekiwi_navi_sim.yaml"
    if nav2_params_override:
        document = yaml.safe_load((package / "test" / "config" / "lekiwi_navi_sim.yaml").read_text(encoding="utf-8"))
        navigation = document["robot"]["nav_stages"]["navigation"]["navigation"]["nav2_bringup"]
        navigation["params_file"] = str(Path(nav2_params_override).expanduser().resolve())
        overlay_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    else:
        shutil.copy2(package / "test" / "config" / "lekiwi_navi_sim.yaml", overlay_path)
    return overlay_path


def load_staged_navigation_config(staged_overlay: Path):
    """The resolved navigation stage of the staged overlay (D10 assertions)."""
    from robot_config.loader import load_robot_config_dict

    return load_robot_config_dict(staged_overlay, nav_stage="navigation", defer_interface_binding=True)
