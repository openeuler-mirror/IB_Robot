"""Staging support for the navigation simulation overlay (D10/D15).

The loader resolves ``base_config`` siblings only, so the Gazebo-free
resolved-config test and the navigation end-to-end test stage the shipped
``config/robots/lekiwi_lidar.yaml`` and ``test/config/lekiwi_navi_sim.yaml``
into one directory and launch with ``config_path:=<staged overlay>``. The
same staging covers the ``NAV2_TEST_PARAMS_FILE`` parameter-override
mechanism (a temporary Nav2 params variant), which rewrites the overlay's
``params_file`` before staging, and the per-platform rewrite of
``simulation.platform`` (D15, the D14 gate staging pattern: the shipped
overlay pins ``gazebo``; the staged copy carries the platform under test so
the world owner selects the right provider branch).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml


def _package_dir() -> Path:
    """The robot_config package root of this checkout."""
    return Path(__file__).resolve().parents[1]


def stage_navigation_config(destination: Path, nav2_params_override: str = "", sim_platform: str = "gazebo") -> Path:
    """Stage base + overlay side by side; return the staged overlay path.

    ``destination`` receives a copy of the shipped ``lekiwi_lidar.yaml`` and
    the ``lekiwi_navi_sim.yaml`` overlay with ``simulation.platform`` set to
    ``sim_platform``. With ``nav2_params_override`` the staged overlay's Nav2
    ``params_file`` points at that file instead (the temporary
    parameter-variant mechanism of the e2e run).
    """
    destination.mkdir(parents=True, exist_ok=True)
    package = _package_dir()
    shutil.copy2(package / "config" / "robots" / "lekiwi_lidar.yaml", destination / "lekiwi_lidar.yaml")
    overlay_path = destination / "lekiwi_navi_sim.yaml"
    document = yaml.safe_load((package / "test" / "config" / "lekiwi_navi_sim.yaml").read_text(encoding="utf-8"))
    stage = document["robot"]["nav_stages"]["navigation"]
    simulation = stage.get("simulation")
    if not isinstance(simulation, dict):
        # fail loudly: the silent ``(stage.get("simulation") or {})`` write
        # would stage an overlay whose platform nobody set
        raise ValueError(
            "the lekiwi_navi_sim overlay must define robot.nav_stages.navigation.simulation (a mapping) "
            f"to carry the platform under test; got {type(simulation).__name__}"
        )
    simulation["platform"] = sim_platform
    if nav2_params_override:
        navigation = stage["navigation"]["nav2_bringup"]
        navigation["params_file"] = str(Path(nav2_params_override).expanduser().resolve())
    overlay_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return overlay_path


def load_staged_navigation_config(staged_overlay: Path):
    """The resolved navigation stage of the staged overlay (D10 assertions)."""
    from robot_config.loader import load_robot_config_dict

    return load_robot_config_dict(staged_overlay, nav_stage="navigation", defer_interface_binding=True)
