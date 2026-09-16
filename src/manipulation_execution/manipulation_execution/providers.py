"""Configuration-declared robot-suite provider resolution for the pick pipeline.

Generic pipeline phases consume robot-specific behaviors (gripper mesh
geometry, wrist-orientation guards) through provider modules declared in the
robot configuration instead of direct robot-suite imports, so the generic
packages stay robot-agnostic and a robot without a suite provider runs with
those grasp features disabled (design D8).
"""

from __future__ import annotations

import importlib
from types import ModuleType

GRASP_GEOMETRY_FUNCTIONS = (
    "gripper_mesh_min_z",
    "tabletop_clearance",
    "gripper_geometry_metrics_batch",
)
WRIST_GUARD_FUNCTIONS = (
    "apply_joint5_retry",
    "canonicalize_joint5",
    "joint5_branch_continuity_check",
    "joint5_closing_axis_correction",
    "joint5_within_abs_limit",
)


def load_provider(
    config_path: str,
    module_path: object,
    required_functions: tuple[str, ...],
) -> ModuleType | None:
    """Import a configuration-declared provider module and verify its contract.

    ``config_path`` names the configuration field for error messages.
    Returns ``None`` when no provider is declared; raises ``ValueError``
    naming the field, the import failure, or the missing functions so node
    startup fails fast with a remediation hint.
    """
    declared = str(module_path or "").strip()
    if not declared:
        return None
    try:
        module = importlib.import_module(declared)
    except Exception as exc:
        raise ValueError(f"{config_path} provider '{declared}' cannot be imported: {exc}") from exc
    missing = [name for name in required_functions if not callable(getattr(module, name, None))]
    if missing:
        raise ValueError(
            f"{config_path} provider '{declared}' does not provide required functions: {', '.join(missing)}"
        )
    return module
