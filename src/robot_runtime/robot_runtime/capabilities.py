"""Canonical robot capability vocabulary.

This registry is the single source of truth for capability names used in
RuntimeStatus messages, robot configuration ``capabilities.requires``
declarations, runtime profiles, and skill admission checks.

EXTENSION-ONLY POLICY: names may be added but never removed or renamed.
Reserved names are declared for planned runtimes (e.g. X2/AimDK) so the
vocabulary is stable before their implementations land.
"""

from __future__ import annotations

REGISTRY_VERSION = 2

# Frozen capability names. Domain.channel granularity; sensors and
# peripherals stay in the robot contract (peripherals + observations).
_CAPABILITY_NAMES = frozenset(
    {
        # Joint-level sensing and control
        "joint.state",
        "joint.position_stream",
        "joint.trajectory",
        # End effectors
        "gripper.1d",
        "hand.multi_joint",
        "hand.gesture",
        # Mobile base
        "base.cmd_vel",
        "base.odom",
        "base.navigation_gate",
        # Goal-level navigation (native planner or Nav2 bridged)
        "nav.goal",
        # Runtime-neutral motion services (robot-motion-services spec)
        "motion.fk",
        "motion.ik",
        "motion.move_to_joint",
        "motion.move_to_pose",
        # Named/pre-programmed motions (preset, linkcraft, skill trajectories)
        "motion.named",
        "perception.camera",
        "perception.lidar",
        # Stop service with ordered guarantees (robot-runtime-contract spec)
        "runtime.stop",
        "runtime.status",
        # Reserved (declared, not implemented in this change)
        "body.whole_stream",  # X2 Develop_MC tier whole-body streaming
        "joint.priority_stream",  # per-joint priority arbitration (dropped from v1)
    }
)

# Reserved names for planned runtimes — declared now, implemented later.
_RESERVED_NAMES = frozenset(
    {
        "body.whole_stream",
        "joint.priority_stream",
        "nav.goal",
        "hand.multi_joint",
        "hand.gesture",
    }
)

# Names that carry quantitative parameters in RuntimeStatus.capabilities_json.
_CAPABILITY_PARAMS: dict[str, dict[str, type]] = {
    "joint.state": {"joint_count": int, "rate_hz": float},
    "joint.position_stream": {"joint_count": int, "rate_hz": float},
    "joint.trajectory": {"joint_count": int},
    "gripper.1d": {"count": int},
    "perception.camera": {"topics": list},
    "perception.lidar": {"topics": list},
    "base.cmd_vel": {"max_vx": float, "max_vy": float, "max_wz": float, "staleness_s": float},
    "motion.ik": {"endpoints": list},
    "runtime.stop": {"cancel_bound_s": float, "idle_bound_s": float, "torque_off_bound_s": float},
    "body.whole_stream": {"rate_hz": float},
}


def all_capabilities() -> frozenset[str]:
    """All registered capability names."""
    return _CAPABILITY_NAMES


def reserved_names() -> frozenset[str]:
    """Names reserved for future runtimes (declared, not yet implemented)."""
    return _RESERVED_NAMES


def parameter_schema(name: str) -> dict[str, type]:
    """Declared parameter fields for a capability (empty when none)."""
    return dict(_CAPABILITY_PARAMS.get(name, {}))


def is_valid(name: str) -> bool:
    """True when ``name`` is a registered capability."""
    return name in _CAPABILITY_NAMES


def validate_capability_set(names: list[str] | set[str]) -> list[str]:
    """Return the sorted list of invalid names (empty when all valid)."""
    return sorted(str(n) for n in names if not is_valid(str(n)))


def requires_subset_of_discovered(required: list[str] | set[str], discovered: list[str] | set[str]) -> list[str]:
    """Return the sorted list of required capabilities missing from discovered.

    This is the launch-time reconciliation check: a non-empty result means
    the runtime does not provide what the deployment requires, and launch
    must fail fast listing these names.
    """
    discovered_set = set(discovered)
    return sorted(str(n) for n in required if str(n) not in discovered_set)
