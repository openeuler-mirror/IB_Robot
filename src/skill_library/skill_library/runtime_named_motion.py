"""Delegated skills that run a named motion the robot runtime owns.

The runtime serves ``ibrobot_msgs/action/ExecuteNamedMotion`` and decides how a
name maps onto the platform (a vendor preset gesture, a posture action, ...).
The skill catalog binds one motion name per skill; this module holds the pure
rules the executor applies around that call, so they can be tested without a
ROS graph:

* the bound name must be one the runtime currently advertises for a capability
  the skill requires (``RuntimeStatus.capabilities_json``), so a catalog that
  drifted from the runtime fails with a stable code instead of an unexplained
  goal rejection;
* every runtime result code maps to one explicit public error code.

Named motions cannot be cancelled: the runtime refuses cancel requests because
the platform offers no way to stop a preset motion and stopping a posture
change half-way is a fall. The executor therefore never forwards a cancel; the
operator path to interrupt motion is the runtime stop service.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from ibrobot_msgs.action import ExecuteNamedMotion

EXECUTOR_NAME = "runtime_named_motion"

# Runtime capabilities that advertise named motions, and the parameter listing them.
NAMED_MOTION_CAPABILITY_LISTS: Mapping[str, str] = {
    "motion.named": "names",
    "motion.posture": "postures",
}

INVALID_BINDING = "NAMED_MOTION_INVALID_BINDING"
UNKNOWN = "NAMED_MOTION_UNKNOWN"
BUSY = "NAMED_MOTION_BUSY"
CONTROL_MODE_MISMATCH = "CONTROL_MODE_MISMATCH"
STOP_LATCHED = "NAMED_MOTION_STOP_LATCHED"
PLATFORM_NOT_READY = "NAMED_MOTION_PLATFORM_NOT_READY"
REJECTED = "NAMED_MOTION_REJECTED"
TIMEOUT = "NAMED_MOTION_TIMEOUT"
PREEMPTED = "NAMED_MOTION_PREEMPTED"
FAILED = "NAMED_MOTION_FAILED"
RUNTIME_UNAVAILABLE = "NAMED_MOTION_RUNTIME_UNAVAILABLE"

_RESULT = ExecuteNamedMotion.Result
RUNTIME_ERROR_CODES: Mapping[int, str] = {
    _RESULT.UNKNOWN_MOTION: UNKNOWN,
    _RESULT.INVALID_TARGET: UNKNOWN,
    _RESULT.BUSY: BUSY,
    _RESULT.MODE_NOT_ALLOWED: CONTROL_MODE_MISMATCH,
    _RESULT.STOP_LATCHED: STOP_LATCHED,
    # Includes "the platform is not in the action this motion requires" (for
    # example not standing); the runtime message names the reason.
    _RESULT.RUNTIME_UNAVAILABLE: PLATFORM_NOT_READY,
    _RESULT.REJECTED_BY_PLATFORM: REJECTED,
    _RESULT.UNSAFE_POSTURE: REJECTED,
    _RESULT.TIMEOUT: TIMEOUT,
    # A later goal or the runtime itself superseded this motion.
    _RESULT.CANCELLED: PREEMPTED,
    # The runtime failed internally; nothing more specific can be said.
    _RESULT.INTERNAL_ERROR: FAILED,
}


def bound_motion_name(template: Any) -> str:
    """Return the motion the catalog implementation binds, or "" if malformed."""
    binding = template.get("binding") if isinstance(template, Mapping) else None
    name = binding.get("motion") if isinstance(binding, Mapping) else None
    return name.strip() if isinstance(name, str) else ""


def advertised_motion_names(
    capabilities_json: str, required_capabilities: Iterable[str]
) -> tuple[frozenset[str] | None, str, str]:
    """Names the runtime advertises for the named-motion capabilities a skill requires.

    Returns ``(names, error_code, reason)``; ``names`` is None when the check
    cannot be made, with the public error code and a reason.
    """
    capabilities = [name for name in required_capabilities if name in NAMED_MOTION_CAPABILITY_LISTS]
    if not capabilities:
        return None, INVALID_BINDING, "skill requires no named-motion runtime capability"
    try:
        advertised = json.loads(capabilities_json or "{}")
    except (TypeError, ValueError):
        return None, PLATFORM_NOT_READY, "runtime capabilities_json is not valid JSON"
    if not isinstance(advertised, Mapping):
        return None, PLATFORM_NOT_READY, "runtime capabilities_json is not an object"
    names: set[str] = set()
    for capability in capabilities:
        parameters = advertised.get(capability)
        listed = parameters.get(NAMED_MOTION_CAPABILITY_LISTS[capability]) if isinstance(parameters, Mapping) else None
        if not isinstance(listed, list):
            return None, PLATFORM_NOT_READY, f"runtime does not advertise {capability} names"
        names.update(str(item) for item in listed)
    return frozenset(names), "", ""


def public_error_code(runtime_code: int) -> str:
    """Map an ExecuteNamedMotion result code onto the public skill error code."""
    return RUNTIME_ERROR_CODES.get(int(runtime_code), FAILED)
