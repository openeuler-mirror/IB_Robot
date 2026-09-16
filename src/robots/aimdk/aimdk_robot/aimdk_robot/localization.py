"""Relocalization against a stored vendor map.

The vendor triggers relocalization with a string command on a shared control
topic (``start_relocalization:<map_id>``, SLAM section of the AimDK docs). Map
identifiers come from the profile, so an unknown map name is refused rather
than sent to the platform as an arbitrary id.

Completion is deliberately not faked: unless the deployment configures an
observable relocalization signal, a caller asking to wait is told that this
platform does not report completion, instead of receiving an invented success.
"""

from __future__ import annotations

import time
from typing import Any

from std_msgs.msg import String

from aimdk_robot import projection
from aimdk_robot.projection import CommandRejected

RELOCALIZATION_COMMAND = "start_relocalization"


def _command_publisher(node: Any) -> Any:
    """Create the shared vendor command publisher once, on first use."""
    publisher = getattr(node, "_integrated_command_pub", None)
    if publisher is None:
        topic = str((node.vendor_config.get("slam") or {}).get("command_topic", "/integrated_command"))
        publisher = node.create_publisher(String, topic, 10)
        node._integrated_command_pub = publisher  # noqa: SLF001 - lazily attached to its owner
    return publisher


def relocalize(node: Any, request: Any, response: Any) -> Any:
    """Serve ``ibrobot_msgs/srv/StartRelocalization`` on the X2."""
    slam = node.vendor_config.get("slam") or {}
    maps = {str(name): str(map_id) for name, map_id in (slam.get("maps") or {}).items()}

    if node.runtime_state.stop_latched:
        response.success, response.error_code = False, "STOP_LATCHED"
        response.message = "stop latched"
        return response
    try:
        map_id = projection.resolve_named(str(request.map_name), maps, reason="UNKNOWN_MAP")
    except CommandRejected as rejected:
        response.success, response.error_code, response.message = False, rejected.reason, rejected.detail
        return response

    _command_publisher(node).publish(String(data=f"{RELOCALIZATION_COMMAND}:{map_id}"))

    if not bool(request.wait_for_completion):
        response.success, response.error_code = True, ""
        response.message = f"relocalization requested for map {request.map_name!r} (id {map_id})"
        return response

    observed = getattr(node, "localization_pose_seen_at", None)
    if observed is None:
        # The runtime has no configured completion signal for this platform.
        response.success, response.error_code = False, "LOCALIZATION_FAILED"
        response.message = (
            "relocalization was requested, but this platform exposes no completion signal; "
            "call with wait_for_completion=false and observe /localization/pose instead"
        )
        return response

    deadline = time.time() + max(float(request.timeout_sec), 0.0)
    started = time.time()
    while time.time() < deadline:
        if float(node.localization_pose_seen_at) > started:
            response.success, response.error_code = True, ""
            response.message = f"relocalized in map {request.map_name!r}"
            return response
        time.sleep(0.1)
    response.success, response.error_code = False, "TIMEOUT"
    response.message = f"no localization pose observed within {request.timeout_sec}s"
    return response
