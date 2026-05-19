"""Resolve ROS 2 message types from string identifiers.

Only the small whitelist the mock supports is exposed here. Anything outside
this list must trigger a hard startup failure so it cannot be silently skipped.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Whitelist of fully-qualified message types the mock node knows how to handle.
# Adding new entries here is a deliberate, reviewed change.
SUPPORTED_PUBLISH_TYPES = (
    "sensor_msgs/msg/Image",
    "sensor_msgs/msg/JointState",
)

SUPPORTED_SUBSCRIBE_TYPES = ("std_msgs/msg/Float64MultiArray",)


class UnsupportedMessageTypeError(RuntimeError):
    """Raised when a contract references a message type the mock does not support."""


def resolve_msg_class(type_str: str) -> Any:
    """Resolve a 'pkg/msg/Name' string to the Python message class.

    Args:
        type_str: ROS 2 type identifier such as ``sensor_msgs/msg/Image``.

    Returns:
        The imported message class object.

    Raises:
        UnsupportedMessageTypeError: If the identifier is malformed.
    """
    if not type_str or type_str.count("/") != 2:
        raise UnsupportedMessageTypeError(f"Malformed message type '{type_str}'. Expected 'pkg/msg/Name'.")
    pkg, sub, name = type_str.split("/")
    try:
        module = import_module(f"{pkg}.{sub}")
    except ImportError as exc:  # pragma: no cover - import errors are environment-specific
        raise UnsupportedMessageTypeError(f"Cannot import message module for '{type_str}': {exc}") from exc
    try:
        return getattr(module, name)
    except AttributeError as exc:  # pragma: no cover
        raise UnsupportedMessageTypeError(f"Message class '{name}' not found in '{pkg}.{sub}'.") from exc


def ensure_publish_supported(type_str: str) -> None:
    if type_str not in SUPPORTED_PUBLISH_TYPES:
        raise UnsupportedMessageTypeError(
            f"Observation type '{type_str}' is not supported by hardware_mock. "
            f"Supported: {sorted(SUPPORTED_PUBLISH_TYPES)}"
        )


def ensure_subscribe_supported(type_str: str) -> None:
    if type_str not in SUPPORTED_SUBSCRIBE_TYPES:
        raise UnsupportedMessageTypeError(
            f"Action type '{type_str}' is not supported by hardware_mock. "
            f"Supported: {sorted(SUPPORTED_SUBSCRIBE_TYPES)}"
        )
