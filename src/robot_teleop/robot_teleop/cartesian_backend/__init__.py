"""Cartesian teleoperation backend abstraction.

Backends share a common ``servo(linear, angular)`` interface so device-side
code (``xbox_controller.py``, ``phone_device.py``) is solver-agnostic:

- :class:`PlacoServoBackend` — publishes base-frame linear/angular Vector3Stamped
  to the ``so101_placo_servo_node`` (in-process Placo QP differential IK). For
  SO101 only.
- :class:`MoveItServoBackend` — wraps MoveIt Servo for generic Cartesian
  velocity control.

``placo_servo`` and ``moveit_servo`` honour the project convention::

    linear  ∈ base
    angular ∈ tool_frame

and converts ``angular`` into the base frame via :class:`ToolAngularAdapter`.

Factory entry point: :func:`make_cartesian_backend`.
"""

from .base import CartesianBackend  # noqa: F401
from .frame_adapter import ToolAngularAdapter  # noqa: F401
from .moveit_servo import MoveItServoBackend  # noqa: F401
from .placo_servo import PlacoServoBackend  # noqa: F401


def make_cartesian_backend(solver: str, **kwargs) -> CartesianBackend:
    """Construct a Cartesian backend by name.

    Args:
        solver: ``'placo_servo'`` or ``'moveit_servo'``.
        **kwargs: Forwarded to the concrete backend constructor. Common
            arguments: ``node``, ``tf_buffer``, ``base_link``, ``tool_frame``,
            ``linear_speed``, ``angular_speed``.

    Raises:
        ValueError: If ``solver`` is not a known solver name.
    """
    if solver == "moveit_servo":
        return MoveItServoBackend(**kwargs)
    if solver in ("placo_servo", "runtime"):
        return PlacoServoBackend(**kwargs)
    raise ValueError(f"unknown cartesian solver: {solver!r}")
