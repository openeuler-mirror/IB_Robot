"""DispatchScheduler subpackage.

completion-aware executor contract introduces an independent scheduler capability that decides when the
dispatcher may request inference and when it may submit the next action. The
scheduler only returns permission/state-transition decisions; it does not own
the action queue, smoother, action values, models or reports.

This subpackage deliberately imports only the Python standard library and
``executors.completion`` so it can be unit-tested without rclpy, NumPy, Torch,
benchmark runtime or sim backend.
"""

from .base import (
    ActionDecision,
    CompletionDecision,
    DispatchScheduler,
    SchedulerSnapshot,
    SchedulerTransition,
)
from .registry import (
    SchedulerAlreadyRegisteredError,
    SchedulerNotFoundError,
    SchedulerRegistryError,
    SchedulerTypeMismatchError,
    create_scheduler,
    get_scheduler_factory,
    register_scheduler,
    registered_scheduler_modes,
)

__all__ = [
    "ActionDecision",
    "CompletionDecision",
    "DispatchScheduler",
    "SchedulerSnapshot",
    "SchedulerTransition",
    "SchedulerRegistryError",
    "SchedulerAlreadyRegisteredError",
    "SchedulerNotFoundError",
    "SchedulerTypeMismatchError",
    "create_scheduler",
    "get_scheduler_factory",
    "register_scheduler",
    "registered_scheduler_modes",
]
