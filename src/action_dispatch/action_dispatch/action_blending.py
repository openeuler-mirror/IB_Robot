"""Action selection result; storage and hold-last selection belong to ActivePlan."""

from dataclasses import dataclass
from typing import Any

from robot_config.dispatch_strategies import SUPPORTED_BLENDING_STRATEGIES as SUPPORTED_BLENDING


@dataclass(frozen=True, slots=True)
class BlendedAction:
    """One selected action, with queue/smoother/hold/empty diagnostic source."""

    action: Any | None
    source: str


__all__ = ["BlendedAction", "SUPPORTED_BLENDING"]
