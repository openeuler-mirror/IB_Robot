"""Shared runtime state: lifecycle, faults, stop latch, and RuntimeStatus assembly.

Used by both the ros2_control facade and the mock runtime so status semantics
(publication on change, latch behavior, fault detail) cannot drift between them.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from ibrobot_msgs.msg import RuntimeStatus
from robot_runtime.contract import (
    LIFECYCLE_ACTIVE,
    LIFECYCLE_CONNECTING,
    LIFECYCLE_STOPPED,
    LIFECYCLES,
)
from robot_runtime.interface_description import validate_description
from robot_runtime.modes import ModeModel


class RuntimeState:
    """Thread-safe holder of everything RuntimeStatus reports.

    ``on_change`` is invoked (outside the lock) whenever lifecycle, mode,
    capabilities, latch, or faults change, so the owner can publish
    immediately per the contract.
    """

    def __init__(
        self,
        name: str,
        version: str,
        capabilities: dict[str, dict[str, Any]],
        modes: ModeModel,
        on_change: Callable[[], None] | None = None,
        interface_description: dict | None = None,
        interface_states: Callable[[], dict] | None = None,
    ):
        self._lock = threading.RLock()
        self.name = name
        self.version = version
        self._capabilities: dict[str, dict[str, Any]] = {k: dict(v or {}) for k, v in capabilities.items()}
        self.modes = modes
        self._lifecycle = LIFECYCLE_CONNECTING
        self._faults: list[str] = []
        self._stop_latched = False
        self._stop_policy = ""
        self._stop_epoch = 0
        self._active_controllers: tuple[str, ...] = ()
        self._on_change = on_change
        if interface_description is not None:
            validate_description(interface_description)
        self._interface_description = deepcopy(interface_description)
        self._interface_states = interface_states

    # --- mutation -------------------------------------------------------------

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change()

    @property
    def lifecycle(self) -> str:
        with self._lock:
            return self._lifecycle

    def set_lifecycle(self, lifecycle: str) -> None:
        if lifecycle not in LIFECYCLES:
            raise ValueError(f"unknown lifecycle {lifecycle!r}")
        with self._lock:
            if self._lifecycle == lifecycle:
                return
            self._lifecycle = lifecycle
        self._changed()

    def add_fault(self, detail: str) -> None:
        with self._lock:
            if detail in self._faults:
                return
            self._faults.append(detail)
        self._changed()

    def clear_faults(self) -> None:
        with self._lock:
            if not self._faults:
                return
            self._faults.clear()
        self._changed()

    @property
    def faults(self) -> list[str]:
        with self._lock:
            return list(self._faults)

    @property
    def stop_latched(self) -> bool:
        with self._lock:
            return self._stop_latched

    @property
    def stop_policy(self) -> str:
        with self._lock:
            return self._stop_policy

    @property
    def stop_epoch(self) -> int:
        with self._lock:
            return self._stop_epoch

    def engage_stop(self, policy: str) -> None:
        with self._lock:
            self._stop_latched = True
            self._stop_policy = policy
            self._lifecycle = LIFECYCLE_STOPPED
            self._stop_epoch += 1
        self._changed()

    def clear_stop(self) -> None:
        with self._lock:
            if not self._stop_latched:
                return
            self._stop_latched = False
            self._stop_policy = ""
            self._lifecycle = LIFECYCLE_ACTIVE
        self._changed()

    def clear_stop_if_unchanged(self, epoch: int) -> bool:
        """Clear only the observed stop generation, atomically with the epoch check."""
        with self._lock:
            if self._stop_epoch != epoch:
                return False
            if not self._stop_latched:
                return True
            self._stop_latched = False
            self._stop_policy = ""
            self._lifecycle = LIFECYCLE_ACTIVE
        self._changed()
        return True

    def set_active_controllers(self, controllers: tuple[str, ...] | list[str]) -> None:
        with self._lock:
            new = tuple(controllers)
            if new == self._active_controllers:
                return
            self._active_controllers = new
        self._changed()

    def set_capability_params(self, name: str, params: dict[str, Any]) -> None:
        with self._lock:
            self._capabilities[name] = dict(params)
        self._changed()

    @property
    def capabilities(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self._capabilities.items()}

    def has_capability(self, name: str) -> bool:
        with self._lock:
            return name in self._capabilities

    # --- message assembly ---------------------------------------------------------

    def to_msg(self, stamp) -> RuntimeStatus:
        with self._lock:
            msg = RuntimeStatus()
            msg.runtime_name = self.name
            msg.runtime_version = self.version
            msg.lifecycle = self._lifecycle
            msg.capabilities = sorted(self._capabilities)
            msg.capabilities_json = json.dumps(self._capabilities, sort_keys=True)
            if self._interface_description is not None:
                description = deepcopy(self._interface_description)
                if self._interface_states is not None:
                    description["states"] = self._interface_states()
                if "runtime.status" in description["interfaces"]:
                    description["states"]["runtime.status"] = {
                        "state": "ready",
                        "observed_profile": None,
                        "observed_frame_id": None,
                        "last_seen": time.time(),
                        "detail": "Current RuntimeStatus snapshot",
                    }
                msg.interface_description_json = json.dumps(description, sort_keys=True, allow_nan=False)
            msg.active_mode = self.modes.mode
            msg.declared_modes = self.modes.declared_modes()
            msg.active_controllers = list(self._active_controllers)
            rejections = self.modes.rejections()
            msg.rejected_channels = sorted(rejections)
            msg.rejected_counts = [int(rejections[k]) for k in msg.rejected_channels]
            msg.stop_latched = self._stop_latched
            msg.stop_policy = self._stop_policy
            msg.faults = list(self._faults)
            msg.stamp = stamp
            return msg
