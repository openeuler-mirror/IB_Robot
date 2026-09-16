"""Runtime facade: serves the runtime contract over a ros2_control execution stack.

Non-real-time. It never carries joint commands itself: streaming and
trajectory execution go straight to the controllers. The facade only

- maps ``SetRuntimeMode`` onto ``controller_manager/switch_controller``
  (a mode is a named controller activation set from the runtime profile),
- publishes ``RuntimeStatus`` (1 Hz + on change) and answers ``GetRuntimeStatus``,
- implements ``StopRuntime`` with the ordered guarantees of the contract
  (cancel trajectory goals -> idle activation set -> optional hardware
  component deactivation for TORQUE_OFF), measuring each step's latency,
- counts streaming commands observed on a channel whose mode is not active.

Design D2/D3 of change robot-runtime-abstraction.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from typing import Any

import rclpy
from action_msgs.srv import CancelGoal
from builtin_interfaces.msg import Duration as DurationMsg
from controller_manager_msgs.srv import ListControllers, SetHardwareComponentState, SwitchController
from geometry_msgs.msg import Twist
from lifecycle_msgs.msg import State as LifecycleState
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray

from ibrobot_msgs.msg import RuntimeStatus
from ibrobot_msgs.srv import GetRuntimeStatus, SetRuntimeMode, StopRuntime
from robot_runtime.contract import (
    GET_STATUS_SERVICE,
    IDLE_MODE,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_CONNECTING,
    LIFECYCLE_DEGRADED,
    LIFECYCLE_FAULTED,
    NAVIGATION_ACK_TOPIC,
    SET_MODE_SERVICE,
    STATUS_TOPIC,
    STOP_POLICIES,
    STOP_SERVICE,
    STOP_TORQUE_OFF,
)
from robot_runtime.interface_description import build_description, validate_description
from robot_runtime.interface_monitor import InterfaceMonitor
from robot_runtime.modes import ModeModel, ModeModelConfig
from robot_runtime.peripherals import load_peripherals_file, merge_peripherals, perception_capabilities
from robot_runtime.profile import load_profile, stop_bounds
from robot_runtime.state import RuntimeState

_SERVICE_TIMEOUT_S = 5.0
_SWITCH_TIMEOUT_S = 3.0


class RuntimeFacade(Node):
    def __init__(self, profile_path: str | None = None):
        super().__init__("runtime_facade")
        self.declare_parameter("profile", profile_path or "")
        self.declare_parameter("peripherals", "")
        path = str(self.get_parameter("profile").value)
        if not path:
            raise ValueError("runtime_facade requires the 'profile' parameter (path to the runtime profile YAML)")
        self._profile: dict[str, Any] = load_profile(path)
        self.declare_parameter("simulated", bool(self._profile.get("simulated", False)))
        self.declare_parameter("instance_id", str(self._profile["runtime"].get("instance_id", "")))
        self.declare_parameter("initial_mode", str(self._profile["modes"].get("initial", IDLE_MODE)))
        self._profile["modes"]["initial"] = str(self.get_parameter("initial_mode").value)
        if self.get_parameter("instance_id").value:
            self._profile["runtime"]["instance_id"] = str(self.get_parameter("instance_id").value)
        peripherals_path = str(self.get_parameter("peripherals").value or "").strip()
        fragment = load_peripherals_file(peripherals_path) if peripherals_path else {}
        peripherals = merge_peripherals(self._profile.get("peripherals"), fragment.get("peripherals"))
        capabilities = dict(self._profile["capabilities"])
        capabilities.update(perception_capabilities(peripherals))
        capabilities.setdefault("runtime.status", {})
        self._profile["fast_lio"] = {**(self._profile.get("fast_lio") or {}), **(fragment.get("fast_lio") or {})}
        self.declare_parameter("interface_description_json", "")
        encoded_description = str(self.get_parameter("interface_description_json").value)
        description = (
            json.loads(encoded_description)
            if encoded_description
            else build_description(self._profile, peripherals, simulated=bool(self.get_parameter("simulated").value))
        )
        validate_description(description)
        identity = description["robot"]
        if (
            identity["runtime_name"] != self._profile["runtime"]["name"]
            or identity["runtime_version"] != str(self._profile["runtime"]["version"])
            or description["execution"] != ("simulated" if self.get_parameter("simulated").value else "physical")
        ):
            raise ValueError("effective public description does not match the runtime profile")
        self._interface_monitor = InterfaceMonitor(self, description)
        self._modes = ModeModel(ModeModelConfig.from_profile(self._profile))
        self._state = RuntimeState(
            name=str(self._profile["runtime"]["name"]),
            version=str(self._profile["runtime"]["version"]),
            capabilities=capabilities,
            modes=self._modes,
            on_change=self._publish_status,
            interface_description=description,
            interface_states=self._interface_monitor.states,
        )
        self._bounds = stop_bounds(self._profile)
        self._cm = str(self._profile["controller_manager"])
        self._stop_lock = threading.Lock()

        cb = ReentrantCallbackGroup()
        self._status_pub = self.create_publisher(RuntimeStatus, STATUS_TOPIC, 10)
        self.create_service(SetRuntimeMode, SET_MODE_SERVICE, self._on_set_mode, callback_group=cb)
        self.create_service(GetRuntimeStatus, GET_STATUS_SERVICE, self._on_get_status, callback_group=cb)
        self.create_service(StopRuntime, STOP_SERVICE, self._on_stop, callback_group=cb)

        client_cb = ReentrantCallbackGroup()
        self._switch = self.create_client(SwitchController, f"{self._cm}/switch_controller", callback_group=client_cb)
        self._list = self.create_client(ListControllers, f"{self._cm}/list_controllers", callback_group=client_cb)
        self._hw_state = self.create_client(
            SetHardwareComponentState, f"{self._cm}/set_hardware_component_state", callback_group=client_cb
        )
        self._cancel_clients = [
            self.create_client(CancelGoal, f"{name}/_action/cancel_goal", callback_group=client_cb)
            for name in self._profile["trajectory_actions"]
        ]

        self._navigation_enabled = False
        needs_navigation_ack = False
        for entry in self._profile["command_channels"]:
            channel = str(entry["channel"])
            allowed = {str(m) for m in entry.get("modes", [])}
            gated = bool(entry.get("navigation_gated", False))
            needs_navigation_ack = needs_navigation_ack or gated
            msg_type = Twist if str(entry.get("type", "float64_array")).lower() == "twist" else Float64MultiArray
            self.create_subscription(
                msg_type,
                str(entry["topic"]),
                lambda _msg, ch=channel, allowed=allowed, gated=gated: self._on_command_observed(ch, allowed, gated),
                10,
                callback_group=cb,
            )
        if needs_navigation_ack:
            self.create_subscription(Bool, NAVIGATION_ACK_TOPIC, self._on_navigation_ack, 10, callback_group=cb)

        timer_cb = MutuallyExclusiveCallbackGroup()
        self.create_timer(1.0, self._publish_status, callback_group=timer_cb)
        self.create_timer(1.0, self._check_readiness, callback_group=timer_cb)
        self.get_logger().info(f"runtime facade for {self._state.name} {self._state.version}: profile {path}")

    # --- helpers ---------------------------------------------------------------------

    def _call(self, client, request, timeout_s: float = _SERVICE_TIMEOUT_S):
        """Synchronous service call usable from inside a reentrant callback."""
        if not client.wait_for_service(timeout_sec=timeout_s):
            raise TimeoutError(f"service {client.srv_name} unavailable")
        done = threading.Event()
        future = client.call_async(request)
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout_s):
            raise TimeoutError(f"service {client.srv_name} timed out after {timeout_s}s")
        if future.exception() is not None:
            raise RuntimeError(f"service {client.srv_name} failed: {future.exception()}")
        return future.result()

    def _publish_status(self) -> None:
        self._status_pub.publish(self._state.to_msg(self.get_clock().now().to_msg()))

    def _active_controllers(self) -> set[str]:
        response = self._call(self._list, ListControllers.Request())
        return {c.name for c in response.controller if c.state == "active"}

    def _check_readiness(self) -> None:
        if self._state.stop_latched:
            return
        try:
            active = self._active_controllers()
        except (TimeoutError, RuntimeError) as exc:
            if self._state.lifecycle != LIFECYCLE_CONNECTING:
                self._state.add_fault(f"controller_manager unreachable: {exc}")
                self._state.set_lifecycle(LIFECYCLE_DEGRADED)
            return
        self._state.set_active_controllers(sorted(active))
        expected = set(self._modes.spec().controllers)
        missing = sorted(expected - active)
        if missing:
            self._state.add_fault(f"mode {self._modes.mode!r} controllers not active: {missing}")
            self._state.set_lifecycle(LIFECYCLE_DEGRADED)
            return
        if self._state.lifecycle in (LIFECYCLE_CONNECTING, LIFECYCLE_DEGRADED):
            self._state.clear_faults()
            self._state.set_lifecycle(LIFECYCLE_ACTIVE)

    def _switch_to(self, target: str) -> None:
        """Activate ``target``'s controller set and deactivate the rest (STRICT)."""
        current = set(self._modes.spec().controllers)
        wanted = set(self._modes.spec(target).controllers)
        request = SwitchController.Request()
        request.activate_controllers = sorted(wanted - current)
        request.deactivate_controllers = sorted(current - wanted)
        request.strictness = SwitchController.Request.STRICT
        request.activate_asap = True
        request.timeout = DurationMsg(sec=int(_SWITCH_TIMEOUT_S), nanosec=0)
        response = self._call(self._switch, request, timeout_s=_SWITCH_TIMEOUT_S + _SERVICE_TIMEOUT_S)
        if not response.ok:
            raise RuntimeError(f"switch_controller rejected {self._modes.mode!r} -> {target!r}")
        self._modes.commit(target)
        self._state.set_active_controllers(sorted(wanted))

    def _set_hardware_state(self, label: str, state_id: int) -> None:
        for component in self._profile["hardware_components"]:
            request = SetHardwareComponentState.Request()
            request.name = str(component)
            request.target_state = LifecycleState(id=state_id, label=label)
            response = self._call(self._hw_state, request)
            if not response.ok:
                raise RuntimeError(f"hardware component {component!r} refused state {label!r}")

    def _on_navigation_ack(self, msg: Bool) -> None:
        self._navigation_enabled = bool(msg.data)

    def _on_command_observed(self, channel: str, allowed_modes: set[str], navigation_gated: bool = False) -> None:
        active = self._modes.mode
        permitted = active in allowed_modes if allowed_modes else self._modes.spec().allows_stream
        if navigation_gated and not self._navigation_enabled:
            permitted = False
        if self._state.stop_latched or not permitted:
            self._modes.note_rejected(channel)

    # --- services ---------------------------------------------------------------------

    def _on_get_status(self, _request, response):
        response.status = self._state.to_msg(self.get_clock().now().to_msg())
        return response

    def _on_set_mode(self, request, response):
        target = str(request.mode)
        epoch = self._state.stop_epoch
        if self._state.stop_latched and target != IDLE_MODE:
            response.success = False
            response.message = f"stop latched ({self._state.stop_policy}); request {IDLE_MODE!r} to clear it"
            response.valid_transitions = [IDLE_MODE]
            return response
        decision = self._modes.can_switch(target)
        if not decision.allowed:
            response.success = False
            response.message = decision.reason
            response.valid_transitions = sorted(self._modes.valid_transitions())
            return response
        try:
            if self._state.stop_latched and self._state.stop_policy == STOP_TORQUE_OFF:
                self._set_hardware_state("active", LifecycleState.PRIMARY_STATE_ACTIVE)
            if target != self._modes.mode:
                self._switch_to(target)
            if not self._state.clear_stop_if_unchanged(epoch):
                # _switch_to already committed; undo the late activation without clearing the latch.
                if self._modes.mode != IDLE_MODE:
                    self._switch_to(IDLE_MODE)
                self._state.add_fault(f"mode {target!r} aborted: stop engaged during the controller switch")
                response.success = False
                response.message = (
                    f"stop engaged while switching to {target!r}; runtime returned to {IDLE_MODE!r}, "
                    f"request {IDLE_MODE!r} to clear the latch"
                )
                response.valid_transitions = [IDLE_MODE]
                self._publish_status()
                return response
        except (TimeoutError, RuntimeError) as exc:
            self._state.add_fault(str(exc))
            self._state.set_lifecycle(LIFECYCLE_FAULTED)
            response.success = False
            response.message = str(exc)
            response.valid_transitions = sorted(self._modes.valid_transitions())
            return response
        response.success = True
        response.message = f"mode {target!r} active"
        response.valid_transitions = sorted(self._modes.valid_transitions())
        self._publish_status()
        return response

    def _on_stop(self, request, response):
        policy = str(request.policy or self._profile["stop_default_policy"])
        if policy not in STOP_POLICIES:
            response.success = False
            response.message = f"unknown stop policy {policy!r}; expected one of {STOP_POLICIES}"
            response.cancel_latency_s = response.idle_latency_s = response.torque_off_latency_s = -1.0
            return response
        with self._stop_lock:
            t0 = time.monotonic()
            response.cancel_latency_s = response.idle_latency_s = response.torque_off_latency_s = -1.0
            self._state.engage_stop(policy)
            try:
                # (1) cancel every in-flight trajectory goal (zeroed goal_info = cancel all)
                cancel_failures = []
                current = set(self._modes.spec().controllers)
                for action, client in zip(self._profile["trajectory_actions"], self._cancel_clients, strict=True):
                    controller = str(action).rstrip("/").rsplit("/", 1)[0].rsplit("/", 1)[-1]
                    if controller not in current:
                        continue
                    if not client.wait_for_service(timeout_sec=0.2):
                        cancel_failures.append(f"{client.srv_name}: unreachable")
                        continue
                    try:
                        cancelled = self._call(client, CancelGoal.Request(), timeout_s=_SERVICE_TIMEOUT_S)
                    except (TimeoutError, RuntimeError) as exc:
                        cancel_failures.append(f"{client.srv_name}: {exc}")
                        continue
                    if cancelled.return_code not in (
                        CancelGoal.Response.ERROR_NONE,
                        CancelGoal.Response.ERROR_UNKNOWN_GOAL_ID,
                    ):
                        cancel_failures.append(f"{client.srv_name}: return_code={cancelled.return_code}")
                response.cancel_latency_s = time.monotonic() - t0
                if cancel_failures:
                    raise RuntimeError(f"could not cancel trajectories: {'; '.join(cancel_failures)}")
                # (2) idle activation set: no command controller active, joints hold
                if self._modes.mode != IDLE_MODE:
                    self._switch_to(IDLE_MODE)
                response.idle_latency_s = time.monotonic() - t0
                # (3) TORQUE_OFF: release torque through the hardware lifecycle
                if policy == STOP_TORQUE_OFF:
                    self._set_hardware_state("inactive", LifecycleState.PRIMARY_STATE_INACTIVE)
                    response.torque_off_latency_s = time.monotonic() - t0
                active = self._active_controllers()
                self._state.set_active_controllers(sorted(active))
                command_controllers = {
                    controller
                    for mode in self._modes.declared_modes()
                    for controller in self._modes.spec(mode).controllers
                } - set(self._modes.spec(IDLE_MODE).controllers)
                remaining = sorted(active & command_controllers)
                if remaining:
                    raise RuntimeError(f"command controllers still active: {remaining}")
            except (TimeoutError, RuntimeError) as exc:
                self._state.add_fault(f"stop ({policy}) incomplete: {exc}")
                self._state.set_lifecycle(LIFECYCLE_FAULTED)
                response.success = False
                response.message = f"stop ({policy}) incomplete: {exc}"
                return response
            self._note_bound_violations(response, policy)
        response.success = True
        response.message = f"stop ({policy}) engaged; latched until mode {IDLE_MODE!r} is requested"
        self._publish_status()
        return response

    def _note_bound_violations(self, response, policy: str) -> None:
        checks = [("cancel_bound_s", response.cancel_latency_s), ("idle_bound_s", response.idle_latency_s)]
        if policy == STOP_TORQUE_OFF:
            checks.append(("torque_off_bound_s", response.torque_off_latency_s))
        for key, measured in checks:
            bound = self._bounds.get(key)
            if bound is not None and measured > bound:
                self._state.add_fault(f"stop {key.removesuffix('_bound_s')} took {measured:.3f}s > bound {bound:.3f}s")


def main(args=None):
    rclpy.init(args=args)
    try:
        node = RuntimeFacade()
    except (ValueError, OSError) as exc:
        print(f"runtime_facade: {exc}", file=sys.stderr)
        rclpy.shutdown()
        return 2
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
