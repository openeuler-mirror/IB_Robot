"""Policy ownership of the public runtime mode, shared by both dispatchers."""

import json
import threading
import time

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType

from ibrobot_msgs.msg import RuntimeStatus
from ibrobot_msgs.srv import GetRuntimeStatus, SetRuntimeMode
from robot_config.contract_utils import qos_profile_from_dict
from robot_config.interface_binding import InterfaceBindingError
from robot_config.loader import load_robot_section


class PolicyAdmission:
    """Acquire only from observed idle; a revoked producer never resumes itself.

    Callbacks and publication use the owner's lock. Service waits must happen
    outside that lock so status notifications can revoke an acquisition.
    """

    def __init__(self, node, config_path, runtime, lock, on_revoke):
        self.lock = lock
        self.on_revoke = on_revoke
        self.owned = False
        self.pending = False
        self._releasing = False
        self.status = None
        self.generation = 0
        self._completion = None
        self._timeout_timer = None
        self._rpc_timeout_s = 2.0
        self._rpc_deadline = 0.0
        self._pending_rpc = None
        self._status_max_age_s = 2.0
        self._last_status_received = None
        self._clock = node.get_clock()
        _, config = load_robot_section(config_path)
        self.mode = (config.get("control_modes", {}).get("model_inference") or {}).get("runtime_mode")
        if not isinstance(self.mode, str) or not self.mode or self.mode == "idle":
            raise InterfaceBindingError(
                "mode_required", "control_modes.model_inference.runtime_mode", "explicit non-idle mode required"
            )
        description = runtime["interface_description"]
        identity = description.get("robot") or {}
        self._identity = dict(identity)
        self.runtime_name = identity.get("runtime_name")
        self.runtime_version = identity.get("runtime_version")
        if any(
            not isinstance(identity.get(key), str) or not identity[key]
            for key in ("id", "type", "runtime_name", "runtime_version")
        ):
            raise InterfaceBindingError(
                "identity_required", "runtime.interface_description.robot", "complete runtime identity is required"
            )
        interfaces = description["interfaces"]
        selected = {}
        for key, kind, direction, message_type in (
            ("runtime.status", "topic", "publish", "ibrobot_msgs/msg/RuntimeStatus"),
            ("runtime.get_status", "service", "serve", "ibrobot_msgs/srv/GetRuntimeStatus"),
            ("runtime.set_mode", "service", "serve", "ibrobot_msgs/srv/SetRuntimeMode"),
        ):
            item = interfaces.get(key, {})
            if (item.get("kind"), item.get("direction"), item.get("message_type")) != (kind, direction, message_type):
                raise InterfaceBindingError("invalid_interface", key, f"requires {kind} {message_type} {direction}")
            selected[key] = item
        group = ReentrantCallbackGroup()
        self.get_client = node.create_client(
            GetRuntimeStatus, selected["runtime.get_status"]["endpoint"], callback_group=group
        )
        self.set_client = node.create_client(
            SetRuntimeMode, selected["runtime.set_mode"]["endpoint"], callback_group=group
        )
        status = selected["runtime.status"]
        self.subscription = node.create_subscription(
            RuntimeStatus, status["endpoint"], self.observe, qos_profile_from_dict(status["qos"]), callback_group=group
        )
        self._watchdog = node.create_timer(
            0.1, self._check_freshness, callback_group=group, clock=Clock(clock_type=ClockType.STEADY_TIME)
        )

    @staticmethod
    def healthy(status):
        return status is not None and status.lifecycle == "ACTIVE" and not status.stop_latched and not status.faults

    def _status_matches(self, status):
        if status is None:
            return False
        try:
            description = json.loads(status.interface_description_json)
        except (TypeError, ValueError):
            return False
        if not isinstance(description, dict) or description.get("robot") != self._identity:
            return False
        stamp_ns = status.stamp.sec * 1_000_000_000 + status.stamp.nanosec
        age_ns = self._clock.now().nanoseconds - stamp_ns
        return (
            self.healthy(status)
            and status.runtime_name == self.runtime_name
            and status.runtime_version == self.runtime_version
            and 0 <= status.stamp.nanosec < 1_000_000_000
            and 0 <= age_ns <= self._status_max_age_s * 1_000_000_000
        )

    def _check_freshness(self):
        with self.lock:
            if self.owned and (
                self._last_status_received is None
                or time.monotonic() - self._last_status_received > self._status_max_age_s
            ):
                self._finish(False, "runtime status heartbeat expired; explicit start required")

    def _timeout(self, generation):
        with self.lock:
            if self.pending and self.generation == generation:
                self._finish(False, "runtime admission RPC timed out")

    def observe(self, status):
        with self.lock:
            previous = self.status
            self.status = status
            valid = self._status_matches(status)
            if valid:
                self._last_status_received = time.monotonic()
            if self.owned and (not valid or status.active_mode != self.mode):
                self.owned = False
                self._finish(False, "runtime admission was revoked")
                self.on_revoke()
            elif self.pending and (
                not valid
                or status.active_mode not in ("idle", self.mode)
                or (
                    not self._releasing
                    and status.active_mode == "idle"
                    and previous is not None
                    and previous.active_mode == self.mode
                )
            ):
                self._finish(False, "runtime changed during admission")
                self.on_revoke()

    def _finish(self, success, message):
        callback = self._completion
        self._completion = None
        self.pending = False
        if self._timeout_timer is not None:
            self._timeout_timer.cancel()
            self._timeout_timer = None
        if self._pending_rpc is not None:
            client, future = self._pending_rpc
            self._pending_rpc = None
            client.remove_pending_request(future)
        self.generation += 1
        if not success and self.owned:
            self.owned = False
            self.on_revoke()
        if callback is not None:
            callback(success, message)

    def acquire(self, callback, *, own_active_session=False):
        with self.lock:
            if self.pending:
                callback(False, "runtime transition in progress")
                return
            self.pending = True
            self._releasing = False
            self.status = None
            self._completion = callback
            generation = self.generation
            own_session = self.owned and own_active_session
            self._rpc_deadline = time.monotonic() + self._rpc_timeout_s
            self._timeout_timer = threading.Timer(self._rpc_timeout_s, self._timeout, args=(generation,))
            self._timeout_timer.daemon = True
            self._timeout_timer.start()

        def checked(response):
            status = response.status
            self.observe(status)
            if generation != self.generation:
                return
            if not self._status_matches(status):
                self._finish(False, "runtime is not ACTIVE or stop is latched")
            elif own_session and self.owned and status.active_mode == self.mode:
                self._finish(True, "policy already owns runtime mode")
            elif status.active_mode != "idle":
                self._finish(False, f"runtime mode {status.active_mode!r} is owned by another producer")
            else:
                self._call(self.set_client, SetRuntimeMode.Request(mode=self.mode), switched, generation)

        def switched(response):
            if not response.success:
                self._finish(False, response.message)
                return
            # Own the transition before verifying it, so an idle/stop notification
            # racing the final status read invalidates this attempt.
            self.owned = True
            self._call(self.get_client, GetRuntimeStatus.Request(), verified, generation)

        def verified(response):
            self.observe(response.status)
            if generation != self.generation:
                return
            self._finish(True, "policy acquired runtime mode")

        self._call(self.get_client, GetRuntimeStatus.Request(), checked, generation)

    def _call(self, client, request, callback, generation):
        def done(future):
            with self.lock:
                if generation != self.generation:
                    return
                self._pending_rpc = None
                if time.monotonic() >= self._rpc_deadline:
                    self._finish(False, "runtime admission RPC timed out")
                    return
                try:
                    response = future.result()
                    if response is None:
                        raise RuntimeError("empty runtime response")
                    callback(response)
                except Exception as exc:
                    self._finish(False, f"runtime request failed: {exc}")

        with self.lock:
            if generation != self.generation:
                return
            try:
                if not client.service_is_ready():
                    raise RuntimeError("runtime service unavailable")
                future = client.call_async(request)
                self._pending_rpc = (client, future)
                future.add_done_callback(done)
            except Exception as exc:
                self._finish(False, f"runtime request failed: {exc}")

    def cancel(self):
        with self.lock:
            if self.pending:
                self._finish(False, "policy admission canceled; explicit start required")
            else:
                self.generation += 1

    def release(self, timeout=2.0, *, spin_once=None):
        """Return an owned mode to idle before another producer can acquire it."""
        event = threading.Event()
        result = [False]

        def finished(success, _message):
            result[0] = success
            event.set()

        def checked(response):
            current = response.status
            if not self._status_matches(current) or current.active_mode != self.mode:
                self._finish(True, "runtime ownership already revoked")
                return
            self._call(
                self.set_client,
                SetRuntimeMode.Request(mode="idle"),
                lambda response: self._finish(bool(response.success), response.message),
                generation,
            )

        with self.lock:
            self.cancel()
            if not self.owned:
                return True
            self.owned = False
            self.pending = True
            self._releasing = True
            self._completion = finished
            generation = self.generation
            self._rpc_deadline = time.monotonic() + timeout
            self._call(self.get_client, GetRuntimeStatus.Request(), checked, generation)
        deadline = time.monotonic() + timeout
        while not event.is_set() and time.monotonic() < deadline:
            if spin_once is None:
                event.wait(min(0.05, max(0, deadline - time.monotonic())))
            else:
                spin_once(timeout_sec=0.05)
        if not event.is_set():
            self.cancel()
        return result[0]
