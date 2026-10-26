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
        # Sim-time freshness skew tolerance. Each node's ROS clock advances only
        # as it processes /clock, so a status publisher's stamp can lead a busy
        # subscriber's clock by a few milliseconds (observed up to ~16 ms during
        # a controller switch under MuJoCo, /clock ticks are ~1 ms). The
        # freshness window exists to reject *stale* status, and a stamp slightly
        # ahead of the local clock is not stale, so admit a bounded negative age.
        self._status_skew_tolerance_s = 0.1
        self._last_status_received = None
        # A set_mode request was sent for the current admission attempt and the
        # facade may have committed it; failure paths consult this (before
        # _finish clears it) to decide whether an idle rollback is needed.
        self._switch_requested = False
        self._rolling_back = False
        self._rollback_timer = None
        # Each rollback attempt gets its own token; a timeout or completion
        # bumps it, so callbacks of an expired attempt can no longer request
        # idle. The event/result let release() wait for an in-flight rollback;
        # _idle_confirmed is False whenever the last idle transition (rollback
        # or release) could not be confirmed, so the next release re-verifies.
        # _inflight_switch keeps a failed attempt's unanswered set_mode so the
        # rollback can wait for the runtime's answer before reading the mode.
        self._rollback_token = 0
        self._rollback_done = threading.Event()
        self._rollback_done.set()
        self._idle_confirmed = True
        self._inflight_switch = None
        # A mode request the runtime never answered within the rollback bound:
        # it may still commit, so acquire and release stay fail-closed until it
        # answers (then a fresh check-then-set cleans up a late commit).
        self._unresolved_switch = None
        # Config-driven auto start (control_modes.model_inference.auto_start,
        # default true): when armed, the first observed healthy idle status
        # acquires the mode automatically. Disarms permanently on the first
        # successful admission, on any explicit start/stop, on foreign-mode or
        # latched observations, and on revocation; recovery afterwards always
        # requires an explicit start.
        self._auto_start_armed = False
        self._auto_start_on_admitted = None
        self._auto_start_attempted_at = 0.0
        self._auto_start_retry_interval_s = 1.0
        # Consecutive-failure cap: a runtime that stays idle-but-unadmittable
        # must not log a retry warning every interval forever; after this many
        # failed attempts the auto start disarms and recovery requires an
        # explicit start_evaluate.
        self._auto_start_max_attempts = 10
        self._auto_start_attempts = 0
        self._logger = node.get_logger()
        self._clock = node.get_clock()
        _, config = load_robot_section(config_path)
        mode_config = config.get("control_modes", {}).get("model_inference") or {}
        self.mode = mode_config.get("runtime_mode")
        if not isinstance(self.mode, str) or not self.mode or self.mode == "idle":
            raise InterfaceBindingError(
                "mode_required", "control_modes.model_inference.runtime_mode", "explicit non-idle mode required"
            )
        auto_start = mode_config.get("auto_start", True)
        if not isinstance(auto_start, bool):
            raise InterfaceBindingError(
                "auto_start_bool", "control_modes.model_inference.auto_start", "must be a boolean"
            )
        self.auto_start = auto_start
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

    def _status_problem(self, status):
        """Why this status cannot admit the policy, or None when it can.

        Single source of the admissibility checks (no active_mode judgement):
        ``_status_matches`` is exactly "_status_problem is None", and
        ``_rejection_reason`` returns the problem itself plus the mode
        fallbacks, so start_evaluate surfaces the actual cause instead of a
        generic "changed during admission".
        """
        if status is None:
            return "runtime status unavailable"
        if status.stop_latched:
            return (
                f"runtime stop is latched ({status.stop_policy or 'unknown policy'}); "
                "request /runtime/set_mode {mode: idle} to clear, then start_evaluate"
            )
        if status.lifecycle != "ACTIVE":
            return f"runtime lifecycle is {status.lifecycle!r}, not ACTIVE"
        if status.faults:
            return f"runtime reports faults: {list(status.faults)}"
        identity_problem = self._identity_problem(status)
        if identity_problem is not None:
            return identity_problem
        if not 0 <= status.stamp.nanosec < 1_000_000_000:
            return f"runtime status stamp nanosec out of range: {status.stamp.nanosec}"
        stamp_ns = status.stamp.sec * 1_000_000_000 + status.stamp.nanosec
        age_ns = self._clock.now().nanoseconds - stamp_ns
        # A bounded negative age is a sim-time clock skew (the publisher's
        # clock leading this node's), not staleness; only ages beyond the
        # skew tolerance or older than max_age are rejected.
        if not -self._status_skew_tolerance_s * 1_000_000_000 <= age_ns <= self._status_max_age_s * 1_000_000_000:
            return (
                f"runtime status age {age_ns / 1e6:+.1f} ms is outside the freshness window "
                f"[-{self._status_skew_tolerance_s * 1e3:.0f} ms, +{self._status_max_age_s * 1e3:.0f} ms]"
            )
        return None

    def _identity_problem(self, status):
        """Why this status is not from the bound runtime, or None when it is.

        Idle transitions (release, rollback) skip the health checks, since a
        faulted runtime should still leave our mode, but never this one.
        """
        if status.runtime_name != self.runtime_name or status.runtime_version != self.runtime_version:
            return (
                f"runtime identity mismatch: expected {self.runtime_name}/{self.runtime_version}, "
                f"status reports {status.runtime_name}/{status.runtime_version}"
            )
        try:
            description = json.loads(status.interface_description_json)
        except (TypeError, ValueError):
            return "runtime status interface description is not valid JSON"
        if not isinstance(description, dict) or description.get("robot") != self._identity:
            return "runtime status advertises a different robot identity"
        return None

    def _status_matches(self, status):
        return self._status_problem(status) is None

    def _rejection_reason(self, status):
        """A specific, actionable reason why this status cannot admit the policy."""
        problem = self._status_problem(status)
        if problem is not None:
            return problem
        if status.active_mode != "idle":
            return f"runtime mode {status.active_mode!r} is owned by another producer"
        return "runtime changed during admission"

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
                # Admission verify-stage revocations (pending) may leave the
                # runtime in our mode; running-state revocations do not roll
                # back: a stopped or faulted runtime is the supervisor's call.
                self.owned = False
                self._auto_start_disarm("runtime admission was revoked")
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
                # The facade may have committed the requested mode before this
                # status invalidated the attempt (the on-change status can lose
                # the race against the set_mode response); _finish decides
                # whether an idle rollback is needed. The message names the
                # actual rejection cause; only a genuinely admissible status
                # that went idle mid-admission keeps "changed during
                # admission".
                self._finish(False, self._rejection_reason(status))
                self.on_revoke()
            self._auto_start_observe(status, valid)

    def _finish(self, success, message, *, rollback=True):
        # Central failure exit for every admission attempt. A failed attempt
        # that had requested (or completed) the mode switch may leave the
        # runtime in the admission mode; return it to idle via the bounded
        # check-then-set rollback. Only admission-window failures (pending and
        # not releasing) roll back — running-state revocations and the
        # heartbeat watchdog keep the "never resume itself" contract.
        needs_rollback = (
            not success and rollback and self.pending and not self._releasing and (self.owned or self._switch_requested)
        )
        switch_in_flight = not success and self.pending and not self._releasing and self._switch_requested
        callback = self._completion
        self._completion = None
        self.pending = False
        self._switch_requested = False
        if self._timeout_timer is not None:
            self._timeout_timer.cancel()
            self._timeout_timer = None
        if self._pending_rpc is not None:
            client, future = self._pending_rpc
            self._pending_rpc = None
            if switch_in_flight and client is self.set_client:
                # The runtime may still commit this switch: keep its response
                # so the rollback waits for it instead of reading a pre-switch mode.
                self._inflight_switch = future
            else:
                client.remove_pending_request(future)
        self.generation += 1
        if not success and self.owned:
            self.owned = False
            self.on_revoke()
        if callback is not None:
            callback(success, message)
        if needs_rollback:
            self._begin_idle_rollback_locked(message)

    def acquire(self, callback, *, own_active_session=False):
        with self.lock:
            if self._rolling_back or self.pending or self._unresolved_switch is not None:
                callback(False, "runtime transition in progress")
                return
            self.pending = True
            self._releasing = False
            self._switch_requested = False
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
                self._finish(False, self._rejection_reason(status))
            elif own_session and self.owned and status.active_mode == self.mode:
                self._finish(True, "policy already owns runtime mode")
            elif status.active_mode != "idle":
                self._finish(False, f"runtime mode {status.active_mode!r} is owned by another producer")
            else:
                # Mark before the call: the facade may commit the switch even if
                # this attempt later fails, and failure paths need to know a
                # mode switch was requested to trigger the idle rollback.
                self._switch_requested = True
                self._call(self.set_client, SetRuntimeMode.Request(mode=self.mode), switched, generation)

        def switched(response):
            if not response.success:
                self._finish(False, response.message)
                return
            # Own the transition before verifying it, so an idle/stop notification
            # racing the final status read invalidates this attempt. From here
            # on our mode may be active: a later release must verify, not assume, idle.
            self.owned = True
            self._idle_confirmed = False
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

    def cancel(self, *, rollback=True):
        """Cancel a pending admission attempt.

        ``rollback=False`` is the release()-internal flavor: the release chain
        performs its own idle transition, so the cancel must not also fire the
        idle rollback from here.
        """
        with self.lock:
            if self.pending:
                self._finish(False, "policy admission canceled; explicit start required", rollback=rollback)
            else:
                self.generation += 1

    # --- config-driven auto start -----------------------------------------------------

    def arm_auto_start(self, on_admitted):
        """Arm the config-driven auto start (control_modes.model_inference.auto_start).

        Once armed, the first observed healthy idle status acquires the policy
        mode automatically. ``on_admitted(success, message)`` runs from the
        admission completion under the owner's lock (RLock) and must not
        block. The first successful admission, any explicit start/stop
        (``disarm``), foreign-mode or latched observations, and revocation
        disarm permanently; recovery afterwards requires an explicit start.
        """
        with self.lock:
            self._auto_start_on_admitted = on_admitted
            self._auto_start_armed = True
            self._logger.info("auto start armed: waiting for a healthy idle runtime")

    def disarm(self, reason):
        """Permanently cancel the armed auto start (explicit operator action)."""
        with self.lock:
            self._auto_start_disarm(reason)

    def _auto_start_disarm(self, reason):
        if self._auto_start_armed:
            self._auto_start_armed = False
            self._logger.info(f"auto start disarmed: {reason}")

    def _auto_start_observe(self, status, valid):
        """Evaluate the armed auto start against one observed status (locked)."""
        if not self._auto_start_armed:
            return
        if status.stop_latched:
            self._auto_start_disarm("runtime stop is latched")
            return
        if not self.pending and not self.owned and status.active_mode not in ("idle", self.mode):
            # Another producer holds the runtime: never grab the mode back
            # after it is released.
            self._auto_start_disarm(f"runtime mode {status.active_mode!r} is owned by another producer")
            return
        if (
            valid
            and status.active_mode == "idle"
            and not self.pending
            and not self.owned
            and not self._rolling_back
            and self._unresolved_switch is None
            and time.monotonic() - self._auto_start_attempted_at >= self._auto_start_retry_interval_s
        ):
            self._auto_start_attempted_at = time.monotonic()
            self._logger.info("auto start: runtime is idle and healthy; acquiring the policy mode")
            self.acquire(self._auto_start_done)

    def _auto_start_done(self, success, message):
        # Invoked from _finish under the owner's lock (RLock).
        if success:
            self._auto_start_disarm("policy mode acquired")
        else:
            self._auto_start_attempts += 1
            if self._auto_start_attempts >= self._auto_start_max_attempts:
                self._auto_start_disarm("auto start retry limit reached")
                self._logger.warning(
                    f"auto start gave up after {self._auto_start_attempts} attempts; call ~/start_evaluate"
                )
            else:
                self._logger.warning(
                    f"auto start failed: {message}; retrying at most once per "
                    f"{self._auto_start_retry_interval_s:.0f} s; manual ~/start_evaluate is available"
                )
        if self._auto_start_on_admitted is not None:
            self._auto_start_on_admitted(success, message)

    def release(self, timeout=2.0, *, spin_once=None):
        """Return an owned mode to idle before another producer can acquire it."""
        event = threading.Event()
        result = [False]

        def finished(success, _message):
            result[0] = success
            event.set()

        def checked(response):
            current = response.status
            identity_problem = self._identity_problem(current)
            if identity_problem is not None:
                self._finish(False, f"{identity_problem}; idle not requested")
                return
            if current.active_mode != self.mode:
                self._finish(True, "runtime already left the policy mode")
                return
            if current.stop_latched:
                # A stop is engaging; idle would clear its latch.
                self._finish(False, "runtime stop is latched; idle is never requested over a latch")
                return
            self._call(
                self.set_client,
                SetRuntimeMode.Request(mode="idle"),
                lambda response: self._finish(bool(response.success), response.message),
                generation,
            )

        with self.lock:
            # Capture before the cancel: a pending admission whose set_mode was
            # already in flight may have committed on the facade, and when this
            # release bails out before its own idle transition (not owned),
            # only the rollback would return the mode. When release does run
            # its own chain below, the internal cancel never double-fires it.
            switch_in_flight = self.pending and self._switch_requested
            self.cancel(rollback=False)
            if not self.owned:
                if self._unresolved_switch is not None:
                    return False  # an unanswered mode request may still commit
                if switch_in_flight or not self._idle_confirmed:
                    # Our mode may still be active (a canceled switch, a failed
                    # rollback, a revocation): verify instead of assuming idle.
                    self._begin_idle_rollback_locked("release verifies the runtime left the policy mode")
                if not self._rolling_back:
                    return self._idle_confirmed  # no rollback needed, or it already completed
                # A rollback (started here or by an earlier cancel) is still in
                # flight: report success only once the runtime is confirmed to
                # have left our mode, never before.
                rollback_done = self._rollback_done
            else:
                rollback_done = None
                self.owned = False
                self.pending = True
                self._releasing = True
                self._completion = finished
                generation = self.generation
                self._rpc_deadline = time.monotonic() + timeout
                self._call(self.get_client, GetRuntimeStatus.Request(), checked, generation)
        deadline = time.monotonic() + timeout
        if rollback_done is not None:
            self._wait(rollback_done, deadline, spin_once)
            with self.lock:
                return rollback_done.is_set() and self._idle_confirmed
        self._wait(event, deadline, spin_once)
        if not event.is_set():
            self.cancel()
        with self.lock:
            self._idle_confirmed = result[0]
        return result[0]

    @staticmethod
    def _wait(event, deadline, spin_once):
        while not event.is_set() and time.monotonic() < deadline:
            if spin_once is None:
                event.wait(min(0.05, max(0, deadline - time.monotonic())))
            else:
                spin_once(timeout_sec=0.05)

    # --- idle rollback -----------------------------------------------------------------

    def _begin_idle_rollback_locked(self, reason):
        """Best-effort return of the runtime to idle after a failed admission.

        The facade may already have committed the requested mode when an
        admission fails (the set_mode response can lose the race against the
        on-change status). Check-then-set: idle is only requested when the
        runtime verifiably still holds *our* mode, so a mode owned by another
        producer is never touched, and never over a stop latch (idle clears
        it). The check and the idle request are not atomic on the runtime
        contract; the same residual window applies to ``release`` (#198).
        Called with the owner's lock held; every step is non-blocking and the whole chain is bounded by
        ``_rpc_timeout_s``. A new acquire while the rollback is in flight
        fails fast with "runtime transition in progress".
        """
        if self._rolling_back:
            return
        self._rolling_back = True
        self._rollback_token += 1
        token = self._rollback_token
        self._rollback_done.clear()
        self._logger.warning(f"returning the runtime mode to idle if it still holds {self.mode!r} ({reason})")
        self._rollback_timer = threading.Timer(self._rpc_timeout_s, self._rollback_expired, args=(token,))
        self._rollback_timer.daemon = True
        self._rollback_timer.start()
        inflight, self._inflight_switch = self._inflight_switch, None
        if inflight is not None and not inflight.done():
            # Our own set_mode is still unanswered: read the mode only after
            # the runtime has decided it, or the read may see the pre-switch idle.
            self._unresolved_switch = inflight
            inflight.add_done_callback(lambda _done: self._switch_answered(token, inflight))
            return
        self._rollback_query(token)

    def _switch_answered(self, token, future):
        with self.lock:
            if self._unresolved_switch is future:
                self._unresolved_switch = None
            if token == self._rollback_token:
                self._rollback_query(token)
            elif not (self.pending or self.owned or self._rolling_back):
                # Answered only after its rollback expired: a late commit would
                # strand our mode, so run a fresh check-then-set.
                self._begin_idle_rollback_locked("a timed-out mode request was answered late")

    def _rollback_query(self, token):
        with self.lock:
            if token != self._rollback_token:
                return
            try:
                future = self.get_client.call_async(GetRuntimeStatus.Request())
            except Exception:
                self._rollback_finished(token, False)
                return
            future.add_done_callback(lambda done: self._rollback_status_done(token, done))

    def _rollback_status_done(self, token, future):
        with self.lock:
            if token != self._rollback_token:
                return  # expired attempt: never request idle on its behalf
            try:
                response = future.result()
                status = response.status if response is not None else None
            except Exception:
                status = None
            if status is None or self._identity_problem(status) is not None:
                self._rollback_finished(token, False)
                return
            if status.active_mode != self.mode:
                self._rollback_finished(token, True)  # not in our mode: nothing to return
                return
            if status.stop_latched:
                # A stop is engaging: idle would clear its latch, so the
                # runtime's own stop transition is left to finish.
                self._logger.warning("runtime stop is latched; the idle rollback is never requested over a latch")
                self._rollback_finished(token, False)
                return
            try:
                future = self.set_client.call_async(SetRuntimeMode.Request(mode="idle"))
            except Exception:
                self._rollback_finished(token, False)
                return
            future.add_done_callback(lambda done: self._rollback_mode_done(token, done))

    def _rollback_mode_done(self, token, future):
        with self.lock:
            if token != self._rollback_token:
                return
            try:
                response = future.result()
            except Exception:
                self._logger.warning("runtime idle rollback request failed")
                self._rollback_finished(token, False)
                return
            if response is None or not response.success:
                message = response.message if response is not None else "empty response"
                self._logger.warning(f"runtime rejected the idle rollback: {message}")
                self._rollback_finished(token, False)
                return
            self._rollback_finished(token, True)

    def _rollback_expired(self, token):
        with self.lock:
            if token != self._rollback_token:
                return
            self._logger.warning("runtime idle rollback timed out; the runtime may remain in the admission mode")
            self._rollback_finished(token, False)

    def _rollback_finished(self, token, succeeded):
        with self.lock:
            if token != self._rollback_token or not self._rolling_back:
                return
            # Invalidate the attempt so none of its late callbacks can act.
            self._rollback_token += 1
            self._rolling_back = False
            self._idle_confirmed = succeeded
            if self._rollback_timer is not None:
                self._rollback_timer.cancel()
                self._rollback_timer = None
            self._rollback_done.set()
