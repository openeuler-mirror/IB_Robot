"""Scene-task service behaviour: reset sequencing, ownership, safety (D9).

Regression tests for the review findings 1–5 drive the ACTUAL service
callbacks (``_svc_reset`` / ``_svc_random_set`` / ``_svc_autotest``) with fake
services and action clients:

1. readiness is revalidated before any dispatcher call, mode change or goal;
2. cleanup is exception-safe and a failed mode restore fails the operation;
3. restoration fails closed on unknown status / stop latch (idle re-check);
4. a goal accepted after the acceptance timeout is cancelled, not orphaned;
5. competing operations are rejected immediately with ``busy``; AutoTest
   holds ownership through evaluation; shutdown stops and joins the worker.
"""

import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from action_msgs.msg import GoalStatus
from std_srvs.srv import Trigger

from sim_models.tasks.base import EvalResult
from sim_models.tasks.pick_banana import PickBananaTask


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))

    def error(self, message):
        self.messages.append(("error", message))


class FakeNode:
    logger = FakeLogger()

    def __init__(self):
        self.clients = {}
        self.subscriptions = []
        self.parameters = {"autotest_indices": []}

    def get_logger(self):
        return self.logger

    def get_parameter(self, name):
        from types import SimpleNamespace

        return SimpleNamespace(value=self.parameters.get(name, []))

    def create_client(self, srv_type, name, callback_group=None):
        return self.clients.get(name, FakeServiceClient())

    def create_subscription(self, msg_type, topic, callback, qos, callback_group=None):
        self.subscriptions.append((topic, callback))


class FakeFuture:
    def __init__(self, result):
        self._result = result

    def add_done_callback(self, callback):
        callback(self)

    def result(self):
        return self._result


class DeferredFuture:
    """Completes only when the test says so (late acceptance, finding 4)."""

    def __init__(self):
        self._done = threading.Event()
        self._result = None
        self._callbacks = []

    def set_result(self, result):
        self._result = result
        self._done.set()
        for callback in list(self._callbacks):
            callback(self)

    def add_done_callback(self, callback):
        self._callbacks.append(callback)
        if self._done.is_set():
            callback(self)

    def result(self):
        return self._result


class _Request:
    pass


def _set_pose_request_stub():
    return SimpleNamespace(
        entity=SimpleNamespace(name="", type=0, id=0),
        pose=SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )


class FakeServiceClient:
    """Scripted service client: records requests, returns canned responses."""

    def __init__(self, responses=None, available=True, availability_script=None, request_factory=_Request):
        self.responses = list(responses or [])
        self.available = available
        self.availability_script = list(availability_script or [])
        self.request_factory = request_factory
        self.requests = []

    @property
    def srv_type(self):
        return SimpleNamespace(Request=self.request_factory)

    def wait_for_service(self, timeout_sec=None):
        if self.availability_script:
            self.available = self.availability_script.pop(0)
        return self.available

    def service_is_ready(self):
        return self.available

    def call_async(self, request):
        self.requests.append(request)
        response = self.responses.pop(0) if self.responses else SimpleNamespace(success=True, message="ok")
        return FakeFuture(response)


class FakeGoalHandle:
    def __init__(self, client, *, accepted=True, error_code=0, error_string="", status=None):
        self._client = client
        self.accepted = accepted
        self._error_code = error_code
        self._error_string = error_string
        self._status = GoalStatus.STATUS_SUCCEEDED if status is None else status
        self.cancel_requests = 0

    def get_result_async(self):
        return FakeFuture(
            SimpleNamespace(
                status=self._status,
                result=SimpleNamespace(error_code=self._error_code, error_string=self._error_string),
            )
        )

    def cancel_goal_async(self):
        self.cancel_requests += 1
        return FakeFuture(None)


class FakeActionClient:
    def __init__(self, *, server=True, handles=None, goal_future=None):
        self.server = server
        self.handles = handles or [FakeGoalHandle(self)]
        self.goal_future = goal_future
        self.goals = []

    def wait_for_server(self, timeout_sec=None):
        return self.server

    def send_goal_async(self, goal):
        self.goals.append(goal)
        if self.goal_future is not None:
            return self.goal_future
        return FakeFuture(self.handles[min(len(self.goals) - 1, len(self.handles) - 1)])


class FakeBackend:
    def __init__(self, ready_result=(True, "ready"), place_result=(True, "placed")):
        self.ready_result = ready_result
        self.place_result = place_result
        self.place_calls = []
        self.place_gate = None  # threading.Event: place blocks until set

    def check_ready(self):
        return self.ready_result

    def place(self, index, stop_event=None):
        self.place_calls.append(index)
        if self.place_gate is not None:
            self.place_gate.wait(timeout=10.0)
        return self.place_result

    def object_position(self, name):
        return None

    def shutdown(self):
        pass


def _status(active_mode="idle", stop_latched=False, stop_policy=""):
    return SimpleNamespace(
        status=SimpleNamespace(active_mode=active_mode, stop_latched=stop_latched, stop_policy=stop_policy)
    )


def make_task(
    *,
    active_mode="idle",
    stop_latched=False,
    status_available=True,
    set_mode_responses=None,
    status_responses=None,
    availability_script=None,
    arm_handles=None,
    gripper_handles=None,
    backend=None,
    dispatcher=True,
    eval_root=None,
):
    task = object.__new__(PickBananaTask)
    task._node = FakeNode()
    task._trajectory_mode = "trajectory"
    task._cleanup_budget_s = 5.0
    task._rest_duration_s = 0.0
    task._accept_timeout_s = 0.2
    task._mode_timeout_s = 0.5
    task._placement_idx = -1
    task._n_placements = 210
    task._placements = [(0.0, 0.0, 0.05, 0.0)] * task._n_placements
    task._rest_positions = {"1": 0.0, "2": -1.5854, "3": 1.5708, "4": 0.7854, "5": 0.0, "6": 0.0}
    task._arm_endpoint = "/arm_trajectory_controller/follow_joint_trajectory"
    task._arm_joints = ["1", "2", "3", "4", "5"]
    task._gripper_endpoint = "/gripper_trajectory_controller/follow_joint_trajectory"
    task._gripper_joints = ["6"]
    task._goal_clients = {
        task._arm_endpoint: FakeActionClient(handles=arm_handles),
        task._gripper_endpoint: FakeActionClient(handles=gripper_handles),
    }
    task._goal_lock = threading.Lock()
    task._outstanding_goals = []
    if status_responses is not None:
        task._get_status = FakeServiceClient(responses=status_responses, availability_script=availability_script)
    elif status_available:
        task._get_status = FakeServiceClient(
            responses=[_status(active_mode, stop_latched)] * 8, availability_script=availability_script
        )
    else:
        task._get_status = FakeServiceClient(available=False)
    task._set_mode = FakeServiceClient(responses=set_mode_responses)
    task._backend = backend if backend is not None else FakeBackend()
    task._disp_stop = FakeServiceClient() if dispatcher else FakeServiceClient(available=False)
    task._disp_start = FakeServiceClient() if dispatcher else FakeServiceClient(available=False)
    task._disp_reset = FakeServiceClient() if dispatcher else FakeServiceClient(available=False)
    task._operation = threading.Lock()
    task._autotest_thread = None
    task._autotest_stop = threading.Event()
    task._eval_root = eval_root
    return task


def sent_modes(task):
    return [request.mode for request in task._set_mode.requests]


def call_reset(task):
    response = Trigger.Response()
    task._svc_reset(Trigger.Request(), response)
    return response


def call_random_set(task):
    response = Trigger.Response()
    task._svc_random_set(Trigger.Request(), response)
    return response


# ── finding 1: readiness is revalidated before any motion ─────────────────────


def test_svc_reset_refuses_before_motion_when_backend_not_ready():
    task = make_task(backend=FakeBackend(ready_result=(False, "set_pose bridge not available")))

    response = call_reset(task)

    assert response.success is False and "backend not ready" in response.message
    assert task._disp_stop.requests == [], "no dispatcher call before readiness"
    assert sent_modes(task) == [], "no mode change before readiness"
    assert not task._goal_clients[task._arm_endpoint].goals, "no goal before readiness"
    assert task._backend.place_calls == []


def test_svc_random_set_revalidates_readiness_per_call():
    backend = FakeBackend()
    task = make_task(active_mode="trajectory", backend=backend)

    first = call_random_set(task)
    assert first.success and backend.place_calls == [0]

    backend.ready_result = (False, "no fresh object_poses sample for banana, plate")
    second = call_random_set(task)
    assert second.success is False and "backend not ready" in second.message
    assert backend.place_calls == [0], "no placement while not ready"


def test_svc_autotest_refuses_when_backend_not_ready():
    task = make_task(backend=FakeBackend(ready_result=(False, "shadow model unavailable")))
    response = Trigger.Response()

    task._svc_autotest(Trigger.Request(), response)

    assert response.success is False and "backend not ready" in response.message
    assert task._autotest_thread is None
    assert not task._operation.locked(), "ownership released"


# ── finding 2: exception-safe cleanup; failed restore fails the operation ─────


def test_svc_reset_converts_operational_exceptions_to_failure_and_restores():
    backend = FakeBackend()

    def exploding_place(index, stop_event=None):
        raise RuntimeError("reset_world transport exploded")

    backend.place = exploding_place
    task = make_task(active_mode="idle", backend=backend)

    response = call_reset(task)

    assert response.success is False
    assert "reset sequence error" in response.message and "transport exploded" in response.message
    assert "mode restored" in response.message
    assert sent_modes(task) == ["trajectory", "idle"]


def test_svc_random_set_fails_when_mode_restore_is_rejected():
    task = make_task(
        active_mode="stream",
        set_mode_responses=[
            SimpleNamespace(success=True, message="ok"),
            SimpleNamespace(success=False, message="transition not allowed"),
        ],
    )

    response = call_random_set(task)

    assert response.success is False, "a rejected restore must fail the operation"
    assert "mode restore rejected: transition not allowed" in response.message


# ── finding 3: fail-closed restore ────────────────────────────────────────────


def test_svc_reset_fails_closed_when_status_is_unknown_during_restore():
    # First status read (reset entry) succeeds; the restore-time read finds
    # the service gone: restoration must be refused, nothing sent to idle.
    task = make_task(
        status_responses=[_status("idle", stop_latched=False)],
        availability_script=[True, False, False, False],
    )

    response = call_reset(task)

    assert response.success is False
    assert "runtime status unknown" in response.message
    assert sent_modes(task) == ["trajectory"], "restore refused; never idle on unknown status"


def test_svc_reset_rechecks_latch_immediately_before_restoring_idle():
    task = make_task(
        status_responses=[
            _status("idle", stop_latched=False),
            _status("trajectory", stop_latched=False),
            _status("trajectory", stop_latched=True),  # pre-idle recheck sees the latch
        ]
    )

    response = call_reset(task)

    assert response.success is False
    assert "stop-latched (pre-idle recheck)" in response.message
    assert sent_modes(task) == ["trajectory"], "idle never sent once the latch engaged"


# ── finding 4: late goal acceptance ───────────────────────────────────────────


def test_goal_accepted_after_the_timeout_is_cancelled():
    task = make_task(active_mode="trajectory")
    deferred = DeferredFuture()
    task._goal_clients[task._arm_endpoint] = FakeActionClient(goal_future=deferred)

    ok, message = task._send_trajectory_goal(task._arm_endpoint, object())

    assert not ok and "goal acceptance timeout" in message
    late_handle = FakeGoalHandle(None)
    deferred.set_result(late_handle)  # the server accepts after we gave up

    assert late_handle.cancel_requests == 1, "the late-accepted goal is cancelled"
    assert task._outstanding_goals == [], "the cancelled goal is not left outstanding"


# ── finding 5: single operation owner ─────────────────────────────────────────


def test_competing_reset_is_rejected_immediately_with_busy():
    backend = FakeBackend()
    backend.place_gate = threading.Event()
    task = make_task(active_mode="trajectory", backend=backend)
    results = {}

    def first_call():
        results["first"] = call_reset(task)

    thread = threading.Thread(target=first_call)
    thread.start()
    time.sleep(0.3)  # the first call is parked inside place()

    second = call_random_set(task)
    assert second.success is False and "busy" in second.message

    backend.place_gate.set()
    thread.join(timeout=5.0)
    assert results["first"].success, "the owning call still completes"


def test_autotest_holds_ownership_through_evaluation():
    backend = FakeBackend()
    with tempfile.TemporaryDirectory() as tmp:
        task = make_task(active_mode="trajectory", backend=backend, eval_root=Path(tmp))
        task._n_placements = 1
        entered_evaluate = threading.Event()
        release_evaluate = threading.Event()

        def blocking_evaluate(duration_s=30.0):
            entered_evaluate.set()
            release_evaluate.wait(timeout=10.0)
            return EvalResult(success=False, reason="timeout", duration_s=0.1)

        task.evaluate = blocking_evaluate
        response = Trigger.Response()
        task._svc_autotest(Trigger.Request(), response)
        assert response.success

        assert entered_evaluate.wait(timeout=5.0), "autotest reached evaluation"
        competing = call_reset(task)
        assert competing.success is False and "busy" in competing.message

        release_evaluate.set()
        worker = task._autotest_thread
        worker.join(timeout=5.0)
        assert not worker.is_alive()
        assert not task._operation.locked(), "ownership released after the run"

        after = call_reset(task)
        assert after.success, "operations are accepted again after AutoTest"


def test_shutdown_stops_and_joins_the_autotest_worker():
    with tempfile.TemporaryDirectory() as tmp:
        task = make_task(active_mode="trajectory", backend=FakeBackend(), eval_root=Path(tmp))
        task._n_placements = 1
        entered_evaluate = threading.Event()

        def slow_evaluate(duration_s=30.0):
            entered_evaluate.set()
            deadline = time.monotonic() + duration_s
            while time.monotonic() < deadline and not task._autotest_stop.is_set():
                time.sleep(0.05)
            if task._autotest_stop.is_set():
                return EvalResult(success=False, reason="stopped", duration_s=0.1)
            return EvalResult(success=False, reason="timeout", duration_s=duration_s)

        task.evaluate = slow_evaluate
        response = Trigger.Response()
        task._svc_autotest(Trigger.Request(), response)
        assert response.success and entered_evaluate.wait(timeout=5.0)

        task.shutdown()

        assert not task._autotest_thread.is_alive()


# ── reset sequencing (through the callbacks) ──────────────────────────────────


def test_svc_reset_switches_goals_places_and_restores():
    task = make_task(active_mode="policy_stream")

    response = call_reset(task)

    assert response.success, response.message
    assert sent_modes(task) == ["trajectory", "policy_stream"]
    assert len(task._goal_clients[task._arm_endpoint].goals) == 1
    assert len(task._goal_clients[task._gripper_endpoint].goals) == 1
    assert task._backend.place_calls == [None]
    assert "mode restored" in response.message


def test_svc_reset_skips_mode_calls_when_already_in_trajectory_mode():
    task = make_task(active_mode="trajectory")

    response = call_reset(task)

    assert response.success and sent_modes(task) == []
    assert task._backend.place_calls == [None]


def test_svc_reset_refuses_when_stop_latched_without_moving():
    task = make_task(stop_latched=True)

    response = call_reset(task)

    assert response.success is False and "stop-latched" in response.message
    assert sent_modes(task) == []
    assert not task._goal_clients[task._arm_endpoint].goals
    assert task._backend.place_calls == []


def test_svc_reset_reports_mode_rejection_without_further_steps():
    task = make_task(
        active_mode="idle",
        set_mode_responses=[SimpleNamespace(success=False, message="unknown mode")],
    )

    response = call_random_set(task)

    assert response.success is False
    assert "set_mode('trajectory') rejected: unknown mode" in response.message
    assert task._backend.place_calls == []


def test_svc_reset_fails_on_missing_rest_joint():
    task = make_task()
    task._rest_positions.pop("3")

    response = call_reset(task)

    assert response.success is False and "joint '3'" in response.message


def test_trajectory_rejection_after_mode_switch_cancels_and_restores():
    rejected = FakeGoalHandle(None, error_code=-4, error_string="path tolerance violated")
    task = make_task(active_mode="idle", arm_handles=[rejected])

    response = call_reset(task)

    assert response.success is False
    assert "rest trajectory rejected" in response.message and "path tolerance violated" in response.message
    assert "mode restored" in response.message
    assert sent_modes(task) == ["trajectory", "idle"]
    assert task._backend.place_calls == [], "placement never ran"
    assert rejected.cancel_requests >= 1, "the failed goal is cancelled"


def test_backend_failure_after_arm_reached_rest_cancels_and_restores():
    task = make_task(active_mode="stream", backend=FakeBackend(place_result=(False, "reset_world timeout")))

    response = call_reset(task)

    assert response.success is False
    assert "reset_world timeout" in response.message and "mode restored" in response.message
    assert sent_modes(task) == ["trajectory", "stream"]
    assert not task._outstanding_goals


def test_stop_latch_engaged_mid_reset_keeps_latch_and_skips_restore():
    task = make_task(
        status_responses=[
            _status("idle", stop_latched=False),
            _status("trajectory", stop_latched=True),
        ]
    )

    response = call_reset(task)

    assert response.success is False
    assert "mode not restored: runtime is stop-latched" in response.message
    assert sent_modes(task) == ["trajectory"], "no restore (and never idle) while latched"


# ── dispatcher integration ────────────────────────────────────────────────────


def test_clean_world_reset_orders_dispatcher_around_the_runtime_reset():
    task = make_task(active_mode="idle", dispatcher=True)
    labels = []
    original_sequence = task._reset_sequence

    def sequenced(place, **_kwargs):
        labels.append("arm-reset+place")
        return original_sequence(place)

    task._reset_sequence = sequenced
    task._dispatcher_step = lambda client, request, label, **kwargs: labels.append(label) or True

    ok, _ = task._clean_world_reset(randomize=True, resume=True, settle_s=0)

    assert ok
    assert labels == ["dispatcher/stop", "arm-reset+place", "dispatcher/reset", "dispatcher/start"]


def test_clean_world_reset_skips_absent_dispatcher_steps():
    task = make_task(dispatcher=False)

    ok, message = task._clean_world_reset(randomize=False, resume=True, settle_s=0)

    assert ok, message
    assert task._backend.place_calls == [None]
    assert task._disp_stop.requests == [] and task._disp_start.requests == []
    skipping = [text for level, text in task._node.logger.messages if level == "info" and "skipping" in text]
    assert any("dispatcher/stop" in text for text in skipping)
    assert any("dispatcher/start" in text for text in skipping)


def test_dispatcher_step_reports_absence_as_none_and_failure_as_false():
    task = make_task(dispatcher=False)
    assert task._dispatcher_step(task._disp_stop, _Request(), "dispatcher/stop") is None

    failing = make_task(dispatcher=True)
    failing._disp_stop.responses = [SimpleNamespace(success=False, message="pause refused")]
    assert failing._dispatcher_step(failing._disp_stop, _Request(), "dispatcher/stop", check_success=True) is False


# ── verification round A: cleanup itself is exception-safe ────────────────────


class RaisingOnCallClient(FakeServiceClient):
    """Service client whose Nth call_async raises instead of answering."""

    def __init__(self, responses=None, raise_on_call=2):
        super().__init__(responses=responses)
        self._calls = 0
        self._raise_on_call = raise_on_call

    def call_async(self, request):
        self._calls += 1
        if self._calls >= self._raise_on_call:
            raise RuntimeError("get_status transport exploded during cleanup")
        return super().call_async(request)


def test_svc_reset_survives_an_exception_during_cleanup():
    # Entry status answers; the restore-time status read RAISES. The service
    # must still answer, report the cleanup error, and not send anything else.
    task = make_task(
        status_responses=[_status("idle", stop_latched=False)],
    )
    task._get_status = RaisingOnCallClient(responses=[_status("idle", stop_latched=False)], raise_on_call=2)

    response = call_reset(task)

    assert response.success is False
    assert "mode restore error" in response.message and "get_status transport exploded" in response.message
    assert sent_modes(task) == ["trajectory"], "the switch happened; the restore did not"


# ── verification round B: dispatcher stop absent vs failed ────────────────────


def test_svc_reset_fails_when_the_dispatcher_stop_is_present_but_fails():
    task = make_task(dispatcher=True)
    task._disp_stop.responses = [SimpleNamespace(success=False, message="pause refused")]

    response = call_reset(task)

    assert response.success is False and "dispatcher/stop failed" in response.message
    assert sent_modes(task) == [], "no mode change after an unconfirmed pause"
    assert not task._goal_clients[task._arm_endpoint].goals, "no trajectory after an unconfirmed pause"
    assert task._backend.place_calls == [], "no placement after an unconfirmed pause"


def test_svc_reset_proceeds_when_the_dispatcher_stop_succeeds():
    task = make_task(active_mode="trajectory", dispatcher=True)

    response = call_reset(task)

    assert response.success, response.message
    assert task._backend.place_calls == [None]


# ── verification round C: shutdown interrupts waits, backend last ─────────────


class PendingFuture:
    """A future that never completes (the RPC wait must be stop-interruptible)."""

    def add_done_callback(self, callback):  # never fires
        pass


class PendingModeClient(FakeServiceClient):
    def __init__(self, responses=None):
        super().__init__(responses=responses)
        self.blocked = threading.Event()

    def call_async(self, request):
        self.blocked.set()
        return PendingFuture()


class ShutdownTrackingBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.shutdown_called = threading.Event()

    def shutdown(self):
        self.shutdown_called.set()


def test_shutdown_interrupts_a_worker_blocked_in_an_rpc_wait():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        task = make_task(active_mode="idle", backend=ShutdownTrackingBackend(), eval_root=Path(tmp))
        task._n_placements = 1
        task._set_mode = PendingModeClient(responses=None)
        response = Trigger.Response()
        task._svc_autotest(Trigger.Request(), response)
        assert response.success and task._set_mode.blocked.wait(timeout=5.0), "worker parked in the mode wait"

        task.shutdown()

        assert not task._autotest_thread.is_alive(), "stop event interrupts the RPC wait"
        assert task._backend.shutdown_called.is_set(), "backend shut down after the worker exited"


def test_shutdown_leaves_the_backend_alone_when_the_worker_never_exits():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        backend = ShutdownTrackingBackend()
        backend.place_gate = threading.Event()  # stop-blind block inside place()
        task = make_task(active_mode="trajectory", backend=backend, eval_root=Path(tmp))
        task._n_placements = 1
        response = Trigger.Response()
        task._svc_autotest(Trigger.Request(), response)
        assert response.success
        deadline = time.monotonic() + 5.0
        while not backend.place_calls and time.monotonic() < deadline:
            time.sleep(0.05)
        assert backend.place_calls, "worker reached place()"

        task.shutdown()

        assert task._autotest_thread.is_alive(), "the stop-blind block outlives the join budget"
        assert not backend.shutdown_called.is_set(), "backend untouched while the worker is stuck"
        assert any("teardown timeout" in message for _level, message in task._node.logger.messages)

        backend.place_gate.set()  # let the worker finish for a clean test exit
        task._autotest_thread.join(timeout=10.0)


# ── verification round F: post-idle-restore stop_policy report ────────────────


def test_idle_restore_reports_a_surviving_stop_policy_loudly():
    task = make_task(
        status_responses=[
            _status("idle", stop_latched=False),
            _status("trajectory", stop_latched=False),
            _status("trajectory", stop_latched=False),
            _status("idle", stop_latched=False, stop_policy="HOLD"),  # post-restore read
        ]
    )

    response = call_reset(task)

    assert response.success, "the restore itself worked"
    assert "WARNING" in response.message and "stop_policy='HOLD'" in response.message
    assert any(level == "error" and "cleared" in message for level, message in task._node.logger.messages), (
        "the possible latch clear is reported loudly"
    )


def test_idle_restore_without_stop_policy_stays_quiet():
    task = make_task(
        status_responses=[
            _status("idle", stop_latched=False),
            _status("trajectory", stop_latched=False),
            _status("trajectory", stop_latched=False),
            _status("idle", stop_latched=False, stop_policy=""),
        ]
    )

    response = call_reset(task)

    assert response.success and "WARNING" not in response.message


# ── final round 1: ambiguous mode-switch timeout ──────────────────────────────


class PendingThenOkClient(FakeServiceClient):
    """First call never completes; later calls answer normally."""

    def __init__(self):
        super().__init__()
        self._calls = 0

    def call_async(self, request):
        self._calls += 1
        self.requests.append(request)
        if self._calls == 1:
            return PendingFuture()
        return FakeFuture(SimpleNamespace(success=True, message="ok"))


def test_ambiguous_switch_timeout_with_late_effect_restores_the_previous_mode():
    # set_mode(trajectory) times out client-side, but the server switched:
    # the task must notice via status and restore the previous (idle) mode.
    task = make_task(
        status_responses=[
            _status("idle", stop_latched=False),  # reset entry
            _status("trajectory", stop_latched=False),  # post-failure live check
            _status("trajectory", stop_latched=False),  # pre-idle recheck
            _status("idle", stop_latched=False),  # post-restore read
        ]
    )
    task._set_mode = PendingThenOkClient()

    response = call_reset(task)

    assert response.success is False
    assert "runtime switched anyway" in response.message and "mode restored" in response.message
    assert sent_modes(task) == ["trajectory", "idle"], "the late switch is undone"


def test_ambiguous_switch_with_unknown_status_reports_without_restoring():
    # The switch timed out and the follow-up status read fails: nothing is
    # restored, no idle is sent, and the ambiguity is reported.
    task = make_task(
        status_responses=[_status("idle", stop_latched=False)],
        availability_script=[True, False, False, False],
    )
    task._set_mode = PendingThenOkClient()

    response = call_reset(task)

    assert response.success is False
    assert "switch outcome unknown" in response.message and "runtime status unavailable" in response.message
    assert sent_modes(task) == ["trajectory"], "no restore (and never idle) on unknown status"


# ── final round 2: cancellation through the backend and dispatcher ────────────


def _free_joint_samples_message():
    from types import SimpleNamespace

    return SimpleNamespace(
        free_joints=[
            SimpleNamespace(
                name=name,
                pose=SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=x, y=y, z=z))),
            )
            for name, x, y, z in (("banana", 0.0, 0.0, 0.05), ("plate", 0.22, 0.02, 0.02))
        ]
    )


class PendingRpcClient(FakeServiceClient):
    """Service client whose RPC never completes (stop must interrupt it)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.entered = threading.Event()

    def call_async(self, request):
        self.requests.append(request)
        self.entered.set()
        return PendingFuture()


class BackendShutdownRecorder:
    """Wraps a real backend, recording shutdown; everything else delegates."""

    def __init__(self, backend):
        self._backend = backend
        self.shutdown_called = threading.Event()

    def shutdown(self):
        self.shutdown_called.set()
        self._backend.shutdown()

    def __getattr__(self, name):
        return getattr(self._backend, name)


def test_shutdown_interrupts_a_worker_blocked_in_reset_world():
    import tempfile

    from mujoco_ros2_control_msgs.srv import ResetWorld

    from sim_models.tasks.backends import MujocoBackend

    with tempfile.TemporaryDirectory() as tmp:
        reset_client = PendingRpcClient(request_factory=ResetWorld.Request)
        node = FakeNode()
        node.clients["/mujoco_ros2_control_node/reset_world"] = reset_client
        layout = {
            "randomization": {
                "banana": {"x_range": [0, 0.1], "y_range": [0, 0.1], "z_spawn": 0.05},
                "arm_joint_names": ["1", "2", "3", "4", "5", "6"],
                "arm_rest_qpos": [0.0, -1.5854, 1.5708, 0.7854, 0.0, 0.0],
            },
            "objects": [
                {"name": "banana", "type": "dynamic", "position": [0, 0.02, 0.05], "rotation_euler_rad": [0, 0, 0]},
                {"name": "plate", "type": "dynamic", "position": [0.22, 0.02, 0.02], "rotation_euler_rad": [0, 0, 0]},
            ],
        }
        backend = MujocoBackend(node, layout, scene="pick_banana", callback_group=None)
        # readiness needs fresh samples of every placed object
        for topic, callback in node.subscriptions:
            if topic == "/sim/pick_banana/free_joint_states":
                callback(_free_joint_samples_message())
        recorder = BackendShutdownRecorder(backend)

        task = make_task(active_mode="trajectory", backend=recorder, eval_root=Path(tmp))
        task._n_placements = 1
        response = Trigger.Response()
        task._svc_autotest(Trigger.Request(), response)
        assert response.success and reset_client.entered.wait(timeout=5.0), "worker parked in ResetWorld"

        task.shutdown()

        assert not task._autotest_thread.is_alive(), "stop interrupts the pending ResetWorld"
        assert recorder.shutdown_called.is_set(), "backend shut down after the worker exited"


def test_shutdown_interrupts_a_worker_blocked_in_set_entity_pose():
    import tempfile

    from sim_models.tasks.backends import GazeboBackend

    with tempfile.TemporaryDirectory() as tmp:
        set_pose_client = PendingRpcClient(request_factory=_set_pose_request_stub)
        node = FakeNode()
        node.clients["/world/demo/set_pose"] = set_pose_client
        backend = GazeboBackend(
            node, _pick_banana_layout_stub(), world_name="demo", scene="pick_banana", callback_group=None
        )
        for topic, callback in node.subscriptions:
            if topic == "/sim/pick_banana/object_poses":
                callback(_pose_message_stub("banana", 0.0, 0.0, 0.05))
                callback(_pose_message_stub("plate", 0.22, 0.02, 0.02))
        recorder = BackendShutdownRecorder(backend)

        task = make_task(active_mode="trajectory", backend=recorder, eval_root=Path(tmp))
        task._n_placements = 1
        response = Trigger.Response()
        task._svc_autotest(Trigger.Request(), response)
        assert response.success and set_pose_client.entered.wait(timeout=5.0), "worker parked in SetEntityPose"

        task.shutdown()

        assert not task._autotest_thread.is_alive(), "stop interrupts the pending SetEntityPose"
        assert recorder.shutdown_called.is_set()


class CancellingStopClient(FakeServiceClient):
    """Dispatcher stop that engages the cancellation event when called."""

    def __init__(self, cancel_event):
        super().__init__()
        self._cancel_event = cancel_event

    def call_async(self, request):
        self.requests.append(request)
        self._cancel_event.set()
        return FakeFuture(SimpleNamespace(success=True, message="ok"))


def test_cancelled_reset_issues_no_dispatcher_resume():
    cancel_event = threading.Event()
    backend = FakeBackend()

    def stop_aware_place(index, stop_event=None):
        backend.place_calls.append(index)
        if stop_event is not None and stop_event.is_set():
            return False, "reset_world interrupted (stop)"
        return True, "placed"

    backend.place = stop_aware_place
    task = make_task(active_mode="trajectory", backend=backend)
    task._disp_stop = CancellingStopClient(cancel_event)  # cancellation fires after the pause

    ok, message = task._clean_world_reset(randomize=False, resume=True, settle_s=0, stop_event=cancel_event)

    assert not ok and "interrupted" in message
    assert task._disp_reset.requests == [], "no dispatcher reset after cancellation"
    assert task._disp_start.requests == [], "no dispatcher start after cancellation"
    assert any("skipping dispatcher reset/start" in text for _level, text in task._node.logger.messages)


# ── autotest_indices selection (issue #195) ──────────────────────────────────


def test_svc_autotest_rejects_out_of_range_indices():
    task = make_task(active_mode="trajectory")
    task._node.parameters["autotest_indices"] = [0, 999]
    response = Trigger.Response()

    task._svc_autotest(Trigger.Request(), response)

    assert response.success is False and "autotest_indices" in response.message and "999" in response.message
    assert task._autotest_thread is None
    assert not task._operation.locked(), "ownership released"


def test_svc_autotest_releases_the_lock_when_selection_raises_unexpectedly():
    # An exception the service does not translate into a response must still
    # release the operation lock: the finally block runs on every
    # non-hand-over exit, so a later reset is not busy.
    task = make_task(active_mode="trajectory")

    def exploding_indices():
        raise RuntimeError("parameter transport exploded")

    task._selected_indices = exploding_indices
    response = Trigger.Response()

    with pytest.raises(RuntimeError, match="parameter transport exploded"):
        task._svc_autotest(Trigger.Request(), response)

    assert task._autotest_thread is None
    assert not task._operation.locked(), "ownership released despite the exception"
    assert "busy" not in call_reset(task).message


def test_autotest_runs_exactly_the_selected_indices_and_records_them():
    import json

    backend = FakeBackend()
    with tempfile.TemporaryDirectory() as tmp:
        task = make_task(active_mode="trajectory", backend=backend, eval_root=Path(tmp))
        # duplicate, out-of-order indices: episode ordinal and placement index
        # must stay distinct in the records.
        task._placements[7] = (0.07, 0.0, 0.05, 0.7)
        task._placements[3] = (0.03, 0.0, 0.05, 0.3)
        task._node.parameters["autotest_indices"] = [7, 3, 7]
        task.evaluate = lambda duration_s=30.0: EvalResult(success=False, reason="timeout", duration_s=0.1)

        response = Trigger.Response()
        task._svc_autotest(Trigger.Request(), response)
        assert response.success and "3 episodes" in response.message

        worker = task._autotest_thread
        worker.join(timeout=5.0)
        assert not worker.is_alive()
        assert not task._operation.locked(), "ownership released after the run"

        run_dir = next(iter(sorted(Path(tmp).iterdir())))
        records = [json.loads(line) for line in (Path(tmp) / run_dir / "results.jsonl").read_text().splitlines()]

    assert backend.place_calls == [7, 3, 7], "each episode places exactly its selected index"
    assert task._placement_idx == 7, "the counter follows the last placed index"
    assert [record["idx"] for record in records] == [0, 1, 2], "idx stays the episode ordinal (baseline semantics)"
    assert [record["placement_index"] for record in records] == [7, 3, 7]
    assert [record["banana_spawn"] for record in records] == [
        [0.07, 0.0, 0.05, 0.7],
        [0.03, 0.0, 0.05, 0.3],
        [0.07, 0.0, 0.05, 0.7],
    ]
    assert all("episode" not in record for record in records), "no redundant episode field"


def test_selected_indices_default_to_all_when_empty():
    task = make_task(active_mode="trajectory")
    assert task._selected_indices() == list(range(210))

    task._node.parameters["autotest_indices"] = []
    assert task._selected_indices() == list(range(210)), "an empty parameter means the full placement list"


# ── default parameter startup on a real rclpy node ───────────────────────────
#
# The FakeNode stubs an initialized parameter, which hid the real-rclpy bug:
# a type-only declaration leaves the parameter uninitialized and every later
# get_parameter raises ParameterUninitializedException. These tests use a
# real rclpy Node with no overrides — exactly the default launch path.


@contextmanager
def _real_node(name="pick_banana_param_probe", parameter_overrides=None):
    """A real rclpy node on its own context, with deterministic teardown.

    The node never touches the global rclpy context, and the finally block
    destroys it even when construction inside the ``with`` block raises.
    """
    import os

    import rclpy
    from rclpy.node import Node as RclpyNode

    assert os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID"), (
        "IBROBOT_TEST_ROS_DOMAIN_ID must be set by the root conftest (see docs/testing.md)"
    )
    context = rclpy.Context()
    rclpy.init(context=context)
    node = None
    try:
        node = RclpyNode(name, context=context, parameter_overrides=parameter_overrides or [])
        yield node
    finally:
        if node is not None:
            node.destroy_node()
        context.try_shutdown()
        context.destroy()


@pytest.fixture
def real_task_node(tmp_path):
    """A PickBananaTask on a real node, writing eval output to tmp_path."""
    from rclpy.parameter import Parameter as RclpyParameter

    overrides = [RclpyParameter("eval_output_dir", value=str(tmp_path))]
    with _real_node(parameter_overrides=overrides) as node:
        task = None
        try:
            task = PickBananaTask(node)
            yield task
        finally:
            if task is not None:
                task.shutdown()


def _fast_absent_services(task):
    """Replace the facade/dispatcher clients so worker episodes fail fast."""
    task._disp_stop = FakeServiceClient(available=False)
    task._disp_start = FakeServiceClient(available=False)
    task._disp_reset = FakeServiceClient(available=False)
    task._get_status = FakeServiceClient(available=False)


def _fake_ready_runtime(task):
    """Absent dispatcher + an answering runtime status, so episodes place."""
    task._disp_stop = FakeServiceClient(available=False)
    task._disp_start = FakeServiceClient(available=False)
    task._disp_reset = FakeServiceClient(available=False)
    task._get_status = FakeServiceClient(responses=[_status("trajectory", stop_latched=False)] * 8)
    task._set_mode = FakeServiceClient()


def test_real_node_selected_indices_with_default_parameter_returns_all(real_task_node):
    # No autotest_indices override was set: the parameter must still be
    # readable (explicit empty default) and mean "every placement".
    assert real_task_node._selected_indices() == list(range(real_task_node._n_placements))


def test_real_node_default_startup_autotest_start_fails_clearly_and_releases_lock(real_task_node):
    # Default launch: no sim_platform override, so there is no backend —
    # autotest/start must answer with a clear failure (not raise, not hang)
    # and leave the operation lock free for the next reset.
    task = real_task_node
    response = Trigger.Response()

    task._svc_autotest(Trigger.Request(), response)

    assert response.success is False
    assert "backend not ready" in response.message or "no backend" in response.message
    assert task._autotest_thread is None
    assert not task._operation.locked()
    reset_response = call_reset(task)
    assert "busy" not in reset_response.message


def test_real_node_default_startup_autotest_start_runs_and_releases_lock(real_task_node, tmp_path):
    # With a ready (injected) backend and no index override, the default
    # parameter must let AutoTest start; after the run the lock is free,
    # reset is not busy, and the eval output landed in tmp_path — never in
    # the user's ~/ib_robot_eval.
    task = real_task_node
    task._backend = FakeBackend()
    _fast_absent_services(task)
    task.evaluate = lambda duration_s=30.0: EvalResult(success=False, reason="timeout", duration_s=0.1)

    response = Trigger.Response()
    task._svc_autotest(Trigger.Request(), response)

    assert response.success, response.message
    assert f"{task._n_placements} episodes" in response.message

    worker = task._autotest_thread
    worker.join(timeout=30.0)
    assert not worker.is_alive()
    assert not task._operation.locked(), "ownership released after the run"
    assert "busy" not in call_reset(task).message
    results_files = list(tmp_path.glob("pick_banana/*/results.jsonl"))
    assert results_files, "the AutoTest run wrote its results under tmp_path"


def test_real_node_runtime_set_indices_then_start_uses_them(real_task_node, tmp_path):
    # The runtime `ros2 param set` equivalent on a real node: set the
    # parameter after construction, start, and confirm the selection is
    # honoured (one episode, fast).
    from rclpy.parameter import Parameter

    task = real_task_node
    task._node.set_parameters([Parameter("autotest_indices", value=[5])])
    assert task._selected_indices() == [5]

    task._backend = FakeBackend()
    _fake_ready_runtime(task)
    task.evaluate = lambda duration_s=30.0: EvalResult(success=False, reason="timeout", duration_s=0.1)

    response = Trigger.Response()
    task._svc_autotest(Trigger.Request(), response)
    assert response.success and "1 episodes" in response.message

    worker = task._autotest_thread
    worker.join(timeout=30.0)
    assert not worker.is_alive()
    assert task._backend.place_calls == [5]
    assert not task._operation.locked()
    assert list(tmp_path.glob("pick_banana/*/results.jsonl")), "eval output stayed in tmp_path"


def test_real_node_indices_round_trip_via_explicitly_typed_empty_value(real_task_node):
    # `ros2 param set ... "[]"` infers BOOL_ARRAY and silently fails on an
    # INTEGER_ARRAY parameter; the documented restore path is an explicitly
    # typed empty INTEGER_ARRAY (the set_parameters service call in the
    # README goes through the same code).
    from rclpy.parameter import Parameter

    task = real_task_node
    task._node.set_parameters([Parameter("autotest_indices", value=[5, 2, 5])])
    assert task._selected_indices() == [5, 2, 5]

    task._node.set_parameters([Parameter("autotest_indices", type_=Parameter.Type.INTEGER_ARRAY, value=[])])

    assert task._selected_indices() == list(range(task._n_placements)), (
        "an explicitly typed empty INTEGER_ARRAY restores the full placement list"
    )


def test_task_construction_fails_when_the_empty_default_is_rejected():
    # A parameter callback that rejects the empty default makes
    # declare_array_param's set_parameters fail; construction must abort
    # with the parameter name in the error.
    from rcl_interfaces.msg import SetParametersResult

    def reject_empty_arrays(params):
        if any(isinstance(p.value, list) and not p.value for p in params):
            return SetParametersResult(successful=False, reason="empty arrays are rejected")
        return SetParametersResult(successful=True, reason="")

    with _real_node(name="pick_banana_rejected_default_probe") as node:
        node.add_on_set_parameters_callback(reject_empty_arrays)

        with pytest.raises(RuntimeError, match="arm_joint_names.*empty arrays are rejected"):
            PickBananaTask(node)


def test_task_construction_fails_closed_on_invalid_layout(monkeypatch):
    # An invalid randomization config must fail in PickBananaTask's
    # constructor — before params, services, trajectories or any ResetWorld
    # request exist. The old count=0 layout only exploded later, in
    # randomize(), after the arm's rest trajectory had been sent.
    import sim_models.tasks.pick_banana as pick_banana_module

    bad_layout = {
        "randomization": {
            "banana": {
                "mode": "uniform",
                "seed": 1,
                "count": 0,
                "x_range": [0, 0.1],
                "y_range": [0, 0.1],
            }
        },
        "objects": [],
    }
    monkeypatch.setattr(pick_banana_module, "get_scene_layout", lambda scene: bad_layout)

    with (
        _real_node(name="pick_banana_invalid_layout_probe") as node,
        pytest.raises(ValueError, match="randomization.banana.count"),
    ):
        PickBananaTask(node)


def _pick_banana_layout_stub():
    return {
        "randomization": {
            "banana": {"x_range": [0, 0.1], "y_range": [0, 0.1], "z_spawn": 0.05},
            "arm_joint_names": ["1", "2", "3", "4", "5", "6"],
            "arm_rest_qpos": [0.0, -1.5854, 1.5708, 0.7854, 0.0, 0.0],
        },
        "objects": [
            {"name": "banana", "type": "dynamic", "position": [0, 0.02, 0.05], "rotation_euler_rad": [0, 0, 0]},
            {"name": "plate", "type": "dynamic", "position": [0.22, 0.02, 0.02], "rotation_euler_rad": [0, 0, 0]},
        ],
    }


def _pose_message_stub(name, x, y, z):
    return SimpleNamespace(
        transforms=[
            SimpleNamespace(
                child_frame_id=name,
                transform=SimpleNamespace(translation=SimpleNamespace(x=x, y=y, z=z)),
            )
        ]
    )
