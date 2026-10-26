"""PickBananaTask — scene task for the pick_banana scene (D9 refactor).

Exposes three ROS2 services:
  /sim/pick_banana/random_set     (Trigger) — advance the shared placements and place the banana
  /sim/pick_banana/reset          (Trigger) — reset the scene to the layout default
  /sim/pick_banana/autotest/start (Trigger) — run the AutoTest loop (non-blocking)

Platform specifics live in :mod:`sim_models.tasks.backends` (MuJoCo atomic
``ResetWorld`` state overrides with observation through the
FreeJointStatePublisherPlugin stream, Gazebo ``SetEntityPose`` through the
bridge); both use the single ``scene_compiler.randomization_placements``
generator so index *k* is the same banana placement on every platform. The
``autotest_indices`` parameter (INTEGER_ARRAY, empty = all) selects the
episode placement indices and is re-read when AutoTest starts, so a runtime
``ros2 param set`` picks it up.

Safety properties (each covered by a service-callback regression test):

- **Single operation owner**: the three services share one non-blocking
  ownership token; a competing request is rejected immediately with ``busy``
  instead of blocking an executor thread. AutoTest holds ownership for its
  whole run (through each episode's evaluation); shutdown stops and joins the
  worker.
- **Readiness first**: every public operation revalidates backend readiness
  (live, not a latched startup flag) before any dispatcher call, mode change
  or trajectory goal.
- **Exception-safe cleanup**: the mode-switch → goals → placement span runs
  under try/finally; operational exceptions become failure responses, the
  task's own goals are cancelled and the pre-reset mode is restored within a
  bounded cleanup budget. A refused or failed restore fails the operation.
- **Fail-closed restore**: restoration is refused when the runtime status is
  unknown or the stop latch is engaged; restoring to ``idle`` re-checks the
  latch immediately before sending, because the facade treats an idle request
  as a latch clear.
- **Late goal acceptance**: an acceptance future that completes after the
  timeout gets a late-acceptance handler that cancels the goal.
"""

import contextlib
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Empty, Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from sim_models.scene_compiler import get_scene_layout, randomization_placements
from sim_models.tasks.backends import GazeboBackend, MujocoBackend, PickBananaBackend
from sim_models.tasks.base import EvalResult, SceneTask

# Defaults the layout's ``evaluation:`` section may override.
_DEFAULT_SUCCESS_RADIUS_M = 0.15
_DEFAULT_FLEW_OUT_RADIUS_M = 0.40
_DEFAULT_FELL_OFF_Z = -0.05

_IDLE_MODE = "idle"


class ResetFailure(RuntimeError):
    """A reset-sequence step failed after the runtime mode was switched."""


class PickBananaTask(SceneTask):
    scene_name = "pick_banana"

    def __init__(self, node: Node) -> None:
        self._node = node
        self._cbg = ReentrantCallbackGroup()

        layout = get_scene_layout("pick_banana")
        rand = layout.get("randomization", {})
        self._placements = randomization_placements(layout)
        self._n_placements = len(self._placements)
        self._placement_idx = -1

        # Layout-driven evaluation (D9): plate centre from the layout objects,
        # thresholds from the optional evaluation section.
        objects = {str(obj.get("name")): obj for obj in layout.get("objects", []) or []}
        evaluation = layout.get("evaluation", {}) or {}
        self._plate_xy = np.array([float(v) for v in objects["plate"]["position"][:2]])
        self._success_radius_m = float(evaluation.get("success_radius_m", _DEFAULT_SUCCESS_RADIUS_M))
        self._flew_out_radius_m = float(evaluation.get("flew_out_radius_m", _DEFAULT_FLEW_OUT_RADIUS_M))
        self._fell_off_z = float(evaluation.get("fell_off_z", _DEFAULT_FELL_OFF_Z))

        # Rest pose keyed by joint name (D9 arm reset target).
        rest_joints = [str(j) for j in rand.get("arm_joint_names", [])]
        rest_values = [float(v) for v in rand.get("arm_rest_qpos", [])]
        self._rest_positions = dict(zip(rest_joints, rest_values, strict=False))

        # Launch parameters (the scene_task builder passes these from the
        # bound effective config). Array parameters are declared with an
        # explicit type: an inferred empty default becomes BYTE_ARRAY and
        # rejects the builder's string arrays.
        def param(name, default):
            return node.declare_parameter(name, default).value

        def string_array_param(name):
            from rclpy.parameter import Parameter

            # A typed declaration without an override leaves .value None.
            value = node.declare_parameter(name, Parameter.Type.STRING_ARRAY).value
            return [str(j) for j in (value or [])]

        def int_array_param(name):
            from rclpy.parameter import Parameter

            value = node.declare_parameter(name, Parameter.Type.INTEGER_ARRAY).value
            return [int(i) for i in (value or [])]

        sim_platform = str(param("sim_platform", ""))
        self._trajectory_mode = str(param("trajectory_mode", "trajectory"))
        self._rest_duration_s = float(param("rest_trajectory_duration_s", 3.0))
        self._cleanup_budget_s = float(param("cleanup_budget_s", 5.0))
        self._accept_timeout_s = float(param("goal_acceptance_timeout_s", 5.0))
        self._mode_timeout_s = float(param("mode_request_timeout_s", 5.0))
        arm_endpoint = str(param("arm_trajectory_endpoint", ""))
        self._arm_joints = string_array_param("arm_joint_names")
        gripper_endpoint = str(param("gripper_trajectory_endpoint", ""))
        self._gripper_joints = string_array_param("gripper_joint_names")
        self._autotest_indices = int_array_param("autotest_indices")

        # Platform backend (lazy platform imports inside the backends).
        self._backend: PickBananaBackend | None = None
        if sim_platform == "mujoco":
            self._backend = MujocoBackend(
                node,
                layout,
                scene=self.scene_name,
                reset_world_service=str(param("reset_world_service", "")),
                callback_group=self._cbg,
            )
        elif sim_platform == "gazebo":
            self._backend = GazeboBackend(
                node,
                layout,
                world_name=str(param("world_name", "demo")),
                scene=self.scene_name,
                callback_group=self._cbg,
            )
        else:
            node.get_logger().error(f"unsupported sim_platform {sim_platform!r}; services will refuse")
        if self._backend is not None:
            self._backend.start_readiness(timeout_s=float(param("readiness_timeout_s", 10.0)))

        # Runtime facade: mode sequencing and stop-latch respect (D9).
        from ibrobot_msgs.srv import GetRuntimeStatus, SetRuntimeMode

        self._get_status = node.create_client(GetRuntimeStatus, "/runtime/get_status", callback_group=self._cbg)
        self._set_mode = node.create_client(SetRuntimeMode, "/runtime/set_mode", callback_group=self._cbg)

        # Trajectory action clients for the runtime arm reset.
        self._goal_clients: dict[str, ActionClient] = {}
        self._arm_endpoint = arm_endpoint or ""
        self._gripper_endpoint = gripper_endpoint or ""
        for endpoint in (self._arm_endpoint, self._gripper_endpoint):
            if endpoint:
                self._goal_clients[endpoint] = ActionClient(
                    node, FollowJointTrajectory, endpoint, callback_group=self._cbg
                )
        self._goal_lock = threading.Lock()
        self._outstanding_goals: list[object] = []

        # Dispatcher control clients (AutoTest drives clean episodes; each
        # step is skipped when the dispatcher is absent).
        self._disp_start = node.create_client(Trigger, "/action_dispatcher/start_evaluate", callback_group=self._cbg)
        self._disp_stop = node.create_client(Trigger, "/action_dispatcher/stop_evaluate", callback_group=self._cbg)
        self._disp_reset = node.create_client(Empty, "/action_dispatcher/reset", callback_group=self._cbg)

        # Single operation owner (reentrant callbacks + the AutoTest worker
        # otherwise share mode/goal state); AutoTest state and eval output.
        self._operation = threading.Lock()
        self._autotest_thread: threading.Thread | None = None
        self._autotest_stop = threading.Event()
        out_dir = str(param("eval_output_dir", str(Path.home() / "ib_robot_eval")))
        self._eval_root = Path(out_dir) / "pick_banana"

        node.create_service(Trigger, "/sim/pick_banana/random_set", self._svc_random_set, callback_group=self._cbg)
        node.create_service(Trigger, "/sim/pick_banana/reset", self._svc_reset, callback_group=self._cbg)
        node.create_service(Trigger, "/sim/pick_banana/autotest/start", self._svc_autotest, callback_group=self._cbg)

    # ── SceneTask interface ────────────────────────────────────────────────────

    def randomize(self, stop_event: threading.Event | None = None, index: int | None = None) -> tuple[bool, str]:
        """Advance to the next placement (or jump to ``index``) and place it."""
        ready_ok, ready_message = self._check_backend_ready()
        if not ready_ok:
            return False, f"backend not ready: {ready_message}"
        if index is None:
            self._placement_idx = (self._placement_idx + 1) % self._n_placements
        else:
            self._placement_idx = index
        return self._backend.place(self._placement_idx, stop_event=stop_event)

    def reset(self, stop_event: threading.Event | None = None) -> tuple[bool, str]:
        ready_ok, ready_message = self._check_backend_ready()
        if not ready_ok:
            return False, f"backend not ready: {ready_message}"
        return self._backend.place(None, stop_event=stop_event)

    def evaluate(self, duration_s: float = 30.0) -> EvalResult:
        t0 = time.monotonic()
        banana_final = [0.0, 0.0, 0.0]
        start_dist_xy: float | None = None
        saw_data = False

        while (time.monotonic() - t0) < duration_s:
            if self._autotest_stop.is_set():
                return EvalResult(success=False, reason="stopped", duration_s=time.monotonic() - t0)
            pos = self._backend.object_position("banana") if self._backend is not None else None
            if pos is not None:
                saw_data = True
                banana_final = pos.tolist()
                dist_xy = float(np.linalg.norm(pos[:2] - self._plate_xy))
                if start_dist_xy is None:
                    start_dist_xy = dist_xy
                if pos[2] < self._fell_off_z or dist_xy > self._flew_out_radius_m:
                    return EvalResult(
                        success=False,
                        reason="flew_out",
                        banana_final_pos=banana_final,
                        duration_s=time.monotonic() - t0,
                    )
                if start_dist_xy >= self._success_radius_m and dist_xy < self._success_radius_m:
                    return EvalResult(
                        success=True,
                        reason="placed_on_plate",
                        banana_final_pos=banana_final,
                        duration_s=time.monotonic() - t0,
                    )
            time.sleep(0.1)

        return EvalResult(
            success=False,
            reason="timeout" if saw_data else "no_data",
            banana_final_pos=banana_final,
            duration_s=duration_s,
        )

    # ── service handlers (single owner + readiness gate) ──────────────────────

    def _svc_random_set(self, _req: Trigger.Request, resp: Trigger.Response):
        if not self._begin_operation():
            return self._busy(resp)
        try:
            ok, msg = self._clean_world_reset(randomize=True, resume=True)
        finally:
            self._end_operation()
        resp.success = ok
        resp.message = msg
        return resp

    def _svc_reset(self, _req: Trigger.Request, resp: Trigger.Response):
        if not self._begin_operation():
            return self._busy(resp)
        try:
            ok, msg = self._clean_world_reset(randomize=False, resume=True)
        finally:
            self._end_operation()
        resp.success = ok
        resp.message = msg
        return resp

    def _svc_autotest(self, _req: Trigger.Request, resp: Trigger.Response):
        if not self._begin_operation():
            return self._busy(resp)
        ready_ok, ready_message = self._check_backend_ready()
        if not ready_ok:
            self._end_operation()
            resp.success = False
            resp.message = f"backend not ready: {ready_message}"
            return resp
        # Re-read the parameter so a runtime `ros2 param set` between runs is
        # honoured; empty means the full placement list.
        try:
            selected = self._selected_indices()
        except ValueError as exc:
            self._end_operation()
            resp.success = False
            resp.message = f"autotest_indices invalid: {exc}"
            return resp
        # Ownership passes to the worker; it releases in its finally block.
        self._autotest_stop.clear()
        self._autotest_thread = threading.Thread(
            target=self._run_autotest, args=(selected,), daemon=True, name="pick_banana_autotest"
        )
        self._autotest_thread.start()
        resp.success = True
        resp.message = f"AutoTest started ({len(selected)} episodes)"
        return resp

    def _selected_indices(self) -> list[int]:
        """The AutoTest episode placement indices: the live parameter or all."""
        raw = list(self._node.get_parameter("autotest_indices").value or [])
        if not raw:
            return list(range(self._n_placements))
        indices = [int(v) for v in raw]
        invalid = [k for k in indices if not 0 <= k < self._n_placements]
        if invalid:
            raise ValueError(f"index {invalid[0]} outside 0..{self._n_placements - 1}")
        return indices

    def _begin_operation(self) -> bool:
        return self._operation.acquire(blocking=False)

    def _end_operation(self) -> None:
        with contextlib.suppress(RuntimeError):
            self._operation.release()

    @staticmethod
    def _busy(resp: Trigger.Response) -> Trigger.Response:
        resp.success = False
        resp.message = "busy: another scene-task operation is running"
        return resp

    def shutdown(self) -> None:
        """Stop the AutoTest worker, then (only once it has exited) the backend."""
        self._autotest_stop.set()
        worker = self._autotest_thread
        if worker is not None and worker.is_alive():
            worker.join(timeout=5.0)
        if worker is not None and worker.is_alive():
            # Never tear the backend out from under a still-running worker:
            # surface the timeout and leave every resource in place.
            self._node.get_logger().error(
                "AutoTest worker did not exit within the shutdown budget; "
                "leaving the backend running (teardown timeout)"
            )
            return
        if self._backend is not None:
            self._backend.shutdown()

    # ── reset sequence (D9): readiness → dispatcher → runtime arm reset ───────

    def _clean_world_reset(
        self,
        randomize: bool,
        resume: bool,
        settle_s: float = 0.8,
        stop_event: threading.Event | None = None,
        index: int | None = None,
    ) -> tuple[bool, str]:
        """Revalidate readiness → pause inference → park the arm → place → resume.

        The readiness gate comes first: no dispatcher call, mode change or
        goal happens before the backend confirms it can place. Dispatcher
        steps are skipped (and logged) when the dispatcher is not running;
        with it present the order pause → arm reset → place → reset → resume
        is preserved. ``index`` (AutoTest) places exactly that placement
        instead of advancing the shared counter.
        """
        ready_ok, ready_message = self._check_backend_ready()
        if not ready_ok:
            return False, f"backend not ready: {ready_message}"

        stop_outcome = self._dispatcher_step(
            self._disp_stop, Trigger.Request(), "dispatcher/stop", check_success=True, stop_event=stop_event
        )
        if stop_outcome is False:
            # The dispatcher is running but did not confirm the pause: fail
            # before any mode change or trajectory (finding B).
            return False, "dispatcher/stop failed; refusing to reset the scene"

        place = (
            (lambda: self.randomize(stop_event=stop_event, index=index))
            if randomize
            else (lambda: self.reset(stop_event=stop_event))
        )
        ok, msg = self._reset_sequence(place, stop_event=stop_event)

        if ok and settle_s:
            if stop_event is None:
                time.sleep(settle_s)
            else:
                stop_event.wait(settle_s)

        cancelled = stop_event is not None and stop_event.is_set()
        if resume and not cancelled:
            if ok:
                self._dispatcher_step(self._disp_reset, Empty.Request(), "dispatcher/reset", stop_event=stop_event)
            # Preserve the legacy resume behavior even when the reset fails
            # (but never after a cancellation: a stopped run must not resume).
            self._dispatcher_step(
                self._disp_start, Trigger.Request(), "dispatcher/start", check_success=True, stop_event=stop_event
            )
        elif resume and cancelled:
            self._node.get_logger().info("cancelled; skipping dispatcher reset/start")
        return ok, msg

    def _reset_sequence(self, place, stop_event: threading.Event | None = None) -> tuple[bool, str]:
        """The runtime arm reset (D9): status → mode → trajectories → place → restore.

        Any rejection fails with the runtime's message. The mode-switch →
        goals → placement span runs under try/finally: on any failure the
        task's own goals are cancelled and the previous mode is restored
        within the cleanup budget — only while the runtime is not
        stop-latched and its status is known. A refused or failed restore
        fails the operation; the response reports both the original failure
        and the restoration outcome.
        """
        status = self._runtime_status(stop_event=stop_event)
        if status is None:
            return False, "runtime status unavailable (/runtime/get_status)"
        previous_mode, stop_latched, _stop_policy = status
        if stop_latched:
            return False, "runtime is stop-latched; refusing to reset the scene"

        switched = previous_mode != self._trajectory_mode
        if switched:
            ok, message = self._set_runtime_mode(self._trajectory_mode, stop_event=stop_event)
            if not ok:
                # Ambiguous switch: a rejected or timed-out request may still
                # have taken effect server-side. Read the live mode; if it
                # moved, restore the previous mode under the same fail-closed
                # rules (unknown status or stop-latched ⇒ no restore, report).
                status_after = self._runtime_status(stop_event=stop_event)
                if status_after is None:
                    return False, (
                        f"set_mode({self._trajectory_mode!r}) {message}; "
                        "switch outcome unknown (runtime status unavailable)"
                    )
                if status_after[0] != previous_mode:
                    restored, restore_note = self._restore_mode(previous_mode, None, stop_event=stop_event)
                    return False, (
                        f"set_mode({self._trajectory_mode!r}) {message}; runtime switched anyway{restore_note}"
                        if restored
                        else f"set_mode({self._trajectory_mode!r}) {message}; runtime switched "
                        f"anyway but{restore_note.lstrip('; ')}"
                    )
                return False, f"set_mode({self._trajectory_mode!r}) rejected: {message}"

        failure: str | None = None
        cleanup_deadline: float | None = None
        restore_ok, restore_note = True, ""
        cleanup_errors: list[str] = []
        try:
            try:
                ok, message = self._send_rest_trajectories(stop_event=stop_event)
                if not ok:
                    raise ResetFailure(f"rest trajectory rejected: {message}")
                ok, message = place()
                if not ok:
                    raise ResetFailure(message)
            except ResetFailure as exc:
                failure = str(exc)
            except Exception as exc:  # noqa: BLE001 — operational errors become responses
                failure = f"reset sequence error: {exc!r}"
        finally:
            # Cleanup itself must be exception-safe: each step is guarded and
            # its outcome lands in the response; the service always answers.
            if failure is not None:
                cleanup_deadline = time.monotonic() + self._cleanup_budget_s
                try:
                    self._cancel_outstanding_goals(cleanup_deadline)
                except Exception as exc:  # noqa: BLE001
                    cleanup_errors.append(f"goal cancellation failed: {exc!r}")
            if switched:
                try:
                    restore_ok, restore_note = self._restore_mode(
                        previous_mode, cleanup_deadline, stop_event=stop_event
                    )
                except Exception as exc:  # noqa: BLE001
                    restore_ok, restore_note = False, f"; mode restore error: {exc!r}"

        suffix = restore_note + "".join(f"; {note}" for note in cleanup_errors)
        if failure is not None:
            return False, f"{failure}{suffix}"
        if not restore_ok:
            return False, f"reset completed but{suffix}"
        return True, ("ok" + suffix).strip("; ") or "ok"

    def _restore_mode(
        self, previous_mode: str, deadline: float | None = None, stop_event: threading.Event | None = None
    ) -> tuple[bool, str]:
        """Restore the pre-reset mode, failing closed on unknown/latched status.

        Restoring to ``idle`` would clear the facade's stop latch, so that
        path re-checks the status immediately before sending.
        """
        status = self._runtime_status(deadline, stop_event)
        if status is None:
            return False, "; mode not restored: runtime status unknown"
        if status[1]:
            return False, "; mode not restored: runtime is stop-latched"
        if previous_mode == _IDLE_MODE:
            recheck = self._runtime_status(deadline, stop_event)
            if recheck is None:
                return False, "; mode not restored: runtime status unknown (pre-idle recheck)"
            if recheck[1]:
                return False, "; mode not restored: runtime is stop-latched (pre-idle recheck)"
        if deadline is not None and time.monotonic() > deadline:
            return False, "; mode not restored: cleanup budget exhausted"
        ok, message = self._set_runtime_mode(previous_mode, deadline, stop_event)
        if not ok:
            return False, f"; mode restore rejected: {message}"
        if previous_mode == _IDLE_MODE:
            post = self._runtime_status(deadline, stop_event)
            if post is not None and not post[1] and post[2]:
                warning = (
                    f"; WARNING: stop_policy={post[2]!r} survived the idle restore — a stop "
                    "requested in the closing window may have been cleared (runtime follow-up)"
                )
                self._node.get_logger().error(warning.lstrip("; "))
                return True, f"; mode restored{warning}"
        return True, "; mode restored"

    def _check_backend_ready(self) -> tuple[bool, str]:
        if self._backend is None:
            return False, "no backend for this platform"
        return self._backend.check_ready()

    # ── runtime facade helpers ────────────────────────────────────────────────

    def _bounded_timeout(self, timeout_s: float, deadline: float | None) -> float:
        if deadline is None:
            return timeout_s
        return max(min(timeout_s, deadline - time.monotonic()), 0.05)

    def _runtime_status(
        self, deadline: float | None = None, stop_event: threading.Event | None = None
    ) -> tuple[str, bool, str] | None:
        """(active_mode, stop_latched, stop_policy) from /runtime/get_status, or None."""
        if not self._get_status.wait_for_service(timeout_sec=self._bounded_timeout(2.0, deadline)):
            return None
        future = self._get_status.call_async(self._get_status.srv_type.Request())
        future = self._wait_future(future, timeout_s=self._bounded_timeout(5.0, deadline), stop_event=stop_event)
        if future is None:
            return None
        response = future.result()
        status = getattr(response, "status", None)
        if status is None:
            return None
        return str(status.active_mode), bool(status.stop_latched), str(getattr(status, "stop_policy", "") or "")

    def _set_runtime_mode(
        self, mode: str, deadline: float | None = None, stop_event: threading.Event | None = None
    ) -> tuple[bool, str]:
        if not self._set_mode.wait_for_service(timeout_sec=self._bounded_timeout(2.0, deadline)):
            return False, "service not available"
        request = self._set_mode.srv_type.Request()
        request.mode = mode
        future = self._set_mode.call_async(request)
        future = self._wait_future(
            future, timeout_s=self._bounded_timeout(self._mode_timeout_s, deadline), stop_event=stop_event
        )
        if future is None:
            return False, "timeout"
        response = future.result()
        return bool(response.success), str(response.message)

    def _send_rest_trajectories(self, stop_event: threading.Event | None = None) -> tuple[bool, str]:
        """Park arm and gripper at the rest pose through the trajectory actions."""
        for endpoint, joints in (
            (self._arm_endpoint, self._arm_joints),
            (self._gripper_endpoint, self._gripper_joints),
        ):
            if not endpoint:
                continue
            try:
                positions = [self._rest_positions[joint] for joint in joints]
            except KeyError as missing:
                return False, f"rest pose has no value for joint {missing} of {endpoint}"
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = JointTrajectory()
            goal.trajectory.joint_names = list(joints)
            point = JointTrajectoryPoint()
            point.positions = positions
            seconds = int(self._rest_duration_s)
            point.time_from_start.sec = seconds
            point.time_from_start.nanosec = int((self._rest_duration_s - seconds) * 1e9)
            goal.trajectory.points = [point]
            ok, message = self._send_trajectory_goal(endpoint, goal, stop_event=stop_event)
            if not ok:
                return False, message
        return True, "rest pose reached"

    def _send_trajectory_goal(self, endpoint: str, goal, stop_event: threading.Event | None = None) -> tuple[bool, str]:
        client = self._goal_clients[endpoint]
        if not client.wait_for_server(timeout_sec=self._accept_timeout_s):
            return False, f"action server not available ({endpoint})"
        future = client.send_goal_async(goal)
        goal_future = self._wait_future(future, timeout_s=self._accept_timeout_s, stop_event=stop_event)
        if goal_future is None:
            # The server may still accept after our timeout: attach a
            # late-acceptance handler so the goal is cancelled, not orphaned.
            future.add_done_callback(self._on_late_goal_acceptance)
            return False, f"goal acceptance timeout ({endpoint})"
        goal_handle = goal_future.result()
        if not getattr(goal_handle, "accepted", False):
            return False, f"goal rejected ({endpoint})"
        # The goal stays registered as outstanding until it SUCCEEDS: a
        # timed-out or errored goal is left for the failure cleanup to cancel.
        self._track_goal(goal_handle)
        result_future = self._wait_future(
            goal_handle.get_result_async(), timeout_s=self._rest_duration_s + 10.0, stop_event=stop_event
        )
        if result_future is None:
            return False, f"goal result timeout ({endpoint})"
        wrapper = result_future.result()
        if getattr(wrapper, "status", GoalStatus.STATUS_SUCCEEDED) != GoalStatus.STATUS_SUCCEEDED:
            return False, f"goal aborted ({endpoint})"
        result = getattr(wrapper, "result", None)
        error_code = getattr(result, "error_code", FollowJointTrajectory.Result.SUCCESSFUL)
        if error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            return False, f"{endpoint}: {getattr(result, 'error_string', '') or f'error_code={error_code}'}"
        self._untrack_goal(goal_handle)
        return True, "ok"

    def _on_late_goal_acceptance(self, future) -> None:
        """Cancel a goal the server accepted after we stopped waiting."""
        try:
            goal_handle = future.result()
        except Exception:  # noqa: BLE001
            return
        if goal_handle is None or not getattr(goal_handle, "accepted", False):
            return
        self._node.get_logger().warning("goal accepted after the timeout; cancelling")
        self._track_goal(goal_handle)
        self._cancel_goal_handle(goal_handle, time.monotonic() + self._cleanup_budget_s)
        self._untrack_goal(goal_handle)

    def _track_goal(self, goal_handle) -> None:
        with self._goal_lock:
            self._outstanding_goals.append(goal_handle)

    def _untrack_goal(self, goal_handle) -> None:
        with self._goal_lock:
            self._outstanding_goals = [g for g in self._outstanding_goals if g is not goal_handle]

    def _cancel_outstanding_goals(self, deadline: float | None = None) -> None:
        with self._goal_lock:
            goals, self._outstanding_goals = list(self._outstanding_goals), []
        for goal_handle in goals:
            self._cancel_goal_handle(goal_handle, deadline)

    def _cancel_goal_handle(self, goal_handle, deadline: float | None) -> None:
        try:
            cancel_future = goal_handle.cancel_goal_async()
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().warning(f"cancelling rest goal failed: {exc}")
            return
        timeout = 2.0 if deadline is None else max(deadline - time.monotonic(), 0.05)
        self._wait_future(cancel_future, timeout_s=timeout)

    # ── generic helpers ───────────────────────────────────────────────────────

    def _wait_future(self, future, timeout_s: float | None, stop_event: threading.Event | None = None):
        """Block for a future from a non-executor thread; None on timeout.

        Service callbacks run on the MultiThreadedExecutor with a reentrant
        callback group, so another executor thread can deliver the response
        while this one waits on the event.
        """
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        if stop_event is None:
            if not done.wait(timeout=timeout_s):
                return None
            return future
        if timeout_s is None:
            timeout_s = 60.0
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            if done.wait(timeout=min(remaining, 0.1)):
                return future
            if stop_event.is_set():
                return None

    def _dispatcher_step(
        self, client, request, label: str, *, check_success: bool = False, stop_event: threading.Event | None = None
    ) -> bool | None:
        """One dispatcher step: None = absent (skipped, logged), False = failed."""
        if not client.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().info(f"{label}: dispatcher not running; skipping")
            return None
        return self._call_service_sync(client, request, label, check_success=check_success, stop_event=stop_event)

    def _call_service_sync(
        self,
        client,
        request,
        label: str,
        timeout: float = 5.0,
        *,
        check_success: bool = False,
        stop_event: threading.Event | None = None,
    ) -> bool:
        """Call a service and block (safe from a non-executor thread)."""
        future = client.call_async(request)
        if self._wait_future(future, timeout_s=timeout + 5.0, stop_event=stop_event) is None:
            self._node.get_logger().warning(f"{label}: timed out")
            return False
        if not check_success:
            return True
        try:
            response = future.result()
        except Exception as e:  # noqa: BLE001
            self._node.get_logger().warning(f"{label}: result error {e}")
            return False
        if response is None or not getattr(response, "success", False):
            self._node.get_logger().warning(f"{label}: service returned failure")
            return False
        return True

    # ── AutoTest loop ─────────────────────────────────────────────────────────

    def _run_autotest(self, indices: list[int]) -> None:
        log = self._node.get_logger()
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = self._eval_root / timestamp
            run_dir.mkdir(parents=True, exist_ok=True)
            results_path = run_dir / "results.jsonl"
            summary_path = run_dir / "summary.yaml"
            log.info(f"AutoTest: saving to {run_dir}")

            total = len(indices)
            passed = 0
            completed = 0

            with results_path.open("w") as fh:
                for episode, k in enumerate(indices):
                    if self._autotest_stop.is_set():
                        log.info("AutoTest: stop requested; ending the run")
                        break
                    ok, _ = self._clean_world_reset(
                        randomize=True, resume=True, settle_s=1.5, stop_event=self._autotest_stop, index=k
                    )
                    if not ok:
                        log.warning(f"AutoTest {k}: random_set failed, skipping")
                        continue
                    completed += 1

                    result = self.evaluate(duration_s=30.0)
                    if result.success:
                        passed += 1

                    placement = list(self._placements[k]) if 0 <= k < len(self._placements) else []
                    record = {
                        "episode": episode,
                        "idx": k,
                        "placement_index": k,
                        "banana_spawn": placement,
                        "success": result.success,
                        "reason": result.reason,
                        "banana_final_pos": result.banana_final_pos,
                        "duration_s": round(result.duration_s, 2),
                    }
                    fh.write(json.dumps(record) + "\n")
                    fh.flush()

                    sym = "✓" if result.success else "✗"
                    log.info(f"AutoTest {episode + 1}/{total} {sym} {result.reason} ({result.duration_s:.1f}s)")

            # Leave the robot idle after the batch completes.
            self._dispatcher_step(self._disp_stop, Trigger.Request(), "dispatcher/stop", check_success=True)

            rate = passed / completed if completed > 0 else 0.0
            summary_path.write_text(
                f"scene: pick_banana\n"
                f'date: "{datetime.now().isoformat()}"\n'
                f"total: {total}\n"
                f"completed: {completed}\n"
                f"passed: {passed}\n"
                f"failed: {completed - passed}\n"
                f"success_rate: {rate:.3f}\n"
            )
            log.info(f"AutoTest complete: {passed}/{completed} ({rate:.1%})")
        except Exception as exc:  # noqa: BLE001
            log.error(f"AutoTest failed: {exc!r}")
            self._dispatcher_step(self._disp_stop, Trigger.Request(), "dispatcher/stop", check_success=True)
        finally:
            self._end_operation()


# ─── Node entrypoint ──────────────────────────────────────────────────────────


def main() -> None:
    rclpy.init()
    node = Node("pick_banana_task_node")
    task = PickBananaTask(node)  # noqa: F841 — registers services on node
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        task.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
