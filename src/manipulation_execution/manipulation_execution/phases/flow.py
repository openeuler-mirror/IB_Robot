"""Pick action orchestration and shared runtime control helpers."""

from __future__ import annotations

import time
import traceback

import rclpy
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

from embodied_common.dispatch_binding import copy_binding
from ibrobot_msgs.action import PickObject, PrimitiveCommand
from manipulation_execution.pick_executor_models import (
    CandidateSelectionDiagnostics,
    FlowState,
    PickCancelled,
    PickFlowError,
)

EXECUTION_RETRYABLE_CODES = frozenset(
    {
        "IK_FAILED",
        "FK_FAILED",
        "IK_JOINT5_LIMIT",
        "IK_JOINT5_RETRY_FAILED",
        "IK_JOINT5_BRANCH_CHANGED",
        "IK_JOINT5_MISSING",
        "IK_ORIENTATION_REJECTED",
        "CONTACT_REALIGN_FAILED",
        "CONTACT_COMPENSATION_FAILED",
        "CONTACT_Z_ERROR",
        "IK_FK_PREDICTED_CONTACT_Z",
        "WORKSPACE_REJECTED",
        "FK_FIXED_FINGER_BASE_SIDE_REJECTED",
        "FK_FIXED_FINGER_BASE_SIDE_UNAVAILABLE",
        "TARGET_TABLETOP_COLLISION",
        "TF_UNAVAILABLE",
        "RPC_TIMEOUT",
        "RPC_FAILED",
    }
)


def _is_execution_retryable(error: PickFlowError) -> bool:
    """Limit generic RPC retries to calls that explicitly declared them safe."""
    if error.code in {"GRASP_VERIFICATION_FAILED", "GRASP_UNCERTAIN"}:
        return True
    if error.code in {"RPC_TIMEOUT", "RPC_FAILED"}:
        return bool(error.retryable)
    return error.code in EXECUTION_RETRYABLE_CODES


class PickFlowPhase:
    """Orchestrate the phase objects while keeping ROS lifecycle in the node."""

    @staticmethod
    def _execution_token(goal_handle) -> str:
        goal_id = getattr(goal_handle, "goal_id", None)
        raw_uuid = getattr(goal_id, "uuid", None)
        if raw_uuid is None:
            return ""
        try:
            return bytes(raw_uuid).hex()
        except (TypeError, ValueError):
            return ""

    def _check_cancel(self, goal_handle) -> None:
        if goal_handle.is_cancel_requested:
            raise PickCancelled()

    def _publish_feedback(
        self,
        goal_handle,
        state: FlowState,
        phase: str,
        detail: str,
    ) -> None:
        self._check_cancel(goal_handle)
        completed_timing = state.enter_phase(phase)
        if completed_timing is not None and completed_timing[0] != "completed":
            completed_phase, duration = completed_timing
            self.get_logger().info(f"PIPELINE_TIMING stage=phase_{completed_phase} duration_s={duration:.3f}")
        if not state.completed_phases or state.completed_phases[-1] != phase:
            state.completed_phases.append(phase)
        feedback = PickObject.Feedback()
        feedback.phase = phase
        feedback.progress = float(self._PHASE_PROGRESS.get(phase, 0.0))
        feedback.attempt = int(state.attempt)
        feedback.detail = detail
        goal_handle.publish_feedback(feedback)

    def _wait_future(
        self,
        future,
        goal_handle,
        deadline: float,
        timeout_sec: float,
        label: str,
        *,
        retryable: bool = False,
    ):
        local_deadline = min(deadline, time.monotonic() + max(0.1, timeout_sec))
        while rclpy.ok() and not future.done():
            self._check_cancel(goal_handle)
            if time.monotonic() >= local_deadline:
                future.cancel()
                raise PickFlowError("RPC_TIMEOUT", f"{label} timed out", retryable=retryable)
            time.sleep(0.05)
        response = future.result()
        if response is None:
            raise PickFlowError("RPC_FAILED", f"{label} returned no response", retryable=retryable)
        return response

    def _wait_for_service(self, client, deadline: float, service_name: str, *, required: bool = True) -> bool:
        timeout = min(self._ready_timeout, self._remaining(deadline))
        ready = client.service_is_ready() or client.wait_for_service(timeout_sec=max(0.1, timeout))
        if not ready and required:
            raise PickFlowError("SERVICE_UNAVAILABLE", f"required service unavailable: {service_name}")
        return ready

    def _cancel_primitive_and_wait(self, primitive_handle, result_future, deadline: float, primitive_name: str) -> None:
        del deadline
        cleanup_deadline = time.monotonic() + self._rpc_timeout
        try:
            cancel_future = primitive_handle.cancel_goal_async()
            while rclpy.ok() and not cancel_future.done() and time.monotonic() < cleanup_deadline:
                time.sleep(0.02)
            if not cancel_future.done() or cancel_future.result() is None:
                raise PickFlowError(
                    "PRIMITIVE_CANCEL_CLEANUP_TIMEOUT",
                    f"primitive cancellation state is unknown: {primitive_name}",
                )
            cancel_response = cancel_future.result()
            if hasattr(cancel_response, "goals_canceling") and not cancel_response.goals_canceling:
                raise PickFlowError(
                    "PRIMITIVE_CANCEL_CLEANUP_TIMEOUT",
                    f"primitive cancellation was not accepted: {primitive_name}",
                )
            while rclpy.ok() and not result_future.done() and time.monotonic() < cleanup_deadline:
                time.sleep(0.02)
            if not result_future.done() or result_future.result() is None:
                raise PickFlowError(
                    "PRIMITIVE_CANCEL_CLEANUP_TIMEOUT",
                    f"primitive terminal state is unknown: {primitive_name}",
                )
        except PickFlowError:
            raise
        except Exception as exc:
            raise PickFlowError(
                "PRIMITIVE_CANCEL_CLEANUP_TIMEOUT",
                f"primitive cleanup failed: {primitive_name}",
            ) from exc

    def _preflight(self, goal_handle, deadline: float, state: FlowState, mode: int) -> None:
        self._publish_feedback(goal_handle, state, "preflight", "checking grasp services and safe primitive server")
        if not self._primitive_client.wait_for_server(timeout_sec=min(self._ready_timeout, self._remaining(deadline))):
            raise PickFlowError("PRIMITIVE_SERVER_UNAVAILABLE", self._primitive_action_name)
        if mode == PickObject.Goal.MODE_OBSERVE_ONLY:
            return
        self._wait_for_service(self._planner_client, deadline, self._planner_service)
        if mode == PickObject.Goal.MODE_EXECUTE:
            self._wait_for_service(
                self._move_configuration_client,
                deadline,
                self._move_configuration_service,
            )
            verification_required = self._verification_policy == "required"
            self._wait_for_service(
                self._verifier_client,
                deadline,
                self._verifier_service,
                required=verification_required,
            )
        self._wait_for_service(self._ik_client, deadline, self._ik_service)
        self._wait_for_service(self._fk_client, deadline, self._fk_service)
        unhealthy_workers = self._kinematics_unhealthy_snapshot()
        for index, (ik_client, fk_client) in enumerate(
            zip(self._ik_worker_clients, self._fk_worker_clients, strict=True)
        ):
            if index in unhealthy_workers:
                continue
            self._wait_for_service(ik_client, deadline, self._ik_worker_endpoints[index])
            self._wait_for_service(fk_client, deadline, fk_client.srv_name)
        joint_state_deadline = min(deadline, time.monotonic() + self._ready_timeout)
        while self._snapshot_joint_state() is None:
            self._check_cancel(goal_handle)
            if time.monotonic() >= joint_state_deadline:
                raise PickFlowError(
                    "JOINT_STATE_UNAVAILABLE",
                    f"no current joint state received from {self._joint_state_topic}",
                )
            time.sleep(0.05)
        camera_frame = str(self._config.get("camera", {}).get("frame_id", "")).strip()
        if camera_frame:
            self._lookup_base_to_camera(camera_frame)

    def _run_primitive(
        self,
        goal_handle,
        deadline: float,
        task_id: str,
        primitive_name: str,
        *,
        pose: Pose | None = None,
        pose_name: str = "",
        velocity_scaling: float = 0.0,
        gripper_position: float = 0.0,
        joint_state: JointState | None = None,
        duration_sec: float = 0.0,
    ) -> None:
        primitive_goal = PrimitiveCommand.Goal()
        primitive_goal.schema_version = 1
        primitive_goal.dispatch_binding = copy_binding(self._dispatch_binding)
        if getattr(self, "_supervised_direct", False):
            self._direct_primitive_index = int(getattr(self, "_direct_primitive_index", 0)) + 1
            direct_task_id = f"{task_id}/primitive-{self._direct_primitive_index}"
            primitive_goal.dispatch_binding.task_id = direct_task_id
            primitive_goal.dispatch_binding.root_task_id = direct_task_id
            primitive_goal.dispatch_binding.dispatch_nonce = ""
        if not getattr(self, "_supervised_direct", False) and primitive_goal.dispatch_binding.task_id != task_id:
            raise PickFlowError("DISPATCH_BINDING_MISMATCH", "delegated primitive task ID mismatch")
        primitive_goal.execution_token = self._execution_token(goal_handle)
        primitive_goal.primitive_name = primitive_name
        primitive_goal.pose_name = pose_name
        if pose is not None:
            primitive_goal.target_pose = pose
        primitive_goal.velocity_scaling = float(velocity_scaling)
        primitive_goal.gripper_position = float(gripper_position)
        if joint_state is not None:
            position_by_name = dict(zip(joint_state.name, joint_state.position, strict=False))
            missing = [name for name in self._arm_joint_names if name not in position_by_name]
            if missing:
                raise PickFlowError(
                    "IK_JOINTS_MISSING", f"IK result missing arm joints: {', '.join(missing)}", retryable=True
                )
            primitive_goal.joint_names = list(self._arm_joint_names)
            primitive_goal.joint_positions = [float(position_by_name[name]) for name in self._arm_joint_names]
            primitive_goal.primitive_duration_sec = float(duration_sec)
        # Delegated primitives let the Gateway resolve the remaining root
        # budget at admission time; a propagated snapshot is already stale.
        primitive_goal.timeout_sec = (
            float(self._remaining(deadline)) if getattr(self, "_supervised_direct", False) else 0.0
        )

        send_future = self._primitive_client.send_goal_async(primitive_goal)
        primitive_handle = self._wait_future(send_future, goal_handle, deadline, self._rpc_timeout, primitive_name)
        if not primitive_handle.accepted:
            raise PickFlowError("PRIMITIVE_REJECTED", f"primitive rejected: {primitive_name}", retryable=True)

        result_future = primitive_handle.get_result_async()
        try:
            action_result = self._wait_future(
                result_future,
                goal_handle,
                deadline,
                self._remaining(deadline),
                primitive_name,
            )
        except PickCancelled:
            self._cancel_primitive_and_wait(primitive_handle, result_future, deadline, primitive_name)
            raise
        except PickFlowError:
            self._cancel_primitive_and_wait(primitive_handle, result_future, deadline, primitive_name)
            raise
        result = action_result.result
        if not result.success:
            raise PickFlowError(
                result.error_code or "PRIMITIVE_FAILED",
                result.message or f"primitive failed: {primitive_name}",
                retryable=primitive_name in {"move_to_named_pose", "move_to_pose"},
            )
        actual_identity = (
            result.actual_registry_epoch,
            int(result.actual_registry_generation),
            result.actual_registry_digest,
        )
        expected_identity = (
            primitive_goal.dispatch_binding.expected_registry_epoch,
            int(primitive_goal.dispatch_binding.expected_registry_generation),
            primitive_goal.dispatch_binding.expected_registry_digest,
        )
        if actual_identity != expected_identity:
            raise PickFlowError(
                "SKILL_REGISTRY_VERSION_MISMATCH",
                f"primitive used a different registry identity: {primitive_name}",
            )

    def _move_to_observe(self, goal_handle, deadline: float, state: FlowState, task_id: str) -> None:
        observe_pose = str(self._config.get("observe_pose", "observe_table"))
        if not observe_pose:
            return
        self._publish_feedback(goal_handle, state, "observe", f"moving to {observe_pose}")
        self._run_primitive(
            goal_handle,
            deadline,
            task_id,
            "move_to_named_pose",
            pose_name=observe_pose,
            velocity_scaling=float(self._config.get("observe_velocity_scaling", 0.05)),
        )
        self._sleep_with_cancel(goal_handle, deadline, float(self._config.get("observe_settle_sec", 0.6)))
        camera_frame = str(self._config.get("camera", {}).get("frame_id", "")).strip()
        if camera_frame:
            self._record_frame_diagnostic(state, "observe_settled_latest", camera_frame)

    def _sleep_with_cancel(self, goal_handle, deadline: float, duration_sec: float) -> None:
        end = min(deadline, time.monotonic() + max(0.0, duration_sec))
        while time.monotonic() < end:
            self._check_cancel(goal_handle)
            time.sleep(0.05)

    def _plan_and_prepare_candidates(
        self,
        goal_handle,
        deadline: float,
        state: FlowState,
        target_query: str,
    ):
        selection = self._config.get("candidate_selection", {})
        selection_attempts = max(1, int(selection.get("selection_attempts", 1)))
        retryable_codes = {
            "GRASP_PLANNING_FAILED",
            "NO_SAFE_GRASP_CANDIDATES",
            "NO_EXECUTABLE_CANDIDATE",
            "ALL_CANDIDATES_FAILED",
            "RPC_TIMEOUT",
            "RPC_FAILED",
            "TF_UNAVAILABLE",
        }
        last_error: PickFlowError | None = None
        selection_started = time.monotonic()

        for selection_attempt in range(1, selection_attempts + 1):
            diagnostics = CandidateSelectionDiagnostics(selection_attempt=selection_attempt)
            diagnostics_recorded = False
            self.get_logger().info(
                f"grasp selection attempt={selection_attempt}/{selection_attempts} target={target_query!r}"
            )
            try:
                stage_started = time.monotonic()
                grasp_header, candidates, scene = self._request_grasps(
                    goal_handle,
                    deadline,
                    state,
                    target_query,
                )
                stage_duration = time.monotonic() - stage_started
                state.add_timing("graspgen_request", stage_duration)
                self.get_logger().info(
                    f"PIPELINE_TIMING stage=graspgen_request duration_s={stage_duration:.3f} "
                    f"selection_attempt={selection_attempt}"
                )
                default_frame = str(grasp_header.frame_id)
                capture_transform = self._lookup_base_transform(default_frame, grasp_header.stamp)
                base_to_camera = self._transform_to_matrix(capture_transform)
                self._record_frame_diagnostic(
                    state,
                    f"grasp_capture_stamp_attempt_{selection_attempt}",
                    default_frame,
                    grasp_header.stamp,
                    camera_transform=capture_transform,
                )
                self._record_frame_diagnostic(
                    state,
                    f"post_plan_latest_attempt_{selection_attempt}",
                    default_frame,
                )
                scene_base = self._scene_geometry_base(base_to_camera, scene)
                self._publish_feedback(goal_handle, state, "selecting", f"evaluating {len(candidates)} candidates")
                stage_started = time.monotonic()
                diagnostics.raw_candidates = len(candidates)
                ranked = self._rank_candidates(
                    default_frame,
                    grasp_header.stamp,
                    base_to_camera,
                    candidates,
                    scene,
                    scene_base,
                    diagnostics=diagnostics,
                )
                stage_duration = time.monotonic() - stage_started
                state.add_timing("candidate_geometry_ranking", stage_duration)
                self.get_logger().info(
                    f"PIPELINE_TIMING stage=candidate_geometry_ranking "
                    f"duration_s={stage_duration:.3f} candidates={len(ranked)} "
                    f"selection_attempt={selection_attempt}"
                )
                candidate_seed = self._snapshot_joint_state()
                if candidate_seed is None:
                    raise PickFlowError(
                        "JOINT_STATE_UNAVAILABLE",
                        f"no current joint state received from {self._joint_state_topic}",
                    )
                stage_started = time.monotonic()
                prepared_candidates, preparation_error = self._prepare_ranked_candidates(
                    ranked,
                    scene_base,
                    candidate_seed,
                    goal_handle,
                    deadline,
                    diagnostics=diagnostics,
                )
                state.add_timing("candidate_ik_fk", time.monotonic() - stage_started)
                if prepared_candidates:
                    diagnostics.terminal_code = "ok"
                    state.candidate_selection_diagnostics.append(diagnostics.as_dict())
                    diagnostics_recorded = True
                    state.pipeline_timings["candidate_selection_total"] = time.monotonic() - selection_started
                    return prepared_candidates, scene_base
                if preparation_error is None:
                    raise PickFlowError("NO_EXECUTABLE_CANDIDATE", "no candidate could be prepared")
                preparation_rejections = diagnostics.preparation_rejections
                rejected_total = sum(preparation_rejections.values())
                workspace_rejected = preparation_rejections.get("WORKSPACE_REJECTED", 0)
                if rejected_total > 0 and workspace_rejected == rejected_total:
                    raise PickFlowError(
                        "TARGET_OUTSIDE_WORKSPACE",
                        f"target is visible but all {workspace_rejected} prepared grasp candidates are outside the robot workspace",
                    )
                raise PickFlowError(
                    "ALL_CANDIDATES_FAILED",
                    f"all {len(ranked)} ranked candidates failed preparation; "
                    f"last={preparation_error.code}: {preparation_error}",
                )
            except PickCancelled:
                raise
            except PickFlowError as exc:
                last_error = exc
                if not diagnostics_recorded:
                    diagnostics.terminal_code = exc.code
                    state.candidate_selection_diagnostics.append(diagnostics.as_dict())
                    diagnostics_recorded = True
                if exc.code not in retryable_codes or selection_attempt >= selection_attempts:
                    raise
                self.get_logger().warning(
                    f"grasp selection retry: attempt={selection_attempt}/{selection_attempts} "
                    f"code={exc.code} message={exc}"
                )
                self._publish_feedback(
                    goal_handle,
                    state,
                    "planning",
                    f"retrying grasp planning after {exc.code} ({selection_attempt + 1}/{selection_attempts})",
                )
                self._sleep_with_cancel(
                    goal_handle,
                    deadline,
                    float(selection.get("retry_settle_sec", self._config.get("observe_settle_sec", 0.6))),
                )

        if last_error is None:
            raise PickFlowError("NO_EXECUTABLE_CANDIDATE", "no candidate could be prepared")
        raise last_error

    def _order_prepared_candidates(self, state: FlowState, prepared_candidates: list) -> None:
        prepared_scoring = self._config.get("prepared_candidate_scoring", {})
        if bool(prepared_scoring.get("enabled", False)):
            prepared_candidates.sort(
                key=lambda item: (
                    -item.selection_score,
                    item.contact_z_error_m,
                    item.contact_residual_xy_m,
                    -item.ranked.score,
                    item.ranked.index,
                )
            )
        else:
            prepared_candidates.sort(
                key=lambda item: (
                    item.contact_z_error_m,
                    item.contact_residual_xy_m,
                )
            )
        self._record_prepared_ranking(state, prepared_candidates)
        summary_parts = []
        for item in prepared_candidates[:10]:
            envelope = item.fixed_finger_envelope
            fixed_text = "n/a" if envelope is None else f"{envelope.fixed_gap_m:.4f}/{envelope.score:.3f}"
            headroom_text = (
                "n/a" if item.predicted_robust_gap_headroom_m is None else f"{item.predicted_robust_gap_headroom_m:.4f}"
            )
            base_side_text = (
                "n/a"
                if item.ranked.fixed_finger_base_side is None
                else f"{item.ranked.fixed_finger_base_side.alignment_cos:.3f}"
            )
            fk_base_side_text = (
                "n/a"
                if item.fk_fixed_finger_base_side is None
                else f"{item.fk_fixed_finger_base_side.alignment_cos:.3f}"
            )
            summary_parts.append(
                f"{item.ranked.index}:score={item.selection_score:.3f}/fixed={fixed_text}/"
                f"robust_headroom={headroom_text}/"
                f"base_side={base_side_text}/fk_base_side={fk_base_side_text}/"
                f"z={item.contact_z_error_m:.4f}/xy={item.contact_residual_xy_m:.4f}/"
                f"approach_axis={item.approach_axis_error_deg}/closing_axis={item.closing_axis_error_deg:.1f}"
            )
        self.get_logger().info(f"prepared candidate rank: {', '.join(summary_parts)}")

    def _execute_pick(self, goal_handle):
        goal = goal_handle.request
        state = FlowState(completed_phases=[])
        task_deadline_unix = (
            goal.dispatch_binding.task_budget.deadline.sec
            + goal.dispatch_binding.task_budget.deadline.nanosec / 1_000_000_000
        )
        remaining_budget = task_deadline_unix - self.get_clock().now().nanoseconds / 1_000_000_000
        timeout_sec = min(float(goal.timeout_sec), remaining_budget)
        if timeout_sec <= 0.0:
            result = self._result_from_state(state)
            result.success = False
            result.error_code = "TASK_TIMEOUT"
            result.message = "shared task budget expired before pick execution"
            goal_handle.abort()
            return result
        deadline = time.monotonic() + timeout_sec
        target_query = str(goal.target_query).strip()
        task_id = str(goal.dispatch_binding.task_id).strip() or f"pick-{int(time.time() * 1000)}"
        mode = int(goal.mode)
        try:
            if len(target_query) > 200:
                raise PickFlowError("INVALID_TARGET", "target_query is too long")
            self._preflight(goal_handle, deadline, state, mode)
            self._move_to_observe(goal_handle, deadline, state, task_id)
            if mode == PickObject.Goal.MODE_OBSERVE_ONLY:
                self._publish_feedback(goal_handle, state, "completed", "observation pose reached")
                result = self._result_from_state(state)
                result.success = True
                result.error_code = ""
                result.message = "observation pose reached"
                goal_handle.succeed()
                return result
            if mode == PickObject.Goal.MODE_PLAN_ONLY:
                prepared_candidates, _scene_base = self._plan_and_prepare_candidates(
                    goal_handle,
                    deadline,
                    state,
                    target_query,
                )
                self._order_prepared_candidates(state, prepared_candidates)
                state.candidate_index = int(prepared_candidates[0].ranked.index)
                self._publish_feedback(goal_handle, state, "completed", "grasp plan prepared without execution")
                result = self._result_from_state(state)
                result.success = True
                result.error_code = ""
                result.message = (
                    f"planned {target_query!r}; candidate={state.candidate_index}; prepared={len(prepared_candidates)}"
                )
                goal_handle.succeed()
                return result

            max_attempts = int(self._config.get("max_execution_attempts", 1))
            execution_attempts = max(1, max_attempts)
            last_error: PickFlowError | None = None
            for attempt in range(1, execution_attempts + 1):
                prepared_candidates, scene_base = self._plan_and_prepare_candidates(
                    goal_handle,
                    deadline,
                    state,
                    target_query,
                )
                self._order_prepared_candidates(state, prepared_candidates)
                prepared = prepared_candidates[0]
                state.attempt = attempt
                candidate = prepared.ranked
                state.candidate_index = int(candidate.index)
                state.recovery_completed = False
                try:
                    self._publish_feedback(
                        goal_handle,
                        state,
                        "selecting",
                        f"preparing execution candidate {candidate.index} ({attempt}/{execution_attempts})",
                    )
                    self._execute_candidate(
                        goal_handle,
                        deadline,
                        state,
                        task_id,
                        target_query,
                        prepared,
                        scene_base,
                        release_after_success=bool(goal.release_after_success),
                    )
                    self._publish_feedback(goal_handle, state, "completed", "object grasped and verified")
                    result = self._result_from_state(state)
                    result.success = True
                    result.error_code = ""
                    result.message = (
                        f"picked {target_query!r}; candidate={candidate.index}; attempts={attempt}; "
                        f"verification_confidence={state.verification_confidence:.3f}"
                    )
                    goal_handle.succeed()
                    return result
                except PickCancelled:
                    raise
                except PickFlowError as exc:
                    last_error = exc
                    self.get_logger().warning(
                        f"pick candidate failed: attempt={attempt} candidate={candidate.index} "
                        f"code={exc.code} retryable={exc.retryable} message={exc}"
                    )
                    if not _is_execution_retryable(exc):
                        raise
                    if not state.recovery_completed:
                        recovery_started = time.monotonic()
                        try:
                            self._move_to_observe(goal_handle, deadline, state, task_id)
                            state.recovery_completed = True
                        finally:
                            state.add_timing("subphase_recovery", time.monotonic() - recovery_started)
            if last_error is None:
                raise PickFlowError("NO_EXECUTION_RESULT", "prepared candidates produced no execution result")
            raise PickFlowError(
                last_error.code,
                f"grasp retry budget exhausted after {state.attempt} attempts; "
                f"last failure: {last_error.code}: {last_error}",
            )
        except PickCancelled as exc:
            result = self._result_from_state(state)
            result.success = False
            result.error_code = exc.code
            result.message = str(exc)
            goal_handle.canceled()
            return result
        except PickFlowError as exc:
            result = self._result_from_state(state)
            result.success = False
            result.error_code = exc.code
            result.message = str(exc)
            goal_handle.abort()
            return result
        except Exception as exc:
            self.get_logger().error(f"unexpected pick execution failure:\n{traceback.format_exc()}")
            result = self._result_from_state(state)
            result.success = False
            result.error_code = "INTERNAL_ERROR"
            result.message = str(exc)
            goal_handle.abort()
            return result
        finally:
            with self._goal_lock:
                self._goal_active = False
                self._dispatch_nonce = ""
                self._supervised_direct = False
                self._direct_primitive_index = 0
                self._dispatch_binding = None
