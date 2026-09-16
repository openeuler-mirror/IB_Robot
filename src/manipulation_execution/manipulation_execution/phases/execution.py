"""Physical candidate execution, verification, diagnostics, and recovery phase."""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path

from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import JointState

from ibrobot_msgs.action import PickObject
from ibrobot_msgs.msg import GraspCandidate
from ibrobot_msgs.srv import VerifyGrasp
from manipulation_execution.geometry import axis_error_deg, quaternion_error_deg
from manipulation_execution.grasp_geometry import fixed_finger_robust_gap, xyz_within_workspace
from manipulation_execution.pick_executor_models import (
    BaseSceneGeometry,
    FlowState,
    IKPayload,
    PickCancelled,
    PickFlowError,
    PreparedCandidate,
)


class ExecutionPhase:
    """Execute one prepared candidate and enforce post-motion safety checks."""

    def _post_grasp_motion_target(self) -> tuple[dict, JointState]:
        config = self._config.get("post_grasp_motion", {})
        if not isinstance(config, dict):
            raise PickFlowError("INVALID_GRASP_CONFIG", "post_grasp_motion must be a mapping")
        names = [str(name) for name in config.get("joint_names", [])]
        positions_by_name = config.get("joint_positions", {})
        if not names or not isinstance(positions_by_name, dict) or set(names) != set(positions_by_name):
            raise PickFlowError(
                "INVALID_GRASP_CONFIG",
                "post_grasp_motion must define matching joint_names and joint_positions",
            )
        if len(set(names)) != len(names) or any(
            not isinstance(value, int | float) or not math.isfinite(float(value))
            for value in positions_by_name.values()
        ):
            raise PickFlowError(
                "INVALID_GRASP_CONFIG",
                "post_grasp_motion joint names must be unique and positions finite",
            )
        if set(names) != set(self._arm_joint_names):
            raise PickFlowError(
                "INVALID_GRASP_CONFIG",
                "post_grasp_motion must target exactly the configured arm joints",
            )
        target = JointState()
        target.name = names
        target.position = [float(positions_by_name[name]) for name in names]
        return config, target

    def _record_verification(
        self,
        state: FlowState,
        phase: str,
        candidate_index: int,
        candidate: GraspCandidate,
        response: VerifyGrasp.Response,
    ) -> None:
        status = int(response.status)
        record = {
            "label": {
                "verify_close": "close",
            }.get(phase, phase.removeprefix("verify_")),
            "success": bool(response.success),
            "status": status,
            "status_name": {
                VerifyGrasp.Response.STATUS_FAILED: "failed",
                VerifyGrasp.Response.STATUS_SUCCESS: "success",
                VerifyGrasp.Response.STATUS_UNCERTAIN: "uncertain",
            }.get(status, f"unknown_{status}"),
            "confidence": float(response.confidence),
            "message": response.message,
            "expected_target_width_m": float(candidate.target_width_m),
            "evidence": list(response.evidence),
            "candidate_index": candidate_index,
        }
        state.verification_records.append(record)
        self.get_logger().info(
            f"grasp verification phase={phase} candidate={candidate_index} evidence={record['evidence']}"
        )
        if state.debug_output_dir:
            output_path = Path(state.debug_output_dir) / "grasp_verification.json"
            payload = {
                "service": self._verifier_service,
                "policy": self._verification_policy,
                "records": state.verification_records,
            }
            try:
                output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            except OSError as exc:
                self.get_logger().warning(f"failed to write {output_path}: {exc}")

    def _verify(
        self,
        goal_handle,
        deadline: float,
        state: FlowState,
        phase: str,
        task_id: str,
        target_query: str,
        candidate_index: int,
        candidate: GraspCandidate,
    ) -> None:
        if self._verification_policy == "disabled":
            return
        if not self._verifier_client.service_is_ready():
            if self._verification_policy == "optional":
                return
            raise PickFlowError("VERIFIER_UNAVAILABLE", self._verifier_service)
        self._publish_feedback(goal_handle, state, phase, "sampling grasp retention evidence")
        request = VerifyGrasp.Request()
        request.task_id = task_id
        request.text_prompt = target_query
        request.grasp = candidate
        request.expected_target_width_m = float(candidate.target_width_m)
        request.post_grasp_wait_s = float(self._config.get("verification_wait_sec", 0.1))
        future = self._verifier_client.call_async(request)
        response = self._wait_future(
            future,
            goal_handle,
            deadline,
            min(float(self._config.get("verification_timeout_sec", 5.0)), self._remaining(deadline)),
            phase,
        )
        state.verification_confidence = float(response.confidence)
        self._record_verification(state, phase, candidate_index, candidate, response)
        if response.status == VerifyGrasp.Response.STATUS_SUCCESS and response.success:
            state.verification_status = PickObject.Result.VERIFICATION_SUCCESS
            return
        if response.status == VerifyGrasp.Response.STATUS_UNCERTAIN:
            state.verification_status = PickObject.Result.VERIFICATION_UNCERTAIN
            raise PickFlowError("GRASP_UNCERTAIN", response.message)
        state.verification_status = PickObject.Result.VERIFICATION_FAILED
        raise PickFlowError("GRASP_VERIFICATION_FAILED", response.message)

    def _recover_after_close_failure(
        self,
        goal_handle,
        deadline: float,
        task_id: str,
        prepared: PreparedCandidate,
        pregrasp_xyz: tuple[float, float, float],
        state: FlowState | None = None,
    ) -> None:
        retreat = (
            prepared.plan.grasp[0],
            prepared.plan.grasp[1],
            max(prepared.plan.grasp[2], pregrasp_xyz[2]),
        )
        self._run_primitive(
            goal_handle,
            deadline,
            task_id,
            "move_to_pose",
            pose=self._pose(retreat, prepared.plan.quaternion),
            velocity_scaling=float(self._config.get("descend_velocity_scaling", 0.05)),
        )
        self._run_primitive(
            goal_handle,
            deadline,
            task_id,
            "open_gripper",
            gripper_position=self._gripper_open,
        )
        if state is not None:
            self._move_to_observe(goal_handle, deadline, state, task_id)
            state.recovery_completed = True
        else:
            observe_pose = str(self._config.get("observe_pose", "observe_table"))
            if observe_pose:
                self._run_primitive(
                    goal_handle,
                    deadline,
                    task_id,
                    "move_to_named_pose",
                    pose_name=observe_pose,
                    velocity_scaling=float(self._config.get("observe_velocity_scaling", 0.05)),
                )

    def _current_ee_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        try:
            transform = self._tf_buffer.lookup_transform(
                self._base_frame,
                self._ee_frame,
                Time(),
                timeout=Duration(seconds=self._rpc_timeout),
            )
        except Exception as exc:
            raise PickFlowError(
                "TF_UNAVAILABLE",
                f"cannot read current {self._ee_frame} pose in {self._base_frame}: {exc}",
                retryable=True,
            ) from exc
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return (
            (float(translation.x), float(translation.y), float(translation.z)),
            (float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)),
        )

    def _record_pose_diagnostic(
        self,
        goal_handle,
        deadline: float,
        state: FlowState,
        label: str,
        commanded_xyz: tuple[float, float, float],
        commanded_quaternion: tuple[float, float, float, float],
        contact_ee: tuple[float, float, float],
        *,
        planned_contact: tuple[float, float, float] | None = None,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
        config = self._config.get("pose_diagnostics", {})
        robust_gap_config = self._config.get("target_gripper", {}).get("fixed_finger_robust_gap", {})
        if not bool(config.get("enabled", False)) and not (
            label == "grasp" and bool(robust_gap_config.get("enabled", False))
        ):
            return None
        self._sleep_with_cancel(goal_handle, deadline, float(config.get("settle_sec", 0.25)))
        try:
            actual_xyz, actual_quaternion = self._current_ee_pose()
        except PickFlowError as exc:
            self.get_logger().warning(f"pose diagnostic skipped for {label}: {exc}")
            return None

        position_delta = (
            actual_xyz[0] - commanded_xyz[0],
            actual_xyz[1] - commanded_xyz[1],
            actual_xyz[2] - commanded_xyz[2],
        )
        actual_contact = self._contact_for_pose(self._pose(actual_xyz, actual_quaternion), contact_ee)
        target_contact = planned_contact or self._contact_for_pose(
            self._pose(commanded_xyz, commanded_quaternion), contact_ee
        )
        contact_residual = (
            target_contact[0] - actual_contact[0],
            target_contact[1] - actual_contact[1],
            target_contact[2] - actual_contact[2],
        )
        contact_xy_error = math.hypot(contact_residual[0], contact_residual[1])
        action = "continue"
        if label == "grasp":
            abort_threshold = max(0.0, float(config.get("grasp_abort_log_threshold_m", 0.03)))
            realign_threshold = max(0.0, float(config.get("grasp_realign_log_threshold_m", 0.01)))
            warn_threshold = max(0.0, float(config.get("grasp_warn_threshold_m", 0.008)))
            if abort_threshold > 0.0 and contact_xy_error > abort_threshold:
                action = "log_only_abort_threshold_exceeded"
            elif realign_threshold > 0.0 and contact_xy_error > realign_threshold:
                action = "log_only_realign_threshold_exceeded"
            elif warn_threshold > 0.0 and contact_xy_error > warn_threshold:
                action = "warn_continue"
            else:
                action = "continue_without_low_height_realign"

        record = {
            "label": label,
            "commanded_xyz": list(commanded_xyz),
            "commanded_quaternion_xyzw": list(commanded_quaternion),
            "actual_xyz": list(actual_xyz),
            "actual_quaternion_xyzw": list(actual_quaternion),
            "actual_minus_command_xyz": list(position_delta),
            "actual_minus_command_norm_m": math.sqrt(sum(value * value for value in position_delta)),
            "rotation_error_deg": quaternion_error_deg(commanded_quaternion, actual_quaternion),
            "closing_axis_error_deg": axis_error_deg(
                commanded_quaternion,
                actual_quaternion,
                self._config.get("target_gripper", {}).get("closing_axis_ee", [1.0, 0.0, 0.0]),
            ),
            "planned_contact_base": list(target_contact),
            "actual_contact_base": list(actual_contact),
            "contact_residual_xyz": list(contact_residual),
            "contact_xy_error_m": contact_xy_error,
            "action": action,
        }
        state.pose_diagnostics.append(record)
        self.get_logger().info(
            f"pose diagnostic label={label} position_delta={position_delta} "
            f"rotation_error_deg={record['rotation_error_deg']:.2f} "
            f"closing_axis_error_deg={record['closing_axis_error_deg']:.2f} "
            f"contact_residual={contact_residual} contact_xy_error_m={contact_xy_error:.4f} action={action}"
        )
        if state.debug_output_dir:
            output_path = Path(state.debug_output_dir) / "pick_pose_diagnostics.json"
            try:
                output_path.write_text(json.dumps(state.pose_diagnostics, indent=2) + "\n", encoding="utf-8")
            except OSError as exc:
                self.get_logger().warning(f"failed to write {output_path}: {exc}")
        return contact_residual, actual_quaternion

    def _move_branch_locked_pose(
        self,
        goal_handle,
        deadline: float,
        task_id: str,
        xyz: tuple[float, float, float],
        quaternion: tuple[float, float, float, float],
        velocity_scaling: float,
        seed: JointState | None,
        *,
        validate_orientation: bool = True,
    ) -> IKPayload:
        pose = self._pose(xyz, quaternion)
        payload = self._solve_grasp_ik_fk(
            pose,
            goal_handle,
            deadline,
            seed,
            validate_orientation=validate_orientation,
        )
        self._validate_joint5_branch_continuity(seed, payload.joint_state)
        self._run_primitive(
            goal_handle,
            deadline,
            task_id,
            "move_to_configuration",
            joint_state=payload.joint_state,
            velocity_scaling=velocity_scaling,
        )
        return payload

    def _realign_contact(
        self,
        goal_handle,
        deadline: float,
        task_id: str,
        phase: str,
        command_xyz: tuple[float, float, float],
        quaternion: tuple[float, float, float, float],
        contact_ee: tuple[float, float, float],
        velocity_scaling: float,
        joint_seed: JointState | None,
    ) -> tuple[tuple[float, float, float], JointState | None]:
        config = self._config.get("contact_realign", {})
        if not bool(config.get("enabled", True)):
            return command_xyz, joint_seed
        target_contact = self._contact_for_pose(self._pose(command_xyz, quaternion), contact_ee)
        current_command = command_xyz
        current_seed = joint_seed
        tolerance = max(0.0, float(config.get("tolerance_m", 0.008)))
        maximum_correction = max(0.0, float(config.get("max_correction_m", 0.03)))
        for iteration in range(max(0, int(config.get("max_iterations", 4)))):
            self._sleep_with_cancel(goal_handle, deadline, float(config.get("settle_sec", 0.25)))
            actual_xyz, actual_quaternion = self._current_ee_pose()
            actual_contact = self._contact_for_pose(self._pose(actual_xyz, actual_quaternion), contact_ee)
            correction = (
                target_contact[0] - actual_contact[0],
                target_contact[1] - actual_contact[1],
                target_contact[2] - actual_contact[2],
            )
            error_norm = math.sqrt(sum(value * value for value in correction))
            self.get_logger().info(
                f"contact realign phase={phase} iteration={iteration} error="
                f"({correction[0]:.4f},{correction[1]:.4f},{correction[2]:.4f}) norm={error_norm:.4f}"
            )
            if error_norm <= tolerance:
                return current_command, current_seed
            next_command = (
                current_command[0] + correction[0],
                current_command[1] + correction[1],
                current_command[2] + correction[2],
            )
            cumulative = math.sqrt(sum((next_command[index] - command_xyz[index]) ** 2 for index in range(3)))
            if cumulative > maximum_correction:
                raise PickFlowError(
                    "CONTACT_REALIGN_FAILED",
                    f"{phase} contact correction {cumulative:.4f}m exceeds {maximum_correction:.4f}m",
                    retryable=True,
                )
            allowed, reason = xyz_within_workspace(next_command, self._workspace)
            if not allowed:
                raise PickFlowError("WORKSPACE_REJECTED", f"{phase} realign: {reason}", retryable=True)
            payload = self._move_branch_locked_pose(
                goal_handle,
                deadline,
                task_id,
                next_command,
                quaternion,
                velocity_scaling,
                current_seed,
            )
            current_seed = payload.joint_state
            current_command = next_command
        return current_command, current_seed

    def _pregrasp_pose(
        self,
        prepared: PreparedCandidate,
        scene_base: BaseSceneGeometry,
    ) -> tuple[float, float, float]:
        config = self._config.get("contact_realign", {})
        if not bool(config.get("pregrasp_enabled", False)) or prepared.mesh_min_z is None:
            return prepared.plan.approach
        clearance = max(0.0, float(config.get("pregrasp_clearance_m", 0.02)))
        target_reference_z = prepared.mesh_min_z + clearance
        if scene_base.object_top_base is not None:
            target_reference_z = scene_base.object_top_base[2] + clearance
        dz = target_reference_z - prepared.mesh_min_z
        return (
            prepared.plan.grasp[0],
            prepared.plan.grasp[1],
            prepared.plan.grasp[2] + dz,
        )

    def _execute_candidate(
        self,
        goal_handle,
        deadline: float,
        state: FlowState,
        task_id: str,
        target_query: str,
        prepared: PreparedCandidate,
        scene_base: BaseSceneGeometry,
        *,
        release_after_success: bool = False,
    ) -> None:
        candidate_seed = prepared.final_joint_state
        prepared = self._prepare_candidate(
            prepared.ranked,
            scene_base,
            goal_handle,
            deadline,
            apply_compensation=True,
            initial_seed=candidate_seed,
        )
        plan = prepared.plan
        candidate = prepared.ranked.candidate
        final_grasp_joint_state = prepared.final_joint_state
        current_joint_state = final_grasp_joint_state
        self._publish_feedback(goal_handle, state, "open", "opening gripper")
        self._run_primitive(
            goal_handle,
            deadline,
            task_id,
            "open_gripper",
            gripper_position=self._gripper_open,
        )
        self._sleep_with_cancel(goal_handle, deadline, float(self._config.get("open_settle_sec", 0.3)))

        self._publish_feedback(goal_handle, state, "approach", f"approaching candidate {prepared.ranked.index}")
        approach_payload = self._move_branch_locked_pose(
            goal_handle,
            deadline,
            task_id,
            plan.approach,
            plan.quaternion,
            float(self._config.get("approach_velocity_scaling", 0.05)),
            current_joint_state,
        )
        current_joint_state = approach_payload.joint_state

        realign_started = time.monotonic()
        try:
            aligned_approach, current_joint_state = self._realign_contact(
                goal_handle,
                deadline,
                task_id,
                "approach",
                plan.approach,
                plan.quaternion,
                plan.target_contact_ee,
                float(self._config.get("approach_velocity_scaling", 0.05)),
                current_joint_state,
            )
        finally:
            state.add_timing("subphase_contact_realign", time.monotonic() - realign_started)
        self._record_pose_diagnostic(
            goal_handle,
            deadline,
            state,
            "approach",
            aligned_approach,
            plan.quaternion,
            plan.target_contact_ee,
        )

        realign_delta_x = aligned_approach[0] - plan.approach[0]
        realign_delta_y = aligned_approach[1] - plan.approach[1]
        realign_delta_z = aligned_approach[2] - plan.approach[2]
        adjusted_plan = replace(
            plan,
            approach=aligned_approach,
            grasp=(
                plan.grasp[0] + realign_delta_x,
                plan.grasp[1] + realign_delta_y,
                plan.grasp[2] + realign_delta_z,
            ),
        )
        # Contact compensation is complete. Re-run only IK/FK and safety checks
        # after the measured XYZ correction; compensating again would undo it.
        prepared = self._prepare_candidate(
            replace(prepared.ranked, plan=adjusted_plan),
            scene_base,
            goal_handle,
            deadline,
            enforce_contact_error=False,
            enforce_fixed_finger_robust_gap=False,
            initial_seed=current_joint_state,
        )
        plan = prepared.plan
        final_grasp_joint_state = prepared.final_joint_state

        pregrasp = self._pregrasp_pose(prepared, scene_base)
        try:
            self._publish_feedback(goal_handle, state, "descend", "moving through safe target-relative waypoint")
            pregrasp_payload = self._move_branch_locked_pose(
                goal_handle,
                deadline,
                task_id,
                pregrasp,
                plan.quaternion,
                float(self._config.get("descend_velocity_scaling", 0.03)),
                current_joint_state,
            )
            current_joint_state = pregrasp_payload.joint_state
            self._record_pose_diagnostic(
                goal_handle,
                deadline,
                state,
                "pregrasp",
                pregrasp,
                plan.quaternion,
                plan.target_contact_ee,
            )

            self._publish_feedback(goal_handle, state, "descend", "descending to compensated grasp configuration")
            self._validate_joint5_branch_continuity(current_joint_state, final_grasp_joint_state)
            self._run_primitive(
                goal_handle,
                deadline,
                task_id,
                "move_to_configuration",
                joint_state=final_grasp_joint_state,
                velocity_scaling=float(self._config.get("descend_velocity_scaling", 0.03)),
                duration_sec=float(self._config.get("descend_duration_sec", 2.0)),
            )
            current_joint_state = final_grasp_joint_state
        except PickCancelled:
            raise
        except PickFlowError:
            recovery_started = time.monotonic()
            try:
                self._move_to_observe(goal_handle, deadline, state, task_id)
                state.recovery_completed = True
            finally:
                state.add_timing("subphase_recovery", time.monotonic() - recovery_started)
            raise
        self.get_logger().info(
            f"grasp prediction candidate={prepared.ranked.index} commanded_xyz={plan.grasp} "
            f"predicted_xyz={prepared.actual_ee_xyz} "
            f"predicted_closing_axis_error_deg={prepared.closing_axis_error_deg:.2f} "
            f"contact_xy_residual_m={prepared.contact_residual_xy_m:.4f} "
            f"contact_z_error_m={prepared.contact_z_error_m:.4f}"
        )
        try:
            self._publish_feedback(goal_handle, state, "close", "closing gripper on target")
            self._run_primitive(
                goal_handle,
                deadline,
                task_id,
                "close_gripper",
                gripper_position=self._gripper_closed,
            )
            self._sleep_with_cancel(goal_handle, deadline, float(self._config.get("hold_sec", 0.8)))

            # The grasp pose is now a committed physical state. Record diagnostics
            # after closing so the measured gap cannot trigger a pre-close retreat.
            grasp_measurement = self._record_pose_diagnostic(
                goal_handle,
                deadline,
                state,
                "grasp",
                prepared.actual_ee_xyz,
                prepared.actual_ee_quaternion,
                plan.target_contact_ee,
                planned_contact=plan.target_contact_base,
            )
            target_gripper = self._config.get("target_gripper", {})
            robust_gap_config = target_gripper.get("fixed_finger_robust_gap", {})
            if bool(robust_gap_config.get("enabled", False)) and prepared.fixed_finger_envelope is not None:
                robust_gap = None
                if grasp_measurement is not None:
                    contact_residual, actual_quaternion = grasp_measurement
                    robust_gap = fixed_finger_robust_gap(
                        prepared.fixed_finger_envelope.fixed_gap_m,
                        prepared.fixed_finger_envelope.target_gap_m,
                        contact_residual,
                        actual_quaternion,
                        target_gripper.get("closing_axis_ee", [1.0, 0.0, 0.0]),
                        max_target_gap_deficit_m=float(robust_gap_config.get("max_target_gap_deficit_m", 0.003)),
                        measurement_tolerance_m=float(robust_gap_config.get("measurement_tolerance_m", 0.0)),
                    )
                if robust_gap is not None and state.pose_diagnostics:
                    state.pose_diagnostics[-1].update(
                        {
                            "fixed_finger_contact_error_m": robust_gap.contact_error_along_closing_axis_m,
                            "fixed_finger_effective_gap_m": robust_gap.effective_gap_m,
                            "fixed_finger_required_gap_m": robust_gap.required_gap_m,
                            "fixed_finger_robust_gap_passed": robust_gap.passed,
                            "action": "record_fixed_finger_robust_gap",
                        }
                    )
                    if state.debug_output_dir:
                        output_path = Path(state.debug_output_dir) / "pick_pose_diagnostics.json"
                        try:
                            output_path.write_text(
                                json.dumps(state.pose_diagnostics, indent=2) + "\n", encoding="utf-8"
                            )
                        except OSError as exc:
                            self.get_logger().warning(f"failed to write {output_path}: {exc}")

            self._verify(
                goal_handle,
                deadline,
                state,
                "verify_close",
                task_id,
                target_query,
                prepared.ranked.index,
                candidate,
            )
            self._record_pose_diagnostic(
                goal_handle,
                deadline,
                state,
                "close",
                prepared.actual_ee_xyz,
                prepared.actual_ee_quaternion,
                plan.target_contact_ee,
                planned_contact=plan.target_contact_base,
            )
        except PickCancelled:
            raise
        except PickFlowError:
            if bool(self._config.get("recover_after_close_failure", True)):
                recovery_started = time.monotonic()
                try:
                    self._recover_after_close_failure(
                        goal_handle,
                        deadline,
                        task_id,
                        prepared,
                        pregrasp,
                        state,
                    )
                finally:
                    state.add_timing("subphase_recovery", time.monotonic() - recovery_started)
            raise

        post_grasp_motion, transport_state = self._post_grasp_motion_target()
        self._publish_feedback(goal_handle, state, "transport", "moving verified grasp to configured container pose")
        self._run_primitive(
            goal_handle,
            deadline,
            task_id,
            "move_to_joint_positions",
            joint_state=transport_state,
            velocity_scaling=float(post_grasp_motion.get("velocity_scaling", 0.05)),
            duration_sec=float(post_grasp_motion.get("duration_sec", 5.0)),
        )
        self.get_logger().info(
            f"post-grasp transport completed: pose={post_grasp_motion.get('pose_name', '')} "
            f"joints={list(transport_state.name)}"
        )
        if release_after_success:
            self._release_at_transport_pose(goal_handle, deadline, state, task_id)

    def _release_at_transport_pose(self, goal_handle, deadline: float, state: FlowState, task_id: str) -> None:
        """Open the gripper at the configured container pose without another arm motion."""
        self._publish_feedback(goal_handle, state, "release", "opening gripper at transported container pose")
        try:
            self._run_primitive(
                goal_handle,
                deadline,
                task_id,
                "open_gripper",
                gripper_position=self._gripper_open,
            )
            self._sleep_with_cancel(goal_handle, deadline, float(self._config.get("release_settle_sec", 0.3)))
        except PickCancelled:
            raise
        except PickFlowError as exc:
            raise PickFlowError("RELEASE_FAILED", str(exc)) from exc
        state.released_after_success = True
