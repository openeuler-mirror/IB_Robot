"""Grasp request, frame lookup, geometry filtering, and source ranking phase."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from rclpy.duration import Duration
from rclpy.time import Time

from ibrobot_msgs.msg import GraspCandidate
from ibrobot_msgs.srv import DetectSegment, PlanGrasp
from manipulation_execution.grasp_geometry import (
    build_candidate_plan,
    contact_distance_score,
    fixed_finger_base_side_alignment,
    source_contact_camera,
    xyz_within_workspace,
)
from manipulation_execution.pick_executor_models import (
    BaseSceneGeometry,
    CandidateSelectionDiagnostics,
    FlowState,
    PickFlowError,
    PlannerSceneGeometry,
    RankedCandidate,
)


class PlanningPhase:
    """Plan and rank source-gripper candidates before IK preparation."""

    @staticmethod
    def _planner_failure(response) -> PickFlowError:
        diagnostics = [str(detail).strip() for detail in response.diagnostic_details]
        details = "; ".join(detail for detail in diagnostics if detail)
        message = str(response.message).strip() or "grasp planning failed"
        failure_stage = next(
            (detail.split(":", 1)[1].strip() for detail in diagnostics if detail.startswith("failure_stage:")),
            "",
        )
        detection_status = next(
            (detail.split(":", 1)[1].strip() for detail in diagnostics if detail.startswith("detection_status:")),
            "",
        )
        failure_reason = next(
            (detail.split(":", 1)[1].strip() for detail in diagnostics if detail.startswith("failure_reason:")),
            "",
        )
        if failure_reason in {"detect_service_unavailable", "segment_service_unavailable"}:
            return PickFlowError(
                "SERVICE_UNAVAILABLE",
                message + (f" ({details})" if details else ""),
            )
        if (
            failure_stage == "detection"
            or detection_status in {"not_found", "low_confidence"}
            or "no grasps generated" in message.lower()
            or "no grasp candidates" in message.lower()
        ):
            return PickFlowError(
                "TARGET_NOT_VISIBLE",
                f"target is not visible from the observation pose: {message}" + (f" ({details})" if details else ""),
            )
        return PickFlowError(
            "GRASP_PLANNING_FAILED",
            message + (f" ({details})" if details else ""),
        )

    def _request_grasps(self, goal_handle, deadline: float, state: FlowState, target_query: str):
        self._publish_feedback(goal_handle, state, "planning", f"planning grasps for {target_query!r}")
        planner = self._config.get("planner", {})
        request = PlanGrasp.Request()
        request.text_prompt = target_query
        request.confidence_threshold = float(planner.get("confidence_threshold", 0.10))
        request.grasp_threshold = float(planner.get("grasp_threshold", 0.50))
        request.debug_output_mode = str(planner.get("debug_output_mode", "diagnostic"))
        future = self._planner_client.call_async(request)
        response = self._wait_future(
            future,
            goal_handle,
            deadline,
            min(float(planner.get("timeout_sec", 120.0)), self._remaining(deadline)),
            "grasp planning",
        )
        state.debug_output_dir = str(response.debug_output_dir)
        if not response.success:
            raise self._planner_failure(response)
        candidates = list(response.grasps.grasps)
        if not candidates:
            raise PickFlowError(
                "TARGET_NOT_VISIBLE",
                "target was not detected or no grasp candidate could be generated from the observation frame",
            )
        scoring = self._config.get("execution_scoring", {})
        centroid_source = str(scoring.get("centroid_source", "volume")).strip().lower()
        use_volume = centroid_source == "volume" and float(response.object_volume_m3) > 0.0
        centroid_msg = response.object_volume_centroid_xyz if use_volume else response.object_centroid_xyz
        object_centroid = None
        if int(response.object_point_count) > 0:
            values = (float(centroid_msg.x), float(centroid_msg.y), float(centroid_msg.z))
            if all(np.isfinite(value) for value in values):
                object_centroid = values
        if object_centroid is None and float(scoring.get("contact_distance_weight", 0.0)) > 0.0:
            object_centroid = self._request_fallback_detection_centroid(
                goal_handle,
                deadline,
                target_query,
                centroid_source,
                float(planner.get("confidence_threshold", 0.10)),
            )
        table_normal, table_offset, table_inlier_ratio = self._table_geometry_from_response(response)
        object_top = None
        top_values = (
            float(response.object_top_xyz.x),
            float(response.object_top_xyz.y),
            float(response.object_top_xyz.z),
        )
        if table_normal is not None and all(np.isfinite(value) for value in top_values):
            object_top = top_values
        scene = PlannerSceneGeometry(
            object_centroid_camera=object_centroid,
            table_normal_camera=table_normal,
            table_offset_camera=table_offset,
            table_inlier_ratio=table_inlier_ratio,
            object_top_camera=object_top,
        )
        return response.grasps.header, candidates, scene

    def _request_fallback_detection_centroid(
        self,
        goal_handle,
        deadline: float,
        target_query: str,
        centroid_source: str,
        confidence_threshold: float,
    ) -> tuple[float, float, float] | None:
        if not self._detect_client.service_is_ready():
            self.get_logger().warning(f"fallback detection skipped: service unavailable: {self._detect_service}")
            return None
        request = DetectSegment.Request()
        request.text_prompt = target_query
        request.confidence_threshold = confidence_threshold
        try:
            response = self._wait_future(
                self._detect_client.call_async(request),
                goal_handle,
                deadline,
                min(float(self._config.get("planner", {}).get("timeout_sec", 120.0)), self._remaining(deadline)),
                "fallback detection",
            )
        except PickFlowError as exc:
            self.get_logger().warning(f"fallback detection skipped: {exc}")
            return None
        if not response.success:
            self.get_logger().warning(f"fallback detection failed: {response.message}")
            return None
        minimum_points = int(self._config.get("candidate_selection", {}).get("min_point_count", 100))
        matching = [
            detection
            for detection in response.detections.detections
            if target_query.lower() in str(detection.label).lower() and int(detection.point_count) >= minimum_points
        ]
        if not matching:
            return None
        detection = max(matching, key=lambda item: (float(item.confidence), int(item.point_count)))
        use_volume = centroid_source == "volume" and float(detection.volume_m3) > 0.0
        centroid = detection.volume_centroid_xyz if use_volume else detection.centroid_xyz
        values = (float(centroid.x), float(centroid.y), float(centroid.z))
        if not all(np.isfinite(value) for value in values):
            return None
        self.get_logger().info(
            f"fallback detection centroid source={'volume' if use_volume else 'surface'} xyz={values}"
        )
        return values

    def _lookup_base_transform(self, frame_id: str, stamp=None):
        if not frame_id:
            raise PickFlowError("GRASP_FRAME_MISSING", "grasp candidates have no frame_id")
        stamp_ns = self._stamp_to_ns(stamp)
        lookup_time = Time() if stamp_ns == 0 else Time.from_msg(stamp)
        try:
            return self._tf_buffer.lookup_transform(
                self._base_frame,
                frame_id,
                lookup_time,
                timeout=Duration(seconds=self._rpc_timeout),
            )
        except Exception as exc:
            lookup_label = "latest" if stamp_ns == 0 else f"stamp_ns={stamp_ns}"
            raise PickFlowError(
                "TF_UNAVAILABLE",
                f"cannot transform {frame_id} to {self._base_frame} at {lookup_label}: {exc}",
            ) from exc

    def _lookup_base_to_camera(self, frame_id: str, stamp=None) -> np.ndarray:
        return self._transform_to_matrix(self._lookup_base_transform(frame_id, stamp))

    def _record_frame_diagnostic(
        self,
        state: FlowState,
        label: str,
        camera_frame: str,
        stamp=None,
        *,
        camera_transform=None,
    ) -> None:
        config = self._config.get("frame_diagnostics", {})
        if not bool(config.get("enabled", False)):
            return
        try:
            if camera_transform is None:
                camera_transform = self._lookup_base_transform(camera_frame, stamp)
            ee_transform = self._lookup_base_transform(self._ee_frame, stamp)
        except PickFlowError as exc:
            self.get_logger().warning(f"frame diagnostic skipped for {label}: {exc}")
            return

        base_to_camera = self._transform_to_matrix(camera_transform)
        base_to_ee = self._transform_to_matrix(ee_transform)
        ee_to_camera = np.linalg.inv(base_to_ee) @ base_to_camera
        requested_stamp_ns = self._stamp_to_ns(stamp)
        record = {
            "label": label,
            "base_frame": self._base_frame,
            "ee_frame": self._ee_frame,
            "camera_frame": camera_frame,
            "lookup_mode": "latest" if requested_stamp_ns == 0 else "capture_stamp",
            "requested_stamp_ns": requested_stamp_ns,
            "camera_transform_stamp_ns": self._stamp_to_ns(camera_transform.header.stamp),
            "ee_transform_stamp_ns": self._stamp_to_ns(ee_transform.header.stamp),
            "base_to_ee": self._matrix_diagnostic(base_to_ee),
            "base_to_camera": self._matrix_diagnostic(base_to_camera),
            "ee_to_camera": self._matrix_diagnostic(ee_to_camera),
        }
        state.frame_diagnostics.append(record)
        self.get_logger().info(
            f"frame diagnostic label={label} lookup={record['lookup_mode']} stamp_ns={requested_stamp_ns} "
            f"base_to_ee_xyz={record['base_to_ee']['translation_xyz']} "
            f"base_to_ee_q={record['base_to_ee']['quaternion_xyzw']} "
            f"base_to_camera_xyz={record['base_to_camera']['translation_xyz']}"
        )
        if state.debug_output_dir:
            output_path = Path(state.debug_output_dir) / "pick_frame_diagnostics.json"
            try:
                output_path.write_text(json.dumps(state.frame_diagnostics, indent=2) + "\n", encoding="utf-8")
            except OSError as exc:
                self.get_logger().warning(f"failed to write {output_path}: {exc}")

    def _rank_candidates(
        self,
        default_frame: str,
        default_stamp,
        default_transform: np.ndarray,
        candidates: list[GraspCandidate],
        scene: PlannerSceneGeometry,
        scene_base: BaseSceneGeometry,
        *,
        diagnostics: CandidateSelectionDiagnostics | None = None,
    ) -> list[RankedCandidate]:
        selection = self._config.get("candidate_selection", {})
        scoring = self._config.get("execution_scoring", {})
        min_confidence = float(selection.get("min_confidence", 0.0))
        require_collision_free = bool(selection.get("require_collision_free", False))
        min_contact_z = float(selection.get("min_contact_z", 0.0))
        min_approach_z = float(selection.get("min_approach_z", 0.04))
        min_topdown_score = float(selection.get("min_topdown_score", 0.0))
        topdown_min_z = float(selection.get("topdown_min_z", -0.25))
        min_topdown_dot = max(0.0, min(0.999, -topdown_min_z))
        confidence_weight = float(scoring.get("confidence_weight", selection.get("confidence_weight", 1.0)))
        topdown_weight = float(scoring.get("topdown_weight", selection.get("topdown_weight", 0.35)))
        contact_weight = max(0.0, float(scoring.get("contact_distance_weight", 0.0)))
        contact_scale = max(1e-6, float(scoring.get("contact_distance_scale_m", 0.06)))
        source_contact = self._config.get("source_contact_point", [0.0, 0.0, 0.195])
        target_gripper = self._config.get("target_gripper", {})
        base_side_config = target_gripper.get("fixed_finger_base_side", {})
        check_fixed_finger_base_side = (
            bool(base_side_config.get("enabled", False))
            and str(target_gripper.get("type", "")) == "asymmetric_single_moving_jaw"
        )
        minimum_base_side_alignment = max(
            -1.0,
            min(1.0, float(base_side_config.get("min_alignment_cos", 0.0))),
        )
        max_candidates = int(selection.get("max_candidates", 80))
        default_stamp_ns = self._stamp_to_ns(default_stamp)
        transforms: dict[tuple[str, int], np.ndarray] = {(default_frame, default_stamp_ns): default_transform}
        ranked: list[RankedCandidate] = []

        def reject(code: str) -> None:
            if diagnostics is not None:
                diagnostics.reject_geometry(code)

        for index, candidate in enumerate(candidates):
            confidence = float(candidate.confidence)
            if confidence < min_confidence:
                reject("MIN_CONFIDENCE")
                continue
            if require_collision_free and not bool(candidate.collision_free):
                reject("COLLISION_NOT_FREE")
                continue
            frame_id = candidate.header.frame_id or default_frame
            candidate_stamp = candidate.header.stamp
            if self._stamp_to_ns(candidate_stamp) == 0:
                candidate_stamp = default_stamp
            transform_key = (frame_id, self._stamp_to_ns(candidate_stamp))
            if transform_key not in transforms:
                transforms[transform_key] = self._lookup_base_to_camera(frame_id, candidate_stamp)
            try:
                plan = build_candidate_plan(
                    candidate.pose_matrix,
                    transforms[transform_key],
                    float(candidate.target_width_m),
                    float(candidate.target_width_quality),
                    self._config,
                    width_axis_camera=candidate.width_axis_camera,
                    target_width_min_offset_m=float(candidate.target_width_min_offset_m),
                    target_width_max_offset_m=float(candidate.target_width_max_offset_m),
                )
            except (ValueError, ArithmeticError):
                reject("CANDIDATE_GEOMETRY_INVALID")
                continue
            normalized_topdown = max(
                0.0,
                min(1.0, (plan.topdown_score - min_topdown_dot) / (1.0 - min_topdown_dot)),
            )
            plan = replace(plan, topdown_score=normalized_topdown)
            if (
                plan.target_contact_base[2] < min_contact_z
                or plan.approach[2] < min_approach_z
                or plan.topdown_score < min_topdown_score
            ):
                reject("HEIGHT_OR_APPROACH_REJECTED")
                continue
            if any(not xyz_within_workspace(xyz, self._workspace)[0] for xyz in (plan.approach, plan.grasp)):
                reject("WORKSPACE_REJECTED")
                continue
            fixed_finger_base_side = None
            if check_fixed_finger_base_side:
                if plan.target_width_min_base is None or plan.target_width_max_base is None:
                    reject("FIXED_FINGER_BASE_SIDE_UNAVAILABLE")
                    continue
                try:
                    fixed_finger_base_side = fixed_finger_base_side_alignment(
                        plan.grasp,
                        plan.quaternion,
                        target_gripper.get("fixed_finger_contact_ee", [-0.014, 0.0, -0.080]),
                        plan.target_width_min_base,
                        plan.target_width_max_base,
                        base_side_config.get("reference_point_base", [0.0, 0.0, 0.0]),
                    )
                except ValueError:
                    reject("FIXED_FINGER_BASE_SIDE_UNAVAILABLE")
                    continue
                if fixed_finger_base_side.alignment_cos < minimum_base_side_alignment:
                    reject("FIXED_FINGER_BASE_SIDE_REJECTED")
                    continue
            contact_distance = None
            contact_score = 0.0
            if scene.object_centroid_camera is not None:
                contact = source_contact_camera(candidate.pose_matrix, source_contact)
                contact_distance, contact_score = contact_distance_score(
                    contact,
                    scene.object_centroid_camera,
                    contact_scale,
                )
            score = (
                confidence_weight * confidence + topdown_weight * plan.topdown_score + contact_weight * contact_score
            )
            ranked.append(
                RankedCandidate(
                    index=index,
                    candidate=candidate,
                    plan=plan,
                    score=score,
                    contact_distance_m=contact_distance,
                    fixed_finger_base_side=fixed_finger_base_side,
                )
            )

        if diagnostics is not None:
            diagnostics.geometry_surviving_candidates = len(ranked)

        if bool(self._target_geometry.get("tabletop_filter", False)) and ranked:
            if self._mesh_directory is None or scene_base.table_plane is None:
                raise PickFlowError(
                    "TARGET_TABLETOP_UNAVAILABLE",
                    "SO101 tabletop filter requires target meshes and a fitted table plane",
                )
            minimum_clearance = float(self._target_geometry.get("tabletop_clearance_m", 0.0))
            try:
                geometry_metrics = self._grasp_geometry.gripper_geometry_metrics_batch(
                    self._mesh_directory,
                    [
                        (
                            item.plan.approach,
                            item.plan.grasp,
                            item.plan.quaternion,
                            float(item.candidate.target_width_m),
                        )
                        for item in ranked
                    ],
                    scene_base.table_plane,
                    clearance_threshold_m=minimum_clearance,
                )
            except Exception as exc:
                raise PickFlowError("TARGET_GEOMETRY_FAILED", str(exc)) from exc
            tabletop_candidates = len(ranked)
            ranked = [
                item
                for item, (_, clearance) in zip(ranked, geometry_metrics, strict=True)
                if clearance is not None and clearance >= minimum_clearance
            ]
            if diagnostics is not None:
                for _ in range(tabletop_candidates - len(ranked)):
                    diagnostics.reject_geometry("TARGET_TABLETOP_COLLISION")

        if diagnostics is not None:
            diagnostics.ranked_candidates = len(ranked)

        ranked.sort(
            key=lambda item: (
                -item.score,
                float("inf") if item.contact_distance_m is None else item.contact_distance_m,
                -float(item.candidate.confidence),
                item.index,
            )
        )
        if max_candidates > 0:
            if diagnostics is not None:
                diagnostics.truncated_by_candidate_budget = max(0, len(ranked) - max_candidates)
            ranked = ranked[:max_candidates]
        if not ranked:
            rejection_counts = diagnostics.geometry_rejections if diagnostics is not None else {}
            rejected_total = sum(rejection_counts.values())
            workspace_rejected = rejection_counts.get("WORKSPACE_REJECTED", 0)
            if rejected_total > 0 and workspace_rejected == rejected_total:
                raise PickFlowError(
                    "TARGET_OUTSIDE_WORKSPACE",
                    f"target is visible but all {workspace_rejected} grasp candidates are outside the robot workspace",
                )
            raise PickFlowError("NO_SAFE_GRASP_CANDIDATES", "no candidate passed geometry and safety checks")
        return ranked
