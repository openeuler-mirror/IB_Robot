"""Pure helpers shared by pick execution phases."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

from ibrobot_msgs.action import PickObject
from manipulation_execution.geometry import orient_table_plane_upward, transform_point, transform_table_plane
from manipulation_execution.grasp_geometry import quaternion_from_matrix, quaternion_matrix, transform_matrix
from manipulation_execution.pick_executor_models import BaseSceneGeometry, FlowState, PlannerSceneGeometry


class PickExecutorHelpers:
    """Stateless conversions exposed to every pick phase."""

    @staticmethod
    def _load_json_object(raw_value: str) -> dict[str, Any]:
        parsed = json.loads(raw_value or "{}")
        if not isinstance(parsed, dict):
            raise ValueError("expected JSON object")
        return parsed

    @staticmethod
    def _load_json_list(raw_value: str) -> list[str]:
        parsed = json.loads(raw_value or "[]")
        if not isinstance(parsed, list):
            raise ValueError("expected JSON list")
        return [str(item) for item in parsed]

    @staticmethod
    def _copy_joint_state(message: JointState) -> JointState:
        copied = JointState()
        copied.header = message.header
        copied.name = list(message.name)
        copied.position = [float(value) for value in message.position]
        copied.velocity = [float(value) for value in message.velocity]
        copied.effort = [float(value) for value in message.effort]
        return copied

    @staticmethod
    def _remaining(deadline: float) -> float:
        return max(0.0, deadline - time.monotonic())

    @staticmethod
    def _pose(xyz: tuple[float, float, float], quaternion: tuple[float, float, float, float]) -> Pose:
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = (float(value) for value in xyz)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (
            float(value) for value in quaternion
        )
        return pose

    @staticmethod
    def _pose_components(pose: Pose) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        return (
            (float(pose.position.x), float(pose.position.y), float(pose.position.z)),
            (
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ),
        )

    @staticmethod
    def _table_geometry_from_response(response) -> tuple[tuple[float, float, float] | None, float, float]:
        if bool(response.execution_table_plane_found):
            normal_msg = response.execution_table_plane_normal
            offset = float(response.execution_table_plane_offset)
            inlier_ratio = float(response.execution_table_plane_inlier_ratio)
        elif bool(response.table_plane_found):
            normal_msg = response.table_plane_normal
            offset = float(response.table_plane_offset)
            inlier_ratio = float(response.table_plane_inlier_ratio)
        else:
            return None, 0.0, 0.0

        normal = (float(normal_msg.x), float(normal_msg.y), float(normal_msg.z))
        if not all(np.isfinite(value) for value in (*normal, offset, inlier_ratio)):
            return None, 0.0, 0.0
        if np.linalg.norm(normal) <= 1e-9:
            return None, 0.0, 0.0
        return normal, offset, inlier_ratio

    @staticmethod
    def _stamp_to_ns(stamp) -> int:
        if stamp is None:
            return 0
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    @staticmethod
    def _transform_to_matrix(transform) -> np.ndarray:
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return transform_matrix(
            (float(translation.x), float(translation.y), float(translation.z)),
            (float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)),
        )

    @staticmethod
    def _matrix_diagnostic(matrix: np.ndarray) -> dict[str, Any]:
        return {
            "translation_xyz": [float(value) for value in matrix[:3, 3]],
            "quaternion_xyzw": [float(value) for value in quaternion_from_matrix(matrix)],
            "matrix_rowmajor": [float(value) for value in matrix.reshape(-1)],
        }

    @staticmethod
    def _scene_geometry_base(base_to_camera: np.ndarray, scene: PlannerSceneGeometry) -> BaseSceneGeometry:
        table_plane = None
        if scene.table_normal_camera is not None:
            table_plane = orient_table_plane_upward(
                transform_table_plane(
                    base_to_camera,
                    scene.table_normal_camera,
                    scene.table_offset_camera,
                    inlier_ratio=scene.table_inlier_ratio,
                )
            )
        object_top_base = None
        if scene.object_top_camera is not None:
            object_top_base = transform_point(base_to_camera, scene.object_top_camera)
        return BaseSceneGeometry(table_plane=table_plane, object_top_base=object_top_base)

    @staticmethod
    def _joint_position(joint_state: JointState, joint_name: str) -> float | None:
        positions = dict(zip(joint_state.name, joint_state.position, strict=False))
        value = positions.get(joint_name)
        return None if value is None else float(value)

    @staticmethod
    def _joint_state_with_joint5(joint_state: JointState, joint5: float) -> JointState:
        seed = PickExecutorHelpers._copy_joint_state(joint_state)
        seed.position = [
            float(joint5) if str(name) == "5" else float(position)
            for name, position in zip(joint_state.name, joint_state.position, strict=False)
        ]
        return seed

    @staticmethod
    def _contact_for_pose(pose: Pose, contact_ee: tuple[float, float, float]) -> tuple[float, float, float]:
        xyz, quaternion = PickExecutorHelpers._pose_components(pose)
        rotation = quaternion_matrix(quaternion)
        contact = np.asarray(xyz, dtype=np.float64) + rotation @ np.asarray(contact_ee, dtype=np.float64)
        return (float(contact[0]), float(contact[1]), float(contact[2]))

    @staticmethod
    def _result_from_state(state: FlowState) -> PickObject.Result:
        state.finish_active_phase()
        result = PickObject.Result()
        result.attempts = int(state.attempt)
        result.verification_status = int(state.verification_status)
        result.verification_confidence = float(state.verification_confidence)
        result.debug_output_dir = state.debug_output_dir
        result.completed_phases = list(state.completed_phases)
        result.candidate_index = int(state.candidate_index)
        result.released_after_success = bool(state.released_after_success)
        result.pipeline_timings_json = json.dumps(state.pipeline_timings, sort_keys=True)
        return result
