"""Validation for the minimal robot-level placement execution contract."""

from __future__ import annotations

import math
from typing import Any


def _positive(value: Any) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0.0
    except (TypeError, ValueError, OverflowError):
        return False


def _finite_number(value: Any) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _unit_interval(value: Any) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0
    except (TypeError, ValueError, OverflowError):
        return False


def validate_placement_execution_config(value: Any) -> list[str]:
    """Return fail-closed schema/range errors for ``placement_execution``."""
    # An absent block and an empty one mean the same thing: the robot does not
    # configure placement execution. RobotConfig defaults this field to {}, not
    # None, so treating only None as absent made every config that never opted in
    # fail with ten missing-field errors.
    if value is None or value == {}:
        return []
    prefix = "placement_execution"
    if not isinstance(value, dict):
        return [f"{prefix} must be a mapping"]
    errors: list[str] = []
    allowed_keys = {
        "enabled",
        "executor",
        "required_args",
        "action_name",
        "detect_service",
        "segment_service",
        "rgb_topic",
        "joint_state_topic",
        "debug_output_root",
        "motion",
        "sensor",
        "verification",
    }
    unknown = sorted(set(value) - allowed_keys)
    if unknown:
        errors.append(f"{prefix} contains unsupported key(s): {', '.join(unknown)}")
    if not isinstance(value.get("enabled"), bool):
        errors.append(f"{prefix}.enabled must be a boolean")
    if value.get("enabled") is True:
        if value.get("executor") != "placement_pipeline":
            errors.append(f"{prefix}.executor must be placement_pipeline")
        if value.get("required_args") != ["target_name", "container_name"]:
            errors.append(f"{prefix}.required_args must equal [target_name, container_name]")
    for key in ("action_name", "detect_service", "rgb_topic", "joint_state_topic"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            errors.append(f"{prefix}.{key} must be a non-empty string")
    if not isinstance(value.get("debug_output_root"), str) or not value["debug_output_root"].strip():
        errors.append(f"{prefix}.debug_output_root must be a non-empty string")
    if not isinstance(value.get("segment_service"), str):
        errors.append(f"{prefix}.segment_service must be a string")

    motion = value.get("motion")
    if not isinstance(motion, dict):
        errors.append(f"{prefix}.motion must be a mapping")
    else:
        allowed_motion = {
            "place_pose",
            "place_joint_names",
            "place_joint_positions",
            "place_duration_sec",
            "post_release",
        }
        unknown_motion = sorted(set(motion) - allowed_motion)
        if unknown_motion:
            errors.append(f"{prefix}.motion contains unsupported key(s): {', '.join(unknown_motion)}")
        if not isinstance(motion.get("place_pose"), str) or not motion["place_pose"].strip():
            errors.append(f"{prefix}.motion.place_pose must be a non-empty string")
        joint_names = motion.get("place_joint_names")
        if (
            not isinstance(joint_names, list)
            or not joint_names
            or any(not isinstance(name, str) or not name.strip() for name in joint_names)
        ):
            errors.append(f"{prefix}.motion.place_joint_names must be a non-empty string list")
        elif len(set(joint_names)) != len(joint_names):
            errors.append(f"{prefix}.motion.place_joint_names must not contain duplicates")
        joint_positions = motion.get("place_joint_positions")
        if not isinstance(joint_positions, dict):
            errors.append(f"{prefix}.motion.place_joint_positions must be a mapping")
        else:
            invalid_names = [name for name in joint_positions if not isinstance(name, str) or not name.strip()]
            if invalid_names:
                errors.append(f"{prefix}.motion.place_joint_positions keys must be non-empty strings")
            if isinstance(joint_names, list) and set(joint_positions) != set(joint_names):
                errors.append(f"{prefix}.motion.place_joint_positions keys must exactly match place_joint_names")
            for name, position in joint_positions.items():
                if not _finite_number(position):
                    errors.append(f"{prefix}.motion.place_joint_positions.{name} must be finite")
        if not _positive(motion.get("place_duration_sec")):
            errors.append(f"{prefix}.motion.place_duration_sec must be finite and positive")
        post_release = motion.get("post_release")
        if not isinstance(post_release, dict):
            errors.append(f"{prefix}.motion.post_release must be a mapping")
        else:
            unknown_post_release = sorted(
                set(post_release)
                - {
                    "verify_joint_name",
                    "verify_joint_position",
                    "verify_duration_sec",
                    "return_duration_sec",
                }
            )
            if unknown_post_release:
                errors.append(
                    f"{prefix}.motion.post_release contains unsupported key(s): {', '.join(unknown_post_release)}"
                )
            verify_joint_name = post_release.get("verify_joint_name")
            if not isinstance(verify_joint_name, str) or not verify_joint_name.strip():
                errors.append(f"{prefix}.motion.post_release.verify_joint_name must be a non-empty string")
            elif isinstance(joint_names, list) and verify_joint_name not in joint_names:
                errors.append(f"{prefix}.motion.post_release.verify_joint_name must be listed in place_joint_names")
            if not _finite_number(post_release.get("verify_joint_position")):
                errors.append(f"{prefix}.motion.post_release.verify_joint_position must be finite")
            for key in ("verify_duration_sec", "return_duration_sec"):
                if not _positive(post_release.get(key)):
                    errors.append(f"{prefix}.motion.post_release.{key} must be finite and positive")

    sensor = value.get("sensor")
    if not isinstance(sensor, dict):
        errors.append(f"{prefix}.sensor must be a mapping")
    else:
        unknown_sensor = sorted(set(sensor) - {"gripper_feedback_timeout_sec"})
        if unknown_sensor:
            errors.append(f"{prefix}.sensor contains unsupported key(s): {', '.join(unknown_sensor)}")
        if not _positive(sensor.get("gripper_feedback_timeout_sec")):
            errors.append(f"{prefix}.sensor.gripper_feedback_timeout_sec must be finite and positive")

    verification = value.get("verification")
    if not isinstance(verification, dict):
        errors.append(f"{prefix}.verification must be a mapping")
        return errors
    allowed_verification = {
        "required",
        "confidence_threshold",
        "post_release_wait_sec",
        "max_resamples",
        "resample_interval_sec",
        "required_confirmations",
        "min_container_mask_pixels",
        "min_target_mask_pixels",
        "min_inside_mask_fraction",
        "container_inset_ratio",
        "target_exclusion",
    }
    unknown_verification = sorted(set(verification) - allowed_verification)
    if unknown_verification:
        errors.append(f"{prefix}.verification contains unsupported key(s): {', '.join(unknown_verification)}")
    if verification.get("required") is not True:
        errors.append(f"{prefix}.verification.required must be true")
    for key in ("confidence_threshold", "min_inside_mask_fraction"):
        if not _unit_interval(verification.get(key)):
            errors.append(f"{prefix}.verification.{key} must be in [0, 1]")
    inset_ratio = verification.get("container_inset_ratio")
    if not _unit_interval(inset_ratio) or float(inset_ratio) >= 0.5:
        errors.append(f"{prefix}.verification.container_inset_ratio must be in [0, 0.5)")
    if not _positive(verification.get("post_release_wait_sec")):
        errors.append(f"{prefix}.verification.post_release_wait_sec must be finite and positive")
    if not _positive(verification.get("resample_interval_sec")):
        errors.append(f"{prefix}.verification.resample_interval_sec must be finite and positive")
    if not isinstance(verification.get("max_resamples"), int) or verification["max_resamples"] < 0:
        errors.append(f"{prefix}.verification.max_resamples must be a non-negative integer")
    for key in ("required_confirmations", "min_container_mask_pixels", "min_target_mask_pixels"):
        if not isinstance(verification.get(key), int) or verification[key] <= 0:
            errors.append(f"{prefix}.verification.{key} must be a positive integer")
    confirmations = verification.get("required_confirmations")
    resamples = verification.get("max_resamples")
    if isinstance(confirmations, int) and isinstance(resamples, int) and confirmations > resamples + 1:
        errors.append(f"{prefix}.verification.required_confirmations cannot exceed max_resamples + 1")

    exclusion = verification.get("target_exclusion")
    if not isinstance(exclusion, dict):
        errors.append(f"{prefix}.verification.target_exclusion must be a mapping")
    else:
        unknown_exclusion = sorted(set(exclusion) - {"enabled", "mask_path", "polygons", "min_detection_overlap"})
        if unknown_exclusion:
            errors.append(
                f"{prefix}.verification.target_exclusion contains unsupported key(s): {', '.join(unknown_exclusion)}"
            )
        if not isinstance(exclusion.get("enabled"), bool):
            errors.append(f"{prefix}.verification.target_exclusion.enabled must be a boolean")
        if not isinstance(exclusion.get("mask_path", ""), str):
            errors.append(f"{prefix}.verification.target_exclusion.mask_path must be a string")
        polygons = exclusion.get("polygons")
        mask_path = str(exclusion.get("mask_path", "")).strip()
        if not isinstance(polygons, list) or (not polygons and not mask_path):
            errors.append(
                f"{prefix}.verification.target_exclusion.polygons must be a non-empty list when mask_path is empty"
            )
        elif not isinstance(polygons, list):
            pass
        else:
            for polygon_index, polygon in enumerate(polygons):
                if not isinstance(polygon, list) or len(polygon) < 3:
                    errors.append(
                        f"{prefix}.verification.target_exclusion.polygons[{polygon_index}] must have at least 3 points"
                    )
                    continue
                for point_index, point in enumerate(polygon):
                    if (
                        not isinstance(point, list | tuple)
                        or len(point) != 2
                        or any(not _unit_interval(coordinate) for coordinate in point)
                    ):
                        errors.append(
                            f"{prefix}.verification.target_exclusion.polygons[{polygon_index}][{point_index}] "
                            "must be a [x, y] pair in [0, 1]"
                        )
        if not _unit_interval(exclusion.get("min_detection_overlap")):
            errors.append(f"{prefix}.verification.target_exclusion.min_detection_overlap must be in [0, 1]")
    return errors
