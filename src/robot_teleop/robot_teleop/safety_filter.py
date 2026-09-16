"""
Safety filter for joint limit enforcement

Ensures all teleoperation commands stay within safe joint limits
before being published to controllers.
"""

import math


class SafetyFilter:
    """
    Enforces joint limits on teleoperation commands.

    The safety filter clips joint target values to their configured limits,
    providing defense-in-depth safety for the robot.

    Attributes:
        joint_limits (Dict): Dictionary of joint limits {joint_name: {"min": float, "max": float}}
    """

    def __init__(self, joint_limits: dict[str, dict[str, float]]):
        """
        Initialize safety filter with joint limits.

        Args:
            joint_limits: Joint limits from robot_config
                Example: {
                    "1": {"min": -3.14, "max": 3.14},
                    "2": {"min": -1.57, "max": 1.57},
                }
        """
        self.joint_limits = joint_limits
        self._clip_count = {}  # Track clipping frequency per joint
        for name, limits in joint_limits.items():
            lower, upper = limits.get("min", -math.inf), limits.get("max", math.inf)
            if math.isnan(lower) or math.isnan(upper) or lower > upper or lower == math.inf or upper == -math.inf:
                raise ValueError(f"Invalid joint limits for {name!r}")

    def apply_limits(self, joint_targets: dict[str, float]) -> dict[str, float]:
        """
        Apply joint limits to target positions.

        Clips each joint target to its configured [min, max] range.
        Logs a warning when clipping occurs.

        Args:
            joint_targets: Dictionary of {joint_name: target_angle} in radians

        Returns:
            Dict[str, float]: Safe joint targets within limits

        Example:
            >>> filter = SafetyFilter({"1": {"min": -1.0, "max": 1.0}})
            >>> filter.apply_limits({"1": 1.5, "2": 0.5})
            {"1": 1.0, "2": 0.5}  # Joint "1" clipped to max
        """
        safe_targets = {}

        for joint_name, target_angle in joint_targets.items():
            # Reject the entire command, including unlimited joints, on corrupt input.
            if not math.isfinite(target_angle):
                return {}
            # Get limits for this joint (default to no limit if not specified)
            if joint_name in self.joint_limits:
                limits = self.joint_limits[joint_name]
                min_limit = limits.get("min", -math.inf)
                max_limit = limits.get("max", math.inf)

                # Clip to limits
                original_angle = target_angle
                safe_angle = float(min(max(target_angle, min_limit), max_limit))
                safe_targets[joint_name] = safe_angle

                # Log if clipping occurred
                # Match numpy.isclose's asymmetric default rtol without scalar array work.
                if abs(original_angle - safe_angle) > 1e-6 + 1e-5 * abs(safe_angle):
                    self._log_clip(joint_name, original_angle, safe_angle, min_limit, max_limit)
            else:
                # No limits defined - pass through
                safe_targets[joint_name] = target_angle

        return safe_targets

    def _log_clip(self, joint_name: str, original: float, clipped: float, min_limit: float, max_limit: float):
        """
        Log joint clipping event.

        Args:
            joint_name: Name of the clipped joint
            original: Original target value (radians)
            clipped: Clipped value (radians)
            min_limit: Minimum joint limit
            max_limit: Maximum joint limit
        """
        # Track clip count
        if joint_name not in self._clip_count:
            self._clip_count[joint_name] = 0
        self._clip_count[joint_name] += 1

        # Log warning (rate-limited to avoid spam)
        if self._clip_count[joint_name] <= 3 or self._clip_count[joint_name] % 100 == 0:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Joint '{joint_name}' clipped: {original:.3f} → {clipped:.3f} "
                f"(limits: [{min_limit:.3f}, {max_limit:.3f}]) "
                f"[clip count: {self._clip_count[joint_name]}]"
            )

    def get_clip_statistics(self) -> dict[str, int]:
        """
        Get statistics on joint clipping frequency.

        Returns:
            Dict mapping joint names to clip counts
        """
        return self._clip_count.copy()

    def reset_statistics(self):
        """Reset clip statistics counter."""
        self._clip_count.clear()
