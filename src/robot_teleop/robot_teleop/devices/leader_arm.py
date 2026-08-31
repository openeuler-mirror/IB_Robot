"""
SO-101 Leader Arm teleoperation device

Implements teleoperation via SO-101 leader arm hardware,
reading joint positions from serial port and applying calibration.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any

from ..base_teleop import BaseTeleopDevice


class LeaderArmDevice(BaseTeleopDevice):
    """
    SO-101 leader arm teleoperation device.

    Reads joint positions from SO-101 leader arm via serial port,
    applies calibration offsets, and maps to follower joint names.
    """

    def __init__(self, config: dict, node=None):
        super().__init__(config, node=node)

        # Configuration
        self.port = config.get("port", "/dev/ttyACM1")
        calib_str = config.get("calib_file", "")
        self.calib_file = Path(calib_str).expanduser() if calib_str else None
        self.joint_mapping = config.get("joint_mapping", {})
        self.gripper_joints = set(config.get("gripper_joint_names", ["6"]))
        # Follower calibration: maps the leader gripper 0~1 percentage onto the
        # follower gripper's physical radian stroke so the command unit matches
        # the follower's radian position controller. Without it the 0~1 value is
        # executed as radians, leaving the gripper near tick 2048 short of its
        # mechanical limit, i.e. only closing about halfway.
        follower_calib_str = config.get("follower_calib_file", "")
        self.follower_calib_file = Path(follower_calib_str).expanduser() if follower_calib_str else None

        # Physical constants (4096 steps per 360 degrees)
        self.rad_per_step = (2 * math.pi) / 4096.0

        # Hardware interface
        self.motors_bus = None
        self.calibration = None
        # Follower gripper radian stroke [rad_min, rad_max]; loaded in connect().
        # When None, the gripper target falls back to the legacy 0~1 behavior.
        self.gripper_rad_min = None
        self.gripper_rad_max = None
        # Joint definitions (SO-101 leader arm)
        self.joints = {
            "1": {"id": 1, "model": "sts3215"},
            "2": {"id": 2, "model": "sts3215"},
            "3": {"id": 3, "model": "sts3215"},
            "4": {"id": 4, "model": "sts3215"},
            "5": {"id": 5, "model": "sts3215"},
            "6": {"id": 6, "model": "sts3215"},
        }
        self.joint_names = list(self.joints.keys())
        self.logger = logging.getLogger(__name__)

    def connect(self) -> bool:
        try:
            from lerobot.motors import Motor, MotorNormMode
            from lerobot.motors.feetech import FeetechMotorsBus

            motors = {}
            for joint_name, joint_info in self.joints.items():
                norm_mode = (
                    MotorNormMode.RANGE_0_100 if self._is_gripper_joint(joint_name) else MotorNormMode.RANGE_M100_100
                )
                motors[joint_name] = Motor(
                    id=joint_info["id"],
                    model=joint_info["model"],
                    norm_mode=norm_mode,
                )

            self.motors_bus = FeetechMotorsBus(port=self.port, motors=motors)
            # Connect to motors
            self.motors_bus.connect()
            self.logger.info(f"Motors bus connected on {self.port}")

            # Load and write calibration to firmware (CRITICAL for Feetech coordinate system)
            if self.calib_file and self.calib_file.exists():
                self.calibration = self._load_calibration()
                self.logger.info(f"Loaded calibration from {self.calib_file}")

                self.logger.info("Writing calibration to motor firmware...")
                self.motors_bus.write_calibration(self.calibration)
                self.logger.info("Calibration written to firmware. Motors will now output ~2048 at physical zero.")
            else:
                self.logger.warning("No calibration file found, using raw encoder positions")
                self.calibration = None

            # Resolve the follower gripper radian stroke used to emit radian
            # commands. Falls back silently to legacy 0~1 behavior when the
            # follower calibration is unavailable.
            self._load_follower_gripper_stroke()

            self._is_connected = True
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect leader arm: {e}")
            self._is_connected = False
            raise ConnectionError(f"Cannot connect to leader arm on {self.port}: {e}") from e

    def get_joint_targets(self) -> dict[str, float]:
        if not self._is_connected or self.motors_bus is None:
            return {}

        try:
            # Read absolute positions. Because we wrote calibration to firmware,
            # the motors will output 2048 when they are at their home physical position.
            raw_positions = self.motors_bus.sync_read("Present_Position", normalize=False)

            joint_targets = {}
            for name in self.joint_names:
                if name not in raw_positions:
                    continue

                raw = raw_positions[name]
                follower_joint = self._map_joint(name)

                if self._is_gripper_joint(name):
                    gripper_target = self._normalize_gripper_target(name, raw)
                    if gripper_target is None:
                        self.logger.warning(
                            f"Gripper joint '{name}' calibration unavailable or degenerate; "
                            "skipping publish to avoid bad radians target"
                        )
                        continue
                    # Map the leader 0~1 percentage onto the follower gripper's
                    # physical radian stroke so the command matches the follower's
                    # radian position controller. Falls back to the legacy 0~1
                    # value when the follower stroke is unknown (backward compat).
                    if self.gripper_rad_min is not None and self.gripper_rad_max is not None:
                        gripper_target = self.gripper_rad_min + gripper_target * (
                            self.gripper_rad_max - self.gripper_rad_min
                        )
                    joint_targets[follower_joint] = gripper_target
                else:
                    # EXACTLY matching so101_hardware.cpp logic:
                    # rad = (raw - 2048.0) * rad_per_step
                    joint_targets[follower_joint] = (raw - 2048.0) * self.rad_per_step

            return joint_targets

        except Exception as e:
            self.logger.error(f"Failed to read leader arm positions: {e}")
            return {}

    def _gripper_calibration_keys(self) -> list[str]:
        """Candidate keys for the gripper entry in the follower calibration.

        ``gripper_joint_names`` may hold either the leader joint id ("6") or the
        mapped follower name ("joint6_left") — ``_is_gripper_joint`` accepts both
        — while the follower calibration is always keyed by joint id. Trying the
        reverse mapping too keeps a follower-name configuration from silently
        falling back to the 0~1 command this method exists to replace.
        """
        candidates: list[str] = []
        for name in sorted(self.gripper_joints):
            candidates.append(name)
            candidates.extend(leader_id for leader_id, mapped in self.joint_mapping.items() if mapped == name)
        candidates.append("6")
        seen: set[str] = set()
        return [key for key in candidates if not (key in seen or seen.add(key))]

    def _find_gripper_calibration(self, follower_calib: dict) -> tuple[str, dict | None]:
        """Return the first candidate key that carries a usable gripper stroke."""
        keys = self._gripper_calibration_keys()
        for key in keys:
            entry = follower_calib.get(key)
            if isinstance(entry, dict) and "range_min" in entry and "range_max" in entry:
                return key, entry
        return keys[0], None

    def _load_follower_gripper_stroke(self) -> None:
        """Resolve the follower gripper radian stroke from follower calibration.

        Reads the follower calibration file and converts the gripper joint's
        ``range_min``/``range_max`` ticks into radians using the same formula as
        the follower hardware (``(ticks - 2048) / TICKS_PER_RAD``). The stroke
        is then used by ``get_joint_targets`` to map the leader gripper 0~1
        percentage onto the follower's physical radian range so the command
        unit matches the follower's radian position controller.

        When the follower calibration is not configured or the gripper entry is
        missing, the stroke stays ``None`` and the gripper command falls back
        to the legacy 0~1 value (the original half-close behavior), keeping the
        change backward compatible.
        """
        if not self.follower_calib_file or not self.follower_calib_file.exists():
            self.logger.warning(
                "No follower_calib_file configured; gripper command stays 0~1 and "
                "the follower will execute it as radians, closing only about halfway. "
                "Set follower_calib_file to the follower calibration JSON to enable "
                "the radian mapping."
            )
            return

        try:
            with self.follower_calib_file.open("r", encoding="utf-8") as f:
                follower_calib = json.load(f)
            gripper_joint, entry = self._find_gripper_calibration(follower_calib)
            if entry is None:
                self.logger.warning(
                    f"Follower calibration '{self.follower_calib_file}' has no "
                    f"range_min/range_max for gripper joint '{gripper_joint}'; "
                    "gripper command stays 0~1"
                )
                return

            ticks_per_rad = 4096.0 / (2 * math.pi)
            self.gripper_rad_min = (int(entry["range_min"]) - 2048.0) / ticks_per_rad
            self.gripper_rad_max = (int(entry["range_max"]) - 2048.0) / ticks_per_rad
            self.logger.info(
                f"Follower gripper stroke: [{self.gripper_rad_min:.3f}, "
                f"{self.gripper_rad_max:.3f}] rad "
                f"(from {self.follower_calib_file.name})"
            )
        except (OSError, ValueError, KeyError) as exc:
            self.logger.warning(
                f"Failed to load follower gripper stroke from {self.follower_calib_file}: {exc}; "
                "gripper command stays 0~1"
            )

    def get_gripper_limits(self) -> dict[str, dict[str, float]]:
        """Return radian safety limits for gripper joints, from follower calibration.

        Overrides the base no-op so the TeleopNode can keep the gripper safety
        limits in lockstep with the follower's physical radian stroke (the same
        ``[rad_min, rad_max]`` used to map the leader's 0~1 percentage). This way
        swapping or re-calibrating the follower needs no ``leader_teleop_params``
        YAML edit: the limits follow the follower calibration file automatically.

        Returns:
            Empty dict when the follower stroke is unknown (no override; the
            YAML limits stand as-is, preserving backward compatibility).
        """
        if self.gripper_rad_min is None or self.gripper_rad_max is None:
            return {}
        gripper_joint = next(iter(self.gripper_joints), "6") if self.gripper_joints else "6"
        return {gripper_joint: {"min": self.gripper_rad_min, "max": self.gripper_rad_max}}

    def disconnect(self):
        if self.motors_bus is not None:
            try:
                self.motors_bus.disconnect()
                self.logger.info("Leader arm disconnected")
            except Exception:
                pass
            finally:
                self.motors_bus = None
                self._is_connected = False

    def _load_calibration(self) -> dict[str, Any]:
        from so101_hardware.calibration.interactive import load_calibration as load_calib_so101

        return load_calib_so101(self.calib_file, self.joint_names, self.logger)

    def _normalize_gripper_target(self, joint_name: str, raw: float) -> float | None:
        if not self.calibration or joint_name not in self.calibration:
            return None

        calib = self.calibration[joint_name]
        range_min = calib.range_min
        range_max = calib.range_max
        if range_max == range_min:
            return None

        bounded = min(range_max, max(range_min, raw))
        norm = (bounded - range_min) / (range_max - range_min)
        return 1.0 - norm if calib.drive_mode else norm

    def _is_gripper_joint(self, leader_joint: str) -> bool:
        mapped_joint = self._map_joint(leader_joint)
        return leader_joint in self.gripper_joints or mapped_joint in self.gripper_joints

    def _map_joint(self, leader_joint: str) -> str:
        return self.joint_mapping.get(leader_joint, leader_joint)
