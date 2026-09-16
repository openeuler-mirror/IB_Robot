"""Interactive calibration checker helpers for the SO-101 follower arm."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from so101_hardware.calibration.constants import DEFAULT_SERIAL_PORT, FOLLOWER_CALIB_FILE

DEFAULT_FOLLOWER_PORT = DEFAULT_SERIAL_PORT
DEFAULT_FOLLOWER_CALIB_FILE = FOLLOWER_CALIB_FILE
DEFAULT_LEGACY_ROBOT_TYPE = "so101_follower"

FOLLOWER_JOINTS = {
    "1": {"id": 1, "model": "sts3215", "mode": "DEGREES"},
    "2": {"id": 2, "model": "sts3215", "mode": "DEGREES"},
    "3": {"id": 3, "model": "sts3215", "mode": "DEGREES"},
    "4": {"id": 4, "model": "sts3215", "mode": "DEGREES"},
    "5": {"id": 5, "model": "sts3215", "mode": "DEGREES"},
    "6": {"id": 6, "model": "sts3215", "mode": "RANGE_0_100"},
}


@dataclass(frozen=True)
class CheckStep:
    prompt: str
    updates: dict[str, float]


class CheckerBackend(Protocol):
    """Backend abstraction shared by hardware and simulation modes."""

    def connect(self) -> None:
        """Prepare resources."""

    def disconnect(self) -> None:
        """Release resources."""

    def get_current_action(self) -> dict[str, float]:
        """Return the current checker action in checker-native units."""

    def send_action(self, action: Mapping[str, float]) -> None:
        """Send a checker action to the target backend."""


def default_lerobot_calibration_dir(robot_type: str = DEFAULT_LEGACY_ROBOT_TYPE) -> Path:
    """Return the legacy LeRobot calibration directory for a robot type."""
    hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    lerobot_home = Path(os.getenv("HF_LEROBOT_HOME", hf_home / "lerobot"))
    calibration_root = Path(
        os.getenv("HF_LEROBOT_CALIBRATION", lerobot_home / "calibration"),
    )
    return calibration_root.expanduser() / "robots" / robot_type


def resolve_checker_calibration_file(
    calib_file: str | Path | None = None,
    robot_id: str | None = None,
    calibration_dir: str | Path | None = None,
    robot_type: str = DEFAULT_LEGACY_ROBOT_TYPE,
) -> Path:
    """Resolve the calibration file path with legacy LeRobot compatibility."""
    if calib_file is not None:
        return Path(calib_file).expanduser()

    if robot_id:
        if calibration_dir is not None:
            calibration_base = Path(calibration_dir).expanduser()
            return calibration_base / f"{robot_id}.json"

        legacy_path = default_lerobot_calibration_dir(robot_type) / f"{robot_id}.json"
        if legacy_path.is_file():
            return legacy_path

    return FOLLOWER_CALIB_FILE


def build_default_check_steps() -> list[CheckStep]:
    """Build the default interactive calibration verification flow."""
    return [
        CheckStep("<<按回车机械臂到30度测量动作>> ", {"1": 90, "2": -60, "3": 60, "4": 60}),
        CheckStep("<<按回车机械臂到60度测量动作>> ", {"1": 60, "2": -30, "3": 30, "4": 30}),
        CheckStep("<<按回车机械臂到90度测量动作>> ", {"1": 30, "2": 0, "3": 0, "4": 0}),
        CheckStep("<<按回车机械臂到0位>> ", {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}),
        CheckStep("<<按回车wrist_roll关节转动,机械臂镜头在正下方>> ", {"5": 90}),
        CheckStep("<<按回车wrist_roll关节转动,机械臂镜头在正上方>> ", {"5": -90}),
        CheckStep("<<按回车夹抓完全打开>> ", {"5": 0, "6": 100}),
        CheckStep("<<按回车结束,机械臂回到最初位置>> ", {"1": 0, "2": -50, "3": 40, "4": 30, "5": 0, "6": 0}),
        CheckStep("<<按回车完成收臂动作>> ", {"1": 0, "2": -95, "3": 85, "4": 65, "5": 0, "6": 0}),
    ]


def apply_joint_updates(
    current_action: Mapping[str, float],
    updates: Mapping[str, float],
) -> dict[str, float]:
    """Return a new action dict with updates applied."""
    merged = dict(current_action)
    merged.update(updates)
    return merged


class HardwareArmCalibrationBackend:
    """Hardware backend that talks to the Feetech follower arm."""

    def __init__(
        self,
        port: str = DEFAULT_FOLLOWER_PORT,
        calib_file: str | Path = DEFAULT_FOLLOWER_CALIB_FILE,
    ):
        self.port = port
        self.calib_file = Path(calib_file).expanduser()
        self.bus = None

    def connect(self) -> None:
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

        from so101_hardware.calibration.interactive import load_calibration

        motors = {
            joint_name: Motor(defn["id"], defn["model"], MotorNormMode[defn["mode"]])
            for joint_name, defn in FOLLOWER_JOINTS.items()
        }
        self.bus = FeetechMotorsBus(self.port, motors)
        self.bus.connect()

        calibration = load_calibration(self.calib_file, list(FOLLOWER_JOINTS.keys()))
        self.bus.write_calibration(calibration)
        self.bus.disable_torque()
        for joint_name in self.bus.motors:
            self.bus.write("Operating_Mode", joint_name, OperatingMode.POSITION.value)
        self.bus.enable_torque()

    def disconnect(self) -> None:
        if self.bus is not None:
            self.bus.disconnect()
            self.bus = None

    def get_current_action(self) -> dict[str, float]:
        if self.bus is None:
            raise RuntimeError("HardwareArmCalibrationBackend is not connected.")
        action = self.bus.sync_read("Present_Position")
        return {joint_name: float(value) for joint_name, value in action.items()}

    def send_action(self, action: Mapping[str, float]) -> None:
        if self.bus is None:
            raise RuntimeError("HardwareArmCalibrationBackend is not connected.")
        self.bus.sync_write("Goal_Position", dict(action))


class ArmCalibrationChecker:
    """Interactive checker that replays a fixed set of follower arm poses."""

    def __init__(
        self,
        port: str = DEFAULT_FOLLOWER_PORT,
        calib_file: str | Path = DEFAULT_FOLLOWER_CALIB_FILE,
    ):
        self.backend: CheckerBackend = HardwareArmCalibrationBackend(
            port=port,
            calib_file=calib_file,
        )

    def connect(self) -> None:
        self.backend.connect()

    def disconnect(self) -> None:
        self.backend.disconnect()

    def get_current_action(self) -> dict[str, float]:
        return self.backend.get_current_action()

    def send_action(self, action: Mapping[str, float]) -> None:
        self.backend.send_action(action)

    def run(
        self,
        input_func: Callable[[str], str] = input,
        sleep_func: Callable[[float], None] | None = None,
    ) -> None:
        if sleep_func is None:
            import time

            sleep_func = time.sleep

        current_action = self.get_current_action()
        steps = build_default_check_steps()
        last_index = len(steps) - 1

        for index, step in enumerate(steps):
            input_func(f"\n{step.prompt}")
            current_action = apply_joint_updates(current_action, step.updates)
            self.send_action(current_action)
            sleep_func(1.0 if index == last_index else 0.3)
