"""Read-only SDK acquisition; provisioning is a separate operator operation."""

import math
from pathlib import Path


class LeaderInput:
    def __init__(self, config, sdk=None):
        if type(config.get("calibration_version")) is not int or config["calibration_version"] != 1:
            raise ValueError("explicit provisioned calibration_version: 1 is required")
        self.config = dict(config)
        self.names = list(config["joint_order"])
        self.gripper = config["gripper_joint"]
        if not self.names or len(set(self.names)) != len(self.names) or self.gripper not in self.names:
            raise ValueError("leader joint_order/gripper_joint are incomplete or ambiguous")
        self.sdk = sdk
        self.leader = None

    def connect(self):
        if not Path(self.config["calibration_file"]).expanduser().is_file():
            raise ValueError("leader calibration missing; run explicit calibrate_arm provisioning first")
        if self.sdk is None:
            import so101_sdk_py

            self.sdk = so101_sdk_py
        config = self.sdk.LeaderConfig()
        for key in (
            "port",
            "baudrate",
            "calibration_version",
            "joint_order",
            "gripper_joint",
            "motor_ids",
            "simulated",
        ):
            if key in self.config:
                setattr(config, key, self.config[key])
        config.calibration_file = str(Path(self.config["calibration_file"]).expanduser())
        self.leader = self.sdk.LeaderArm(config)
        if not self.leader.connect():
            detail = self.leader.health().detail
            self.disconnect()
            raise ConnectionError(f"leader provisioning verification failed: {detail}")

    def read(self):
        if self.leader is None:
            return None
        try:
            state = self.leader.read()
            if state is None or set(state) != set(self.names):
                return None
            values = {name: float(state[name]["position"]) for name in self.names}
            if not all(math.isfinite(value) for value in values.values()):
                return None
            if not 0.0 <= values[self.gripper] <= 1.0:
                return None
            return values
        except (KeyError, TypeError, ValueError, RuntimeError, OSError):
            return None

    def disconnect(self):
        if self.leader is not None:
            try:
                self.leader.disconnect()
            finally:
                self.leader = None
