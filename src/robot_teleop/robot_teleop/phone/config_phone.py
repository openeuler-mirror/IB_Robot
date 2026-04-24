"""
Phone teleoperation device configuration.

Defines configuration options for iOS and Android phone-based teleoperation.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np


class PhoneOS(Enum):
    """Supported phone operating systems."""
    ANDROID = "android"
    IOS = "ios"


@dataclass
class PhoneConfig:
    """
    Configuration for phone-based teleoperation.

    Attributes:
        phone_os: The operating system of the phone (iOS or Android)
        camera_offset: Offset from camera to phone physical center [x, y, z] in meters
        linear_scale: Linear displacement amplification factor
        end_effector_bounds: Workspace bounds for end-effector position
        max_ee_step_m: Maximum allowed end-effector step in meters
        max_angular_step_rad: Maximum allowed angular step per control cycle in radians
        gripper_speed_factor: Speed factor for gripper velocity integration
        gripper_range: Min and max gripper positions
    """
    phone_os: PhoneOS = PhoneOS.IOS
    camera_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, -0.02, 0.04]))
    linear_scale: float = 2.0
    end_effector_bounds: Dict[str, list] = field(
        default_factory=lambda: {"min": [-0.5, -0.5, 0.0], "max": [0.5, 0.5, 0.5]}
    )
    max_ee_step_m: float = 0.05
    max_angular_step_rad: float = 0.1
    gripper_speed_factor: float = 20.0
    gripper_range: tuple = (0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for ROS2 parameter."""
        return {
            "phone_os": self.phone_os.value,
            "camera_offset": self.camera_offset.tolist(),
            "linear_scale": self.linear_scale,
            "end_effector_bounds": self.end_effector_bounds,
            "max_ee_step_m": self.max_ee_step_m,
            "max_angular_step_rad": self.max_angular_step_rad,
            "gripper_speed_factor": self.gripper_speed_factor,
            "gripper_range": list(self.gripper_range),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhoneConfig":
        """Create config from dictionary."""
        return cls(
            phone_os=PhoneOS(data.get("phone_os", "ios")),
            camera_offset=np.array(data.get("camera_offset", [0.0, -0.02, 0.04])),
            linear_scale=data.get("linear_scale", 2.0),
            end_effector_bounds=data.get(
                "end_effector_bounds", {"min": [-0.5, -0.5, 0.0], "max": [0.5, 0.5, 0.5]}
            ),
            max_ee_step_m=data.get("max_ee_step_m", 0.05),
            max_angular_step_rad=data.get("max_angular_step_rad", 0.1),
            gripper_speed_factor=data.get("gripper_speed_factor", 20.0),
            gripper_range=tuple(data.get("gripper_range", [0.0, 1.0])),
        )
