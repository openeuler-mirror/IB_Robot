"""
Base teleoperation device interface

Defines the abstract interface that all teleoperation devices must implement.
This enables the "One Node, Many Devices" design pattern where a single
TeleopNode can work with any device implementing this interface.
"""

import logging
from abc import ABC, abstractmethod


class BaseTeleopDevice(ABC):
    """
    Abstract base class for all teleoperation devices.

    All teleoperation devices (leader arms, gamepads, VR controllers, phones) must
    inherit from this class and implement the required abstract methods.

    The class provides a standardized interface for:
    - Device connection management
    - Joint target acquisition
    - Resource cleanup

    Design Pattern: Strategy Pattern
    - TeleopNode is the Context
    - BaseTeleopDevice is the Strategy interface
    - Concrete devices (LeaderArmDevice, PhoneDevice) are Concrete Strategies

    Attributes:
        _is_connected (bool): Internal connection status flag
        _config (dict): Device configuration
        _node: ROS 2 node reference (optional; required by devices that need ROS)
        logger: Python logger for device messages
    """

    def __init__(self, config: dict, node=None):
        """
        Initialize the teleoperation device.

        Args:
            config (dict): Device configuration from robot_config YAML
            node: Optional ROS 2 node instance for creating subscribers/publishers
        """
        self._is_connected = False
        self._config = config
        self._node = node
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def is_connected(self) -> bool:
        """
        Check if the device is currently connected.

        Returns:
            bool: True if device is connected and operational, False otherwise
        """
        return self._is_connected

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the teleoperation device.

        This method should:
        - Open serial ports, Bluetooth connections, or other communication channels
        - Initialize device hardware
        - Verify device is responsive

        Returns:
            bool: True if connection successful, False otherwise

        Raises:
            ConnectionError: If device cannot be reached
        """
        pass

    @abstractmethod
    def get_joint_targets(self) -> dict[str, float]:
        """
        Read current joint targets from the teleoperation device.

        This is the core method called at each control cycle. It should:
        - Read device state (joint positions, button states, poses, etc.)
        - Apply device-specific transformations (calibration, mapping, IK)
        - Return joint-angle mapping in a standardized format

        Device-specific behaviour:
        - Leader Arm: Direct joint mapping from serial readings (returns all joints)
        - Phone: Drives the selected Cartesian backend for arm control; returns only
          gripper target (arm keys absent → TeleopNode skips arm publisher)

        Returns:
            Dict[str, float]: Mapping from joint names to target angles (radians)
                             Example: {"1": 0.5, "2": 1.2, "3": -0.3, ...}

        Note:
            If device is disconnected or read fails, return empty dict {}
            The teleop node will handle the failure gracefully
        """
        pass

    def get_gripper_limits(self) -> dict[str, dict[str, float]]:
        """
        Return radian safety limits for gripper joints, derived from the follower
        calibration so they stay in sync with the follower's physical stroke.

        Devices that know the follower's calibrated stroke (e.g. the SO-101
        leader arm, which reads the follower calibration) override this to
        report the actual ``[rad_min, rad_max]``. The TeleopNode uses these to
        override the static YAML limits for the gripper joints, so changing or
        re-calibrating the follower needs no YAML edit.

        Returns:
            dict[str, dict[str, float]]: Mapping from gripper joint name to
                ``{"min": float, "max": float}`` in radians. Empty by default
                (no override), letting the YAML limits stand as-is.
        """
        return {}

    def get_gripper_stroke(self) -> tuple[float, float] | None:
        """
        Return the follower gripper stroke ``(closed_rad, open_rad)`` or None.

        Ratio-emitting devices that know the follower's calibrated stroke
        override this. TeleopNode uses it as the ratio -> radian endpoint
        source only when neither the device config nor the runtime public
        description provides endpoints (provider-less legacy deployments).
        """
        return None

    @abstractmethod
    def disconnect(self):
        """
        Disconnect from the teleoperation device and release resources.

        This method should:
        - Close communication channels (serial ports, Bluetooth)
        - Release file handles and hardware resources
        - Reset internal state

        Safe to call multiple times.
        """
        pass

    def __enter__(self):
        """Context manager entry - connect to device."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - disconnect from device."""
        self.disconnect()
        return False

    def emergency_stop(self) -> None:
        """Stop device-owned motion immediately when the node receives E-stop."""
        return None

    def emergency_stop_released(self) -> None:
        """Notify the device that E-stop was released and output may re-arm."""
        return None
