"""
Device factory for dynamic teleoperation device instantiation

Provides a factory pattern for creating teleoperation devices based on
configuration, enabling easy extension without modifying core code.
"""

from .base_teleop import BaseTeleopDevice
from .devices.hand_retarget import HandRetargetDevice
from .devices.leader_arm import LeaderArmDevice
from .devices.leader_topic import LeaderTopicDevice
from .devices.xbox_controller import XboxTeleopDevice
from .phone.phone_device import PhoneDevice

# Device registry - add new device types here
DEVICE_MAP: dict[str, type[BaseTeleopDevice]] = {
    "hand_retarget": HandRetargetDevice,
    "leader_arm": LeaderArmDevice,  # SO-101 leader arm
    "leader_topic": LeaderTopicDevice,
    "phone": PhoneDevice,
    "xbox_controller": XboxTeleopDevice,
}


def device_factory(config: dict, node=None) -> BaseTeleopDevice:
    """
    Create a teleoperation device instance based on configuration.

    Args:
        config: Device configuration dictionary with at minimum a 'type' key
        node: Optional ROS 2 node instance

    Returns:
        BaseTeleopDevice: Instantiated device object
    """
    if not config:
        raise ValueError("Device configuration cannot be empty")

    dev_type = config.get("type")
    if not dev_type:
        raise ValueError("Device configuration must include 'type' field")

    if dev_type not in DEVICE_MAP:
        available_types = ", ".join(DEVICE_MAP.keys())
        raise ValueError(f"Unknown teleop device type: '{dev_type}'. Available types: [{available_types}]")

    device_class = DEVICE_MAP[dev_type]
    return device_class(config, node=node)


def register_device(device_type: str, device_class: type[BaseTeleopDevice]) -> None:
    """
    Register a new device type in the global device map.

    This allows external packages to extend the teleop system with custom devices.

    Args:
        device_type: String identifier for the device (e.g., "leader_arm")
        device_class: Class that inherits from BaseTeleopDevice

    Example:
        >>> class CustomDevice(BaseTeleopDevice):
        ...     # ... implementation
        >>> register_device("custom", CustomDevice)
    """
    if device_type in DEVICE_MAP:
        import warnings

        warnings.warn(f"Overwriting existing device type: {device_type}", UserWarning, stacklevel=2)

    DEVICE_MAP[device_type] = device_class
