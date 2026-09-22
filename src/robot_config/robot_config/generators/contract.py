"""Contract extension loader for robot_config integration.

This module extends the rosetta contract system to support peripheral references
from robot_config, eliminating duplication between hardware configuration and ML I/O.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from robot_config.contract_utils import (
    ActionSpec,
    Contract,
    ObservationSpec,
    _as_align,
    contract_from_dict,
)
from robot_config.interface_binding import resolve_robot_interfaces
from robot_config.observation_transport import (
    observation_transport_to_dict,
    parse_observation_transport,
    require_valid_observation_transports,
    resolve_observation_transport,
)

logger = logging.getLogger(__name__)


def load_contract_with_robot_config(
    contract_path: str | Path,
    robot_config: Any | None = None,
) -> Contract:
    """Load contract with robot_config peripheral integration.

    This function loads a rosetta contract and resolves peripheral references
    from the robot_config, eliminating the need to duplicate camera
    configurations in both places.

    Args:
        contract_path: Path to contract YAML file
        robot_config: RobotConfig object from robot_config package

    Returns:
        Contract with resolved peripheral metadata

    Example:
        In contract YAML:
        ```yaml
        observations:
          - key: observation.images.top
            topic: /camera/top
            peripheral: top  # References camera 'top' from robot_config
        ```

        The peripheral metadata (width, height, fps, frame_id) will be
        automatically loaded from robot_config.peripherals.
    """
    # Load base contract
    contract_data = yaml.safe_load(Path(contract_path).read_text(encoding="utf-8")) or {}
    # Standalone contracts must contain resolved endpoints, not unchecked logical IDs.
    contract_data = resolve_robot_interfaces({"contract": contract_data})["contract"]

    # Resolve peripheral references if robot_config is provided
    if robot_config:
        contract_data = _resolve_peripheral_references(contract_data, robot_config)

    return contract_from_dict(contract_data)


def _resolve_peripheral_references(contract_data: dict, robot_config: Any) -> dict:
    """Resolve peripheral references in contract using robot_config.

    For observations that reference a peripheral (by name), this function
    adds peripheral metadata from robot_config to the observation.

    Args:
        contract_data: Contract dictionary
        robot_config: RobotConfig object

    Returns:
        Updated contract dictionary with resolved peripheral metadata
    """
    # Build peripheral lookup
    peripherals = {}
    for cam in robot_config.peripherals:
        peripherals[cam.name] = {
            "type": "camera",
            "driver": cam.driver,
            "width": cam.width,
            "height": cam.height,
            "fps": cam.fps,
            "frame_id": cam.frame_id,
            "optical_frame_id": cam.optical_frame_id,
            "pixel_format": cam.pixel_format,
        }

    # Resolve peripheral references in observations
    for obs in contract_data.get("observations", []):
        peripheral_name = obs.get("peripheral")
        if peripheral_name and peripheral_name in peripherals and obs.get("_interface_source") is None:
            # Add peripheral metadata
            obs["_peripheral"] = peripherals[peripheral_name]

            # Auto-fill image resize if not specified
            if "image" not in obs or obs["image"] is None:
                obs["image"] = {}
            if "resize" not in obs["image"] or obs["image"]["resize"] is None:
                # Default to camera resolution (height, width)
                cam = peripherals[peripheral_name]
                obs["image"]["resize"] = [cam["height"], cam["width"]]

            # Infer encoding from pixel format
            if "encoding" not in obs["image"] or obs["image"]["encoding"] is None:
                pixel_format = peripherals[peripheral_name]["pixel_format"]
                obs["image"]["encoding"] = pixel_format

    return contract_data


def validate_contract_peripheral_consistency(
    contract_data: dict,
    robot_config: Any,
) -> list[str]:
    """Validate that all peripheral references in contract exist in robot_config.

    Args:
        contract_data: Contract dictionary
        robot_config: RobotConfig object

    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    peripheral_names = {cam.name for cam in robot_config.peripherals}

    for obs in contract_data.get("observations", []):
        peripheral_name = obs.get("peripheral")
        if peripheral_name and obs.get("_interface_source") is None and peripheral_name not in peripheral_names:
            errors.append(f"Observation '{obs.get('key')}' references undefined peripheral: {peripheral_name}")

    return errors


def build_contract_from_robot_config_dict(robot_config: dict[str, Any]) -> Contract:
    """Build a runtime ``Contract`` directly from raw robot_config dict data."""
    robot_config = resolve_robot_interfaces(robot_config)

    def _camera_lookup(name: str) -> dict[str, Any] | None:
        for periph in robot_config.get("peripherals", []) or []:
            if not isinstance(periph, dict):
                continue
            if periph.get("type") == "camera" and periph.get("name") == name:
                return periph
        return None

    contract_config = robot_config.get("contract", {}) or {}
    obs_specs: list[ObservationSpec] = []
    for obs in contract_config.get("observations", []) or []:
        if not isinstance(obs, dict):
            continue

        image_meta = obs.get("image")
        topic = str(obs.get("topic", "") or "")
        topic_type = obs.get("type") or "sensor_msgs/msg/JointState"
        peripheral_name = obs.get("peripheral")
        camera = None
        if peripheral_name and obs.get("_interface_source") is None:
            topic_type = "sensor_msgs/msg/Image"
            camera = _camera_lookup(str(peripheral_name))
            if camera is None:
                logger.warning(
                    "Observation '%s' references peripheral '%s' but no camera found",
                    obs.get("key"),
                    peripheral_name,
                )
            if camera and not image_meta:
                cam_h = int(camera.get("height", 0) or 0) or 480
                cam_w = int(camera.get("width", 0) or 0) or 640
                image_meta = {
                    "resize": [cam_h, cam_w],
                    "encoding": camera.get("pixel_format", "bgr8"),
                }
        elif not topic:
            raise ValueError(f"Observation '{obs.get('key', '?')}' must specify a topic when no peripheral is set")

        transport = resolve_observation_transport(
            parse_observation_transport(obs.get("transport")),
            image=image_meta,
            camera_width=int(camera.get("width")) if camera and camera.get("width") is not None else None,
            camera_height=int(camera.get("height")) if camera and camera.get("height") is not None else None,
            camera_fps=float(camera.get("fps")) if camera and camera.get("fps") is not None else None,
            interface_source=obs.get("_interface_source"),
        )
        obs_specs.append(
            ObservationSpec(
                key=obs["key"],
                topic=topic,
                type=topic_type,
                selector=obs.get("selector"),
                image=image_meta,
                align=_as_align(obs.get("align")),
                qos=obs.get("qos"),
                transport=transport,
                _interface_source=obs.get("_interface_source"),
            )
        )

    act_specs: list[ActionSpec] = []
    for act in contract_config.get("actions", []) or []:
        if not isinstance(act, dict):
            continue
        publish = act.get("publish", {}) or {}
        safety_behavior = str(act.get("safety_behavior", "zeros")).lower().strip()
        if safety_behavior not in ("zeros", "hold"):
            safety_behavior = "zeros"
        act_specs.append(
            ActionSpec(
                key=act["key"],
                publish_topic=publish.get("topic", ""),
                type=publish.get("type", ""),
                selector=act.get("selector"),
                from_tensor=act.get("from_tensor"),
                publish_qos=publish.get("qos"),
                publish_strategy=publish.get("strategy"),
                safety_behavior=safety_behavior,
                _interface_source=publish.get("_interface_source"),
            )
        )

    contract = Contract(
        name=str(robot_config.get("name", "contract")),
        version=1,
        rate_hz=float(contract_config.get("rate_hz", 20.0)),
        max_duration_s=float(contract_config.get("max_duration_s", 30.0)),
        observations=obs_specs,
        actions=act_specs,
        tasks=[],
        recording={"storage": "mcap"},
        robot_type=robot_config.get("robot_type", robot_config.get("type")),
        timestamp_source="receive",
        process={},
    )
    require_valid_observation_transports(contract.observations)
    return contract


def generate_contract_from_robot_config(robot_config: Any) -> str:
    """Generate a complete rosetta contract from robot_config.

    This function creates a contract YAML that references peripherals
    defined in robot_config, suitable for use with PolicyBridge and EpisodeRecorder.

    Args:
        robot_config: RobotConfig object

    Returns:
        Contract YAML as string
    """
    contract = {
        "name": robot_config.name,
        "version": 1,
        "robot_type": robot_config.robot_type,
        "rate_hz": robot_config.contract.rate_hz,
        "max_duration_s": robot_config.contract.max_duration_s,
        "observations": [],
        "actions": [],
        "recording": {"storage": "mcap"},
    }

    # Generate observations from contract extension config
    for obs in robot_config.contract.observations:
        obs_dict = {
            "key": obs.key,
            "topic": obs.topic,
            "type": obs.type or ("sensor_msgs/msg/Image" if obs.peripheral else "sensor_msgs/msg/JointState"),
        }

        if obs.peripheral and obs._interface_source is None:
            obs_dict["peripheral"] = obs.peripheral
            # Image parameters
            if obs.image:
                obs_dict["image"] = obs.image
            else:
                # Use camera defaults from peripheral
                cam = robot_config.get_camera(obs.peripheral)
                if cam:
                    obs_dict["image"] = {"resize": [cam.height, cam.width]}

            if obs.align:
                obs_dict["align"] = obs.align
            if obs.qos:
                obs_dict["qos"] = obs.qos
        else:
            # Non-peripheral observation (e.g., joint states)
            if obs.selector:
                obs_dict["selector"] = obs.selector
            if obs.image:
                obs_dict["image"] = obs.image
            if obs.align:
                obs_dict["align"] = obs.align
            if obs.qos:
                obs_dict["qos"] = obs.qos

        if obs.transport is not None:
            obs_dict["transport"] = observation_transport_to_dict(obs.transport)
        if obs._interface_source is not None:
            obs_dict["_interface_source"] = obs._interface_source

        contract["observations"].append(obs_dict)

    # Generate actions from contract extension config
    for action in robot_config.contract.actions:
        publish = dict(action.publish)
        if publish.get("_interface_source") is not None:
            publish.pop("interface", None)
            publish.pop("requires", None)
        action_dict = {
            "key": action.key,
            "publish": publish,
            "safety_behavior": action.safety_behavior,
        }

        if action.selector:
            action_dict["selector"] = action.selector
        if action.from_tensor:
            action_dict["from_tensor"] = action.from_tensor

        contract["actions"].append(action_dict)

    return yaml.safe_dump(contract, default_flow_style=False, sort_keys=False)
