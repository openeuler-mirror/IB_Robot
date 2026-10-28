"""Public, versioned robot interfaces, projected from effective runtime configuration.

This module is ROS-free. YAML snapshots and RuntimeStatus JSON use exactly the
same schema. Device paths, serials, driver parameters and IPs are not exported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from copy import deepcopy
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

from robot_runtime import contract as C
from robot_runtime.gazebo import advertised_camera_frames
from robot_runtime.profile import LIDAR_SIM_PLATFORMS, PHYSICS_SIM_PLATFORMS, SIM_PLATFORMS, lidar_driver_simulated


class InterfaceDescriptionError(ValueError):
    """A public description violates its schema or semantic contract."""


def description_sim_platform(description: dict) -> str | None:
    """Effective simulation platform of a description, without mutating the document.

    The ``sim_platform`` field is optional (schema_version stays 1), so
    snapshots exported before it existed remain valid with their original
    digest: they are read as ``sdk`` when execution is simulated and as no
    platform when physical. An explicitly present but unknown value fails
    closed instead of silently defaulting.
    """
    sim_platform = description.get("sim_platform")
    if sim_platform is None:
        return "sdk" if description.get("execution") == "simulated" else None
    sim_platform = str(sim_platform)
    if sim_platform not in SIM_PLATFORMS:
        raise InterfaceDescriptionError(f"interface description sim_platform: unknown value {sim_platform!r}")
    return sim_platform


def validate_interface_requirements(requirements) -> dict:
    """The consumer's constraints are not hardware settings or preprocessing."""
    if requirements is None:
        return {}
    allowed = {"width", "height", "encoding", "min_fps", "message_type", "capability"}
    if not isinstance(requirements, dict) or set(requirements) - allowed:
        raise ValueError(f"requires must be a mapping of {sorted(allowed)}")
    for key, value in requirements.items():
        if key in ("width", "height"):
            valid = type(value) is int and value > 0
        elif key == "min_fps":
            valid = type(value) in (int, float) and math.isfinite(value) and value > 0
        else:
            valid = isinstance(value, str) and bool(value.strip())
        if not valid:
            raise ValueError(f"invalid requires.{key}: {value!r}")
    return requirements


def check_interface_requirements(interface: dict, source_profile: dict | None, requirements) -> None:
    for field, expected in validate_interface_requirements(requirements).items():
        actual = (
            interface[field]
            if field in ("message_type", "capability")
            else (source_profile or {}).get("fps" if field == "min_fps" else field)
        )
        if actual is None or not (actual >= expected if field == "min_fps" else actual == expected):
            raise ValueError(f"requires.{field}={expected!r}, source={actual!r}")


@lru_cache(maxsize=1)
def _validator():
    schema = json.loads(files("robot_runtime").joinpath("schemas/interface_description.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def description_digest(description: dict) -> str:
    immutable = {key: value for key, value in description.items() if key not in ("digest", "states")}
    try:
        data = json.dumps(immutable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise InterfaceDescriptionError(f"description is not finite JSON: {exc}") from exc
    return hashlib.sha256(data).hexdigest()


def validate_description(description: dict) -> None:
    try:
        json.dumps(description, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise InterfaceDescriptionError(f"description is not finite JSON: {exc}") from exc
    errors = sorted(_validator().iter_errors(description), key=lambda error: str(list(error.path)))
    if errors:
        error = errors[0]
        path = ".".join(str(part) for part in error.path) or "<root>"
        raise InterfaceDescriptionError(f"interface description {path}: {error.message}")
    if description_digest(description) != description["digest"]:
        raise InterfaceDescriptionError("interface description digest does not match its immutable content")
    model = description.get("model")
    if model is not None:
        from robot_runtime.model_metadata import validate_model_metadata

        try:
            validate_model_metadata(model)
        except ValueError as exc:
            raise InterfaceDescriptionError(f"interface description model: {exc}") from exc
    interfaces = description["interfaces"]
    endpoints = set()
    from robot_runtime.capabilities import all_capabilities

    for name, interface in interfaces.items():
        if interface["capability"] not in all_capabilities():
            raise InterfaceDescriptionError(f"interface {name}: unknown capability {interface['capability']!r}")
        key = (interface["kind"], interface["direction"], interface["endpoint"])
        if key in endpoints:
            raise InterfaceDescriptionError(f"interface {name}: duplicate endpoint/direction {key}")
        endpoints.add(key)
        if "camera_info_topic" in interface:
            infos = [entry for entry in interfaces.values() if entry["endpoint"] == interface["camera_info_topic"]]
            if not any(entry["message_type"] == "sensor_msgs/msg/CameraInfo" for entry in infos):
                raise InterfaceDescriptionError(
                    f"interface {name}: camera_info_topic must reference a CameraInfo interface"
                )
        supported = interface.get("supported_profiles")
        if (
            supported is not None
            and interface["configured_profile"] is not None
            and interface["configured_profile"] not in supported
        ):
            raise InterfaceDescriptionError(f"interface {name}: configured profile is absent from supported_profiles")
        target_group = interface.get("target_group")
        if target_group is not None:
            if model is None:
                raise InterfaceDescriptionError(f"interface {name}: target_group requires public model metadata")
            groups = model["joint_groups"]
            # A command channel may address the target group alone or the full
            # commanded joint set (arm + gripper) of that target.
            if target_group not in groups or interface.get("joint_names") not in (
                groups[target_group],
                groups["all"],
            ):
                raise InterfaceDescriptionError(f"interface {name}: target_group joint order does not match model")
            for frame_field in ("base_frame", "tool_frame"):
                frame = interface.get(frame_field)
                if frame is not None and frame not in model["frames"].values():
                    raise InterfaceDescriptionError(f"interface {name}: unknown {frame_field} {frame!r}")
    for name, state in description["states"].items():
        if name not in interfaces:
            raise InterfaceDescriptionError(f"state references unknown interface {name}")
        if state["state"] in ("ready", "mismatch", "stale") and state["last_seen"] is None:
            raise InterfaceDescriptionError(f"interface {name}: observed state requires last_seen")
        is_image = interfaces[name]["message_type"] == "sensor_msgs/msg/Image"
        if is_image and state["state"] == "ready" and state["observed_profile"] is None:
            raise InterfaceDescriptionError(f"interface {name}: ready image requires an observed_profile")
        if not is_image and state["observed_profile"] is not None:
            raise InterfaceDescriptionError(f"interface {name}: non-image must not carry an image profile")


def load_description(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        description = yaml.safe_load(handle)
    validate_description(description)
    return description


def write_description(description: dict, path: str | Path) -> None:
    """Write a complete, validated YAML snapshot atomically."""
    validate_description(description)
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
        try:
            yaml.safe_dump(description, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def canonicalise_peripheral_entry(entry: dict) -> dict:
    """Canonicalise one peripheral entry's toggle spelling (D3).

    ``disabled: X`` becomes ``enabled: not X`` and the ``disabled`` key is
    removed; an entry setting both keys with contradictory values is rejected
    naming the entry. The input is not mutated. This is the one canonicaliser
    the runtime merge applies to every layer, the runtime profile included.
    """
    key = (entry.get("type"), entry.get("name"))
    canonical = deepcopy(entry)
    if "disabled" in canonical:
        disabled = canonical.pop("disabled")
        if "enabled" in canonical and bool(canonical["enabled"]) != bool(disabled):
            # Both present and agreeing (``disabled: X`` == ``enabled: not X``):
            # the canonical ``enabled`` value wins.
            return canonical
        if "enabled" in canonical:
            raise InterfaceDescriptionError(
                f"peripheral {key} sets contradictory disabled={disabled!r} and "
                f"enabled={canonical['enabled']!r}; pass at most one"
            )
        canonical["enabled"] = not bool(disabled)
    return canonical


def _check_camera_values(camera: dict) -> None:
    """Value checks on one enabled merged camera entry (D5).

    ``width``/``height``/``depth_width``/``depth_height`` must be positive
    integers (bool rejected); ``fps``/``depth_fps`` positive finite numbers
    (bool, NaN and infinity rejected); the depth dimensions are paired and
    ``depth_fps`` requires them (without them the RealSense driver ignores it
    while the description would advertise it). Absent fields keep their
    documented defaults and are not checked; unknown keys are not rejected.
    """
    name = camera.get("name")
    integer_fields = ("width", "height", "depth_width", "depth_height")
    number_fields = ("fps", "depth_fps")
    for field in integer_fields:
        if field not in camera:
            continue
        value = camera[field]
        if type(value) is not int or value <= 0:
            raise InterfaceDescriptionError(f"camera {name}: {field} must be a positive integer, got {value!r}")
    for field in number_fields:
        if field not in camera:
            continue
        value = camera[field]
        if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise InterfaceDescriptionError(f"camera {name}: {field} must be a positive finite number, got {value!r}")
    if ("depth_width" in camera) != ("depth_height" in camera):
        raise InterfaceDescriptionError(f"camera {name}: depth_width and depth_height must be given together")
    if "depth_fps" in camera and "depth_width" not in camera:
        raise InterfaceDescriptionError(f"camera {name}: depth_fps requires depth_width/depth_height")


def merge_peripherals(defaults: list | None, overrides: list | None) -> list[dict]:
    """Resolve partial instance overrides without mutating either configuration.

    Every entry of every layer (defaults included) is canonicalised first
    (``disabled: X`` == ``enabled: not X``; contradictory spellings in one
    entry are rejected, D3). An override entry for an ``(type, name)`` the
    defaults do not declare must declare ``driver`` (D5): a partial entry
    cannot silently add a default device, which catches misspelled names. A
    changed driver replaces the base device settings and inherits only the
    base ``enabled`` state unless the override sets one (toggle state is not
    vendor-specific, D3). Merged enabled cameras get value checks (D5).
    """

    def merge(base, override):
        result = deepcopy(base)
        for key, value in override.items():
            result[key] = (
                merge(result[key], value)
                if isinstance(result.get(key), dict) and isinstance(value, dict)
                else deepcopy(value)
            )
        return result

    result = []
    index = {}
    for entries, override in ((defaults or [], False), (overrides or [], True)):
        if not isinstance(entries, list):
            raise InterfaceDescriptionError("peripherals must be a list")
        seen = set()
        for raw_entry in entries:
            if not isinstance(raw_entry, dict) or not raw_entry.get("type") or not raw_entry.get("name"):
                raise InterfaceDescriptionError("peripherals entries require type and name")
            key = (raw_entry["type"], raw_entry["name"])
            if key in seen:
                raise InterfaceDescriptionError(f"duplicate peripheral {key}")
            seen.add(key)
            entry = canonicalise_peripheral_entry(raw_entry)
            if override and key in index:
                pos = index[key]
                base = result[pos]
                if entry.get("driver", base.get("driver")) != base.get("driver"):
                    # A changed driver must not inherit vendor-specific
                    # settings (D3); the base's enabled state carries over
                    # unless the override sets one itself (toggle state is
                    # not vendor-specific).
                    if "enabled" not in entry and "enabled" in base:
                        entry["enabled"] = base["enabled"]
                    result[pos] = entry
                else:
                    result[pos] = merge(base, entry)
            else:
                if override and "driver" not in entry:
                    raise InterfaceDescriptionError(
                        f"peripheral {key} is not declared by the runtime: an override must name a "
                        "declared peripheral, or declare driver to add a new device"
                    )
                index[key] = len(result)
                result.append(entry)
    for entry in result:
        if entry.get("type") == "camera" and entry.get("enabled", True):
            _check_camera_values(entry)
    return [entry for entry in result if entry.get("enabled", True)]


def camera_streams(camera: dict) -> list[str]:
    if camera.get("driver", "opencv") != "realsense":
        if "streams" in camera and camera["streams"] != ["color"]:
            raise InterfaceDescriptionError(f"camera {camera.get('name')}: this driver supports only streams: [color]")
        return ["color"]
    streams = camera.get("streams")
    if streams is None:
        streams = ["color"] + (["depth"] if camera.get("enable_depth", True) else [])
        if camera.get("enable_pointcloud", False):
            streams.append("pointcloud")
    if not isinstance(streams, list) or set(streams) - {"color", "depth", "pointcloud"}:
        raise InterfaceDescriptionError(f"camera {camera.get('name')}: invalid streams")
    if "pointcloud" in streams and "depth" not in streams:
        raise InterfaceDescriptionError(f"camera {camera.get('name')}: pointcloud requires depth")
    return streams


def _topic(endpoint, message_type, capability, direction="publish", **metadata):
    sensor = message_type.startswith("sensor_msgs/") or message_type.startswith("livox_ros_driver2/")
    return {
        "kind": "topic",
        "direction": direction,
        "endpoint": endpoint,
        "message_type": message_type,
        "capability": capability,
        "qos": {
            "reliability": "best_effort" if sensor else "reliable",
            "durability": "volatile",
            "history": "keep_last",
            "depth": 5 if sensor else 10,
        },
        **metadata,
    }


def peripheral_interfaces(
    peripherals: list[dict] | None,
    *,
    sim_platform: str | None = None,
    camera_rate_hz: float | None = None,
    imu_rate_hz: float | None = None,
    scan_source_runs: bool = True,
) -> dict[str, dict]:
    """Project the peripheral surface.

    ``sim_platform`` adjusts streams the platform cannot produce (the gz
    camera gaps, the lidar wire type) and drops lidars the platform does not
    stand in for (D5). ``camera_rate_hz``/``imu_rate_hz`` carry the rates the
    simulators actually produce when they differ from the device defaults.
    ``scan_source_runs`` says whether the derived scan's source (FAST-LIO's
    registered body cloud) runs on the selected platform; a physics platform
    without it advertises no scan (D5).
    """
    interfaces: dict[str, dict] = {}
    audio_counts = {"microphone": 0, "speaker": 0}
    merged = merge_peripherals(peripherals, [])
    for periph in merged:
        name, kind = periph["name"], periph["type"]
        if kind in ("microphone", "speaker"):
            # A runtime exposes at most one logical audio input and output:
            # the fixed interface IDs (audio.capture / audio.play) map to a
            # single device each, so a second device of the same kind has no
            # surface to project onto.
            audio_counts[kind] += 1
            if audio_counts[kind] > 1:
                raise InterfaceDescriptionError(
                    f"{kind} peripheral {name}: at most one {kind} is supported "
                    f"(found {audio_counts[kind]}; single-value mapping to a fixed logical ID)"
                )
            if periph.get("driver") != "alsa":
                raise InterfaceDescriptionError(
                    f"{kind} peripheral {name}: unsupported driver {periph.get('driver')!r}"
                )
            params = periph.get("params") or {}
            if kind == "microphone":
                metadata = {
                    "sample_rate_hz": params.get("sample_rate", 16000),
                    "channels": params.get("channels", 6),
                    "sample_format": params.get("sample_format", "S16LE"),
                }
                if params.get("channel_indices") is not None:
                    # Consumer-facing channel selection only; the device stays
                    # the interleaved stream the ALSA owner publishes.
                    metadata["channel_indices"] = params["channel_indices"]
                interfaces["audio.capture"] = _topic(
                    "/audio/capture_stamped",
                    "audio_common_msgs/msg/AudioDataStamped",
                    "interaction.audio_capture",
                    **metadata,
                )
            else:
                interfaces["audio.play"] = _topic(
                    "/audio/play",
                    "audio_common_msgs/msg/AudioDataStamped",
                    "interaction.audio_playback",
                    "subscribe",
                    sample_rate_hz=params.get("sample_rate", 24000),
                    channels=params.get("channels", 1),
                    sample_format=params.get("sample_format", "S16LE"),
                )
        elif kind == "camera" and periph.get("driver") != "virtual":
            driver = periph.get("driver", "opencv")
            if driver not in ("opencv", "camera_ros", "realsense"):
                raise InterfaceDescriptionError(f"camera {name}: unsupported driver {driver!r}")
            streams = camera_streams(periph)
            if sim_platform == "gazebo":
                # Streams the gz simulator cannot produce are not advertised on
                # this platform (runtime-simulation-platform D6): the
                # rgbd_camera publishes no depth camera_info (its depth image
                # shares the color intrinsics) and no color-aligned pointcloud.
                streams = [stream for stream in streams if stream != "pointcloud"]
            if sim_platform == "mujoco":
                # The MuJoCo plugin produces no color-aligned pointcloud
                # (so101-mujoco-runtime-platform D5); depth it does produce
                # (32FC1, spike T0).
                streams = [stream for stream in streams if stream != "pointcloud"]
            prefix = f"/camera/{name}"
            capability = "perception.camera"
            # the shared advertised-frame convention (5b.3): one helper for
            # the description, the gz frame_id and the MuJoCo sensor frame_name
            image_frame, info_frame = advertised_camera_frames(periph)
            for stream in ("color", "depth", "aligned_depth"):
                aligned = stream == "aligned_depth"
                if stream == "color" and "color" not in streams:
                    continue
                if stream != "color" and "depth" not in streams:
                    continue
                if aligned and not (periph.get("align_depth") and "color" in streams):
                    continue
                is_color = stream == "color"
                width = (
                    periph.get("width", 640)
                    if is_color or aligned
                    else periph.get("depth_width", periph.get("width", 640))
                )
                height = (
                    periph.get("height", 480)
                    if is_color or aligned
                    else periph.get("depth_height", periph.get("height", 480))
                )
                if sim_platform == "mujoco" and not (is_color or aligned):
                    # the plugin renders one MJCF camera (width x height) for
                    # every stream (D5): the depth stream dims are the rendered
                    # camera's dims, never the physical device's
                    # depth_width/depth_height
                    width = periph.get("width", 640)
                    height = periph.get("height", 480)
                if sim_platform == "gazebo" and not is_color:
                    # the gz rgbd sensor renders one camera at the colour
                    # dimensions and rate (D8): the depth/aligned-depth
                    # streams advertise what the sensor actually renders,
                    # never the physical device's depth_* overrides (the
                    # MuJoCo branch above applies the same principle)
                    width = periph.get("width", 640)
                    height = periph.get("height", 480)
                    fps = periph.get("fps", 30)
                else:
                    fps = periph.get("fps", 30) if is_color else periph.get("depth_fps", periph.get("fps", 30))
                if driver == "realsense":
                    wire_format = periph.get("color_format", "RGB8") if is_color else periph.get("depth_format", "Z16")
                else:
                    wire_format = (
                        periph.get("pixel_format", "mjpeg") if driver == "opencv" else periph.get("format", "MJPEG")
                    )
                encoding = {
                    "RGB8": "rgb8",
                    "BGR8": "bgr8",
                    "Z16": "16UC1",
                    "Y8": "mono8",
                    "mjpeg2rgb": "rgb8",
                    "yuyv2rgb": "rgb8",
                }.get(wire_format)
                if wire_format in ("rgb8", "bgr8", "mono8", "16UC1", "32FC1"):
                    encoding = wire_format
                if sim_platform == "gazebo" and not is_color:
                    # gz rgbd depth is float32 and ros_gz_bridge maps
                    # R_FLOAT32 to 32FC1; advertising the physical device's
                    # encoding (RealSense Z16 -> 16UC1) would mismatch every
                    # observed depth message in the interface monitor
                    # (PR !515 P2). The contract on this platform is what the
                    # simulator actually publishes.
                    encoding = "32FC1"
                if sim_platform == "mujoco":
                    # the plugin always publishes rgb8 color and 32FC1 metre
                    # depth (spike T0); the contract on this platform is what
                    # the simulator actually publishes (D5)
                    encoding = "rgb8" if is_color else "32FC1"
                if sim_platform == "mujoco" and camera_rate_hz is not None:
                    # the plugin has one rate for all cameras (D5): every
                    # stream advertises the effective rate
                    fps = float(camera_rate_hz)
                suffix = "" if is_color else ("/aligned_depth_to_color" if aligned else "/depth")
                endpoint = f"{prefix}{suffix}/{'image_raw' if is_color or aligned else 'image_rect_raw'}"
                info_topic = f"{prefix}{suffix}/camera_info"
                if sim_platform == "gazebo" and not is_color:
                    # gz rgbd_camera has no depth camera_info topic; the depth
                    # image shares the color intrinsics, so the color info
                    # interface carries them (D6: unproducible streams are not
                    # advertised).
                    info_topic = f"{prefix}/camera_info"
                stream_frame, stream_info_frame = image_frame, info_frame
                if driver == "realsense":
                    native_prefix = periph.get(
                        "driver_topic_prefix", f"/camera/{periph.get('driver_camera_name', f'{name}_camera')}"
                    )
                    # Equal source/target endpoints bypass the frame-normalizing relay.
                    if not is_color and native_prefix == prefix:
                        stream_frame = stream_info_frame = None
                key = f"camera.{name}.{stream}"
                configured = {"width": width, "height": height, "fps": fps, "encoding": encoding}
                # Only explicitly declared device modes count as supported.
                supplied_modes = periph.get("supported_profiles") or {}
                if not isinstance(supplied_modes, dict):
                    raise InterfaceDescriptionError(f"camera {name}: supported_profiles must be keyed by stream")
                supported = supplied_modes.get(stream)
                if supported is not None:
                    fields = ("width", "height", "fps", "encoding")
                    if not isinstance(supported, list) or not all(
                        isinstance(mode, dict) and all(key in mode for key in fields) for mode in supported
                    ):
                        raise InterfaceDescriptionError(
                            f"camera {name}.{stream}: supported_profiles entries require {fields}"
                        )
                    supported = [{k: mode[k] for k in ("width", "height", "fps", "encoding")} for mode in supported]
                interfaces[key] = _topic(
                    endpoint,
                    "sensor_msgs/msg/Image",
                    capability,
                    frame_id=stream_frame,
                    camera_info_topic=info_topic,
                    configured_profile=configured,
                    supported_profiles=supported,
                )
                if sim_platform == "mujoco":
                    # the plugin's camera publishers are reliable (depth 1):
                    # large images lose fragments without repair under
                    # best-effort, so the contract on this platform offers
                    # reliable delivery (what the simulator actually publishes)
                    interfaces[key]["qos"]["reliability"] = "reliable"
                if sim_platform == "gazebo" and not is_color:
                    # the shared color info interface above already carries the
                    # depth intrinsics; a dedicated depth info topic has no
                    # producer on this platform
                    continue
                interfaces[f"{key}_info"] = _topic(
                    info_topic, "sensor_msgs/msg/CameraInfo", capability, frame_id=stream_info_frame, rate_hz=fps
                )
                if sim_platform == "mujoco":
                    interfaces[f"{key}_info"]["qos"]["reliability"] = "reliable"
            if "pointcloud" in streams:
                interfaces[f"camera.{name}.points"] = _topic(
                    f"{prefix}/depth/color/points",
                    "sensor_msgs/msg/PointCloud2",
                    capability,
                    frame_id=periph.get("optical_frame_id") if not periph.get("direct_topic_remap") else None,
                    units={"position": "m"},
                )
        elif kind == "lidar":
            driver = periph.get("driver")
            simulated_lidar = sim_platform in LIDAR_SIM_PLATFORMS and lidar_driver_simulated(str(driver), sim_platform)
            # A physics platform that simulates lidar but not THIS driver
            # neither simulates nor advertises it (D5): same rule as an
            # unsupported driver, so the declared surface never claims a
            # stream nothing produces (MuJoCo: MID-360 only).
            if sim_platform in LIDAR_SIM_PLATFORMS and not simulated_lidar:
                continue
            if driver == "livox_mid360":
                if periph.get("multi_topic", 0):
                    raise InterfaceDescriptionError(f"lidar {name}: multi_topic endpoints require an explicit adapter")
                transfer = periph.get("xfer_format", 1)
                types = {0: "sensor_msgs/msg/PointCloud2", 1: "livox_ros_driver2/msg/CustomMsg"}
                if transfer not in types:
                    raise InterfaceDescriptionError(f"lidar {name}: unsupported xfer_format {transfer}")
                # The simulator publishes a plain point cloud; the physical
                # driver's CustomMsg transfer cannot be produced (D5).
                cloud_type = "sensor_msgs/msg/PointCloud2" if simulated_lidar else types[transfer]
                # The advertised rate is the rate the platform runs: the
                # composed MJCF honours simulation.mujoco.update_rate over
                # publish_freq (robot_runtime.mujoco), so the description
                # must resolve the same precedence — never claim a rate the
                # simulation does not produce (review round 1).
                points_rate_hz = periph.get("publish_freq", 10.0)
                if sim_platform == "mujoco":
                    mujoco_section = (periph.get("simulation") or {}).get("mujoco") or {}
                    if isinstance(mujoco_section, dict) and mujoco_section.get("update_rate") is not None:
                        points_rate_hz = mujoco_section["update_rate"]
                interfaces[f"lidar.{name}.points"] = _topic(
                    periph.get("pointcloud_topic", "/livox/lidar"),
                    cloud_type,
                    "perception.lidar",
                    frame_id=periph.get("frame_id", "livox_frame"),
                    rate_hz=points_rate_hz,
                    units={"position": "m"},
                )
                # The integrated IMU's advertised rate is the rate it actually
                # produces: on MuJoCo the imu_sensor_broadcaster publishes on
                # the controller-manager clock (D5.3, no second hard-coded 100).
                interfaces[f"lidar.{name}.imu"] = _topic(
                    periph.get("imu_topic", "/livox/imu"),
                    "sensor_msgs/msg/Imu",
                    "perception.lidar",
                    frame_id=periph.get("frame_id", "livox_frame"),
                    **({"rate_hz": imu_rate_hz} if imu_rate_hz is not None else {}),
                )
                interfaces[f"lidar.{name}.points"]["qos"]["reliability"] = "reliable"
                interfaces[f"lidar.{name}.imu"]["qos"]["reliability"] = "reliable"
                scan = periph.get("scan_converter") or {}
                if scan.get("enabled") and scan_source_runs:
                    interfaces[f"lidar.{name}.scan"] = _topic(
                        scan.get("scan_topic", "/scan"),
                        "sensor_msgs/msg/LaserScan",
                        "perception.lidar",
                        frame_id=scan.get("target_frame", periph.get("frame_id", "livox_frame")),
                        units={"range": "m", "angle": "rad"},
                    )
            elif driver == "ldlidar":
                interfaces[f"lidar.{name}.scan"] = _topic(
                    periph.get("scan_topic", "/scan"),
                    "sensor_msgs/msg/LaserScan",
                    "perception.lidar",
                    frame_id=periph.get("frame_id"),
                    units={"range": "m", "angle": "rad"},
                )
            else:
                raise InterfaceDescriptionError(f"lidar {name}: unsupported driver {driver!r}")
    return interfaces


def audio_capability_entries(interfaces: dict[str, dict]) -> dict[str, dict]:
    """RuntimeStatus ``interaction.audio_*`` entries for projected audio interfaces.

    Field names follow the capability parameter schema (robot_runtime.capabilities):
    capture carries ``topics`` plus the capture format, playback ``endpoints`` plus
    the playback format. Audio interfaces a runtime does not project (e.g. the
    vendor-bridged X2 declares them in its profile instead) contribute nothing.
    """
    entries: dict[str, dict] = {}
    capture = interfaces.get("audio.capture")
    if capture is not None:
        entries["interaction.audio_capture"] = {
            "topics": [capture["endpoint"]],
            "sample_rate_hz": capture["sample_rate_hz"],
            "channels": capture["channels"],
        }
    playback = interfaces.get("audio.play")
    if playback is not None:
        entries["interaction.audio_playback"] = {
            "endpoints": [playback["endpoint"]],
            "sample_rate_hz": playback["sample_rate_hz"],
            "channels": playback["channels"],
        }
    return entries


def _add_profile_interfaces(interfaces: dict[str, dict], declared: object) -> None:
    """Merge explicitly public profile interfaces without accepting duplicate IDs."""
    if declared is None:
        return
    if not isinstance(declared, dict):
        raise InterfaceDescriptionError("profile.interfaces must be a mapping keyed by public interface ID")
    for name, interface in declared.items():
        if not isinstance(name, str) or not name or name in interfaces:
            raise InterfaceDescriptionError(f"duplicate or invalid profile interface ID: {name!r}")
        if not isinstance(interface, dict):
            raise InterfaceDescriptionError(f"profile interface {name!r} must be an object")
        interfaces[name] = deepcopy(interface)


def _teleoperation_interfaces(profile: dict, model: dict) -> dict[str, dict]:
    """Project public single-target commands; private servo configuration stays private."""
    teleop = profile.get("teleoperation") or {}
    if not teleop or not teleop.get("enabled", False):
        return {}
    target = str(teleop.get("target_group", ""))
    groups, frames = model["joint_groups"], model["frames"]
    if target != "arm" or target not in groups or not groups[target]:
        raise InterfaceDescriptionError("teleoperation.target_group must name the single public arm group")
    prefix = str(teleop.get("endpoint_prefix", "")).rstrip("/")
    stale = teleop.get("command_stale_s")
    if not prefix.startswith("/") or type(stale) not in (int, float) or not math.isfinite(stale) or not 0 < stale <= 1:
        raise InterfaceDescriptionError(
            "teleoperation requires absolute endpoint_prefix and finite command_stale_s in (0, 1]"
        )
    try:
        base_frame, tool_frame = frames["base_link"], frames["ee_link"]
    except KeyError as exc:
        raise InterfaceDescriptionError("teleoperation requires model base_link and ee_link frames") from exc
    shared = {
        "target_group": target,
        "joint_names": list(groups[target]),
        "base_frame": base_frame,
        "tool_frame": tool_frame,
        "command_stale_s": float(stale),
    }
    return {
        "motion.arm.pose": _topic(
            f"{prefix}/pose",
            "geometry_msgs/msg/PoseStamped",
            "motion.move_to_pose",
            "subscribe",
            units={"position": "m", "orientation": "quaternion"},
            pose_reference="clutch_relative",
            **shared,
        ),
        "motion.arm.linear": _topic(
            f"{prefix}/linear",
            "geometry_msgs/msg/Vector3Stamped",
            "motion.move_to_pose",
            "subscribe",
            units={"linear": "m/s"},
            **shared,
        ),
        "motion.arm.angular": _topic(
            f"{prefix}/angular",
            "geometry_msgs/msg/Vector3Stamped",
            "motion.move_to_pose",
            "subscribe",
            units={"angular": "rad/s"},
            **shared,
        ),
        "motion.arm.joints": _topic(
            f"{prefix}/joints",
            "sensor_msgs/msg/JointState",
            "joint.position_stream",
            "subscribe",
            units={"position": "rad"},
            target_group=target,
            joint_names=list(groups["all"]),
            base_frame=base_frame,
            tool_frame=tool_frame,
            command_stale_s=float(stale),
        ),
        "motion.arm.lease": _topic(f"{prefix}/lease", "std_msgs/msg/Empty", "runtime.stop", "subscribe", **shared),
        "motion.arm.start": {
            "capability": "runtime.stop",
            "kind": "service",
            "direction": "serve",
            "endpoint": f"{prefix}/start",
            "message_type": "std_srvs/srv/Trigger",
            **shared,
        },
        "motion.arm.stop": {
            "capability": "runtime.stop",
            "kind": "service",
            "direction": "serve",
            "endpoint": f"{prefix}/stop",
            "message_type": "std_srvs/srv/Trigger",
            **shared,
        },
        "motion.arm.home": {
            "capability": "motion.move_to_joint",
            "kind": "action",
            "direction": "serve",
            "endpoint": f"{prefix}/home",
            "message_type": "ibrobot_msgs/action/ArmReturnHome",
            **shared,
        },
    }


def _mujoco_camera_rate(profile: dict) -> float:
    """The effective MuJoCo camera rate for the description (lazy import: the
    rate algorithm lives next to the MuJoCo camera generation, D5)."""
    from robot_runtime.mujoco import mujoco_camera_publish_rate  # noqa: PLC0415

    return mujoco_camera_publish_rate(profile)


def _mujoco_imu_rate(profile: dict, peripherals: list | None) -> float | None:
    """The rate the simulated IMU produces on MuJoCo (D5.3), when one runs.

    The ``imu_sensor_broadcaster`` publishes on the controller-manager clock,
    so the produced rate is the effective ``controller_manager.update_rate``
    of the controllers configuration the embodiment generates from — read
    from the profile's ``ros2_control.controllers_config`` (the generated
    file adds the broadcaster entry, never a different rate). Computed only
    when the EFFECTIVE inventory (the merged profile + deployment fragment,
    exactly the list the description itself iterates — review round 1)
    carries a lidar whose integrated IMU the MuJoCo platform simulates; a
    profile without one needs no rate, and a missing update_rate on a
    profile that does fails closed: no rate is better than a wrong one.
    """
    from robot_runtime.gazebo import LIDAR_INTEGRATED_IMU_DRIVERS  # noqa: PLC0415
    from robot_runtime.mujoco import MUJOCO_LIDAR_SIM_DRIVERS  # noqa: PLC0415

    integrated_here = set(MUJOCO_LIDAR_SIM_DRIVERS) & set(LIDAR_INTEGRATED_IMU_DRIVERS)
    effective = peripherals if peripherals is not None else (profile.get("peripherals") or [])
    effective = [entry for entry in effective if entry.get("enabled", True)]
    if not any(
        periph.get("type") == "lidar" and str(periph.get("driver", "")) in integrated_here for periph in effective
    ):
        return None
    import yaml

    from robot_runtime.path_subst import resolve_path

    config = str(((profile.get("ros2_control") or {}).get("controllers_config") or "").strip())
    if not config:
        raise InterfaceDescriptionError(
            "the MuJoCo IMU rate needs ros2_control.controllers_config in the runtime profile"
        )
    path = resolve_path(config)
    try:
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise InterfaceDescriptionError(
            f"cannot read the controllers configuration for the MuJoCo IMU rate: {exc}"
        ) from exc
    rate = ((data.get("controller_manager") or {}).get("ros__parameters") or {}).get("update_rate")
    if not isinstance(rate, int | float) or rate <= 0:
        raise InterfaceDescriptionError(
            f"controller_manager.update_rate missing or invalid in {path}: the MuJoCo IMU rate cannot be advertised"
        )
    return float(rate)


def build_description(
    profile: dict,
    peripherals: list[dict] | None = None,
    *,
    use_sim: bool | None = None,
    robot_description: str | None = None,
    sim_platform: str | None = None,
) -> dict:
    """Project only public ROS interfaces; do not serialize private configuration.

    ``use_sim``/``sim_platform`` (runtime-simulation-platform D1/D7) select the
    execution identity recorded in the description: physical execution carries
    no platform, simulated execution carries the platform it runs on. With
    neither given the profile's declared selection applies; ``use_sim:=false``
    together with a platform fails naming both.
    """
    runtime = profile["runtime"]
    caps = profile["capabilities"]
    if use_sim is None:
        use_sim = bool(profile.get("use_sim", profile.get("simulated", False)))
    if not use_sim:
        if sim_platform:
            raise InterfaceDescriptionError(
                f"conflicting execution selection: use_sim=false with sim_platform={sim_platform!r}"
            )
        sim_platform = None
    else:
        sim_platform = str(sim_platform or profile.get("sim_platform") or "sdk").strip()
        if sim_platform not in SIM_PLATFORMS:
            raise InterfaceDescriptionError(
                f"unknown sim_platform {sim_platform!r}: expected one of {list(SIM_PLATFORMS)}"
            )
    interfaces = peripheral_interfaces(
        profile.get("peripherals") if peripherals is None else peripherals,
        sim_platform=sim_platform,
        camera_rate_hz=(
            _mujoco_camera_rate({**profile, "peripherals": peripherals or profile.get("peripherals") or []})
            if sim_platform == "mujoco"
            else None
        ),
        imu_rate_hz=(_mujoco_imu_rate(profile, peripherals) if sim_platform == "mujoco" else None),
        # The derived scan's source (FAST-LIO's registered body cloud) runs
        # on a physics platform only when lidar odometry itself runs there;
        # without it no scan is advertised or published (D5).
        scan_source_runs=(
            bool((profile.get("fast_lio") or {}).get("enabled")) if sim_platform in PHYSICS_SIM_PLATFORMS else True
        ),
    )
    # Audio peripherals derive the interaction.audio_* capability entries. A
    # profile that additionally declares them explicitly must agree with the
    # devices it also lists, or the RuntimeStatus surface would contradict the
    # interface description (openspec change runtime-owned-audio-io, D2).
    for capability, derived in audio_capability_entries(interfaces).items():
        declared = caps.get(capability)
        if declared is not None and declared != derived:
            raise InterfaceDescriptionError(
                f"profile capability {capability} {declared!r} contradicts the audio peripherals {derived!r}; "
                "align the profile entry with the peripheral parameters or drop it"
            )
    for name, endpoint, service in (
        ("get_status", C.GET_STATUS_SERVICE, "GetRuntimeStatus"),
        ("set_mode", C.SET_MODE_SERVICE, "SetRuntimeMode"),
        ("stop", C.STOP_SERVICE, "StopRuntime"),
    ):
        interfaces[f"runtime.{name}"] = {
            "capability": "runtime.stop" if name == "stop" else "runtime.status",
            "kind": "service",
            "direction": "serve",
            "endpoint": endpoint,
            "message_type": f"ibrobot_msgs/srv/{service}",
        }
    interfaces["runtime.status"] = _topic(C.STATUS_TOPIC, "ibrobot_msgs/msg/RuntimeStatus", "runtime.status")
    if "joint.state" in caps:
        interfaces["joint.state"] = _topic(
            profile.get("joint_state_topic", "/joint_states"),
            "sensor_msgs/msg/JointState",
            "joint.state",
            joint_names=[str(j) for j in profile["joints"]],
            units={"position": "rad", "velocity": "rad/s"},
            joint_limits=deepcopy(profile.get("joint_limits")),
        )
        interfaces["joint.state"]["qos"]["reliability"] = "reliable"
    for channel in profile.get("command_channels", []):
        twist = str(channel.get("type", "float64_array")).lower() == "twist"
        name = "base.cmd_vel" if twist else f"joint.{channel['channel']}"
        if name in interfaces:
            raise InterfaceDescriptionError(f"duplicate logical command interface {name}")
        interfaces[name] = _topic(
            channel["topic"],
            "geometry_msgs/msg/Twist" if twist else "std_msgs/msg/Float64MultiArray",
            "base.cmd_vel" if twist else "joint.position_stream",
            "subscribe",
            modes=channel.get("modes", []),
        )
        if not twist:
            interfaces[name].update(
                joint_names=[str(j) for j in channel.get("joints", profile["joints"])], units={"position": "rad"}
            )
        else:
            interfaces[name].update(
                units={"linear": "m/s", "angular": "rad/s"},
                limits={key: value for key, value in caps.get("base.cmd_vel", {}).items() if key.startswith("max_")},
            )
    # Trajectory actions carry target identity: the endpoint is matched to a
    # command channel group (arm/gripper/...) so consumers bind named targets
    # instead of guessing by array index.
    channel_groups = {}
    for channel in profile.get("command_channels", []):
        name = str(channel.get("channel", ""))
        if name:
            channel_groups[name] = [str(j) for j in channel.get("joints", [])]
    for index, endpoint in enumerate(profile.get("trajectory_actions", [])):
        target_group, joints = "", []
        for name, group_joints in channel_groups.items():
            base = name.removesuffix("_stream")
            if f"/{base}_trajectory_controller/" in str(endpoint):
                target_group = base
                joints = group_joints
                break
        interface_id = f"joint.trajectory_{target_group}" if target_group else f"joint.trajectory_{index}"
        if interface_id in interfaces:
            raise InterfaceDescriptionError(f"duplicate logical command interface {interface_id}")
        interfaces[interface_id] = {
            "capability": "joint.trajectory",
            "kind": "action",
            "direction": "serve",
            "endpoint": endpoint,
            "message_type": "control_msgs/action/FollowJointTrajectory",
            # Target identity is only projected when the public model backs it.
            **({"target_group": target_group, "joint_names": joints} if target_group and robot_description else {}),
        }
    for cap, name, endpoint, service in (
        ("motion.fk", "compute_fk", C.COMPUTE_FK_SERVICE, "ComputeFk"),
        ("motion.ik", "compute_ik", C.COMPUTE_IK_SERVICE, "ComputeIk"),
        ("motion.move_to_pose", "move_to_pose", C.MOVE_TO_POSE_SERVICE, "MoveToPose"),
        ("motion.move_to_joint", "move_to_joint", C.MOVE_TO_JOINT_SERVICE, "MoveToConfiguration"),
    ):
        if cap in caps:
            for index, address in enumerate(caps[cap].get("endpoints", [endpoint])):
                key = f"motion.{name}" + (f".worker_{index}" if index else "")
                interfaces[key] = {
                    "capability": cap,
                    "kind": "service",
                    "direction": "serve",
                    "endpoint": address,
                    "message_type": f"ibrobot_msgs/srv/{service}",
                }
    if "base.odom" in caps:
        interfaces["base.odom"] = _topic(
            C.ODOM_TOPIC,
            "nav_msgs/msg/Odometry",
            "base.odom",
            frame_id=(profile.get("base") or {}).get("odom_frame", "odom"),
        )
    if (profile.get("description") or {}).get("xacro"):
        interfaces["robot.tf"] = _topic("/tf", "tf2_msgs/msg/TFMessage", "joint.state")
        interfaces["robot.tf_static"] = _topic("/tf_static", "tf2_msgs/msg/TFMessage", "joint.state")
        interfaces["robot.description"] = _topic("/robot_description", "std_msgs/msg/String", "joint.state")
        for key in ("robot.tf_static", "robot.description"):
            interfaces[key]["qos"]["durability"] = "transient_local"
    if "base.navigation_gate" in caps:
        interfaces["base.navigation_enabled"] = _topic(
            C.NAVIGATION_ACK_TOPIC, "std_msgs/msg/Bool", "base.navigation_gate"
        )
        interfaces["base.set_navigation_enabled"] = {
            "capability": "base.navigation_gate",
            "kind": "service",
            "direction": "serve",
            "endpoint": C.NAVIGATION_ENABLE_SERVICE,
            "message_type": "std_srvs/srv/SetBool",
        }
    lidar_odom = profile.get("fast_lio") or {}
    if lidar_odom.get("enabled"):
        interfaces["localization.odometry"] = _topic(
            lidar_odom.get("output_topic", "/odometry/filtered"),
            "nav_msgs/msg/Odometry",
            "base.odom",
            frame_id=lidar_odom.get("odom_frame", "odom"),
        )
        interfaces["localization.cloud"] = _topic(
            "/cloud_registered_body",
            "sensor_msgs/msg/PointCloud2",
            "perception.lidar",
            frame_id=lidar_odom.get("source_body_frame", "body"),
        )
        interfaces["localization.cloud"]["qos"]["reliability"] = "reliable"
    description = {
        "schema_version": 1,
        "robot": {
            "id": runtime.get("instance_id", runtime["name"]),
            "type": runtime.get("type", runtime["name"].removesuffix("_robot")),
            "runtime_name": runtime["name"],
            "runtime_version": str(runtime["version"]),
        },
        "execution": "simulated" if use_sim else "physical",
        **({"sim_platform": sim_platform} if use_sim else {}),
        "interfaces": interfaces,
        "states": {},
    }
    # A rendered URDF denotes a migrated runtime descriptor. Generic descriptors
    # without a robot description remain valid but cannot be normalization authority.
    if robot_description is not None:
        from robot_runtime.model_metadata import build_model_metadata

        description["model"] = build_model_metadata(
            profile, use_sim=use_sim, sim_platform=sim_platform, robot_description=robot_description
        )
        for name, interface in _teleoperation_interfaces(profile, description["model"]).items():
            if name in interfaces:
                raise InterfaceDescriptionError(f"duplicate logical command interface {name}")
            interfaces[name] = interface
    _add_profile_interfaces(interfaces, profile.get("interfaces"))
    description["digest"] = description_digest(description)
    validate_description(description)
    return description


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export or validate the public robot interface description")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--profile")
    source.add_argument("--validate")
    parser.add_argument("--peripherals", default="")
    parser.add_argument("--simulated", choices=("true", "false"))
    parser.add_argument("--sim_platform", choices=SIM_PLATFORMS)
    parser.add_argument("--output")
    parser.add_argument("--instance-id")
    args = parser.parse_args(argv)
    try:
        if args.validate:
            description = load_description(args.validate)
        else:
            from robot_runtime.profile import load_profile

            profile = load_profile(args.profile)
            if args.instance_id:
                profile["runtime"]["instance_id"] = args.instance_id
            if args.peripherals:
                # One structural loader for every fragment (D6): the CLI
                # rejects the same malformed files as the runtime launches
                # and the facade/mock nodes, with the same messages.
                from robot_runtime.peripherals import load_peripherals_file

                fragment = load_peripherals_file(args.peripherals)
                profile["peripherals"] = merge_peripherals(profile.get("peripherals"), fragment.get("peripherals"))
                profile["fast_lio"] = {**(profile.get("fast_lio") or {}), **(fragment.get("fast_lio") or {})}
            description = build_description(
                profile,
                use_sim=None if args.simulated is None else args.simulated == "true",
                sim_platform=args.sim_platform,
            )
        if args.output:
            write_description(description, args.output)
        else:
            print(yaml.safe_dump(description, sort_keys=False), end="")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    return 0
