"""Bind consumer contracts to the public robot ROS interface description, without ROS I/O."""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml


class InterfaceBindingError(ValueError):
    """A named consumer binding failure, including the affected contract path."""

    def __init__(self, code: str, path: str, detail: str):
        self.code = code
        self.path = path
        super().__init__(f"{code}: {path}: {detail}")


def _bindings(config: Mapping[str, Any]) -> Iterator[tuple[dict, str, str]]:
    contract = config.get("contract") or {}
    if not isinstance(contract, Mapping):
        return
    for section, direction in (("observations", "publish"), ("actions", "subscribe")):
        for index, item in enumerate(contract.get(section, []) or []):
            if not isinstance(item, dict):
                continue
            path = f"contract.{section}[{index}] ({item.get('key', '?')})"
            if section == "actions":
                item = item.get("publish") or {}
                path += ".publish"
            if not isinstance(item, Mapping):
                continue
            if "interface" not in item:
                continue
            interface_id = item["interface"]
            if not isinstance(interface_id, str) or not interface_id or interface_id.strip() != interface_id:
                raise InterfaceBindingError("invalid_interface", path, "interface must be a non-empty logical ID")
            yield item, f"{path} interface={interface_id!r}", direction


def required_interface_ids(config: Mapping[str, Any]) -> list[str]:
    """Return all explicitly bound interfaces, in contract order (never guessed from topics)."""
    return list(dict.fromkeys(item["interface"] for item, _, _ in _bindings(config)))


def _requirements(value: Any, path: str) -> Mapping[str, Any]:
    from robot_runtime.interface_description import validate_interface_requirements

    try:
        return validate_interface_requirements(value)
    except ValueError as exc:
        raise InterfaceBindingError("invalid_requires", path, str(exc)) from exc


def required_interface_requirements(config: Mapping[str, Any]) -> dict:
    requirements = {}
    for item, path, _ in _bindings(config):
        merged = requirements.setdefault(item["interface"], {})
        for key, value in _requirements(item.get("requires"), path).items():
            if key in merged and key != "min_fps" and merged[key] != value:
                raise InterfaceBindingError("contradictory_requirements", path, f"different constraints for {key}")
            merged[key] = max(value, merged.get(key, value)) if key == "min_fps" else value
    return requirements


def bind_robot_interfaces(
    config: dict[str, Any], descriptor: dict[str, Any], *, require_ready: bool = False
) -> dict[str, Any]:
    """Return a bound copy, preserving policy preprocessing and all unrelated configuration.

    ``require_ready`` is used by live launch after the runtime waiter. Offline
    binding may use configured profiles, but records their uncertainty explicitly.
    A ready observation with an unknown FPS never borrows a configured FPS.
    """
    from robot_runtime.interface_description import description_digest, validate_description

    try:
        validate_description(descriptor)
    except ValueError as exc:
        raise InterfaceBindingError("invalid_descriptor", "runtime.interface_description", str(exc)) from exc

    result = copy.deepcopy(config)
    runtime = result.get("runtime") or {}
    if not isinstance(runtime, dict):
        raise InterfaceBindingError("invalid_runtime", "runtime", "must be a mapping")
    result["runtime"] = runtime
    identity = descriptor["robot"]
    model = descriptor.get("model")
    if runtime.get("require_model") and model is None:
        raise InterfaceBindingError("model_required", "runtime.interface_description.model", "public model is absent")
    if model is not None:
        if result.get("robot_model") is not None and result["robot_model"] != model:
            raise InterfaceBindingError("model_mismatch", "robot_model", "differs from the runtime description")
        for name, joints in (result.get("joints") or {}).items():
            if model["joint_groups"].get(name) != joints:
                raise InterfaceBindingError("model_mismatch", f"joints.{name}", "differs from public joint groups")
        result["robot_model"] = copy.deepcopy(model)
        result["joints"] = copy.deepcopy(model["joint_groups"])
        declared_moveit = result.get("moveit") or {}
        for name, frame in model["frames"].items():
            if name in declared_moveit and declared_moveit[name] != frame:
                raise InterfaceBindingError(
                    "model_mismatch",
                    f"moveit.{name}",
                    "differs from public runtime frames; remove the application override or fix the runtime profile",
                )
        result["moveit"] = {**declared_moveit, **model["frames"]}
    for label, expected, actual in (
        ("runtime.provider", runtime.get("provider"), identity["runtime_name"]),
        ("runtime.instance_id", runtime.get("instance_id"), identity["id"]),
        ("runtime.runtime_version", runtime.get("runtime_version"), identity["runtime_version"]),
        ("type", result.get("type"), identity["type"]),
    ):
        if expected is not None and expected != actual:
            raise InterfaceBindingError("identity_mismatch", label, f"expected {expected!r}, described {actual!r}")
    execution = {"hardware": "physical", "simulation": "simulated"}.get(runtime.get("target"))
    if execution and execution != descriptor["execution"]:
        raise InterfaceBindingError(
            "identity_mismatch", "runtime.target", f"expected {execution!r}, described {descriptor['execution']!r}"
        )

    digest = description_digest(descriptor)
    for item, path, direction in _bindings(result):
        interface_id = item["interface"]
        interface = descriptor["interfaces"].get(interface_id)
        if interface is None:
            raise InterfaceBindingError("unknown_interface", path, "ID is not in the runtime description")
        if interface["kind"] != "topic":
            raise InterfaceBindingError("kind_mismatch", path, f"expected topic, described {interface['kind']!r}")
        if interface["direction"] != direction:
            raise InterfaceBindingError(
                "direction_mismatch", path, f"expected provider {direction!r}, described {interface['direction']!r}"
            )
        for field, source in (("topic", "endpoint"), ("type", "message_type")):
            explicit = item.get(field)
            if explicit is not None and explicit != interface[source]:
                raise InterfaceBindingError(
                    f"{field}_mismatch", path, f"explicit {field} {explicit!r} differs from {interface[source]!r}"
                )
            item[field] = interface[source]

        qos = copy.deepcopy(interface["qos"])
        explicit_qos = item.get("qos")
        if explicit_qos is not None:
            if not isinstance(explicit_qos, Mapping) or set(explicit_qos) - set(qos):
                raise InterfaceBindingError("qos_mismatch", path, "unsupported explicit QoS fields")
            for field, value in explicit_qos.items():
                if field == "depth":
                    valid = type(value) is int and value > 0
                else:
                    valid = (
                        value
                        in {
                            "reliability": ("best_effort", "reliable"),
                            "durability": ("volatile", "transient_local"),
                            "history": ("keep_last",),
                        }[field]
                    )
                if not valid:
                    raise InterfaceBindingError("qos_mismatch", path, f"invalid qos.{field}: {value!r}")
            qos.update(explicit_qos)
            # DDS requested/offered compatibility is directional; queue depth is local.
            offered, requested = (interface["qos"], qos) if direction == "publish" else (qos, interface["qos"])
            for field, stronger in (("reliability", "reliable"), ("durability", "transient_local")):
                if requested[field] == stronger and offered[field] != stronger:
                    raise InterfaceBindingError(
                        "qos_mismatch", path, f"qos.{field}: requested {requested[field]!r}, offered {offered[field]!r}"
                    )
        item["qos"] = qos

        # Selector <-> joint consistency. `<field>.<joint>` selectors resolve
        # by joint name (tensormsg dot_get) and array-typed command streams
        # consume selector values in the interface's declared joint order, so
        # a selector that names no joint of the bound interface, or a count
        # mismatch on a command stream, is a wiring error that must fail at
        # binding time instead of silently reading the wrong joint.
        joint_names = interface.get("joint_names")
        selector = item.get("selector")
        selectors = selector.get("names") if isinstance(selector, Mapping) else None
        if joint_names and selectors:
            if interface["message_type"] in ("sensor_msgs/msg/JointState", "ibrobot_msgs/msg/JointCurrent"):
                invalid = [
                    str(name)
                    for name in selectors
                    if (str(name).partition(".")[2] if "." in str(name) else str(name)) not in set(joint_names)
                ]
                if invalid:
                    raise InterfaceBindingError(
                        "selector_mismatch",
                        path,
                        f"selector names {invalid} are not joints of {interface_id}: {sorted(joint_names)}",
                    )
            elif direction == "subscribe" and len(selectors) != len(joint_names):
                raise InterfaceBindingError(
                    "selector_mismatch",
                    path,
                    f"{len(selectors)} selector values for {len(joint_names)} joints of {interface_id}",
                )

        state = descriptor.get("states", {}).get(interface_id, {})
        ready = state.get("state", "unknown") == "ready"
        observed = state.get("observed_profile") if ready else None
        is_image = interface["message_type"] == "sensor_msgs/msg/Image"
        if require_ready and direction == "publish" and (not ready or (is_image and not observed)):
            raise InterfaceBindingError(
                "interface_not_ready", path, f"state={state.get('state', 'unknown')}: {state.get('detail', '')}"
            )
        configured = interface.get("configured_profile")
        profile = observed if observed is not None else configured
        profile_origin = "observed" if observed is not None else "configured" if configured is not None else "unknown"
        observed_frame = state.get("observed_frame_id") if ready else None
        uncertainty = []
        if direction == "publish" and not ready:
            uncertainty.append("readiness_not_verified")
        if is_image:
            if observed is None:
                uncertainty.append("profile_not_observed")
            if not observed_frame:
                uncertainty.append("frame_not_observed")
            for field in ("width", "height", "encoding", "fps"):
                if (profile or {}).get(field) in (None, ""):
                    uncertainty.append(f"{field}_unknown")

        from robot_runtime.interface_description import check_interface_requirements

        try:
            check_interface_requirements(interface, profile, _requirements(item.get("requires"), path))
        except InterfaceBindingError:
            raise
        except ValueError as exc:
            raise InterfaceBindingError("requirement_unsatisfied", path, f"{exc} ({profile_origin})") from exc
        item["_interface_source"] = {
            "id": interface_id,
            "description_digest": digest,
            "state": state.get("state", "unknown"),
            "profile_origin": profile_origin,
            "profile": copy.deepcopy(profile),
            "configured_profile": copy.deepcopy(configured),
            "observed_profile": copy.deepcopy(observed),
            "frame_id": observed_frame or interface.get("frame_id"),
            "observed_frame_id": observed_frame,
            "camera_info_topic": interface.get("camera_info_topic"),
            "qos": copy.deepcopy(interface["qos"]),
            "uncertainty": uncertainty,
        }

    runtime["interface_description"] = copy.deepcopy(descriptor)
    result.pop("_interfaces_deferred", None)
    return result


def resolve_robot_interfaces(config: dict[str, Any], *, defer: bool = False) -> dict[str, Any]:
    """Resolve an embedded or file-backed snapshot. Deferral is explicit and structural only."""
    bindings = list(_bindings(config))
    if not bindings and not (config.get("runtime") or {}).get("require_model"):
        return config
    for item, path, _ in bindings:
        _requirements(item.get("requires"), path)
    if defer:
        result = copy.deepcopy(config)
        result["_interfaces_deferred"] = True
        return result
    runtime = config.get("runtime") or {}
    if not isinstance(runtime, Mapping):
        raise InterfaceBindingError("invalid_runtime", "runtime", "must be a mapping")
    descriptor = runtime.get("interface_description")
    if descriptor is None:
        raise InterfaceBindingError(
            "description_required",
            "runtime.interface_description",
            "logical interfaces require a snapshot offline or explicit deferred live launch binding",
        )
    if isinstance(descriptor, str | Path):
        from robot_runtime.interface_description import load_description

        path = Path(descriptor).expanduser()
        if not path.is_absolute():
            path = Path(config.get("_config_path", ".")).parent / path
        try:
            descriptor = load_description(path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise InterfaceBindingError("invalid_descriptor", str(path), str(exc)) from exc
    return bind_robot_interfaces(config, descriptor)
