"""Resolve one teleoperation target exclusively from a bound public snapshot."""

import math


def resolve_public_target(config):
    teleop = config["teleoperation"]
    target = teleop["target"]
    model = config["robot_model"]
    interfaces = config["runtime"]["interface_description"]["interfaces"]
    group = target["group"]
    arm = list(model["joint_groups"][group])
    gripper = list(model["joint_groups"][target["gripper_group"]])
    if group != "arm" or not arm or len(gripper) != 1 or set(arm) & set(gripper):
        raise ValueError("teleoperation requires one explicit non-overlapping arm and gripper target")
    selected = {}
    types = {
        "joints": "sensor_msgs/msg/JointState",
        "pose": "geometry_msgs/msg/PoseStamped",
        "linear": "geometry_msgs/msg/Vector3Stamped",
        "angular": "geometry_msgs/msg/Vector3Stamped",
        "start": "std_srvs/srv/Trigger",
        "stop": "std_srvs/srv/Trigger",
        "home": "ibrobot_msgs/action/ArmReturnHome",
        "lease": "std_msgs/msg/Empty",
    }
    for key, message_type in types.items():
        item = interfaces[target["interfaces"][key]]
        units = {
            "joints": {"position": "rad"},
            "pose": {"position": "m", "orientation": "quaternion"},
            "linear": {"linear": "m/s"},
            "angular": {"angular": "rad/s"},
        }
        kind = "action" if key == "home" else "service" if key in ("start", "stop") else "topic"
        if (
            item["message_type"] != message_type
            or item["target_group"] != group
            or item["kind"] != kind
            or item["direction"] != ("subscribe" if kind == "topic" else "serve")
            or (key in units and item.get("units") != units[key])
            or item["base_frame"] != model["frames"]["base_link"]
            or item["tool_frame"] != model["frames"]["ee_link"]
            or item["joint_names"] != (arm + gripper if key == "joints" else arm)
        ):
            raise ValueError(f"public {key} interface target/type/frame/order mismatch")
        selected[key] = item
        if not math.isfinite(item["command_stale_s"]) or not 0 < item["command_stale_s"] <= 1:
            raise ValueError("public command liveness bound must be finite and at most one second")
    if selected["pose"].get("pose_reference") != "clutch_relative":
        raise ValueError("teleoperation requires clutch_relative pose semantics")
    limits = {name: dict(model["joint_limits"][name]) for name in arm + gripper}
    for name, limit in teleop.get("safety", {}).get("joint_limits", {}).items():
        physical = limits[name]
        if not (physical["min"] <= limit["min"] < limit["max"] <= physical["max"]):
            raise ValueError("application limits may narrow but not enlarge public physical limits")
        limits[name] = dict(limit)
    # Relative narrowing: a margin applied inward on the public limits, so a
    # recalibration that shifts the physical range never conflicts with the
    # application configuration (no absolute snapshots on the app side).
    margin = float(teleop.get("safety", {}).get("joint_limits_margin", 0.0))
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("safety joint_limits_margin must be finite and non-negative")
    if margin:
        for name, limit in limits.items():
            narrowed = {"min": limit["min"] + margin, "max": limit["max"] - margin}
            if narrowed["min"] >= narrowed["max"]:
                raise ValueError(f"safety margin leaves no travel for joint {name!r}")
            limits[name] = narrowed
    conversion = model["joint_conversions"]["modes"]["range_m100_100"][gripper[0]]
    closed, opened = conversion["min"], conversion["max"]
    if not all(math.isfinite(v) for v in (closed, opened)) or closed == opened:
        raise ValueError("gripper conversion endpoints are missing or degenerate")
    return {
        "arm": arm,
        "gripper": gripper,
        "limits": limits,
        "closed": closed,
        "open": opened,
        "interfaces": selected,
        "frames": model["frames"],
    }


def map_gripper_ratio(value, closed, opened, limits):
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("gripper input must be a finite opening ratio in [0,1]")
    return min(limits["max"], max(limits["min"], closed + value * (opened - closed)))
