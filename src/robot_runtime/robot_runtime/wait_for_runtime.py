"""Wait for runtime readiness and reconcile declared vs. discovered capabilities.

Replaces wait_for_controllers as the upper-layer start condition. Exits non-zero with a
named list of missing capabilities when reconciliation fails; exits zero
when the runtime is ready and all required capabilities are discovered.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import rclpy
from rclpy.utilities import remove_ros_args

from ibrobot_msgs.msg import RuntimeStatus
from ibrobot_msgs.srv import GetRuntimeStatus
from robot_runtime.capabilities import requires_subset_of_discovered
from robot_runtime.contract import GET_STATUS_SERVICE
from robot_runtime.interface_description import check_interface_requirements, validate_description, write_description


def validate_runtime_snapshot(status, *, runtime_name, required=(), instance_id="", interfaces=(), requirements=None):
    """Check identity, capabilities and requested public streams before admitting consumers."""
    if status.runtime_name != runtime_name:
        raise ValueError(f"runtime identity {status.runtime_name!r} != expected {runtime_name!r}")
    if status.lifecycle not in ("ACTIVE", "DEGRADED"):
        raise ValueError(f"runtime lifecycle is {status.lifecycle!r}")
    missing = requires_subset_of_discovered(required, status.capabilities)
    if missing:
        raise ValueError(f"runtime {runtime_name!r} is missing required capabilities: {missing}")
    try:
        description = json.loads(status.interface_description_json)
    except (ValueError, AttributeError) as exc:
        raise ValueError("runtime published no valid public interface description") from exc
    validate_description(description)
    identity = description["robot"]
    if identity["runtime_name"] != status.runtime_name or identity["runtime_version"] != status.runtime_version:
        raise ValueError("descriptor identity does not match RuntimeStatus")
    if instance_id and identity["id"] != instance_id:
        raise ValueError(f"robot instance {identity['id']!r} != expected {instance_id!r}")
    for name in interfaces:
        spec = description["interfaces"].get(name)
        if spec is None:
            raise ValueError(f"runtime has no interface {name!r}")
        if spec["kind"] == "topic" and spec["direction"] == "publish":
            state = description["states"].get(name, {})
            if state.get("state") != "ready":
                raise ValueError(
                    f"interface {name!r} is {state.get('state', 'unknown')}: {state.get('detail', 'no sample')}"
                )
            if spec["qos"]["durability"] != "transient_local" and abs(time.time() - state["last_seen"]) > 3.0:
                raise ValueError(f"interface {name!r}: stale sample or unsynchronized runtime clock")
        try:
            source = description["states"].get(name, {}).get("observed_profile")
            check_interface_requirements(spec, source, (requirements or {}).get(name))
        except ValueError as exc:
            raise ValueError(f"interface {name!r}: {exc}") from exc
    return description


def validate_endpoint_graph(node, description, interfaces):
    """Required command/service/action declarations need matching ROS graph endpoints."""
    for name in interfaces:
        spec = description["interfaces"][name]
        endpoint, message_type = spec["endpoint"], spec["message_type"]
        if spec["kind"] == "topic":
            getter = (
                node.get_publishers_info_by_topic
                if spec["direction"] == "publish"
                else node.get_subscriptions_info_by_topic
            )
            found = False
            for info in getter(endpoint):
                if info.topic_type != message_type:
                    continue
                actual = {key: getattr(info.qos_profile, key).name.lower() for key in ("reliability", "durability")}
                offered, requested = (actual, spec["qos"]) if spec["direction"] == "publish" else (spec["qos"], actual)
                if all(
                    requested[key] != strict or offered[key] == strict
                    for key, strict in (("reliability", "reliable"), ("durability", "transient_local"))
                ):
                    found = True
                    break
        elif spec["kind"] == "service":
            found = message_type in dict(node.get_service_names_and_types()).get(endpoint, [])
        else:
            from rclpy.action import get_action_names_and_types

            found = message_type in dict(get_action_names_and_types(node)).get(endpoint, [])
        if not found:
            raise ValueError(
                f"interface {name!r}: ROS graph has no {message_type} {spec['direction']} endpoint {endpoint}"
            )


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-name", default="mock_runtime")
    parser.add_argument("--required", nargs="*", default=[], help="required capabilities")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--status-topic", default="/runtime_status")
    parser.add_argument("--status-service", default=GET_STATUS_SERVICE)
    parser.add_argument("--instance-id", default="")
    parser.add_argument(
        "--description-output", default="", help="atomically export the validated effective description"
    )
    parser.add_argument("--require-interfaces", nargs="*", default=[])
    parser.add_argument(
        "--interface-requirements",
        type=json.loads,
        default={},
        help="JSON mapping of interface IDs to consumer requirements",
    )
    raw_args = sys.argv if args is None else ["wait_for_runtime", *args]
    ns = parser.parse_args(remove_ros_args(args=raw_args)[1:])
    if not isinstance(ns.interface_requirements, dict):
        parser.error("--interface-requirements must be an object")
    ns.require_interfaces = list(dict.fromkeys([*ns.require_interfaces, *ns.interface_requirements]))

    rclpy.init(args=raw_args[1:])
    node = rclpy.create_node("wait_for_runtime")
    latest = None

    def on_status(msg: RuntimeStatus):
        nonlocal latest
        if msg.runtime_name == ns.runtime_name:
            latest = msg

    node.create_subscription(RuntimeStatus, ns.status_topic, on_status, 10)

    client = node.create_client(GetRuntimeStatus, ns.status_service)
    future = None
    next_query = 0.0
    error = f"runtime {ns.runtime_name!r} did not publish status"
    try:
        deadline = time.monotonic() + ns.timeout
        while time.monotonic() < deadline and rclpy.ok():
            if future is not None and future.done():
                try:
                    on_status(future.result().status)
                except Exception as exc:
                    error = f"status query failed: {exc}"
                future = None
            if future is None and time.monotonic() >= next_query and client.service_is_ready():
                future = client.call_async(GetRuntimeStatus.Request())
                next_query = time.monotonic() + 1.0
            if latest is not None:
                try:
                    description = validate_runtime_snapshot(
                        latest,
                        runtime_name=ns.runtime_name,
                        required=ns.required,
                        instance_id=ns.instance_id,
                        interfaces=ns.require_interfaces,
                        requirements=ns.interface_requirements,
                    )
                    validate_endpoint_graph(node, description, ns.require_interfaces)
                    if ns.description_output:
                        write_description(description, ns.description_output)
                    print(f"Runtime ready: {ns.runtime_name} ({description['digest']}, {latest.lifecycle})")
                    return 0
                except (ValueError, OSError) as exc:
                    error = str(exc)
            rclpy.spin_once(node, timeout_sec=0.1)
        print(f"ERROR: {error} (timeout {ns.timeout}s)", file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
