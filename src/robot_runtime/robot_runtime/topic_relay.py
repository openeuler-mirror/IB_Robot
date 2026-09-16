"""Minimal topic relay for message-compatible topic renames (runtime side).

robot_runtime copy used by runtime peripheral composition (D12): the runtime
must not depend on robot_config, so the relay executable shipped by the
contract layer keeps the camera contract stable when upstream drivers publish
under driver-specific topic trees. robot_config keeps its own copy for
provider-less legacy configurations.
"""

import importlib
import sys

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


def _load_message_type(type_name: str):
    parts = type_name.split("/")
    if len(parts) != 3 or parts[1] != "msg":
        raise ValueError("message type must be in the form '<package>/msg/<MessageName>'")

    package_name, _, message_name = parts
    module = importlib.import_module(f"{package_name}.msg")
    return getattr(module, message_name)


class TopicRelay(Node):
    """Relays messages from one topic to another, optionally normalizing frame_id."""

    def __init__(
        self,
        source_topic: str,
        target_topic: str,
        message_type: str,
        target_frame_id: str | None = None,
    ):
        super().__init__("topic_relay")

        self._target_frame_id = target_frame_id
        msg_cls = _load_message_type(message_type)
        self._publisher = self.create_publisher(msg_cls, target_topic, qos_profile_sensor_data)
        self._subscription = self.create_subscription(
            msg_cls,
            source_topic,
            self._relay,
            qos_profile_sensor_data,
        )

        frame_suffix = f" (frame_id -> {target_frame_id})" if target_frame_id else ""
        self.get_logger().info(f"Relaying {message_type}: {source_topic} -> {target_topic}{frame_suffix}")

    def _relay(self, msg):
        if self._target_frame_id and hasattr(msg, "header"):
            msg.header.frame_id = self._target_frame_id
        self._publisher.publish(msg)


def main(argv=None):
    raw_args = argv if argv is not None else sys.argv[1:]
    ros_args_index = raw_args.index("--ros-args") if "--ros-args" in raw_args else len(raw_args)
    args = raw_args[:ros_args_index]

    if len(args) not in (3, 4):
        raise SystemExit("usage: topic_relay <source_topic> <target_topic> <package/msg/MessageName> [target_frame_id]")

    source_topic, target_topic, message_type = args[:3]
    target_frame_id = args[3] if len(args) == 4 else None

    rclpy.init(args=raw_args[ros_args_index:])
    node = TopicRelay(source_topic, target_topic, message_type, target_frame_id)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
