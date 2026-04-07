#!/usr/bin/env python3

"""Wait until a Clock message is observed on a topic."""

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock


class ClockWaiter(Node):
    """Block until a Clock message arrives or the timeout expires."""

    def __init__(self, topic: str) -> None:
        super().__init__("wait_for_clock")
        self._seen_clock = False
        self.create_subscription(Clock, topic, self._clock_cb, 10)

    def _clock_cb(self, _msg: Clock) -> None:
        self._seen_clock = True

    @property
    def seen_clock(self) -> bool:
        return self._seen_clock


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv
    parser = argparse.ArgumentParser(description="Wait for a Clock message on a ROS topic.")
    parser.add_argument("--topic", default="/clock", help="Clock topic to wait for.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Wall-clock timeout in seconds.",
    )
    parser.add_argument(
        "--poll-period",
        type=float,
        default=0.1,
        help="Spin interval while waiting for the topic.",
    )
    args, _unknown = parser.parse_known_args(argv[1:])

    rclpy.init(args=argv)
    node = ClockWaiter(args.topic)
    deadline = time.monotonic() + args.timeout

    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=args.poll_period)
            if node.seen_clock:
                node.get_logger().info(f"Received clock on {args.topic}.")
                return 0

        node.get_logger().error(
            f"Timed out after {args.timeout:.1f}s waiting for a Clock message on {args.topic}."
        )
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
