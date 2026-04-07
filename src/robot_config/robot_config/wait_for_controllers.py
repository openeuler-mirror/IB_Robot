#!/usr/bin/env python3

"""Wait until the requested ros2_control controllers reach the active state."""

import argparse
import sys
import time
from typing import Iterable

import rclpy
from controller_manager_msgs.srv import ListControllers
from rclpy.node import Node


def missing_inactive_controllers(
    controllers: Iterable[object], required_names: Iterable[str]
) -> list[str]:
    """Return required controllers that are missing or not active."""
    states = {controller.name: controller.state for controller in controllers}
    return [
        name
        for name in required_names
        if states.get(name) != "active"
    ]


class ControllerWaiter(Node):
    """Poll controller_manager until all required controllers are active."""

    def __init__(self, controller_manager: str, required_names: list[str]) -> None:
        super().__init__("wait_for_controllers")
        self._required_names = required_names
        service_name = f"{controller_manager.rstrip('/')}/list_controllers"
        self._client = self.create_client(ListControllers, service_name)

    def wait_until_ready(
        self,
        timeout: float,
        service_wait_timeout: float,
        poll_period: float,
    ) -> int:
        deadline = time.monotonic() + timeout

        while rclpy.ok() and time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            if self._client.wait_for_service(timeout_sec=min(service_wait_timeout, remaining)):
                break
        else:
            self.get_logger().error(
                f"Timed out after {timeout:.1f}s waiting for controller_manager services."
            )
            return 1

        while rclpy.ok() and time.monotonic() < deadline:
            future = self._client.call_async(ListControllers.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=service_wait_timeout)
            if not future.done() or future.result() is None:
                self.get_logger().warning("list_controllers call timed out, retrying...")
                time.sleep(poll_period)
                continue

            pending = missing_inactive_controllers(
                future.result().controller,
                self._required_names,
            )
            if not pending:
                self.get_logger().info(
                    "Controllers are active: " + ", ".join(self._required_names)
                )
                return 0

            self.get_logger().info(
                "Waiting for controllers to become active: " + ", ".join(pending)
            )
            time.sleep(poll_period)

        self.get_logger().error(
            "Timed out waiting for controllers: " + ", ".join(self._required_names)
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv
    parser = argparse.ArgumentParser(
        description="Wait for ros2_control controllers to reach the active state."
    )
    parser.add_argument("controllers", nargs="+", help="Controller names to wait for.")
    parser.add_argument(
        "--controller-manager",
        default="controller_manager",
        help="Controller manager name or relative namespace.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Wall-clock timeout in seconds.",
    )
    parser.add_argument(
        "--service-wait-timeout",
        type=float,
        default=5.0,
        help="Per-attempt timeout when waiting for or calling services.",
    )
    parser.add_argument(
        "--poll-period",
        type=float,
        default=0.5,
        help="Polling interval while controller states are still pending.",
    )
    args, _unknown = parser.parse_known_args(argv[1:])

    rclpy.init(args=argv)
    node = ControllerWaiter(args.controller_manager, args.controllers)
    try:
        return node.wait_until_ready(
            timeout=args.timeout,
            service_wait_timeout=args.service_wait_timeout,
            poll_period=args.poll_period,
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
