"""Launch-time navigation admission for runtime-provider stages.

A stage declares ``navigation.open_gate_on_startup`` when its launch may drive
the base from boot (mapping/navigation/hybrid sessions with an operator
standing by). The runtime still boots idle with the navigation gate closed --
``lekiwi-navigation-integration`` requires the closed default regardless of the
initial mode -- so this one-shot client performs the admission explicitly
after the runtime is ready:

1. select the stage's runtime mode through the contract mode service
   (``/runtime/set_mode``), waiting for ``RuntimeStatus`` to confirm it;
2. open the navigation gate through the contract gate service
   (``/motion_mode/set_navigation_enabled``) and wait for the contract
   acknowledgment topic (``/motion_mode/base_navigation_enabled``).

Every step is a contract call, so the opening is auditable and the runtime's
clear-and-stop-before-ack semantics apply. Any timeout or rejection exits
nonzero so the launch fails loudly: cross-robot semantics differ (X2 rejects
enabling the gate while stop is latched; openspec
lekiwi-application-migration design A4) and the difference must surface
rather than be retried away.
"""

import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Bool
from std_srvs.srv import SetBool

from ibrobot_msgs.msg import RuntimeStatus
from ibrobot_msgs.srv import SetRuntimeMode
from robot_runtime import contract as C


class NavigationAdmission(Node):
    def __init__(self):
        super().__init__("navigation_admission")
        self.declare_parameter("runtime_mode", "base_navigation")
        self.declare_parameter("open_gate", True)
        self.declare_parameter("timeout_s", 60.0)
        self.declare_parameter("status_topic", C.STATUS_TOPIC)
        self.declare_parameter("set_mode_service", C.SET_MODE_SERVICE)
        self.declare_parameter("enable_service", C.NAVIGATION_ENABLE_SERVICE)
        self.declare_parameter("ack_topic", C.NAVIGATION_ACK_TOPIC)

        self._runtime_mode = str(self.get_parameter("runtime_mode").value).strip()
        self._open_gate = bool(self.get_parameter("open_gate").value)
        self._timeout = max(float(self.get_parameter("timeout_s").value), 1.0)
        self._status = None

        reliable = QoSProfile(depth=10)
        self.create_subscription(
            RuntimeStatus,
            str(self.get_parameter("status_topic").value),
            self._on_status,
            reliable,
        )
        self._ack_seen = None
        self.create_subscription(
            Bool,
            str(self.get_parameter("ack_topic").value),
            lambda msg: setattr(self, "_ack_seen", bool(msg.data)),
            reliable,
        )
        self._mode_client = self.create_client(SetRuntimeMode, str(self.get_parameter("set_mode_service").value))
        self._gate_client = self.create_client(SetBool, str(self.get_parameter("enable_service").value))

    def _on_status(self, msg: RuntimeStatus) -> None:
        self._status = msg

    def _spin_until(self, predicate, what: str) -> None:
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            rclpy.spin_once(self, timeout_sec=0.2)
        raise TimeoutError(f"navigation admission timed out waiting for {what}")

    def _call(self, client, request, what: str):
        if not client.wait_for_service(timeout_sec=self._timeout):
            raise TimeoutError(f"navigation admission: service unavailable for {what}")
        future = client.call_async(request)
        self._spin_until(lambda: future.done(), f"the {what} response")
        response = future.result()
        if response is None or not bool(getattr(response, "success", False)):
            message = str(getattr(response, "message", "") or "").strip()
            raise RuntimeError(f"navigation admission rejected by {what}: {message or response!r}")
        return response

    def run(self) -> int:
        if not self._runtime_mode:
            raise ValueError("navigation admission requires a non-empty runtime_mode")
        self._spin_until(lambda: self._status is not None, "the first RuntimeStatus")
        if self._status.active_mode != self._runtime_mode:
            request = SetRuntimeMode.Request()
            request.mode = self._runtime_mode
            self._call(self._mode_client, request, f"mode service ({self._runtime_mode!r})")
            self._spin_until(
                lambda: self._status is not None and self._status.active_mode == self._runtime_mode,
                f"RuntimeStatus reporting mode {self._runtime_mode!r}",
            )
        self.get_logger().info(f"runtime mode {self._runtime_mode!r} active")
        if self._open_gate:
            request = SetBool.Request()
            request.data = True
            # Drop any acknowledgment seen before the call: the ack topic is a
            # live state stream, so a True from before this enable must not
            # satisfy the wait for the enable's own acknowledgment.
            self._ack_seen = None
            self._call(self._gate_client, request, "navigation gate service")
            self._spin_until(lambda: self._ack_seen is True, "the navigation gate acknowledgment")
        self.get_logger().info("navigation gate open; base commands are admitted")
        return 0


def main(args=None):
    rclpy.init(args=args)
    node = NavigationAdmission()
    try:
        return node.run()
    except Exception as exc:  # noqa: BLE001 - the launch must fail loudly on any admission error
        node.get_logger().error(f"{exc}")
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
