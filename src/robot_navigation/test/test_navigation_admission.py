"""Graph tests for the launch-time navigation admission client (R2-3).

The acknowledgment topic is a live state stream (the LeKiwi base node
heartbeats it at ~10 Hz). A ``True`` observed before the gate-enable call must
not satisfy the wait for that enable's own acknowledgment, so the client
resets ``_ack_seen`` immediately before the call. The tests drive the real
node against mock contract servers:

- a stale ``True`` acknowledgment plus a gate service that never acknowledges
  makes admission time out -- without the reset, the stale value would satisfy
  the wait instantly and admission would wrongly report success;
- a gate service that acknowledges on the response completes admission;
- an empty runtime_mode is rejected before any ROS interaction matters.

The stale value is injected directly (``node._ack_seen = True``): it is
byte-for-byte what a heartbeat-style ``True`` leaves behind, and it avoids
racing reliable-QoS redelivery of real messages against the wait.
"""

import threading
import time

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Bool
from std_srvs.srv import SetBool

from ibrobot_msgs.msg import RuntimeStatus
from ibrobot_msgs.srv import SetRuntimeMode
from robot_navigation.navigation_admission import NavigationAdmission


class _MockRuntime:
    """Serves the contract surfaces the admission client talks to."""

    def __init__(self, *, acknowledge: bool):
        self.node = rclpy.create_node("mock_runtime_for_admission")
        self.acknowledge = acknowledge
        self.gate_calls = []
        self.status_pub = self.node.create_publisher(RuntimeStatus, "/runtime_status", 10)
        self.ack_pub = self.node.create_publisher(Bool, "/motion_mode/base_navigation_enabled", 10)
        self.gate_service = self.node.create_service(SetBool, "/motion_mode/set_navigation_enabled", self._on_gate)
        self.mode_service = self.node.create_service(SetRuntimeMode, "/runtime/set_mode", self._on_set_mode)
        self.status_timer = self.node.create_timer(0.05, self._publish_status)
        self._publish_status()

    def _publish_status(self):
        status = RuntimeStatus()
        status.runtime_name = "mock"
        status.lifecycle = "ACTIVE"
        status.active_mode = "base_navigation"
        status.declared_modes = ["idle", "base_navigation"]
        self.status_pub.publish(status)

    def _on_set_mode(self, request, response):
        response.success = True
        response.message = f"mode {request.mode!r} active"
        return response

    def _on_gate(self, request, response):
        self.gate_calls.append((time.monotonic(), bool(request.data)))
        if self.acknowledge:
            self.ack_pub.publish(Bool(data=True))
        response.success = True
        response.message = "navigation enabled"
        return response


@pytest.fixture
def rclpy_ctx():
    rclpy.init()
    yield
    rclpy.shutdown()


def _run_in_thread(node: NavigationAdmission, join_timeout: float):
    result = {}

    def _target():
        try:
            result["rc"] = node.run()
        except Exception as exc:  # noqa: BLE001 - surfaced to the test
            result["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=join_timeout)
    return result, thread


def _start_mock(acknowledge: bool):
    mock = _MockRuntime(acknowledge=acknowledge)
    executor = SingleThreadedExecutor()
    executor.add_node(mock.node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    return mock, executor, spin_thread


def _stop(mock, executor, spin_thread, node):
    node.destroy_node()
    executor.shutdown()
    spin_thread.join(timeout=5.0)
    mock.node.destroy_node()


def test_stale_ack_true_cannot_satisfy_the_enable_wait(rclpy_ctx):
    """A pre-call True plus a silent gate service must time out.

    This is the R2-3 regression: before the reset-before-call, the stale True
    satisfied the wait immediately and admission returned success even though
    the enable was never acknowledged.
    """
    mock, executor, spin_thread = _start_mock(acknowledge=False)
    node = NavigationAdmission()
    node._timeout = 2.0
    try:
        time.sleep(0.5)  # discovery
        # Simulate a heartbeat-style True observed before the enable.
        node._ack_seen = True

        result, thread = _run_in_thread(node, join_timeout=10.0)

        assert not thread.is_alive()
        assert isinstance(result.get("error"), TimeoutError), result
        assert len(mock.gate_calls) == 1, mock.gate_calls
        assert mock.gate_calls[0][1] is True
    finally:
        _stop(mock, executor, spin_thread, node)


def test_acknowledged_enable_completes_admission(rclpy_ctx):
    mock, executor, spin_thread = _start_mock(acknowledge=True)
    node = NavigationAdmission()
    node._timeout = 5.0
    try:
        time.sleep(0.5)  # discovery

        result, thread = _run_in_thread(node, join_timeout=10.0)

        assert not thread.is_alive()
        assert result.get("rc") == 0, result
        assert len(mock.gate_calls) == 1, mock.gate_calls
    finally:
        _stop(mock, executor, spin_thread, node)


def test_admission_rejects_an_empty_runtime_mode():
    rclpy.init()
    node = NavigationAdmission()
    node._runtime_mode = ""
    try:
        with pytest.raises(ValueError, match="non-empty runtime_mode"):
            node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
