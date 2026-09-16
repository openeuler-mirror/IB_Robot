import importlib
import json
import threading
from types import MethodType, SimpleNamespace

import pytest
from std_msgs.msg import Bool

from robot_teleop.teleop_node import TeleopNode, connect_device_or_raise

teleop_node_module = importlib.import_module("robot_teleop.teleop_node")


class _Logger:
    def __init__(self):
        self.warnings = []
        self.errors = []
        self.infos = []

    def info(self, message):
        self.infos.append(message)

    def warn(self, message):
        self.warnings.append(message)

    def error(self, message):
        self.errors.append(message)


class _Device:
    def __init__(self):
        self.estop_calls = 0
        self.estop_release_calls = 0
        self.is_connected = True

    def emergency_stop(self):
        self.estop_calls += 1

    def emergency_stop_released(self):
        self.estop_release_calls += 1


def _estop_harness():
    logger = _Logger()
    node = SimpleNamespace(
        estop_active=False,
        _estop_state_lock=threading.Lock(),
        _estop_stop_pending=False,
        _estop_release_pending=False,
        _device_lock=threading.Lock(),
        device=_Device(),
        get_logger=lambda: logger,
    )
    node._estop_is_active = MethodType(TeleopNode._estop_is_active, node)
    node._try_dispatch_estop = MethodType(TeleopNode._try_dispatch_estop, node)
    return node, logger


def test_device_connection_failure_aborts_node_startup():
    device = SimpleNamespace(connect=lambda: False)

    with pytest.raises(RuntimeError, match="connection failed"):
        connect_device_or_raise(device)


def test_successful_device_connection_returns_normally():
    device = SimpleNamespace(connect=lambda: True)

    connect_device_or_raise(device)


def test_bool_estop_latches_stops_once_and_releases_explicitly():
    node, logger = _estop_harness()

    TeleopNode.estop_callback(node, Bool(data=True))
    TeleopNode.estop_callback(node, Bool(data=True))

    assert node.estop_active is True
    assert node.device.estop_calls == 1

    TeleopNode.estop_callback(node, Bool(data=False))

    assert node.estop_active is False
    assert node.device.estop_release_calls == 1
    assert any("released" in message for message in logger.warnings)


def test_estop_callback_does_not_block_when_control_loop_owns_device_lock():
    node, _logger = _estop_harness()
    node._device_lock.acquire()
    try:
        TeleopNode.estop_callback(node, Bool(data=True))
        TeleopNode.estop_callback(node, Bool(data=False))

        assert node.estop_active is True
        assert node._estop_stop_pending is True
        assert node._estop_release_pending is True
        assert node.device.estop_calls == 0
    finally:
        node._device_lock.release()

    assert node._try_dispatch_estop() is True
    assert node.device.estop_calls == 1
    assert node.device.estop_release_calls == 1
    assert node.estop_active is False


def test_estop_arriving_during_device_read_discards_inflight_command():
    node, _logger = _estop_harness()
    arm_messages = []
    gripper_messages = []
    node.arm_joint_names = ["1"]
    node.gripper_joint_names = ["6"]
    node.arm_cmd_pub = SimpleNamespace(publish=arm_messages.append)
    node.gripper_cmd_pub = SimpleNamespace(publish=gripper_messages.append)
    node.safety_filter = SimpleNamespace(apply_limits=lambda targets: targets)

    def get_joint_targets():
        TeleopNode.estop_callback(node, Bool(data=True))
        return {"1": 0.5, "6": 0.25}

    node.device.get_joint_targets = get_joint_targets

    TeleopNode.control_loop_callback(node)

    assert node.device.estop_calls == 1
    assert node.estop_active is True
    assert arm_messages == []
    assert gripper_messages == []


def test_main_keeps_ros_alive_until_device_stop_is_acknowledged(monkeypatch):
    events = []

    class _ShutdownNode:
        def __init__(self):
            self.stop_complete = False

        def disconnect_device(self):
            events.append("disconnect")

        def device_shutdown_complete(self):
            return self.stop_complete

        def destroy_node(self):
            events.append("destroy")

        def get_logger(self):
            return SimpleNamespace(error=lambda message: events.append(message))

    node = _ShutdownNode()

    def init(**kwargs):
        events.append(("init", kwargs["signal_handler_options"]))

    def spin(_node):
        raise KeyboardInterrupt

    def spin_once(_node, timeout_sec):
        assert timeout_sec == 0.05
        events.append("spin_once")
        node.stop_complete = True

    monkeypatch.setattr(teleop_node_module, "TeleopNode", lambda: node)
    monkeypatch.setattr(teleop_node_module.rclpy, "init", init)
    monkeypatch.setattr(teleop_node_module.rclpy, "spin", spin)
    monkeypatch.setattr(teleop_node_module.rclpy, "spin_once", spin_once)
    monkeypatch.setattr(teleop_node_module.rclpy, "ok", lambda: True)
    monkeypatch.setattr(teleop_node_module.rclpy, "shutdown", lambda: events.append("shutdown"))

    teleop_node_module.main()

    assert events[0] == ("init", teleop_node_module.SignalHandlerOptions.NO)
    assert events.index("disconnect") < events.index("spin_once") < events.index("destroy")


class _SafetyFilter:
    def __init__(self):
        self.seen = None

    def apply_limits(self, targets):
        self.seen = dict(targets)
        return dict(targets)


def _publish_harness(mapping, ratio_input=True, managed=False, targets=None):
    """Bind the shared control-loop mapping to a bare namespace node."""
    logger = _Logger()
    device = SimpleNamespace(is_connected=True, get_joint_targets=lambda: dict(targets or {}))

    def disable():
        node._managed_backend.is_enabled = False
        node._managed_backend.stops += 1

    node = SimpleNamespace(
        _device_lock=threading.Lock(),
        _estop_is_active=lambda: False,
        _try_dispatch_estop=lambda: None,
        device=device,
        _managed=managed,
        _managed_backend=SimpleNamespace(
            _home_pending=False, is_enabled=True, disable=disable, keepalive=lambda: None, stops=0
        ),
        _managed_config={"type": "leader_topic" if managed else "leader_arm"},
        _ratio_input=ratio_input,
        _gripper_mapping=mapping,
        gripper_joint_names=["6"],
        safety_filter=_SafetyFilter(),
        _publish_targets=lambda safe: node.published.update(safe),
        _update_diagnostics=lambda loop_time: None,
        get_logger=lambda: logger,
    )
    node.published = {}
    node.control_loop_callback = MethodType(TeleopNode.control_loop_callback, node)
    return node


def test_standalone_leader_ratio_is_mapped_to_calibrated_radians():
    node = _publish_harness((-0.6, 1.6, {"min": -2.0, "max": 2.0}), targets={"1": 0.3, "6": 0.75})
    node.control_loop_callback()
    assert node.published["6"] == pytest.approx(-0.6 + 0.75 * (1.6 - -0.6))


def test_standalone_ratio_without_endpoints_is_dropped_fail_closed():
    node = _publish_harness(None, targets={"1": 0.3, "6": 0.75})
    node.control_loop_callback()
    assert "6" not in node.published
    assert node.published["1"] == pytest.approx(0.3)


def test_out_of_range_ratio_drops_gripper_target_only():
    node = _publish_harness((-0.6, 1.6, {"min": -2.0, "max": 2.0}), targets={"1": 0.3, "6": 1.2})
    node.control_loop_callback()
    assert "6" not in node.published


def test_non_ratio_devices_bypass_the_gripper_mapping():
    node = _publish_harness((-0.6, 1.6, {"min": -2.0, "max": 2.0}), ratio_input=False, targets={"6": 0.4})
    node.control_loop_callback()
    assert node.published["6"] == pytest.approx(0.4)


def test_managed_ratio_mapping_uses_the_public_endpoints():
    mapping = (-0.6, 1.6, {"min": -2.0, "max": 2.0})
    node = _publish_harness(mapping, managed=True, targets={"1": 0.3, "6": 0.0})
    node.control_loop_callback()
    assert node.published["6"] == pytest.approx(-0.6)


@pytest.mark.parametrize(
    "connected,targets,mapped,reason",
    [
        (True, {}, True, "device returned no joint targets"),
        (False, {"6": 0.5}, True, "device unavailable or disconnected"),
        (None, {"6": 0.5}, True, "device unavailable or disconnected"),
        (True, {"6": 0.5}, False, "gripper ratio mapping unavailable"),
        (True, {"6": 1.2}, True, "invalid gripper ratio for '6': 1.2"),
        (True, {"6": float("nan")}, True, "invalid gripper ratio for '6': nan"),
    ],
)
def test_managed_input_failure_logs_once_per_disable(connected, targets, mapped, reason):
    mapping = (-0.6, 1.6, {"min": -2.0, "max": 2.0}) if mapped else None
    node = _publish_harness(mapping, managed=True, targets=targets)
    if connected is None:
        node.device = None
    else:
        node.device.is_connected = connected
        node.device.last_invalid_reason = "unexpected frame_id: 'wrong_units'"

    for _ in range(3):
        node.control_loop_callback()

    assert node._managed_backend.is_enabled is False
    assert node._managed_backend.stops == 1
    assert node.published == {}
    assert node.safety_filter.seen is None
    assert len(node.get_logger().warnings) == 1
    assert f"Disabling managed teleop: {reason}" in node.get_logger().warnings[0]
    if not targets:
        assert "last_invalid_reason=unexpected frame_id: 'wrong_units'" in node.get_logger().warnings[0]

    node._managed_backend.is_enabled = True  # Simulate explicit rearm; a new failure must log again.
    node.control_loop_callback()
    assert node._managed_backend.stops == 2
    assert len(node.get_logger().warnings) == 2
    assert node.published == {}


def test_fetch_runtime_gripper_model_parses_the_public_description(monkeypatch):
    monkeypatch.setattr(teleop_node_module.rclpy, "spin_until_future_complete", lambda *a, **k: None)
    description = {
        "model": {
            "joint_conversions": {"modes": {"range_m100_100": {"6": {"min": -0.6, "max": 1.6}}}},
            "joint_limits": {"6": {"min": -0.6, "max": 1.6}, "1": {"min": -2.0, "max": 2.0}},
        }
    }
    client = SimpleNamespace(
        wait_for_service=lambda timeout_sec: True,
        call_async=lambda request: SimpleNamespace(
            result=lambda: SimpleNamespace(status=SimpleNamespace(interface_description_json=json.dumps(description)))
        ),
    )
    node = SimpleNamespace(
        gripper_joint_names=["6"],
        create_client=lambda *args, **kwargs: client,
        get_logger=lambda: _Logger(),
    )
    fetch = MethodType(TeleopNode._fetch_runtime_gripper_model, node)
    assert fetch() == (-0.6, 1.6, {"6": {"min": -0.6, "max": 1.6}, "1": {"min": -2.0, "max": 2.0}})


def test_fetch_runtime_gripper_model_fails_closed_when_runtime_is_absent(monkeypatch):
    client = SimpleNamespace(wait_for_service=lambda timeout_sec: False)
    node = SimpleNamespace(
        gripper_joint_names=["6"],
        create_client=lambda *args, **kwargs: client,
        get_logger=lambda: _Logger(),
    )
    fetch = MethodType(TeleopNode._fetch_runtime_gripper_model, node)
    assert fetch() is None


def _resolution_harness(device_config, joint_limits, device=None, fetch_result="unset"):
    """Bind the mapping resolution to a bare node with injectable collaborators."""
    node = SimpleNamespace(
        gripper_joint_names=["6"],
        device=device,
        _ratio_input=device_config.get("type") in ("leader_arm", "leader_topic", "phone"),
        _gripper_mapping=None,
        get_logger=lambda: _Logger(),
    )
    if fetch_result == "unset":
        node._fetch_runtime_gripper_model = lambda: None
    else:
        node._fetch_runtime_gripper_model = lambda: fetch_result
    node._resolve_gripper_mapping = MethodType(TeleopNode._resolve_gripper_mapping, node)
    node._resolve_gripper_mapping(device_config, joint_limits)
    return node


def test_resolution_prefers_config_endpoints_without_fetch():
    node = _resolution_harness(
        {"type": "leader_topic", "gripper_closed": -0.7, "gripper_open": 1.6},
        {"6": {"min": -0.7, "max": 1.6}},
    )
    assert node._gripper_mapping == (-0.7, 1.6, {"min": -0.7, "max": 1.6})


def test_resolution_falls_back_to_runtime_description_and_fills_limits():
    node = _resolution_harness(
        {"type": "leader_arm"},
        {},
        fetch_result=(-0.7486, 1.5831, {"6": {"min": -0.7486, "max": 1.5831}, "1": {"min": -2.0, "max": 2.0}}),
    )
    assert node._gripper_mapping[0:2] == (-0.7486, 1.5831)


def test_resolution_falls_back_to_device_stroke_without_runtime():
    device = SimpleNamespace(get_gripper_stroke=lambda: (-0.65, 1.6))
    node = _resolution_harness({"type": "leader_arm"}, {"6": {"min": -0.65, "max": 1.6}}, device=device)
    assert node._gripper_mapping[0:2] == (-0.65, 1.6)


def test_resolution_device_stroke_also_fills_missing_limits():
    device = SimpleNamespace(get_gripper_stroke=lambda: (-0.65, 1.6))
    node = _resolution_harness({"type": "leader_arm"}, {}, device=device)
    assert node._gripper_mapping == (-0.65, 1.6, {"min": -0.65, "max": 1.6})


def test_resolution_fail_closed_without_any_endpoint_source():
    node = _resolution_harness({"type": "leader_arm"}, {"6": {"min": 0.0, "max": 1.0}})
    assert node._gripper_mapping is None


def test_resolution_fills_public_limits_for_non_ratio_devices():
    # Xbox/VR standalone: no endpoints needed, but the runtime public limits
    # still become the safety-filter authority instead of running unclamped.
    node = _resolution_harness(
        {"type": "xbox_controller"},
        {},
        fetch_result=(-0.7486, 1.5831, {"6": {"min": -0.7486, "max": 1.5831}, "1": {"min": -2.0, "max": 2.0}}),
    )
    assert node._gripper_mapping is None


def test_diagnostics_use_wall_budget_and_elapsed_rate(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(teleop_node_module.time, "monotonic", lambda: now[0])
    from builtin_interfaces.msg import Time

    logger = _Logger()
    messages = []
    node = SimpleNamespace(
        loop_count=0,
        avg_loop_time=0.0,
        max_loop_time=0.0,
        last_loop_time=10.0,
        latency_warn_s=0.02,
        diagnostics_period_s=1.0,
        diag_pub=SimpleNamespace(publish=messages.append),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=Time)),
        get_logger=lambda: logger,
    )
    for _ in range(100):
        TeleopNode._update_diagnostics(node, 0.018)
    assert not messages
    now[0] = 11.0
    TeleopNode._update_diagnostics(node, 0.018)
    assert messages[-1].status[0].level == teleop_node_module.DiagnosticStatus.OK
    assert "wall time" in messages[-1].status[0].message
    assert not logger.warnings
    node.latency_warn_s = 0.005
    now[0] = 12.0
    TeleopNode._update_diagnostics(node, 0.018)
    assert messages[-1].status[0].level == teleop_node_module.DiagnosticStatus.WARN
    assert len(logger.warnings) == 1
