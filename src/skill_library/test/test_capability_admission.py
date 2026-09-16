"""Runtime capability/mode admission, with explicit provider-less compatibility."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy

from skill_library import skill_executor_node
from skill_library.skill_executor_node import SkillExecutorNode


def _status(**overrides):
    fields = {
        "runtime_name": "test_runtime",
        "lifecycle": "ACTIVE",
        "capabilities": ["joint.trajectory", "gripper.1d"],
        "active_mode": "trajectory",
        "declared_modes": ["idle", "trajectory", "stream"],
        "stop_latched": False,
        "faults": [],
        "stamp": SimpleNamespace(sec=100, nanosec=0),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


class _FakeModeClient:
    def __init__(self, node, *, accept=True, status=None, available=True, error=None, response_error=None):
        self.node = node
        self.accept = accept
        self.status = status
        self.available = available
        self.error = error
        self.response_error = response_error
        self.requests = []

    def wait_for_service(self, timeout_sec):
        return self.available

    def call_async(self, request):
        self.requests.append(request.mode)
        if self.error:
            raise self.error
        if self.status is not None:
            self.node._on_runtime_status(self.status)

        def result():
            if self.response_error:
                raise self.response_error
            return SimpleNamespace(success=self.accept, message="injected mode rejection")

        return SimpleNamespace(result=result)


def _node(*, enabled=True, status=None, required=("joint.trajectory", "gripper.1d")):
    node = SimpleNamespace(
        _runtime_enabled=enabled,
        _runtime_name="test_runtime",
        _runtime_status_snapshot=None,
        _runtime_status_event=threading.Event(),
        _runtime_status_freshness_sec=3.0,
        _runtime_mode_service="/runtime/set_mode",
        _runtime_mode_map={"moveit_planning": "trajectory", "teleop": "stream", "model_inference": "stream"},
        _motion_mode_switch_lock=threading.RLock(),
        _rpc_timeout=0.001,
        _skill_required_control_mode="moveit_planning",
        _active_runtime_bundle=SimpleNamespace(
            snapshot=SimpleNamespace(capability_view={"pick_object": {"required_capabilities": list(required)}})
        ),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=100_000_000_000)),
        _wait_for_future=lambda *_args, **_kwargs: True,
    )
    for name in (
        "_on_runtime_status",
        "_runtime_status_for_admission",
        "_required_capabilities_for_skill",
        "_ensure_skill_capabilities",
        "_required_control_mode_for_skill",
        "_runtime_mode_for",
        "_request_runtime_mode",
        "_ensure_skill_control_mode",
    ):
        setattr(node, name, getattr(SkillExecutorNode, name).__get__(node))
    node._runtime_mode_client = _FakeModeClient(node)
    if status is not None:
        node._on_runtime_status(status)
    return node


@pytest.mark.parametrize("required", [(), ("joint.trajectory",)])
def test_explicit_runtime_without_status_fails_closed(required):
    node = _node(required=required)
    for check in (node._ensure_skill_capabilities, lambda name: node._ensure_skill_control_mode(name, None)):
        ready, reason = check("pick_object")
        assert not ready
        assert "not been received" in reason
    assert node._runtime_mode_client.requests == []


def test_providerless_node_ignores_observed_runtime_and_keeps_legacy_check():
    node = _node(enabled=False, status=_status(stop_latched=True))
    assert node._runtime_status_snapshot is None
    assert node._ensure_skill_capabilities("pick_object") == (True, "")
    node._active_control_mode = "moveit_planning"
    node._supported_control_modes = ()
    node._motion_mode_service = ""
    node._motion_mode_client = SimpleNamespace()
    assert node._ensure_skill_control_mode("pick_object", None) == (True, "")
    node._active_control_mode = "teleop"
    assert not node._ensure_skill_control_mode("pick_object", None)[0]
    assert node._runtime_mode_client.requests == []


def test_other_runtime_cannot_admit_or_refresh_selected_runtime():
    node = _node(status=_status(runtime_name="another_runtime"))
    assert node._runtime_status_snapshot is None
    node._on_runtime_status(_status())
    previous = node._runtime_status_snapshot
    node._on_runtime_status(_status(runtime_name="another_runtime", stop_latched=True))
    assert node._runtime_status_snapshot is previous


@pytest.mark.parametrize("lifecycle", ["CONNECTING", "DEGRADED", "STOPPED", "FAULTED", "unknown"])
def test_non_active_runtime_rejected(lifecycle):
    node = _node(status=_status(lifecycle=lifecycle))
    ready, reason = node._ensure_skill_control_mode("pick_object", None)
    assert not ready
    assert lifecycle in reason
    assert node._runtime_mode_client.requests == []


@pytest.mark.parametrize(
    ("overrides", "reason_text"),
    [({"stop_latched": True}, "stop is latched"), ({"faults": ["read failure"]}, "read failure")],
)
def test_latch_and_faults_block_even_when_mode_matches(overrides, reason_text):
    node = _node(status=_status(**overrides))
    ready, reason = node._ensure_skill_control_mode("pick_object", None)
    assert not ready
    assert reason_text in reason
    assert node._runtime_mode_client.requests == []


@pytest.mark.parametrize("stamp_sec", [96, 101])
def test_stale_or_future_source_stamp_rejected_despite_fresh_receipt(stamp_sec):
    node = _node(status=_status(stamp=SimpleNamespace(sec=stamp_sec, nanosec=0)))
    ready, reason = node._ensure_skill_capabilities("pick_object")
    assert not ready
    assert "stale" in reason


def test_receipt_timeout_rejects_status_even_when_ros_clock_is_paused():
    node = _node(status=_status())
    status, receipt = node._runtime_status_snapshot
    node._runtime_status_snapshot = (status, receipt - 4.0)
    assert not node._ensure_skill_control_mode("pick_object", None)[0]


def test_missing_capabilities_named_before_mode_switch():
    node = _node(status=_status(capabilities=["joint.state"], active_mode="idle"))
    ready, reason = node._ensure_skill_control_mode("pick_object", None)
    assert not ready
    assert "gripper.1d" in reason
    assert "joint.trajectory" in reason
    assert node._runtime_mode_client.requests == []


def test_active_trajectory_mode_with_capabilities_accepted_without_request():
    node = _node(status=_status())
    assert node._ensure_skill_control_mode("pick_object", None) == (True, "")
    assert node._runtime_mode_client.requests == []


@pytest.mark.parametrize("mode", ["stream", "policy_stream"])
def test_skill_cannot_take_over_an_active_stream_producer(mode):
    node = _node(status=_status(active_mode=mode))
    ready, reason = node._ensure_skill_control_mode("pick_object", None)
    assert not ready
    assert "enter idle" in reason
    assert node._runtime_mode_client.requests == []


def test_unannotated_skill_still_requires_healthy_runtime():
    node = _node(status=_status(), required=())
    assert node._ensure_skill_capabilities("pick_object") == (True, "")
    node._on_runtime_status(_status(stop_latched=True))
    assert not node._ensure_skill_capabilities("pick_object")[0]


def test_undeclared_mode_rejected_before_request():
    node = _node(status=_status(declared_modes=["idle"]))
    ready, reason = node._ensure_skill_control_mode("pick_object", None)
    assert not ready
    assert "not declared" in reason
    assert node._runtime_mode_client.requests == []


def test_idle_requests_mapped_mode_and_requires_observed_confirmation():
    node = _node(status=_status(active_mode="idle"))
    node._runtime_mode_client = _FakeModeClient(node, status=_status())
    assert node._ensure_skill_control_mode("pick_object", None) == (True, "")
    assert node._runtime_mode_client.requests == ["trajectory"]
    assert node._runtime_status_snapshot[0].active_mode == "trajectory"


@pytest.mark.parametrize("confirmation", [None, _status(active_mode="idle"), _status(runtime_name="other")])
def test_rpc_success_without_matching_status_fails_closed(confirmation):
    node = _node(status=_status(active_mode="idle"))
    node._runtime_mode_client = _FakeModeClient(node, status=confirmation)
    ready, reason = node._ensure_skill_control_mode("pick_object", None)
    assert not ready
    assert "not confirmed" in reason


def test_pre_request_sample_cannot_confirm_mode_even_if_received_after_rpc():
    node = _node(status=_status(active_mode="idle"))
    node._runtime_mode_client = _FakeModeClient(node, status=_status(stamp=SimpleNamespace(sec=99, nanosec=0)))
    assert not node._ensure_skill_control_mode("pick_object", None)[0]


@pytest.mark.parametrize(
    "confirmation", [_status(stop_latched=True), _status(lifecycle="DEGRADED"), _status(capabilities=[])]
)
def test_unhealthy_confirmation_or_lost_capability_rejected(confirmation):
    node = _node(status=_status(active_mode="idle"))
    node._runtime_mode_client = _FakeModeClient(node, status=confirmation)
    assert not node._ensure_skill_control_mode("pick_object", None)[0]


@pytest.mark.parametrize(
    ("options", "reason_text"),
    [
        ({"accept": False}, "injected mode rejection"),
        ({"available": False}, "unavailable"),
        ({"error": RuntimeError("transport failed")}, "transport failed"),
        ({"response_error": RuntimeError("future failed")}, "future failed"),
    ],
)
def test_mode_rpc_failures_are_reported_without_dispatch(options, reason_text):
    node = _node(status=_status(active_mode="idle"))
    node._runtime_mode_client = _FakeModeClient(node, **options)
    ready, reason = node._ensure_skill_control_mode("pick_object", None)
    assert not ready
    assert reason_text in reason


def test_mode_rpc_timeout_discards_previous_mode_evidence():
    node = _node(status=_status(active_mode="idle"))
    node._wait_for_future = lambda *_args, **_kwargs: False
    ready, reason = node._ensure_skill_control_mode("pick_object", None)
    assert not ready
    assert "timed out" in reason
    assert node._runtime_status_snapshot is None


def test_canceled_skill_never_requests_mode():
    node = _node(status=_status(active_mode="idle"))
    assert not node._ensure_skill_control_mode("pick_object", SimpleNamespace(is_cancel_requested=True))[0]
    assert node._runtime_mode_client.requests == []


def test_external_primitive_uses_the_same_runtime_admission():
    node = _node()
    assert not node._ensure_skill_control_mode("__primitive__:moveit_planning", None)[0]
    node._on_runtime_status(_status())
    assert node._ensure_skill_control_mode("__primitive__:moveit_planning", None) == (True, "")


@pytest.mark.parametrize("enabled", [False, True])
def test_constructor_explicit_opt_in_controls_endpoint_and_status_qos(monkeypatch, enabled):
    parameters = {"runtime_enabled": enabled, "runtime_name": "test_runtime"}
    subscriptions = []
    clients = []

    class SetupCaptured(Exception):
        pass

    def parameter(name):
        value = parameters[name]
        return SimpleNamespace(
            value=value,
            get_parameter_value=lambda: SimpleNamespace(
                string_value=value, double_value=value, bool_value=value, integer_value=value
            ),
        )

    def declare(_node, name, default, **_kwargs):
        parameters.setdefault(name, default)
        return parameter(name)

    def get(_node, name):
        if name == "supported_control_modes_json":
            raise SetupCaptured
        return parameter(name)

    monkeypatch.setattr(skill_executor_node, "validate_public_request_wire_contracts", lambda: None)
    monkeypatch.setattr(skill_executor_node.Node, "__init__", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(skill_executor_node.Node, "declare_parameter", declare)
    monkeypatch.setattr(skill_executor_node.Node, "get_parameter", get)
    monkeypatch.setattr(
        skill_executor_node.Node,
        "create_subscription",
        lambda _node, *args, **_kwargs: subscriptions.append(args),
    )
    monkeypatch.setattr(skill_executor_node.Node, "create_client", lambda _node, *args, **_kwargs: clients.append(args))
    with pytest.raises(SetupCaptured):
        SkillExecutorNode()
    assert parameters["move_configuration_service"] == (
        "/motion/move_to_joint" if enabled else "/moveit_gateway/move_to_configuration"
    )
    assert parameters["runtime_status_freshness_sec"] == 3.0
    if enabled:
        assert len(subscriptions) == 1
        assert subscriptions[0][1] == "/runtime_status"
        qos = subscriptions[0][3]
        assert qos.reliability == ReliabilityPolicy.RELIABLE
        assert qos.durability == DurabilityPolicy.VOLATILE
        assert qos.history == HistoryPolicy.KEEP_LAST
        assert qos.depth == 10
        assert clients[0][1] == "/runtime/set_mode"
    else:
        assert subscriptions == []
        assert clients == []


def test_runtime_constructor_requires_expected_identity(monkeypatch):
    monkeypatch.setattr(skill_executor_node, "validate_public_request_wire_contracts", lambda: None)
    monkeypatch.setattr(skill_executor_node.Node, "__init__", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        skill_executor_node.Node,
        "declare_parameter",
        lambda _node, name, default, **_kwargs: SimpleNamespace(value=True if name == "runtime_enabled" else default),
    )
    with pytest.raises(ValueError, match="runtime_name is required"):
        SkillExecutorNode()
