"""ROS adapter tests for the fixed-trigger sound-orientation node."""

from types import SimpleNamespace

import rclpy
from rclpy.parameter import Parameter

from embodied_agent.sound_orientation_node import SoundOrientationNode
from embodied_agent.sound_orientation_policy import OrientationState
from ibrobot_msgs.msg import SkillCapabilityStatus, SpeechDirection


def _make_node():
    return SoundOrientationNode(
        parameter_overrides=[
            Parameter("trigger_phrases", Parameter.Type.STRING_ARRAY, ["转向我"]),
            Parameter("direction_topic", Parameter.Type.STRING, "/test/direction"),
            Parameter("command_topic", Parameter.Type.STRING, "/test/command"),
            Parameter("gateway_status_service", Parameter.Type.STRING, "/test/status"),
            Parameter("skill_action_name", Parameter.Type.STRING, "/test/execute_skill"),
            Parameter("debug_tracing", Parameter.Type.BOOL, False),
        ]
    )


def _status(*, busy=False, authorized=True, ready=True):
    capability = SkillCapabilityStatus()
    capability.name = "nav_turn"
    capability.ready = ready
    capability.required_control_mode = "base_navigation"
    return SimpleNamespace(
        control_plane_ready=True,
        motion_authorized=authorized,
        busy=busy,
        active_control_mode="base_navigation",
        registry_epoch="epoch-1",
        registry_generation=1,
        registry_digest="digest-1",
        task_budget_sec=30.0,
        capabilities=[capability],
    )


def _direction(*, stamp_sec: int = 100, stamp_nanosec: int = 0, azimuth: float = 0.5, seq_id: int = 1):
    message = SpeechDirection()
    message.header.frame_id = "base_link"
    message.header.stamp.sec = stamp_sec
    message.header.stamp.nanosec = stamp_nanosec
    message.azimuth_rad = azimuth
    message.seq_id = seq_id
    return message


def test_gateway_snapshot_reads_nav_turn_readiness():
    rclpy.init()
    node = None
    try:
        node = _make_node()

        snapshot = node._gateway_snapshot(_status(busy=True, authorized=False, ready=False))  # noqa: SLF001

        assert snapshot.busy is True
        assert snapshot.motion_authorized is False
        assert snapshot.capability_ready is False
        assert snapshot.required_control_mode == "base_navigation"
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_trigger_and_direction_wait_for_gateway_status():
    rclpy.init()
    node = None
    try:
        node = _make_node()
        node._now_sec = lambda: 100.0  # noqa: SLF001

        node._direction_callback(_direction())  # noqa: SLF001
        node._command_callback(SimpleNamespace(data="转向我"))  # noqa: SLF001

        assert node._policy.state.value == "waiting_for_direction"  # noqa: SLF001
        assert node._status_needed is True  # noqa: SLF001
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_busy_status_drops_trigger_without_submitting_goal():
    rclpy.init()
    node = None
    try:
        node = _make_node()
        node._now_sec = lambda: 100.0  # noqa: SLF001
        node._direction_callback(_direction())  # noqa: SLF001
        node._command_callback(SimpleNamespace(data="转向我"))  # noqa: SLF001

        decision = node._policy.try_dispatch(now_sec=100.0, gateway=node._gateway_snapshot(_status(busy=True)))  # noqa: SLF001

        assert decision.reason == "SKILL_BUSY"
        assert node._active_task_id == ""  # noqa: SLF001
        assert node._active_goal_handle is None  # noqa: SLF001
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_canceled_action_enters_fault_unknown():
    rclpy.init()
    node = None
    try:
        node = _make_node()
        node._now_sec = lambda: 100.0  # noqa: SLF001
        node._direction_callback(_direction())  # noqa: SLF001
        node._command_callback(SimpleNamespace(data="转向我"))  # noqa: SLF001
        decision = node._policy.try_dispatch(now_sec=100.0, gateway=node._gateway_snapshot(_status()))  # noqa: SLF001
        assert decision.request is not None
        node._policy.mark_action_submitted()  # noqa: SLF001
        result = SimpleNamespace(success=False, error_code="SKILL_CANCELLED")
        node._result_done(  # noqa: SLF001
            SimpleNamespace(done=lambda: True, result=lambda: SimpleNamespace(status=5, result=result)),
            node._goal_generation,  # noqa: SLF001
        )

        assert node._policy.state is OrientationState.FAULT_UNKNOWN  # noqa: SLF001
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_contradictory_success_result_enters_fault_unknown():
    rclpy.init()
    node = None
    try:
        node = _make_node()
        node._now_sec = lambda: 100.0  # noqa: SLF001
        node._direction_callback(_direction())  # noqa: SLF001
        node._command_callback(SimpleNamespace(data="转向我"))  # noqa: SLF001
        decision = node._policy.try_dispatch(  # noqa: SLF001
            now_sec=100.0,
            gateway=node._gateway_snapshot(_status()),
        )
        assert decision.request is not None
        node._policy.mark_action_submitted()  # noqa: SLF001
        result = SimpleNamespace(success=False, error_code="SKILL_BUSY")
        node._result_done(  # noqa: SLF001
            SimpleNamespace(result=lambda: SimpleNamespace(status=4, result=result)),
            node._goal_generation,  # noqa: SLF001
        )
        assert node._policy.state is OrientationState.FAULT_UNKNOWN  # noqa: SLF001
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
