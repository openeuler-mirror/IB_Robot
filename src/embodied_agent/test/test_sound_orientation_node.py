"""ROS adapter tests for the fixed-trigger sound-orientation node."""

from concurrent.futures import Future
from types import SimpleNamespace

import rclpy
from rclpy.parameter import Parameter

from embodied_agent.sound_orientation_node import SoundOrientationNode
from embodied_agent.sound_orientation_policy import OrientationState
from ibrobot_msgs.msg import SkillCapabilityStatus, SpeechDirection
from ibrobot_msgs.srv import SetSoundFollowing


def _make_node(*, periodic: bool = False):
    overrides = [
        Parameter("trigger_phrases", Parameter.Type.STRING_ARRAY, ["转向我"]),
        Parameter("direction_topic", Parameter.Type.STRING, "/test/direction"),
        Parameter("command_topic", Parameter.Type.STRING, "/test/command"),
        Parameter("gateway_status_service", Parameter.Type.STRING, "/test/status"),
        Parameter("skill_action_name", Parameter.Type.STRING, "/test/execute_skill"),
        Parameter("debug_tracing", Parameter.Type.BOOL, False),
    ]
    if periodic:
        overrides.extend(
            [
                Parameter("mode", Parameter.Type.STRING, "periodic"),
                Parameter("periodic_interval_sec", Parameter.Type.DOUBLE, 10.0),
                Parameter("max_direction_age_sec", Parameter.Type.DOUBLE, 30.0),
            ]
        )
    return SoundOrientationNode(parameter_overrides=overrides)


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


def _direction(
    *,
    stamp_sec: int = 100,
    stamp_nanosec: int = 0,
    azimuth: float = 0.5,
    seq_id: int = 1,
    segment_id: int = 0,
    direction_type: str = "",
):
    message = SpeechDirection()
    message.header.frame_id = "base_link"
    message.header.stamp.sec = stamp_sec
    message.header.stamp.nanosec = stamp_nanosec
    message.azimuth_rad = azimuth
    message.seq_id = seq_id
    message.segment_id = segment_id
    message.direction_type = direction_type
    return message


def test_periodic_timer_refreshes_gateway_before_dispatch(monkeypatch):
    rclpy.init()
    node = None
    try:
        node = _make_node(periodic=True)
        now = 100.0
        node._now_sec = lambda: now  # noqa: SLF001
        node._policy.activate_following()  # noqa: SLF001
        node._direction_callback(_direction(segment_id=1, direction_type="seg_end"))  # noqa: SLF001
        status_requests = []
        node._request_status = lambda: status_requests.append(True)  # noqa: SLF001

        node._timer_callback()  # noqa: SLF001

        assert status_requests == [True]
        assert node._policy.state is OrientationState.IDLE_LISTENING  # noqa: SLF001

        dispatched = []
        results = []

        def send_goal(goal, **_kwargs):
            dispatched.append(goal)
            result = Future()
            results.append(result)
            accepted = Future()
            accepted.set_result(SimpleNamespace(accepted=True, get_result_async=lambda: result))
            return accepted

        monkeypatch.setattr(node._skill_client, "server_is_ready", lambda: True)
        monkeypatch.setattr(node._skill_client, "send_goal_async", send_goal)
        node._status_done(SimpleNamespace(result=lambda: _status()), node._status_request_generation)  # noqa: SLF001

        assert len(dispatched) == 1
        assert dispatched[0].dispatch_binding.expected_registry_digest == "digest-1"
        assert node._policy.state is OrientationState.TURNING  # noqa: SLF001

        now = 100.1
        results[0].set_result(SimpleNamespace(status=4, result=SimpleNamespace(success=True, error_code="")))
        now = 102.0
        node._timer_callback()  # noqa: SLF001
        now = 112.0
        node._direction_callback(  # noqa: SLF001
            _direction(stamp_sec=112, seq_id=2, segment_id=2, direction_type="seg_end")
        )
        node._timer_callback()  # noqa: SLF001

        assert status_requests == [True, True]
        assert node._policy.state is OrientationState.IDLE_LISTENING  # noqa: SLF001

        fresh = _status()
        fresh.registry_generation = 2
        fresh.registry_digest = "digest-2"
        node._status_done(SimpleNamespace(result=lambda: fresh), node._status_request_generation)  # noqa: SLF001

        assert len(dispatched) == 2
        assert dispatched[1].dispatch_binding.expected_registry_generation == 2
        assert dispatched[1].dispatch_binding.expected_registry_digest == "digest-2"
        assert node._policy.state is OrientationState.TURNING  # noqa: SLF001

        request = SetSoundFollowing.Request(schema_version=1, enable=False)
        stopped = node._set_following_callback(request, SetSoundFollowing.Response())
        assert stopped.success and stopped.current_state == "shutting_down"
        results[1].set_result(SimpleNamespace(status=4, result=SimpleNamespace(success=True, error_code="")))
        assert node._policy.session_state.value == "inactive"
        assert node._policy.active_request is None
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_default_active_does_not_reactivate_initialized_policy():
    rclpy.init()
    node = None
    try:
        node = SoundOrientationNode(
            parameter_overrides=[
                Parameter("mode", Parameter.Type.STRING, "periodic"),
                Parameter("default_active", Parameter.Type.BOOL, True),
                Parameter("direction_topic", Parameter.Type.STRING, "/test/default_active/direction"),
                Parameter("command_topic", Parameter.Type.STRING, "/test/default_active/command"),
                Parameter("gateway_status_service", Parameter.Type.STRING, "/test/default_active/status"),
                Parameter("skill_action_name", Parameter.Type.STRING, "/test/default_active/execute_skill"),
            ]
        )

        assert node._policy.session_state.value == "active"  # noqa: SLF001
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_periodic_status_response_honors_disable_and_current_admission():
    rclpy.init()
    node = None
    try:
        node = _make_node(periodic=True)
        node._now_sec = lambda: 100.0
        node._policy.activate_following()
        node._last_gateway_snapshot = node._gateway_snapshot(_status())
        node._direction_callback(_direction(segment_id=1, direction_type="seg_end"))
        requests = []
        node._request_status = lambda: requests.append(True)
        sent = []
        node._send_turn_goal = lambda *args: sent.append(args)
        node._timer_callback()
        assert requests == [True]
        assert not sent
        node._status_done(SimpleNamespace(result=lambda: _status(authorized=False)), node._status_request_generation)
        assert not sent
        assert node._policy.state is OrientationState.IDLE_LISTENING

        node._now_sec = lambda: 111.0
        node._direction_callback(_direction(stamp_sec=111, seq_id=2, segment_id=2, direction_type="seg_end"))
        node._timer_callback()
        assert len(requests) == 2
        request = SetSoundFollowing.Request(schema_version=1, enable=False)
        response = node._set_following_callback(request, SetSoundFollowing.Response())
        assert response.success and response.current_state == "inactive"
        node._status_done(SimpleNamespace(result=lambda: _status()), node._status_request_generation)
        assert not sent
        assert node._policy.active_request is None
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


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
