"""Transactional joint-space ArmReturnHome tests for the SO-101 Placo node."""

import importlib.util
import os
import sys
import threading
import types

import numpy as np
import pytest

_ORIGINAL_MODULES = {}
_STUB_MODULES = {}


def _install_stubs():
    def _mod(name):
        module = _STUB_MODULES.get(name)
        if module is None:
            _ORIGINAL_MODULES[name] = sys.modules.get(name)
            module = types.ModuleType(name)
            sys.modules[name] = module
            _STUB_MODULES[name] = module
        return module

    rclpy = _mod("rclpy")
    rclpy.ok = lambda: True
    for submodule in ("action", "node", "callback_groups", "duration", "executors", "time"):
        _mod(f"rclpy.{submodule}")
    _mod("rclpy.node").Node = type("Node", (), {})
    action = _mod("rclpy.action")
    action.ActionClient = type("ActionClient", (), {})
    action.ActionServer = type("ActionServer", (), {})
    action.CancelResponse = types.SimpleNamespace(ACCEPT="accept", REJECT="reject")
    action.GoalResponse = types.SimpleNamespace(ACCEPT="accept", REJECT="reject")
    callback_groups = _mod("rclpy.callback_groups")
    callback_groups.MutuallyExclusiveCallbackGroup = type("MECG", (), {})
    callback_groups.ReentrantCallbackGroup = type("RCG", (), {})
    _mod("rclpy.duration").Duration = type("Duration", (), {})
    executors = _mod("rclpy.executors")
    executors.ExternalShutdownException = type("ExternalShutdownException", (Exception,), {})
    executors.MultiThreadedExecutor = type("MultiThreadedExecutor", (), {})
    _mod("rclpy.time").Time = type("Time", (), {})

    for package, names in (
        ("geometry_msgs.msg", ("PoseStamped", "Vector3Stamped")),
        ("sensor_msgs.msg", ("JointState",)),
        ("std_msgs.msg", ("Bool", "Empty", "Float64MultiArray")),
    ):
        _mod(package.split(".")[0])
        message_module = _mod(package)
        for name in names:
            setattr(message_module, name, type(name, (), {}))

    _mod("std_srvs")
    services = _mod("std_srvs.srv")

    class _Trigger:
        class Request:
            pass

        class Response:
            def __init__(self):
                self.success = False
                self.message = ""

    services.Trigger = _Trigger

    ibrobot = _mod("ibrobot_msgs")
    actions = _mod("ibrobot_msgs.action")
    ibrobot.action = actions
    # The servo node also imports the runtime mode service; stub it so the
    # stub ibrobot_msgs remains importable for that path too.
    srv = _mod("ibrobot_msgs.srv")
    ibrobot.srv = srv
    srv.SetRuntimeMode = types.SimpleNamespace
    srv.StopRuntime = types.SimpleNamespace(Request=types.SimpleNamespace)
    _mod("ibrobot_msgs.msg").RuntimeStatus = types.SimpleNamespace

    class _ArmReturnHome:
        class Goal:
            def __init__(self):
                self.target_name = ""

        class Result:
            def __init__(self):
                self.success = False
                self.error_code = ""
                self.message = ""

        class Feedback:
            def __init__(self):
                self.state = ""
                self.max_joint_error_rad = 0.0

    actions.ArmReturnHome = _ArmReturnHome
    tf2_ros = _mod("tf2_ros")
    tf2_ros.Buffer = type("Buffer", (), {})
    tf2_ros.TransformListener = type("TransformListener", (), {})
    kinematics = _mod("so101_placo_kinematics")
    kinematics.SO101PlacoDiffIK = type("SO101PlacoDiffIK", (), {})


_install_stubs()

_NODE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "so101_placo_servo_node.py",
)
_spec = importlib.util.spec_from_file_location("so101_placo_servo_node", _NODE_PATH)
mod = importlib.util.module_from_spec(_spec)
sys.modules["so101_placo_servo_node"] = mod
_spec.loader.exec_module(mod)

Trigger = mod.Trigger
for _name, _original in _ORIGINAL_MODULES.items():
    if _original is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _original


class _FakeLogger:
    def info(self, *_args, **_kwargs):
        pass

    def warn(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


class _FakeGoalHandle:
    def __init__(self):
        self.is_cancel_requested = False
        self.feedback = []
        self.terminal_state = None

    def publish_feedback(self, feedback):
        self.feedback.append(feedback)

    def succeed(self):
        self.terminal_state = "succeeded"

    def canceled(self):
        self.terminal_state = "canceled"

    def abort(self):
        self.terminal_state = "aborted"


def _make_node():
    node = mod.SO101PlacoServoNode.__new__(mod.SO101PlacoServoNode)
    node.managed_teleop = False
    node.arm_joint_names = ["1", "2", "3", "4", "5"]
    node.joint_lo = np.full(5, -2.0)
    node.joint_hi = np.full(5, 2.0)
    node._home_q = np.array([0.2, -0.1, 0.3, -0.2, 0.1], dtype=np.float64)
    node._home_enabled = True
    node.home_joint_tolerance_rad = 0.05
    node.home_max_joint_speed = 1.0
    node.home_joint_state_stale_s = 0.2
    node.home_stable_duration_s = 0.2
    node.home_timeout_s = 10.0
    node.control_period = 0.02
    node.command_lease_timeout_s = 0.0
    # Runtime-mode integration (no provider in this fixture): disabled client.
    node.runtime_mode_service = ""
    node.runtime_stream_mode = "stream"
    node.runtime_idle_mode = "idle"
    node._runtime_mode_client = None
    node.target_reset_timeout = 2.0
    node.input_mode = "pose"
    node._enabled = True
    node._estop_active = False
    node._accept_velocity_commands = True
    node._accept_pose_commands = True
    node._latest_linear = object()
    node._latest_angular = object()
    node._latest_pose = object()
    node._latest_pose_stamp = 0.0
    node._latest_linear_stamp = 0.0
    node._latest_angular_stamp = 0.0
    node._last_input_time = 0.0
    node._last_lease_time = 123.0
    node._p_ref = np.ones(3)
    node._r_ref = np.eye(3)
    node._ee0_p = np.ones(3)
    node._ee0_R = np.eye(3)
    node._last_cmd = np.zeros(5)
    node._latest_js = object()
    node._latest_js_received_at = 123.0
    node._joint_state_generation = 1
    node._home_active = False
    node._home_started_at = 0.0
    node._home_stable_since = None
    node._home_last_joint_state_generation = 0
    node._home_request_lock = threading.Lock()
    node._home_goal_reserved = False
    node._home_preemption = None
    node._pending_home_request = None
    node._active_home_request = None
    node._dropped_frame_count = 0
    node._solve_count = 0
    node._measured_q = np.zeros(5)
    node._measured_arm_joints = lambda: node._measured_q.copy()
    node._now = lambda: 123.0
    node.get_logger = lambda: _FakeLogger()
    node.published = []
    node.cmd_pub = types.SimpleNamespace(publish=node.published.append)
    return node


def _activate_home(node):
    goal_handle = _FakeGoalHandle()
    request = mod._HomeActionRequest(goal_handle=goal_handle, done=threading.Event())
    node._home_goal_reserved = True
    node._pending_home_request = request
    node._process_home_action(123.0)
    return request


def _fresh_pose():
    return object()


def _managed_node():
    node = _make_node()
    node.managed_teleop = True
    node._managed_owner = True
    node._managed_mode = None
    node.input_mode = "auto"
    node.input_timeout = 0.2
    node.max_joint_speed = 1.0
    node.command_lease_timeout_s = 0.2
    node.planning_frame = "base"
    node._status_received_at = 123.0
    node._last_managed_input = 123.0
    node._managed_start_stamp = 122.9
    node._stop_latched = False
    node._stop_pending = False
    node._stop_confirmed = False
    node._stop_future = None
    node._stop_attempt_at = 0.0
    node.future_tolerance_s = 0.05
    node.gripper_joint_names = ["6"]
    node.gripper_lo = np.array([-0.6])
    node.gripper_hi = np.array([1.6])
    node.gripper_published = []
    node.gripper_pub = types.SimpleNamespace(publish=node.gripper_published.append)
    node._joint_intent = None
    node._joint_intent_stamp = 0.0
    node._runtime_status = types.SimpleNamespace(active_mode="stream", lifecycle="ACTIVE", stop_latched=False)
    node.get_clock = lambda: types.SimpleNamespace(now=lambda: types.SimpleNamespace(nanoseconds=123_000_000_000))
    node.requests = []
    node._runtime_stop_client = types.SimpleNamespace(
        service_is_ready=lambda: True,
        call_async=lambda request: node.requests.append(request) or types.SimpleNamespace(done=lambda: False),
    )
    node._latest_js = _joint_message()
    return node


def _joint_message(stamp=123.0, gripper_only=False):
    names = ["6"] if gripper_only else ["1", "2", "3", "4", "5", "6"]
    return types.SimpleNamespace(
        header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=int(stamp), nanosec=int((stamp % 1) * 1e9))),
        name=names,
        position=[0.5] * len(names),
    )


def _pose_message(stamp=123.0, frame="base", quaternion_w=1.0):
    header = _joint_message(stamp).header
    header.frame_id = frame
    return types.SimpleNamespace(
        header=header,
        pose=types.SimpleNamespace(
            position=types.SimpleNamespace(x=0.1, y=0.0, z=0.0),
            orientation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=quaternion_w),
        ),
    )


def test_managed_joint_intent_routes_gripper_and_bounded_arm():
    node = _managed_node()
    node._on_joint_intent(_joint_message())
    assert node._managed_mode == "joint"
    assert node.gripper_published[-1].data == [0.5]
    node._on_control_tick()
    np.testing.assert_allclose(node.published[-1].data, [0.02] * 5)


@pytest.mark.parametrize("stamp", [122.5, 122.9, 123.2])
def test_managed_queued_stale_and_future_commands_cannot_rearm(stamp):
    node = _managed_node()
    node._on_joint_intent(_joint_message(stamp))
    node._on_pose(_pose_message(stamp))
    assert node._joint_intent is None
    assert node._managed_mode is None
    assert node.gripper_published == []


def test_managed_arm_mode_lock_does_not_treat_gripper_as_joint_mode():
    node = _managed_node()
    node._on_joint_intent(_joint_message(gripper_only=True))
    assert node._managed_mode is None
    node._on_pose(_pose_message())
    assert node._managed_mode == "pose"
    node._on_joint_intent(_joint_message())
    assert node._joint_intent is None


@pytest.mark.parametrize("frame,w", [("tool", 1), ("base", 0), ("base", float("nan"))])
def test_managed_invalid_pose_rejected(frame, w):
    node = _managed_node()
    node._on_pose(_pose_message(frame=frame, quaternion_w=w))
    assert node._managed_mode is None


@pytest.mark.parametrize("home", [False, True])
def test_managed_watchdog_dispatches_actual_runtime_hold_and_clears_queued_targets(home):
    node = _managed_node()
    node._on_joint_intent(_joint_message())
    request = _activate_home(node) if home else None
    node._now = lambda: 123.21
    node._on_control_tick()
    assert not node._enabled and not node._managed_owner
    assert node._joint_intent is None and node._latest_pose is None
    assert node.requests[0].policy == "HOLD"
    assert node._stop_latched and node._stop_pending
    if request is not None:
        assert request.done.is_set()
        assert request.outcome == "aborted"
    response = node._on_stop_srv(None, Trigger.Response())
    assert response.success is False
    node._stop_future = types.SimpleNamespace(done=lambda: True, result=lambda: types.SimpleNamespace(success=True))
    node._retry_runtime_stop()
    response = node._on_stop_srv(None, Trigger.Response())
    assert response.success is True
    assert node._stop_latched  # acknowledgement is not a clear
    assert len(node.requests) == 1


def test_managed_live_lease_cannot_hide_stale_cartesian_targets():
    node = _managed_node()
    node._on_pose(_pose_message())
    node._now = lambda: 123.21
    node._latest_js_received_at = 123.21
    node._on_command_lease(None)
    node._on_control_tick()
    assert node.requests[0].policy == "HOLD"


def test_managed_one_live_velocity_component_cannot_hide_stale_other_component():
    node = _managed_node()
    node._managed_mode = "velocity"
    node._latest_linear_stamp = 123.0
    node._latest_angular_stamp = 122.0
    node._on_control_tick()
    assert node.requests[0].policy == "HOLD"


def test_managed_slow_stop_is_not_duplicated_behind_future_rearm():
    node = _managed_node()
    node._managed_stop("test")
    node._now = lambda: 125.0
    node._retry_runtime_stop()
    assert len(node.requests) == 1
    assert node._stop_pending


def test_managed_losing_stream_preempts_home_and_requires_clear():
    node = _managed_node()
    request = _activate_home(node)
    node._on_runtime_status(types.SimpleNamespace(active_mode="idle", lifecycle="ACTIVE", stop_latched=False))
    assert request.done.is_set()
    assert not node._managed_owner
    assert node.requests[0].policy == "HOLD"
    assert not node._on_start_srv(None, Trigger.Response()).success


@pytest.mark.parametrize("mode,latched", [("policy_stream", False), ("trajectory", False), ("idle", True)])
def test_managed_start_refuses_policy_and_latch(mode, latched):
    node = _managed_node()
    node._managed_owner = False
    node._runtime_status = types.SimpleNamespace(active_mode=mode, lifecycle="ACTIVE", stop_latched=latched)
    assert not node._on_start_srv(None, Trigger.Response()).success
    assert node.published == []


def test_managed_explicit_start_latches_fresh_reference_after_clear():
    node = _managed_node()
    node._managed_owner = False
    node._runtime_status = types.SimpleNamespace(active_mode="idle", lifecycle="ACTIVE", stop_latched=False)
    node._request_runtime_mode = lambda mode: (mode == "stream", "")
    node.diffik = types.SimpleNamespace(ee_position=lambda q: np.zeros(3), ee_rotation=lambda q: np.eye(3))
    assert node._on_start_srv(None, Trigger.Response()).success
    assert node._managed_owner and node._enabled
    assert node._latest_pose is None and node._joint_intent is None
    assert not node._on_start_srv(None, Trigger.Response()).success  # second managed producer rejected
    assert node.published == []  # activation is not HOME


def test_managed_home_requires_owner_and_fresh_lease():
    node = _managed_node()
    goal = types.SimpleNamespace(target_name="home")
    node._managed_owner = False
    assert node._home_goal_callback(goal) == mod.GoalResponse.REJECT
    node._managed_owner = True
    node._last_lease_time = 122.0
    assert node._home_goal_callback(goal) == mod.GoalResponse.REJECT
    node._last_lease_time = 123.0
    assert node._home_goal_callback(goal) == mod.GoalResponse.ACCEPT


def test_managed_old_joint_feedback_not_refreshed():
    node = _managed_node()
    node._latest_js_received_at = 122.0
    node._on_joint_state(_joint_message(122.0))
    assert node._latest_js_received_at == 122.0
    node._on_joint_state(_joint_message())
    assert node._latest_js_received_at == 123.0


def test_home_starts_joint_motion_and_closes_command_gates():
    node = _make_node()
    request = _activate_home(node)

    assert node._home_active is True
    assert node._active_home_request is request
    assert node._latest_pose is None
    assert node._latest_linear is None
    assert node._latest_angular is None
    assert node._accept_pose_commands is False
    assert node._accept_velocity_commands is False


def test_home_publishes_bounded_joint_space_step():
    node = _make_node()
    _activate_home(node)

    node._on_control_tick()

    expected = np.clip(node._home_q, -0.02, 0.02)
    np.testing.assert_allclose(node.published[-1].data, expected)


def test_pose_and_velocity_commands_are_rejected_until_next_start():
    node = _make_node()
    _activate_home(node)
    node._on_pose(_fresh_pose())
    node._on_linear(object())
    node._on_angular(object())
    assert node._latest_pose is None
    assert node._latest_linear is None
    assert node._latest_angular is None


def test_start_is_rejected_while_home_is_reserved():
    node = _make_node()
    node._home_goal_reserved = True

    response = node._on_start_srv(Trigger.Request(), Trigger.Response())

    assert response.success is False
    assert "ArmReturnHome" in response.message


def test_estop_aborts_home_and_blocks_new_home_and_start():
    node = _make_node()
    request = _activate_home(node)

    node._on_estop(types.SimpleNamespace(data=True))

    assert request.error_code == "EMERGENCY_STOP"
    assert node._enabled is False
    goal = types.SimpleNamespace(target_name="home")
    assert node._home_goal_callback(goal) == mod.GoalResponse.REJECT
    response = node._on_start_srv(Trigger.Request(), Trigger.Response())
    assert response.success is False

    node._on_estop(types.SimpleNamespace(data=False))
    assert node._estop_active is False
    assert node._enabled is False


def test_home_completes_only_after_fresh_stable_joint_samples():
    node = _make_node()
    node._measured_q = node._home_q.copy()
    request = _activate_home(node)

    node._joint_state_generation += 1
    node._update_home_progress(node._home_q, 123.0)
    node._update_home_progress(node._home_q, 124.0)
    assert request.done.is_set() is False

    node._joint_state_generation += 1
    node._update_home_progress(node._home_q, 123.21)
    assert request.done.is_set() is True
    assert request.outcome == "succeeded"
    assert node._enabled is False


def test_home_timeout_aborts_transaction():
    node = _make_node()
    request = _activate_home(node)

    assert node._home_timed_out(133.01) is True
    assert request.done.is_set() is True
    assert request.error_code == "TIMEOUT"
    assert request.max_joint_error_rad == 0.3
    assert "joint '3' error=0.3000rad" in request.message
    assert "target=0.3000" in request.message
    assert node._enabled is False


def test_stop_aborts_active_home():
    node = _make_node()
    request = _activate_home(node)

    node._on_stop_srv(Trigger.Request(), Trigger.Response())

    assert request.done.is_set() is True
    assert request.error_code == "STOP_REQUESTED"
    assert node._home_active is False


def test_prepare_shutdown_releases_active_action_waiter():
    node = _make_node()
    request = _activate_home(node)

    node.prepare_shutdown()

    assert request.done.is_set() is True
    assert request.error_code == "ROS_SHUTDOWN"


def test_stop_preempts_reserved_goal_before_execute_callback():
    node = _make_node()
    node._home_goal_reserved = True
    node._on_stop_srv(Trigger.Request(), Trigger.Response())
    goal_handle = _FakeGoalHandle()

    result = node._execute_home_action(goal_handle)

    assert result.success is False
    assert result.error_code == "STOP_REQUESTED"
    assert goal_handle.terminal_state == "aborted"
    assert node._home_goal_reserved is False


def test_cancel_request_aborts_without_starting_motion():
    node = _make_node()
    request = _activate_home(node)
    request.cancel_requested = True

    node._process_home_action(123.01)

    assert request.outcome == "canceled"
    assert node._enabled is False


def test_cancel_preempts_reserved_goal_before_execute_callback():
    node = _make_node()
    node._home_goal_reserved = True
    goal_handle = _FakeGoalHandle()

    assert node._home_cancel_callback(goal_handle) == mod.CancelResponse.ACCEPT
    result = node._execute_home_action(goal_handle)

    assert result.error_code == "CANCELED"
    assert goal_handle.terminal_state == "canceled"


def test_home_requires_fresh_joint_state():
    node = _make_node()
    node._latest_js_received_at = 122.0
    request = _activate_home(node)

    assert request.outcome == "aborted"
    assert request.error_code == "JOINT_STATE_UNAVAILABLE"


def test_home_aborts_on_non_finite_joint_feedback():
    node = _make_node()
    request = _activate_home(node)
    node._measured_q[2] = float("nan")

    node._on_control_tick()

    assert request.error_code == "JOINT_STATE_INVALID"
    assert node._enabled is False


def test_start_refreshes_command_lease():
    node = _make_node()
    node._home_goal_reserved = False
    node._last_lease_time = 0.0
    node.diffik = types.SimpleNamespace(
        ee_position=lambda _q: np.zeros(3),
        ee_rotation=lambda _q: np.eye(3),
    )

    response = node._on_start_srv(Trigger.Request(), Trigger.Response())

    assert response.success is True
    assert node._last_lease_time == 123.0
