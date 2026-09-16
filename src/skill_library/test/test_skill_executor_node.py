import json
import threading
from threading import RLock
from types import SimpleNamespace

import pytest
import rclpy

from embodied_common.dispatch_binding import copy_binding, delegated_executor_identity, new_binding
from ibrobot_msgs.action import ExecuteNavigation, PrimitiveCommand
from ibrobot_msgs.msg import SemanticObject3D
from ibrobot_msgs.srv import GetSemanticObjects
from skill_library import skill_executor_node
from skill_library.gateway_policy import (
    BoundedRequestLedger,
    ExecutionOwner,
    GatewayPolicy,
    GatewayRequest,
    RootExecutionLease,
    RuntimeSnapshot,
    SkillRequirements,
)
from skill_library.resolver import PrimitiveSpec
from skill_library.skill_executor_node import SkillExecutorNode


class _Future:
    def __init__(self, *, done: bool, result=None, exception: Exception | None = None) -> None:
        self._done = done
        self._result = result
        self._exception = exception
        self._callbacks = []

    def done(self) -> bool:
        return self._done

    def result(self):
        if self._exception is not None:
            raise self._exception
        return self._result

    def add_done_callback(self, callback) -> None:
        if self._done:
            callback(self)
            return
        self._callbacks.append(callback)

    def set_result(self, result) -> None:
        self._result = result
        self._done = True
        for callback in self._callbacks:
            callback(self)

    def set_exception(self, exception: Exception) -> None:
        self._exception = exception
        self._done = True
        for callback in self._callbacks:
            callback(self)


class _MotionModeClient:
    def __init__(self, *, success: bool = True, message: str = "") -> None:
        self.success = success
        self.message = message
        self.requests = []
        self.wait_calls = 0

    def wait_for_service(self, **_kwargs) -> bool:
        self.wait_calls += 1
        return True

    def call_async(self, request):
        self.requests.append(request)
        return _Future(done=True, result=SimpleNamespace(success=self.success, message=self.message))


class _ChildGoalHandle:
    accepted = True

    def __init__(self, events=None, *, complete_result_on_cancel: bool = True) -> None:
        self.events = events if events is not None else []
        self.complete_result_on_cancel = complete_result_on_cancel
        self.cancel_count = 0
        self.result_future = _Future(done=False)

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        self.cancel_count += 1
        self.events.append("child_cancel")
        return _Future(done=True)

    def complete_cancel_cleanup(self) -> None:
        if self.complete_result_on_cancel and self.cancel_count and not self.result_future.done():
            self.events.append("child_terminal")
            self.result_future.set_result(SimpleNamespace(result=SimpleNamespace(error_code="")))


class _CancelResponseRaisesGoalHandle(_ChildGoalHandle):
    def cancel_goal_async(self):
        self.cancel_count += 1
        self.events.append("child_cancel")
        self.result_future.set_result(SimpleNamespace(result=SimpleNamespace(error_code="")))
        return _Future(done=True, exception=RuntimeError("cancel response unavailable"))


class _ParentGoalHandle:
    def __init__(self, events=None) -> None:
        self.events = events if events is not None else []
        self.feedback = []
        self.request = _BoundRequest(
            task_id="task-1",
            skill_name="test_skill",
            target_name="",
            place_name="",
            motion_direction="",
            motion_distance=0.0,
            timeout_sec=0.01,
        )
        self.cancel_checks = 0
        self.canceled_count = 0
        self.abort_count = 0
        self.succeeded_count = 0

    @property
    def is_cancel_requested(self) -> bool:
        self.cancel_checks += 1
        return self.cancel_checks >= 2

    def publish_feedback(self, feedback) -> None:
        self.feedback.append(feedback)

    def canceled(self) -> None:
        self.canceled_count += 1
        self.events.append("parent_cancelled")

    def abort(self) -> None:
        self.abort_count += 1
        self.events.append("parent_aborted")

    def succeed(self) -> None:
        self.succeeded_count += 1
        self.events.append("parent_succeeded")


class _PrimitiveGoalHandle:
    def __init__(self) -> None:
        self.request = _BoundRequest(
            primitive_name="move_to_joint_positions",
            pose_name="",
            relative_dx=0.0,
            relative_dy=0.0,
            relative_dz=0.0,
            gripper_position=0.0,
            velocity_scaling=0.0,
            joint_names=["joint_1"],
            joint_positions=[0.1],
            joint_waypoints=[],
            joint_waypoint_count=0,
            primitive_duration_sec=0.4,
            waypoint_duration_sec=0.0,
            timeout_sec=1.0,
            task_id="task-1",
        )
        self.canceled_count = 0
        self.abort_count = 0
        self.succeeded_count = 0

    @property
    def is_cancel_requested(self) -> bool:
        return True

    def publish_feedback(self, _feedback) -> None:
        pass

    def canceled(self) -> None:
        self.canceled_count += 1

    def abort(self) -> None:
        self.abort_count += 1

    def succeed(self) -> None:
        self.succeeded_count += 1


class _NoCancelParentGoalHandle(_ParentGoalHandle):
    @property
    def is_cancel_requested(self) -> bool:
        return False


class _ManualCancelParentGoalHandle(_ParentGoalHandle):
    def __init__(self, events=None) -> None:
        super().__init__(events)
        self.cancel_requested = False

    @property
    def is_cancel_requested(self) -> bool:
        return self.cancel_requested


class _BoundRequest(SimpleNamespace):
    """Test request with the typed binding and a legacy test-only alias."""

    def __init__(self, *, task_id: str, **fields) -> None:
        fields.setdefault("schema_version", 1)
        fields.setdefault("timeout_sec", 0.0)
        fields.setdefault("container_name", "")
        fields.setdefault("direction", "")
        fields.setdefault("distance", 0.0)
        fields.setdefault("degree", 0.0)
        fields.setdefault("x", 0.0)
        fields.setdefault("y", 0.0)
        fields.setdefault("yaw", 0.0)
        fields.setdefault("has_x", False)
        fields.setdefault("has_y", False)
        fields.setdefault("has_yaw", False)
        fields.setdefault("navigation_command_type", 0)
        fields.setdefault("navigation_target_pose", skill_executor_node.PoseStamped())
        fields.setdefault("navigation_value", 0.0)
        super().__init__(dispatch_binding=new_binding(task_id=task_id), **fields)

    @property
    def task_id(self) -> str:
        return self.dispatch_binding.task_id

    @task_id.setter
    def task_id(self, value: str) -> None:
        self.dispatch_binding.task_id = str(value)


def _make_skill_node(send_goal_future) -> SkillExecutorNode:
    node = object.__new__(SkillExecutorNode)
    node._skill_goal_lock = threading.Lock()
    node._skill_goal_active = False
    node._validate_skill = lambda *_args, **_kwargs: (True, "")
    node._current_joint_positions = lambda: {}
    node._named_targets = {}
    node._gripper_open = 1.0
    node._gripper_closed = 0.0
    node._skill_templates = {}
    node._relative_motion_direction_mapping = {}
    node._arm_joint_names = []
    node._rpc_timeout = 0.01
    node._debug = False
    node._primitive_client = SimpleNamespace(
        wait_for_server=lambda **_kwargs: True,
        send_goal_async=lambda _goal, **_kwargs: send_goal_future,
    )
    return node


@pytest.mark.parametrize(("context_schema_version", "has_navigation"), [(1, False), (2, True)])
def test_runtime_snapshot_uses_configured_context_schema(monkeypatch, context_schema_version, has_navigation):
    captured = {}
    node = object.__new__(SkillExecutorNode)
    node._robot_name = "test_robot"
    node._context_schema_version = context_schema_version
    node._robot_config_digest = "a" * 64
    node._named_poses = {}
    node._named_targets = {}
    node._arm_joint_names = []
    node._joint_limits = {}
    node._workspace = {}
    node._skill_required_control_mode = "moveit_planning"
    node._gateway_timeout_policy = {"task_budget_sec": 180.0}
    node._relative_motion_reference_frame = "base"
    node._relative_motion_step_m = 0.03
    node._relative_motion_direction_mapping = {}
    node._gripper_open = 1.0
    node._gripper_closed = 0.0
    node._skill_action_name = "/embodied/execute_skill"
    node._primitive_action_name = "/embodied/execute_primitive"
    node._validate_skill_service = "/embodied/validate_skill"
    node._validate_primitive_service = "/embodied/validate_primitive"
    node._skill_gateway_status_service = "/embodied/get_skill_gateway_status"
    node._begin_workflow_service = "/embodied/begin_workflow_execution"
    node._finalize_workflow_service = "/embodied/finalize_workflow_execution"
    node._task_executor_action_name = "/task_executor/execute_task_plan"
    node._arm_trajectory_action_name = "/arm_trajectory_controller/follow_joint_trajectory"
    node._move_configuration_service = skill_executor_node.RUNTIME.MOVE_TO_JOINT_SERVICE
    node._navigation_action_name = "/navigation/execute"
    node._skill_catalog_source_mode = "development"
    node._skill_catalog_source_root = "."
    node._skill_catalog_profile = "unused"
    node._delegated_executor_descriptors = lambda: {}
    snapshot = type("Snapshot", (), {"enabled_skill_names": ()})()

    def compile_catalog(_compiler, _source, *, profile_name, context):
        captured["profile_name"] = profile_name
        captured["context"] = context
        return snapshot

    monkeypatch.setattr(skill_executor_node.SkillCatalogCompiler, "compile", compile_catalog)

    assert node._compile_runtime_snapshot() is snapshot
    robot_context = captured["context"].robot
    assert robot_context.context_schema_version == context_schema_version
    assert ("navigation_action" in robot_context.execution_endpoints) is has_navigation


def test_semantic_map_executor_descriptor_uses_service_identity():
    node = object.__new__(SkillExecutorNode)
    node._skill_templates = {"resolve_object_pose": {"executor": "semantic_map_query"}}
    node._grasp_execution = {}
    node._placement_execution = {}
    node._pick_action_name = "/manipulation/execute_pick"
    node._place_action_name = "/manipulation/execute_place"
    node._semantic_map_target_service = "/semantic_mapping/get_objects"
    node._semantic_map_stand_off_distance_m = 0.3

    descriptor = node._delegated_executor_descriptors()["semantic_map_query"]
    expected = delegated_executor_identity(
        name="semantic_map_query",
        endpoint_name="/semantic_mapping/get_objects",
        endpoint_kind="ros_service",
        configuration={
            "query_service": "/semantic_mapping/get_objects",
            "stand_off_distance_m": 0.3,
        },
    )

    assert vars(descriptor) == expected


def test_semantic_map_executor_is_registered_before_templates_are_loaded():
    node = object.__new__(SkillExecutorNode)
    node._skill_templates = {}
    node._grasp_execution = {}
    node._placement_execution = {}
    node._pick_action_name = "/manipulation/execute_pick"
    node._place_action_name = "/manipulation/execute_place"
    node._semantic_map_target_service = "/semantic_mapping/resolve_target"
    node._semantic_map_stand_off_distance_m = 0.3

    assert "semantic_map_query" in node._delegated_executor_descriptors()


def test_semantic_map_query_calls_resolve_target_service_and_returns_pose_json():
    requests = []
    response = GetSemanticObjects.Response()
    response.success = True
    response.semantic_map.objects = [SemanticObject3D()]
    response.semantic_map.objects[0].pose.pose.pose.position.x = 1.25
    response.semantic_map.objects[0].pose.pose.pose.position.y = -0.5

    class Client:
        @staticmethod
        def wait_for_service(**_kwargs):
            return True

        @staticmethod
        def call_async(request):
            requests.append(request)
            return _Future(done=True, result=response)

    node = object.__new__(SkillExecutorNode)
    node._semantic_map_target_client = Client()
    node._semantic_map_target_service = "/semantic_mapping/get_objects"
    node._semantic_map_stand_off_distance_m = 0.3
    node._rpc_timeout = 0.1
    node._tf_buffer = SimpleNamespace(
        lookup_transform=lambda *_args, **_kwargs: SimpleNamespace(
            transform=SimpleNamespace(translation=SimpleNamespace(x=0.0, y=0.0))
        )
    )
    node._set_result_catalog_identity = lambda result, bundle=None: None
    goal_handle = SimpleNamespace(
        request=SimpleNamespace(
            skill_name="resolve_object_pose",
            target_name="banana",
            distance=0.0,
            timeout_sec=5.0,
        ),
        is_cancel_requested=False,
        succeed=lambda: None,
    )

    result = node._execute_semantic_map_query(goal_handle)

    assert result.success is True
    assert json.loads(result.message) == pytest.approx([0.9714569927, -0.3885827971, -21.8014094864])
    assert len(requests) == 1
    assert requests[0].label == "banana"
    assert requests[0].include_inactive is True
    assert requests[0].max_age_sec == 0.0


def test_semantic_map_query_uses_request_standoff_distance():
    response = GetSemanticObjects.Response()
    response.success = True
    response.semantic_map.objects = [SemanticObject3D()]
    response.semantic_map.objects[0].pose.pose.pose.position.x = 1.25
    response.semantic_map.objects[0].pose.pose.pose.position.y = -0.5

    class Client:
        @staticmethod
        def wait_for_service(**_kwargs):
            return True

        @staticmethod
        def call_async(_request):
            return _Future(done=True, result=response)

    node = object.__new__(SkillExecutorNode)
    node._semantic_map_target_client = Client()
    node._semantic_map_target_service = "/semantic_mapping/get_objects"
    node._semantic_map_stand_off_distance_m = 0.3
    node._rpc_timeout = 0.1
    node._tf_buffer = SimpleNamespace(
        lookup_transform=lambda *_args, **_kwargs: SimpleNamespace(
            transform=SimpleNamespace(translation=SimpleNamespace(x=0.0, y=0.0))
        )
    )
    node._set_result_catalog_identity = lambda result, bundle=None: None
    goal_handle = SimpleNamespace(
        request=SimpleNamespace(
            skill_name="resolve_object_pose",
            target_name="banana",
            distance=0.3,
            timeout_sec=5.0,
        ),
        is_cancel_requested=False,
        succeed=lambda: None,
    )

    result = node._execute_semantic_map_query(goal_handle)

    assert json.loads(result.message) == pytest.approx([0.9714569927, -0.3885827971, -21.8014094864])


def test_v1_runtime_snapshot_reports_navigation_not_ready_without_client():
    node = object.__new__(SkillExecutorNode)
    node._context_schema_version = 1
    node._gateway_lease = RootExecutionLease()
    node._active_workflow = None
    node._motion_authorized = True
    node._active_control_mode = "moveit_planning"
    node._skill_required_control_mode = "moveit_planning"
    node._validate_skill_client = SimpleNamespace(service_is_ready=lambda: True)
    node._task_executor_client = SimpleNamespace(server_is_ready=lambda: True)
    node._arm_trajectory_client = SimpleNamespace(server_is_ready=lambda: True)
    node._ee_pose_is_fresh = lambda: True

    snapshot = node._runtime_snapshot()

    assert not hasattr(node, "_navigation_client")
    assert snapshot.navigation_ready is False


@pytest.mark.parametrize(
    ("context_schema_version", "supported_control_modes", "expected"),
    [
        (1, (), False),
        (2, ("moveit_planning", "base_navigation"), False),
        (3, (), False),
        (3, ("moveit_planning", "base_navigation"), True),
    ],
)
def test_runtime_snapshot_reports_control_mode_switching_only_for_hybrid_context(
    context_schema_version, supported_control_modes, expected
):
    node = object.__new__(SkillExecutorNode)
    node._context_schema_version = context_schema_version
    node._supported_control_modes = supported_control_modes
    node._motion_mode_service = "motion_mode/set_navigation_enabled"
    node._gateway_lease = RootExecutionLease()
    node._active_workflow = None
    node._motion_authorized = True
    node._active_control_mode = "moveit_planning"
    node._skill_required_control_mode = "moveit_planning"
    node._validate_skill_client = SimpleNamespace(service_is_ready=lambda: True)
    node._task_executor_client = SimpleNamespace(server_is_ready=lambda: True)
    node._arm_trajectory_client = SimpleNamespace(server_is_ready=lambda: True)
    node._navigation_client = SimpleNamespace(server_is_ready=lambda: True)
    node._ee_pose_is_fresh = lambda: True

    assert node._runtime_snapshot().control_mode_switching_enabled is expected


def _make_motion_mode_node(active_mode: str, required_mode: str, *, success: bool = True):
    node = object.__new__(SkillExecutorNode)
    node._active_control_mode = active_mode
    node._supported_control_modes = ("moveit_planning", "base_navigation")
    node._motion_mode_service = "motion_mode/set_navigation_enabled"
    node._motion_mode_switch_lock = RLock()
    node._rpc_timeout = 0.1
    node._skill_control_modes = {"test_skill": required_mode}
    node._motion_mode_client = _MotionModeClient(success=success, message="injected rejection")
    return node


def test_control_mode_switch_is_skipped_when_required_mode_is_already_active():
    node = _make_motion_mode_node("moveit_planning", "moveit_planning")

    ready, reason = node._ensure_skill_control_mode("test_skill", _NoCancelParentGoalHandle())

    assert ready is True
    assert reason == ""
    assert node._motion_mode_client.wait_calls == 0
    assert node._motion_mode_client.requests == []


@pytest.mark.parametrize(
    ("active_mode", "required_mode", "navigation_enabled"),
    [
        ("moveit_planning", "base_navigation", True),
        ("base_navigation", "moveit_planning", False),
    ],
)
def test_control_mode_switch_routes_skill_domain_through_motion_mode_service(
    active_mode, required_mode, navigation_enabled
):
    node = _make_motion_mode_node(active_mode, required_mode)

    ready, reason = node._ensure_skill_control_mode("test_skill", _NoCancelParentGoalHandle())

    assert ready is True
    assert reason == ""
    assert node._active_control_mode == required_mode
    assert node._motion_mode_client.wait_calls == 1
    assert [request.data for request in node._motion_mode_client.requests] == [navigation_enabled]


def test_rejected_control_mode_switch_fails_closed_with_unknown_active_mode():
    node = _make_motion_mode_node("moveit_planning", "base_navigation", success=False)

    ready, reason = node._ensure_skill_control_mode("test_skill", _NoCancelParentGoalHandle())

    assert ready is False
    assert reason == "injected rejection"
    assert node._active_control_mode == ""
    assert [request.data for request in node._motion_mode_client.requests] == [True]


def test_hybrid_runtime_without_motion_mode_client_fails_closed():
    node = object.__new__(SkillExecutorNode)
    node._active_control_mode = "moveit_planning"
    node._supported_control_modes = ("moveit_planning", "base_navigation")
    node._motion_mode_service = "motion_mode/set_navigation_enabled"
    node._skill_control_modes = {"test_skill": "base_navigation"}

    ready, reason = node._ensure_skill_control_mode("test_skill", _NoCancelParentGoalHandle())

    assert ready is False
    assert reason == "motion-mode client is unavailable"


def test_rejected_control_mode_switch_prevents_skill_primitive_dispatch():
    primitive_dispatches = []
    node = _make_skill_node(_Future(done=False))
    node._primitive_client.send_goal_async = lambda *_args, **_kwargs: primitive_dispatches.append(True)
    node._active_control_mode = "moveit_planning"
    node._supported_control_modes = ("moveit_planning", "base_navigation")
    node._motion_mode_service = "motion_mode/set_navigation_enabled"
    node._motion_mode_switch_lock = RLock()
    node._skill_control_modes = {"test_skill": "base_navigation"}
    node._motion_mode_client = _MotionModeClient(success=False, message="base controller unavailable")
    goal_handle = _NoCancelParentGoalHandle()

    result = node._execute_skill(goal_handle)

    assert result.success is False
    assert result.error_code == "CONTROL_MODE_MISMATCH"
    assert result.message == "base controller unavailable"
    assert goal_handle.abort_count == 1
    assert primitive_dispatches == []


@pytest.mark.parametrize("executor", ["", "grasp_pipeline", "placement_pipeline", "imitate_human_motion"])
def test_missing_explicit_runtime_status_prevents_primitive_and_delegated_dispatch(executor):
    dispatches = []
    node = _make_skill_node(_Future(done=False))
    node._runtime_enabled = True
    node._runtime_status_snapshot = None
    node._motion_mode_switch_lock = RLock()
    node._skill_templates = {"test_skill": {"executor": executor}}
    node._primitive_client.send_goal_async = lambda *_args, **_kwargs: dispatches.append("primitive")
    node._execute_pick_skill = lambda *_args, **_kwargs: dispatches.append("pick")
    node._execute_place_skill = lambda *_args, **_kwargs: dispatches.append("place")
    node._execute_imitate_human_motion_skill = lambda *_args, **_kwargs: dispatches.append("imitate")
    goal_handle = _NoCancelParentGoalHandle()

    result = node._execute_skill(goal_handle)

    assert result.success is False
    assert result.error_code == "CONTROL_MODE_MISMATCH"
    assert "not been received" in result.message
    assert goal_handle.abort_count == 1
    assert dispatches == []


def test_wait_for_future_runs_cancel_callback_once(monkeypatch):
    cancelled = []
    monkeypatch.setattr(rclpy, "ok", lambda: True)

    completed = SkillExecutorNode._wait_for_future(
        _Future(done=False),
        timeout_sec=1.0,
        cancel_requested=lambda: True,
        cancel_callback=lambda: cancelled.append(True),
    )

    assert completed is False
    assert cancelled == [True]


def test_best_effort_cancel_audits_propagation_once():
    events = []
    goal_handle = SimpleNamespace(cancel_goal_async=lambda: events.append("cancel"))
    node = object.__new__(SkillExecutorNode)
    node._audit_cancel_propagated = lambda: events.append("audit")

    node._best_effort_cancel_goal(goal_handle)

    assert events == ["audit", "cancel"]


def test_delegated_primitive_requires_exact_active_nonce_binding():
    node = object.__new__(SkillExecutorNode)
    node._state_lock = RLock()
    node._active_delegated_dispatches = {}
    node._runtime_snapshot = lambda: RuntimeSnapshot(True, "cartesian", "cartesian")
    node._ros_time_sec = lambda: 1.0
    policy = GatewayPolicy(
        {"default_skill_timeout_sec": 5.0, "task_budget_sec": 10.0},
        {"skill": SkillRequirements()},
        parameter_schemas={"skill": {"type": "object", "properties": {}, "required": []}},
        ledger=BoundedRequestLedger(2),
        lease=RootExecutionLease(),
    )
    node._gateway_policy = policy
    owner = ExecutionOwner.skill_command("task-1")
    admission = policy.admit(GatewayRequest("task-1", "skill"), node._runtime_snapshot(), owner)
    binding = new_binding(task_id="task-1")
    binding.task_budget.schema_version = 1
    binding.task_budget.started_at.sec = 1
    binding.task_budget.deadline.sec = 2_000_000_000
    binding.dispatch_nonce = "delegated-nonce"
    node._active_delegated_dispatches[binding.dispatch_nonce] = (admission, copy_binding(binding))
    goal = SimpleNamespace(dispatch_binding=binding, primitive_name="move_to_named_pose", timeout_sec=0.0)

    assert node._admit_primitive(goal, None) == ("", None, admission, True)
    assert goal.timeout_sec > 0.0
    goal.dispatch_binding.root_task_id = "tampered"
    assert node._admit_primitive(goal, None)[0] == "SKILL_BUSY"


def test_pick_skill_sets_internal_execute_policy(monkeypatch):
    sent_goals = []

    class PickClient:
        @staticmethod
        def wait_for_server(**_kwargs):
            return True

        @staticmethod
        def send_goal_async(goal, **_kwargs):
            sent_goals.append(goal)
            raise RuntimeError("stop after inspecting the delegated goal")

    expected_executor = {
        "name": "grasp_pipeline",
        "contract_version": "1",
        "endpoint_kind": "ros_action",
        "endpoint_name": "/manipulation/execute_pick",
        "configuration_digest": "a" * 64,
        "model_deployment_name": "graspgen",
        "model_fingerprint": "b" * 64,
        "model_bundle_digest": "c" * 64,
    }
    executor = SimpleNamespace(**expected_executor)
    node = object.__new__(SkillExecutorNode)
    node._pick_client = PickClient()
    node._rpc_timeout = 0.01
    node._active_runtime_bundle = SimpleNamespace(
        snapshot=SimpleNamespace(delegated_executors={"grasp_pipeline": executor})
    )
    node._active_skill_admission = object()
    node._register_delegated_dispatch = lambda *_args: b"cleanup"
    node._confirm_delegated_terminal = lambda *_args: None
    node._abort_skill = lambda result, _handle, _phases, code, message: SimpleNamespace(
        success=False, error_code=code, message=message
    )
    node.get_logger = lambda: SimpleNamespace(error=lambda _message: None)
    goal_handle = SimpleNamespace(
        request=_BoundRequest(
            task_id="pick-1",
            skill_name="pick_object",
            target_name="banana",
            place_name="",
            motion_direction="",
            motion_distance=0.0,
            timeout_sec=10.0,
        )
    )

    result = node._execute_pick_skill(goal_handle, {"timeout_sec": 10.0})

    assert result.success is False
    assert result.error_code == skill_executor_node.PRIMITIVE_CANCEL_CLEANUP_TIMEOUT
    assert len(sent_goals) == 1
    assert sent_goals[0].mode == skill_executor_node.PickObject.Goal.MODE_EXECUTE
    assert sent_goals[0].supervised_direct is False
    assert sent_goals[0].release_after_success is False


def test_fresh_ee_pose_snapshot_is_independent_and_rejects_stale_pose(monkeypatch):
    monotonic_time = [100.0]
    monkeypatch.setattr(skill_executor_node.time, "monotonic", lambda: monotonic_time[0])
    node = object.__new__(SkillExecutorNode)
    node._state_lock = RLock()
    node._robot_state_freshness_sec = 0.1
    pose = skill_executor_node.PoseStamped()
    pose.pose.position.x = 0.2
    pose.pose.orientation.w = 1.0

    node._handle_ee_pose(pose)
    snapshot = node._fresh_ee_pose_snapshot()

    assert snapshot.pose is not pose
    assert snapshot.pose.pose.position.x == pytest.approx(0.2)
    assert snapshot.received_monotonic == pytest.approx(100.0)
    pose.pose.position.x = 0.4
    assert snapshot.pose.pose.position.x == pytest.approx(0.2)

    monotonic_time[0] += 0.09
    assert node._ee_pose_receipt_is_fresh(snapshot.received_monotonic) is True
    monotonic_time[0] += 0.02
    assert node._ee_pose_receipt_is_fresh(snapshot.received_monotonic) is False
    assert node._fresh_ee_pose_snapshot() is None


def test_execute_skill_waits_for_child_terminal_before_parent_cancel(monkeypatch):
    events = []
    child_goal_handle = _ChildGoalHandle(events)
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    monkeypatch.setattr(skill_executor_node.time, "sleep", lambda _seconds: child_goal_handle.complete_cancel_cleanup())
    node = _make_skill_node(_Future(done=True, result=child_goal_handle))
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [PrimitiveSpec("open_gripper", gripper_position=1.0)],
    )
    parent_goal_handle = _ParentGoalHandle(events)

    result = node._execute_skill(parent_goal_handle)

    assert result.success is False
    assert result.error_code == "SKILL_CANCELLED"
    assert child_goal_handle.cancel_count == 1
    assert child_goal_handle.result_future.done()
    assert parent_goal_handle.canceled_count == 1
    assert parent_goal_handle.abort_count == 0
    assert events.index("child_terminal") < events.index("parent_cancelled")


def test_execute_skill_drains_child_accepted_after_parent_cancel(monkeypatch):
    events = []
    child_goal_handle = _ChildGoalHandle(events)
    send_goal_future = _Future(done=False)
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    node = _make_skill_node(send_goal_future)
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [PrimitiveSpec("open_gripper", gripper_position=1.0)],
    )

    def advance_cancel_cleanup(_seconds) -> None:
        if not send_goal_future.done():
            send_goal_future.set_result(child_goal_handle)
        else:
            child_goal_handle.complete_cancel_cleanup()

    monkeypatch.setattr(skill_executor_node.time, "sleep", advance_cancel_cleanup)
    parent_goal_handle = _ParentGoalHandle(events)

    result = node._execute_skill(parent_goal_handle)

    assert result.error_code == "SKILL_CANCELLED"
    assert child_goal_handle.cancel_count == 1
    assert child_goal_handle.result_future.done()
    assert parent_goal_handle.canceled_count == 1
    assert events.index("child_terminal") < events.index("parent_cancelled")


def test_execute_skill_aborts_when_cancel_cleanup_does_not_reach_terminal(monkeypatch):
    events = []
    child_goal_handle = _ChildGoalHandle(events, complete_result_on_cancel=False)
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    node = _make_skill_node(_Future(done=True, result=child_goal_handle))
    node._rpc_timeout = 0.0
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [PrimitiveSpec("open_gripper", gripper_position=1.0)],
    )
    parent_goal_handle = _ParentGoalHandle(events)

    result = node._execute_skill(parent_goal_handle)

    assert result.error_code == "SKILL_CANCEL_TIMEOUT"
    assert child_goal_handle.cancel_count == 1
    assert child_goal_handle.result_future.done() is False
    assert parent_goal_handle.abort_count == 1
    assert parent_goal_handle.canceled_count == 0
    assert events[-1] == "parent_aborted"


def test_execute_skill_aborts_when_child_cancel_response_is_unknown(monkeypatch):
    events = []
    child_goal_handle = _CancelResponseRaisesGoalHandle(events)
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    node = _make_skill_node(_Future(done=True, result=child_goal_handle))
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [PrimitiveSpec("open_gripper", gripper_position=1.0)],
    )
    parent_goal_handle = _ParentGoalHandle(events)

    result = node._execute_skill(parent_goal_handle)

    assert result.error_code == "SKILL_CANCEL_TIMEOUT"
    assert child_goal_handle.cancel_count == 1
    assert parent_goal_handle.abort_count == 1
    assert parent_goal_handle.canceled_count == 0


def test_execute_skill_send_timeout_fails_closed_when_late_goal_cleanup_is_unknown(monkeypatch):
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    node = _make_skill_node(_Future(done=False))
    node._rpc_timeout = 0.0
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [PrimitiveSpec("open_gripper", gripper_position=1.0)],
    )
    parent_goal_handle = _NoCancelParentGoalHandle()

    result = node._execute_skill(parent_goal_handle)

    assert result.error_code == "SKILL_CANCEL_TIMEOUT"
    assert parent_goal_handle.abort_count == 1


def test_execute_skill_feedback_uses_sanitized_step_progress(monkeypatch):
    child_goal_handle = _ChildGoalHandle()
    child_goal_handle.result_future.set_result(
        SimpleNamespace(result=SimpleNamespace(success=True, error_code="", message=""))
    )
    node = _make_skill_node(_Future(done=True, result=child_goal_handle))
    audit_events = []
    node._active_audit_context = {"task_id": "task-1", "payload_hash": "digest", "skill": "test_skill"}
    node._audit = lambda event, **fields: audit_events.append((event, fields))
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [
            PrimitiveSpec("move_to_named_pose", pose_name="private_pose"),
            PrimitiveSpec("move_to_joint_positions", joint_names=["joint_1"], joint_positions=[0.1]),
        ],
    )
    parent_goal_handle = _NoCancelParentGoalHandle()

    result = node._execute_skill(parent_goal_handle)

    assert result.success is True
    assert [(feedback.state, feedback.detail) for feedback in parent_goal_handle.feedback] == [
        ("executing", "step 1 of 2"),
        ("executing", "step 2 of 2"),
    ]
    assert all(
        value not in feedback.detail
        for feedback in parent_goal_handle.feedback
        for value in ("move_to_named_pose", "move_to_joint_positions", "private_pose", "joint_1")
    )
    assert [fields for event, fields in audit_events if event == "primitive_started"] == [
        {"task_id": "task-1", "payload_hash": "digest", "skill": "test_skill", "step_count": 1},
        {"task_id": "task-1", "payload_hash": "digest", "skill": "test_skill", "step_count": 2},
    ]


@pytest.mark.parametrize(
    ("primitive_name", "expected_schema_version"),
    [("open_gripper", 1), ("nav_straight", 2)],
)
def test_skill_dispatches_primitive_goal_with_selected_contract_version(
    monkeypatch, primitive_name, expected_schema_version
):
    child_goal_handle = _ChildGoalHandle()
    child_goal_handle.result_future.set_result(
        SimpleNamespace(result=SimpleNamespace(success=True, error_code="", message=""))
    )
    node = _make_skill_node(_Future(done=True, result=child_goal_handle))
    sent_goals = []

    def send_goal(goal, **_kwargs):
        sent_goals.append(goal)
        return _Future(done=True, result=child_goal_handle)

    node._primitive_client = SimpleNamespace(wait_for_server=lambda **_kwargs: True, send_goal_async=send_goal)
    if primitive_name == "nav_straight":
        navigation_goal = ExecuteNavigation.Goal()
        navigation_goal.command_type = ExecuteNavigation.Goal.FORWARD
        navigation_goal.value = 1.0
        primitive = PrimitiveSpec(primitive_name, navigation_goal=navigation_goal)
    else:
        primitive = PrimitiveSpec(primitive_name, gripper_position=1.0)
    monkeypatch.setattr(skill_executor_node, "resolve_skill_primitives", lambda *_args, **_kwargs: [primitive])

    result = node._execute_skill(_NoCancelParentGoalHandle())

    assert result.success is True
    assert len(sent_goals) == 1
    assert sent_goals[0].schema_version == expected_schema_version


class _GatewayPolicyStub:
    def __init__(self, events, *, finalize_raises: bool = False) -> None:
        self.events = events
        self.finalize_raises = finalize_raises
        self.admission = SimpleNamespace(
            admitted=True,
            effective_timeout_sec=1.0,
            prepared_request=SimpleNamespace(identity=SimpleNamespace(task_id="task-1", payload_hash="digest")),
        )
        self.finalize_calls = 0

    def admit(self, *_args, **_kwargs):
        return self.admission

    def finalize(self, *_args, **_kwargs):
        self.finalize_calls += 1
        self.events.append("finalize")
        if self.finalize_raises:
            raise RuntimeError("injected finalization failure")


def _make_gateway_wrapper_node(events, *, finalize_raises: bool = False) -> SkillExecutorNode:
    node = object.__new__(SkillExecutorNode)
    node._skill_goal_lock = threading.Lock()
    node._skill_goal_active = False
    node._gateway_policy = _GatewayPolicyStub(events, finalize_raises=finalize_raises)
    node._runtime_snapshot = lambda: SimpleNamespace()
    node._validate_skill = lambda *_args, **_kwargs: (True, "")
    node._audit = lambda *_args, **_kwargs: None
    node.get_logger = lambda: SimpleNamespace(error=lambda _message: None)
    node._active_skill_admission = None
    node._active_skill_owner = None
    node._active_audit_context = None
    node._state_lock = RLock()
    node._pending_internal_primitive_goals = {}
    node._active_internal_primitive_goals = {}
    node._retained_admission_cleanup = {}
    return node


def _make_retained_gateway_node(send_goal_future) -> tuple[SkillExecutorNode, GatewayPolicy, RootExecutionLease]:
    lease = RootExecutionLease()
    node = _make_skill_node(send_goal_future)
    node._gateway_lease = lease
    node._gateway_ledger = BoundedRequestLedger(4)
    node._gateway_policy = GatewayPolicy(
        {"default_skill_timeout_sec": 1.0, "task_budget_sec": 2.0},
        {"test_skill": SkillRequirements()},
        parameter_schemas={
            "test_skill": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
                "required": [],
            }
        },
        ledger=node._gateway_ledger,
        lease=lease,
    )
    node._runtime_snapshot = lambda: RuntimeSnapshot(
        motion_authorized=True,
        active_control_mode="gateway_mode",
        required_control_mode="gateway_mode",
    )
    node._ros_time_sec = lambda: 1.0
    node._audit = lambda *_args, **_kwargs: None
    node.get_logger = lambda: SimpleNamespace(error=lambda _message: None)
    node._active_skill_admission = None
    node._active_skill_owner = None
    node._active_audit_context = None
    node._state_lock = RLock()
    node._internal_goal_lock = node._state_lock
    node._pending_internal_primitive_goals = {}
    node._active_internal_primitive_goals = {}
    node._retained_admission_cleanup = {}
    return node, node._gateway_policy, lease


class _TerminalOnCancelGoalHandle(_ChildGoalHandle):
    def cancel_goal_async(self):
        cancel_future = super().cancel_goal_async()
        self.result_future.set_result(SimpleNamespace(result=SimpleNamespace(error_code="")))
        return cancel_future


class _ResultFutureRaisesGoalHandle:
    accepted = True

    def __init__(self) -> None:
        self.cancel_count = 0

    def get_result_async(self):
        raise RuntimeError("result future unavailable")

    def cancel_goal_async(self):
        self.cancel_count += 1
        return _Future(done=True)


class _ExternalPolicyStub:
    def __init__(self, release_result) -> None:
        self.release_result = release_result
        self.token = object()

    def borrow_internal(self, *_args, **_kwargs):
        return None

    def admit_external_primitive(self, *_args, **_kwargs):
        return "", self.token

    def release_external_primitive(self, _token):
        if isinstance(self.release_result, Exception):
            raise self.release_result
        return self.release_result


class _PrimitiveActionGoalHandle:
    def __init__(self, goal_id, task_id: str = "task-1", binding=None, execution_token: str = "") -> None:
        self.goal_id = goal_id
        self.request = _BoundRequest(
            task_id=task_id,
            execution_token=execution_token,
            primitive_name="open_gripper",
            pose_name="",
        )
        if binding is not None:
            self.request.dispatch_binding = copy_binding(binding)
        self.abort_count = 0
        self.canceled_count = 0
        self.succeeded_count = 0

    def abort(self) -> None:
        self.abort_count += 1

    def canceled(self) -> None:
        self.canceled_count += 1

    def succeed(self) -> None:
        self.succeeded_count += 1

    @property
    def is_cancel_requested(self) -> bool:
        return False

    def publish_feedback(self, _feedback) -> None:
        pass


class _CountingReleasePolicy(_ExternalPolicyStub):
    def __init__(self) -> None:
        super().__init__(True)
        self.release_count = 0

    def release_external_primitive(self, token):
        self.release_count += 1
        return super().release_external_primitive(token)


def test_deferred_terminal_records_child_intent_without_terminalizing_parent():
    parent_goal_handle = _ParentGoalHandle()
    deferred_goal_handle = skill_executor_node._DeferredTerminalGoalHandle(parent_goal_handle)

    deferred_goal_handle.canceled()

    assert deferred_goal_handle.terminal_intent == "canceled"
    assert parent_goal_handle.canceled_count == 0
    assert parent_goal_handle.abort_count == 0
    assert parent_goal_handle.succeeded_count == 0


def test_deferred_terminal_commits_once_using_recorded_intent():
    parent_goal_handle = _ParentGoalHandle()
    deferred_goal_handle = skill_executor_node._DeferredTerminalGoalHandle(parent_goal_handle)

    deferred_goal_handle.succeed()

    assert deferred_goal_handle.commit() is True
    assert deferred_goal_handle.commit() is False
    assert parent_goal_handle.succeeded_count == 1
    assert parent_goal_handle.abort_count == 0


def test_gateway_defers_parent_success_until_policy_finalization():
    events = []
    node = _make_gateway_wrapper_node(events)
    parent_goal_handle = _ParentGoalHandle(events)

    def child(goal_handle, **_kwargs):
        events.append("child_terminal")
        goal_handle.succeed()
        return SimpleNamespace(success=True, error_code="", message="", executed_primitives=[])

    node._execute_skill_child = child

    result = node._execute_skill(parent_goal_handle)

    assert result.success is True
    assert events == ["child_terminal", "finalize", "parent_succeeded"]


def test_gateway_aborts_instead_of_committing_success_when_finalization_fails():
    events = []
    node = _make_gateway_wrapper_node(events, finalize_raises=True)
    parent_goal_handle = _ParentGoalHandle(events)

    def child(goal_handle, **_kwargs):
        events.append("child_terminal")
        goal_handle.succeed()
        return SimpleNamespace(success=True, error_code="", message="", executed_primitives=[])

    node._execute_skill_child = child

    result = node._execute_skill(parent_goal_handle)

    assert result.error_code == "GATEWAY_FINALIZATION_FAILED"
    assert events == ["child_terminal", "finalize", "parent_aborted"]


def test_gateway_cleanup_unknown_aborts_without_finalizing_or_releasing_lease():
    events = []
    node = _make_gateway_wrapper_node(events)
    parent_goal_handle = _ParentGoalHandle(events)

    def child(goal_handle, **_kwargs):
        events.append("child_terminal")
        goal_handle.abort()
        return SimpleNamespace(
            success=False,
            error_code="SKILL_CANCEL_TIMEOUT",
            message="cleanup unknown",
            executed_primitives=[],
        )

    node._execute_skill_child = child

    result = node._execute_skill(parent_goal_handle)

    assert result.error_code == "SKILL_CANCEL_TIMEOUT"
    assert node._gateway_policy.finalize_calls == 0
    assert events == ["child_terminal", "parent_aborted"]


def test_external_primitive_keeps_lease_when_downstream_cleanup_is_unknown():
    lease = RootExecutionLease()
    policy = GatewayPolicy(
        {"default_skill_timeout_sec": 1.0, "task_budget_sec": 2.0},
        {},
        parameter_schemas={},
        ledger=BoundedRequestLedger(1),
        lease=lease,
    )
    node = object.__new__(SkillExecutorNode)
    node._gateway_policy = policy
    node._gateway_lease = lease
    node._active_skill_admission = None
    node._active_skill_owner = None
    node._runtime_snapshot = lambda: RuntimeSnapshot(
        motion_authorized=True,
        active_control_mode="gateway_mode",
        required_control_mode="gateway_mode",
    )
    node._execute_primitive_unchecked = lambda _goal_handle: SimpleNamespace(
        success=False,
        error_code="CANCEL_CLEANUP_TIMEOUT",
        message="cleanup unknown",
    )
    goal_handle = _PrimitiveGoalHandle()

    result = node._execute_primitive(goal_handle)

    assert result.error_code == "CANCEL_CLEANUP_TIMEOUT"
    assert lease.owner == ExecutionOwner.external_primitive("task-1")


@pytest.mark.parametrize("release_result", [False, RuntimeError("release failed")])
def test_external_primitive_aborts_when_lease_release_cannot_be_confirmed(release_result):
    node = object.__new__(SkillExecutorNode)
    node._gateway_policy = _ExternalPolicyStub(release_result)
    node._runtime_snapshot = lambda: RuntimeSnapshot(
        motion_authorized=True,
        active_control_mode="gateway_mode",
        required_control_mode="gateway_mode",
    )
    node._state_lock = RLock()
    node._pending_internal_primitive_goals = {}
    node._active_internal_primitive_goals = {}
    node._execute_primitive_unchecked = lambda goal_handle: (
        goal_handle.succeed() or SimpleNamespace(success=True, error_code="", message="")
    )
    goal_handle = _PrimitiveGoalHandle()

    result = node._execute_primitive(goal_handle)

    assert result.error_code == "GATEWAY_FINALIZATION_FAILED"
    assert goal_handle.abort_count == 1
    assert goal_handle.succeeded_count == 0


def test_gateway_send_future_exception_retains_root_and_revokes_internal_goal(monkeypatch):
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    send_future = _Future(done=True, exception=RuntimeError("send future failed"))
    node, policy, lease = _make_retained_gateway_node(send_future)
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [PrimitiveSpec("open_gripper", gripper_position=1.0)],
    )
    parent_goal_handle = _NoCancelParentGoalHandle()

    result = node._execute_skill(parent_goal_handle)

    assert result.error_code == "SKILL_CANCEL_TIMEOUT"
    assert parent_goal_handle.abort_count == 1
    assert lease.owner == ExecutionOwner.skill_command("task-1")
    assert policy._ledger.query("task-1").state == "active"
    assert node._pending_internal_primitive_goals == {}
    assert node._active_internal_primitive_goals == {}


def test_revoked_internal_uuid_cannot_borrow_after_send_future_exception(monkeypatch):
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    sent_goals = {}
    node, _policy, _lease = _make_retained_gateway_node(_Future(done=True, exception=RuntimeError("send failed")))
    node._primitive_client.send_goal_async = lambda goal, **kwargs: (
        sent_goals.update(goal=goal, goal_id=kwargs["goal_uuid"]) or node._primitive_client_future
    )
    node._primitive_client_future = _Future(done=True, exception=RuntimeError("send failed"))
    downstream_calls = []
    node._execute_primitive_unchecked = lambda _goal_handle: downstream_calls.append(True)
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [PrimitiveSpec("open_gripper", gripper_position=1.0)],
    )

    result = node._execute_skill(_NoCancelParentGoalHandle())
    late_goal_handle = SimpleNamespace(
        goal_id=sent_goals["goal_id"],
        request=_BoundRequest(
            task_id="task-1",
            primitive_name="open_gripper",
            pose_name="",
        ),
        abort=lambda: None,
    )
    late_result = node._execute_primitive(late_goal_handle)

    assert result.error_code == "SKILL_CANCEL_TIMEOUT"
    assert late_result.error_code == "SKILL_BUSY"
    assert downstream_calls == []


def test_gateway_result_future_exception_retains_root_and_revokes_internal_goal(monkeypatch):
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    primitive_handle = _ResultFutureRaisesGoalHandle()
    node, policy, lease = _make_retained_gateway_node(_Future(done=True, result=primitive_handle))
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [PrimitiveSpec("open_gripper", gripper_position=1.0)],
    )

    result = node._execute_skill(_NoCancelParentGoalHandle())

    assert result.error_code == "SKILL_CANCEL_TIMEOUT"
    assert lease.owner == ExecutionOwner.skill_command("task-1")
    assert policy._ledger.query("task-1").state == "active"
    assert node._pending_internal_primitive_goals == {}
    assert node._active_internal_primitive_goals == {}
    assert primitive_handle.cancel_count == 1


def test_late_rejected_internal_goal_converges_retained_root_admission(monkeypatch):
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    send_future = _Future(done=False)
    node, policy, lease = _make_retained_gateway_node(send_future)
    node._rpc_timeout = 0.0
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [PrimitiveSpec("open_gripper", gripper_position=1.0)],
    )

    result = node._execute_skill(_NoCancelParentGoalHandle())
    send_future.set_result(SimpleNamespace(accepted=False))

    assert result.error_code == "SKILL_CANCEL_TIMEOUT"
    assert lease.owner is None
    assert policy._ledger.query("task-1").state == "terminal"
    assert node._pending_internal_primitive_goals == {}
    assert node._active_internal_primitive_goals == {}


def test_late_accepted_terminal_goal_converges_retained_root_admission(monkeypatch):
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    send_future = _Future(done=False)
    node, policy, lease = _make_retained_gateway_node(send_future)
    node._rpc_timeout = 0.0
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [PrimitiveSpec("open_gripper", gripper_position=1.0)],
    )

    result = node._execute_skill(_NoCancelParentGoalHandle())
    send_future.set_result(_TerminalOnCancelGoalHandle())

    assert result.error_code == "SKILL_CANCEL_TIMEOUT"
    assert lease.owner is None
    assert policy._ledger.query("task-1").state == "terminal"
    assert node._pending_internal_primitive_goals == {}
    assert node._active_internal_primitive_goals == {}


def test_late_accepted_goal_with_unknown_terminal_state_keeps_root_busy(monkeypatch):
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    send_future = _Future(done=False)
    node, policy, lease = _make_retained_gateway_node(send_future)
    node._rpc_timeout = 0.0
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [PrimitiveSpec("open_gripper", gripper_position=1.0)],
    )

    result = node._execute_skill(_NoCancelParentGoalHandle())
    send_future.set_result(_ChildGoalHandle(complete_result_on_cancel=False))

    assert result.error_code == "SKILL_CANCEL_TIMEOUT"
    assert lease.owner == ExecutionOwner.skill_command("task-1")
    assert policy._ledger.query("task-1").state == "active"


def test_post_dispatch_timeout_unknown_terminal_blocks_replacement_retry_and_later_workflow(monkeypatch):
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    send_future = _Future(done=False)
    node, policy, lease = _make_retained_gateway_node(send_future)
    node._rpc_timeout = 0.0
    sent_goals = []
    node._primitive_client.send_goal_async = lambda goal, **_kwargs: sent_goals.append(goal) or send_future
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [
            PrimitiveSpec("open_gripper", gripper_position=1.0),
            PrimitiveSpec("close_gripper", gripper_position=0.0),
        ],
    )
    authorization_before = node._runtime_snapshot().motion_authorized

    timed_out = node._execute_skill(_NoCancelParentGoalHandle())
    unknown_child = _ChildGoalHandle(complete_result_on_cancel=False)
    send_future.set_result(unknown_child)
    replacement_parent = _NoCancelParentGoalHandle()
    replacement_parent.request.task_id = "replacement-task"
    replacement = node._execute_skill(replacement_parent)
    workflow_error, workflow_token = policy.admit_workflow(
        ExecutionOwner.workflow("later-workflow"),
        node._runtime_snapshot(),
        timeout_sec=1.0,
    )

    assert timed_out.error_code == "SKILL_CANCEL_TIMEOUT"
    assert replacement.error_code == "SKILL_BUSY"
    assert len(sent_goals) == 1
    assert unknown_child.cancel_count == 1
    assert workflow_error == "SKILL_BUSY"
    assert workflow_token is None
    assert lease.owner == ExecutionOwner.skill_command("task-1")
    assert policy._ledger.query("task-1").state == "active"
    assert policy._ledger.query("replacement-task").state == ""
    assert policy._ledger.query("later-workflow").state == ""
    assert node._runtime_snapshot().motion_authorized == authorization_before


def test_gateway_parent_cancel_drains_pending_internal_goal_before_terminalizing(monkeypatch):
    events = []
    child_goal_handle = _ChildGoalHandle(events)
    send_goal_future = _Future(done=False)
    node, policy, lease = _make_retained_gateway_node(send_goal_future)
    finalized_error_codes = []
    original_finalize = policy.finalize
    policy.finalize = lambda admission, **kwargs: (
        finalized_error_codes.append(kwargs["error_code"]) or original_finalize(admission, **kwargs)
    )
    audit_events = []
    node._audit = lambda event, **_fields: audit_events.append(event)
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [PrimitiveSpec("open_gripper", gripper_position=1.0)],
    )
    parent_goal_handle = _ManualCancelParentGoalHandle(events)

    def advance_pending_goal(_seconds) -> None:
        if not parent_goal_handle.cancel_requested:
            parent_goal_handle.cancel_requested = True
        elif not send_goal_future.done():
            send_goal_future.set_result(child_goal_handle)
        else:
            child_goal_handle.complete_cancel_cleanup()

    monkeypatch.setattr(skill_executor_node.time, "sleep", advance_pending_goal)

    result = node._execute_skill(parent_goal_handle)

    assert result.error_code == "SKILL_CANCELLED"
    assert child_goal_handle.cancel_count == 1
    assert child_goal_handle.result_future.done()
    assert parent_goal_handle.canceled_count == 1
    assert parent_goal_handle.abort_count == 0
    assert events.index("child_terminal") < events.index("parent_cancelled")
    assert finalized_error_codes == ["SKILL_CANCELLED"]
    assert policy._ledger.query("task-1").state == "terminal"
    assert lease.owner is None
    assert node._pending_internal_primitive_goals == {}
    assert node._active_internal_primitive_goals == {}
    assert audit_events.count("cancel_propagated") == 1


def test_gateway_parent_cancel_retains_late_accepted_goal_without_terminal(monkeypatch):
    child_goal_handle = _ChildGoalHandle(complete_result_on_cancel=False)
    send_goal_future = _Future(done=False)
    node, policy, lease = _make_retained_gateway_node(send_goal_future)
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [PrimitiveSpec("open_gripper", gripper_position=1.0)],
    )
    parent_goal_handle = _ManualCancelParentGoalHandle()

    def advance_pending_goal(_seconds) -> None:
        if not parent_goal_handle.cancel_requested:
            parent_goal_handle.cancel_requested = True
        elif not send_goal_future.done():
            send_goal_future.set_result(child_goal_handle)

    monkeypatch.setattr(skill_executor_node.time, "sleep", advance_pending_goal)

    result = node._execute_skill(parent_goal_handle)

    assert result.error_code == "SKILL_CANCEL_TIMEOUT"
    assert child_goal_handle.cancel_count == 1
    assert parent_goal_handle.abort_count == 1
    assert parent_goal_handle.canceled_count == 0
    assert lease.owner == ExecutionOwner.skill_command("task-1")
    assert policy._ledger.query("task-1").state == "active"
    assert len(node._pending_internal_primitive_goals) == 1
    assert node._active_internal_primitive_goals == {}


def _admit_internal_primitive(node, policy):
    owner = ExecutionOwner.skill_command("task-1")
    admission = policy.admit(
        GatewayRequest(task_id="task-1", skill_name="test_skill"),
        node._runtime_snapshot(),
        owner,
    )
    assert admission.admitted
    node._active_skill_admission = admission
    node._active_skill_owner = owner
    node._active_audit_context = {"task_id": "task-1"}
    goal_id = skill_executor_node.UUID(uuid=[1] * 16)
    binding = new_binding(task_id="task-1")
    binding.expected_registry_epoch = "epoch-1"
    binding.expected_registry_generation = 1
    binding.expected_registry_digest = "digest-1"
    binding.task_budget.schema_version = 1
    binding.task_budget.started_at.sec = 1
    binding.task_budget.deadline.sec = 2_000_000_000
    binding.dispatch_nonce = "nonce-1"
    goal_key = node._register_internal_primitive_goal(goal_id, admission, binding)
    return admission, goal_id, goal_key, binding


def test_clean_internal_primitive_terminal_converges_later_retained_root():
    node, policy, lease = _make_retained_gateway_node(_Future(done=False))
    admission, goal_id, goal_key, binding = _admit_internal_primitive(node, policy)
    node._execute_primitive_unchecked = lambda goal_handle: (
        goal_handle.succeed()
        or SimpleNamespace(
            success=True,
            error_code="",
            message="",
            actual_registry_epoch="epoch-1",
            actual_registry_generation=1,
            actual_registry_digest="digest-1",
        )
    )
    child_goal_handle = _PrimitiveActionGoalHandle(goal_id, binding=binding)

    result = node._execute_primitive(child_goal_handle)

    assert result.success is True
    assert child_goal_handle.succeeded_count == 1
    cleanup = node._retained_admission_cleanup[id(admission)]
    assert cleanup.pending_goal_keys == {goal_key}
    assert cleanup.confirmed_goal_keys == {goal_key}
    assert node._pending_internal_primitive_goals == {}
    assert node._active_internal_primitive_goals == {}
    assert lease.owner == ExecutionOwner.skill_command("task-1")

    node._retain_admission_cleanup(admission, {"task_id": "task-1"}, 0.1, 1)

    assert policy._ledger.query("task-1").state == "terminal"
    assert lease.owner is None
    assert node._retained_admission_cleanup == {}
    assert node._pending_internal_primitive_goals == {}
    assert node._active_internal_primitive_goals == {}
    assert goal_key not in node._pending_internal_primitive_goals


def test_cleanup_unknown_internal_primitive_does_not_confirm_retained_root():
    node, policy, lease = _make_retained_gateway_node(_Future(done=False))
    admission, goal_id, _goal_key, binding = _admit_internal_primitive(node, policy)
    node._execute_primitive_unchecked = lambda goal_handle: (
        goal_handle.abort()
        or SimpleNamespace(
            success=False,
            error_code="CANCEL_CLEANUP_TIMEOUT",
            message="cleanup unknown",
            actual_registry_epoch="epoch-1",
            actual_registry_generation=1,
            actual_registry_digest="digest-1",
        )
    )

    result = node._execute_primitive(_PrimitiveActionGoalHandle(goal_id, binding=binding))

    assert result.error_code == "CANCEL_CLEANUP_TIMEOUT"
    cleanup = node._retained_admission_cleanup[id(admission)]
    assert cleanup.pending_goal_keys
    assert cleanup.confirmed_goal_keys == set()

    node._retain_admission_cleanup(admission, {"task_id": "task-1"}, 0.1, 1)

    assert policy._ledger.query("task-1").state == "active"
    assert lease.owner == ExecutionOwner.skill_command("task-1")
    assert node._retained_admission_cleanup[id(admission)].confirmed_goal_keys == set()


def test_only_current_internal_goal_cleanup_can_converge_retained_root():
    node, policy, lease = _make_retained_gateway_node(_Future(done=False))
    admission, first_goal_id, first_goal_key, first_binding = _admit_internal_primitive(node, policy)
    node._execute_primitive_unchecked = lambda goal_handle: (
        goal_handle.succeed()
        or SimpleNamespace(
            success=True,
            error_code="",
            message="",
            actual_registry_epoch="epoch-1",
            actual_registry_generation=1,
            actual_registry_digest="digest-1",
        )
    )

    first_result = node._execute_primitive(_PrimitiveActionGoalHandle(first_goal_id, binding=first_binding))

    assert first_result.success is True
    second_goal_id = skill_executor_node.UUID(uuid=[2] * 16)
    second_binding = new_binding(task_id="task-1")
    second_binding.expected_registry_epoch = "epoch-1"
    second_binding.expected_registry_generation = 1
    second_binding.expected_registry_digest = "digest-1"
    second_binding.task_budget.schema_version = 1
    second_binding.task_budget.started_at.sec = 1
    second_binding.task_budget.deadline.sec = 2_000_000_000
    second_binding.dispatch_nonce = "nonce-2"
    second_goal_key = node._register_internal_primitive_goal(second_goal_id, admission, second_binding)
    node._execute_primitive_unchecked = lambda goal_handle: (
        goal_handle.abort()
        or SimpleNamespace(
            success=False,
            error_code="CANCEL_CLEANUP_TIMEOUT",
            message="cleanup unknown",
            actual_registry_epoch="epoch-1",
            actual_registry_generation=1,
            actual_registry_digest="digest-1",
        )
    )

    second_result = node._execute_primitive(_PrimitiveActionGoalHandle(second_goal_id, binding=second_binding))
    node._retain_admission_cleanup(admission, {"task_id": "task-1"}, 0.1, 2)

    assert second_result.error_code == "CANCEL_CLEANUP_TIMEOUT"
    assert policy._ledger.query("task-1").state == "active"
    assert lease.owner == ExecutionOwner.skill_command("task-1")

    node._confirm_late_internal_cleanup(admission, first_goal_key)

    assert policy._ledger.query("task-1").state == "active"
    assert lease.owner == ExecutionOwner.skill_command("task-1")

    node._confirm_late_internal_cleanup(admission, second_goal_key)

    assert policy._ledger.query("task-1").state == "terminal"
    assert lease.owner is None
    assert node._retained_admission_cleanup == {}
    assert node._pending_internal_primitive_goals == {}
    assert node._active_internal_primitive_goals == {}


def test_external_primitive_commits_child_terminal_intent_after_successful_release():
    node = object.__new__(SkillExecutorNode)
    node._gateway_policy = _ExternalPolicyStub(True)
    node._runtime_snapshot = lambda: RuntimeSnapshot(
        motion_authorized=True,
        active_control_mode="gateway_mode",
        required_control_mode="gateway_mode",
    )
    node._state_lock = RLock()
    node._pending_internal_primitive_goals = {}
    node._active_internal_primitive_goals = {}
    node._execute_primitive_unchecked = lambda goal_handle: (
        goal_handle.succeed() or SimpleNamespace(success=False, error_code="CHILD_ERROR", message="child error")
    )
    goal_handle = _PrimitiveGoalHandle()

    result = node._execute_primitive(goal_handle)

    assert result.error_code == "CHILD_ERROR"
    assert goal_handle.succeeded_count == 1
    assert goal_handle.abort_count == 0


def test_external_primitive_late_terminal_releases_original_token_once():
    downstream_terminal = _Future(done=False)
    node = object.__new__(SkillExecutorNode)
    node._gateway_policy = _CountingReleasePolicy()
    node._runtime_snapshot = lambda: RuntimeSnapshot(
        motion_authorized=True,
        active_control_mode="gateway_mode",
        required_control_mode="gateway_mode",
    )
    node._state_lock = RLock()
    node._pending_internal_primitive_goals = {}
    node._active_internal_primitive_goals = {}

    def unchecked(goal_handle):
        goal_handle.late_cleanup_confirmation.watch_result_future(downstream_terminal)
        goal_handle.abort()
        return SimpleNamespace(success=False, error_code="CANCEL_CLEANUP_TIMEOUT", message="cleanup unknown")

    node._execute_primitive_unchecked = unchecked

    result = node._execute_primitive(_PrimitiveGoalHandle())

    assert result.error_code == "CANCEL_CLEANUP_TIMEOUT"
    assert node._gateway_policy.release_count == 0

    downstream_terminal.set_result(SimpleNamespace(result=SimpleNamespace(error_code="")))
    downstream_terminal.set_result(SimpleNamespace(result=SimpleNamespace(error_code="")))

    assert node._gateway_policy.release_count == 1


def test_gateway_finalization_cannot_clear_a_newer_active_admission():
    node, policy, _lease = _make_retained_gateway_node(_Future(done=False))
    released = threading.Event()
    allow_first_wrapper_to_return = threading.Event()
    original_finalize = policy.finalize

    def finalize(admission, **kwargs):
        terminal = original_finalize(admission, **kwargs)
        if admission.prepared_request.identity.task_id == "task-a":
            released.set()
            assert allow_first_wrapper_to_return.wait(timeout=1.0)
        return terminal

    policy.finalize = finalize

    def child(goal_handle, **_kwargs):
        if goal_handle.request.task_id == "task-a":
            goal_handle.succeed()
            return SimpleNamespace(success=True, error_code="", message="", executed_primitives=[])
        goal_handle.abort()
        return SimpleNamespace(
            success=False,
            error_code="SKILL_CANCEL_TIMEOUT",
            message="cleanup unknown",
            executed_primitives=[],
        )

    node._execute_skill_child = child
    first_parent = _NoCancelParentGoalHandle()
    first_parent.request.task_id = "task-a"
    second_parent = _NoCancelParentGoalHandle()
    second_parent.request.task_id = "task-b"
    first_result = []
    second_result = []
    first_thread = threading.Thread(target=lambda: first_result.append(node._execute_skill(first_parent)))
    second_thread = threading.Thread(target=lambda: second_result.append(node._execute_skill(second_parent)))

    first_thread.start()
    assert released.wait(timeout=1.0)
    second_thread.start()
    allow_first_wrapper_to_return.set()
    first_thread.join(timeout=1.0)
    second_thread.join(timeout=1.0)

    assert first_result[0].success is True
    assert second_result[0].error_code == "SKILL_CANCEL_TIMEOUT"
    assert node._active_skill_admission.prepared_request.identity.task_id == "task-b"


def test_exec_arm_joint_trajectory_waits_for_downstream_terminal_after_cancel(monkeypatch):
    events = []
    child_goal_handle = _ChildGoalHandle(events)
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    monkeypatch.setattr(skill_executor_node.time, "sleep", lambda _seconds: child_goal_handle.complete_cancel_cleanup())
    node = object.__new__(SkillExecutorNode)
    node._rpc_timeout = 0.01
    node._debug = False
    node._arm_trajectory_action_name = "/arm/follow_joint_trajectory"
    node._arm_trajectory_client = SimpleNamespace(
        wait_for_server=lambda **_kwargs: True,
        send_goal_async=lambda _goal: _Future(done=True, result=child_goal_handle),
    )

    ok, message = node._exec_arm_joint_trajectory(_PrimitiveGoalHandle(), ["joint_1"], [0.1], "task-1", 1.0, 0.4)

    assert ok is False
    assert message.startswith("cancelled")
    assert child_goal_handle.cancel_count == 1
    assert child_goal_handle.result_future.done()
    assert events == ["child_cancel", "child_terminal"]


def test_exec_arm_joint_trajectory_reports_cancel_cleanup_timeout(monkeypatch):
    child_goal_handle = _ChildGoalHandle(complete_result_on_cancel=False)
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    node = object.__new__(SkillExecutorNode)
    node._rpc_timeout = 0.0
    node._debug = False
    node._arm_trajectory_action_name = "/arm/follow_joint_trajectory"
    node._arm_trajectory_client = SimpleNamespace(
        wait_for_server=lambda **_kwargs: True,
        send_goal_async=lambda _goal: _Future(done=True, result=child_goal_handle),
    )

    ok, message = node._exec_arm_joint_trajectory(_PrimitiveGoalHandle(), ["joint_1"], [0.1], "task-1", 1.0, 0.4)

    assert ok is False
    assert message.startswith("cancel cleanup timed out")
    assert child_goal_handle.cancel_count == 1
    assert child_goal_handle.result_future.done() is False


def test_exec_arm_joint_trajectory_cancels_when_result_future_creation_fails():
    child_goal_handle = _ResultFutureRaisesGoalHandle()
    node = object.__new__(SkillExecutorNode)
    node._rpc_timeout = 0.0
    node._debug = False
    node._arm_trajectory_action_name = "/arm/follow_joint_trajectory"
    node._arm_trajectory_client = SimpleNamespace(
        wait_for_server=lambda **_kwargs: True,
        send_goal_async=lambda _goal: _Future(done=True, result=child_goal_handle),
    )

    ok, message = node._exec_arm_joint_trajectory(_PrimitiveGoalHandle(), ["joint_1"], [0.1], "task-1", 1.0, 0.4)

    assert ok is False
    assert message.startswith("cancel cleanup timed out")
    assert child_goal_handle.cancel_count == 1


def test_exec_arm_via_task_dispatch_cancels_when_result_future_creation_fails():
    child_goal_handle = _ResultFutureRaisesGoalHandle()
    node = object.__new__(SkillExecutorNode)
    node._rpc_timeout = 0.0
    node._debug = False
    node._task_executor_action_name = "/task_dispatch/execute_task_plan"
    node._task_executor_client = SimpleNamespace(
        wait_for_server=lambda **_kwargs: True,
        send_goal_async=lambda _goal: _Future(done=True, result=child_goal_handle),
    )

    ok, message = node._exec_arm_via_task_dispatch(
        _PrimitiveGoalHandle(),
        "move_to_named_pose",
        skill_executor_node.Pose(),
        "task-1",
        1.0,
    )

    assert ok is False
    assert message.startswith("cancel cleanup timed out")
    assert child_goal_handle.cancel_count == 1


def test_execute_primitive_aborts_when_downstream_cancel_cleanup_times_out():
    node = object.__new__(SkillExecutorNode)
    node._validate_primitive = lambda *_args, **_kwargs: (True, "")
    node._exec_arm_joint_trajectory = lambda *_args, **_kwargs: (
        False,
        "cancel cleanup timed out during arm joint trajectory execution",
    )
    goal_handle = _PrimitiveGoalHandle()

    result = node._execute_primitive(goal_handle)

    assert result.success is False
    assert result.error_code == "CANCEL_CLEANUP_TIMEOUT"
    assert goal_handle.abort_count == 1
    assert goal_handle.canceled_count == 0


def test_move_to_pose_validation_receives_goal_target_pose():
    node = object.__new__(SkillExecutorNode)
    node._debug = False
    captured = {}

    def validate_primitive(*_args, **kwargs):
        captured.update(kwargs)
        return False, "stop after validation"

    node._validate_primitive = validate_primitive
    goal_handle = _PrimitiveGoalHandle()
    goal_handle.request.primitive_name = "move_to_pose"
    goal_handle.request.target_pose = skill_executor_node.Pose()
    goal_handle.request.target_pose.position.x = 0.12
    goal_handle.request.target_pose.position.y = -0.20
    goal_handle.request.target_pose.position.z = 0.05
    goal_handle.request.target_pose.orientation.w = 1.0

    result = node._execute_primitive_unchecked(goal_handle)

    assert captured["target_pose"] is goal_handle.request.target_pose
    assert captured["target_pose"].position.z == pytest.approx(0.05)
    assert result.success is False
    assert result.error_code == "SAFETY_REJECTED"


class _NavigationPrimitiveGoalHandle:
    def __init__(self, *, cancel_requested: bool = False) -> None:
        self.cancel_requested = cancel_requested
        self.feedback = []

    @property
    def is_cancel_requested(self) -> bool:
        return self.cancel_requested

    def publish_feedback(self, feedback) -> None:
        self.feedback.append(feedback)


def _navigation_executor_node(client) -> SkillExecutorNode:
    node = object.__new__(SkillExecutorNode)
    node._context_schema_version = 2
    node._navigation_action_name = "/navigation/execute"
    node._navigation_client = client
    node._rpc_timeout = 0.01
    node._debug = False
    node._audit_cancel_propagated = lambda: None
    node.get_logger = lambda: SimpleNamespace(warning=lambda _message: None, info=lambda _message: None)
    return node


def _navigation_result(*, success: bool, error_code: int, message: str = ""):
    return SimpleNamespace(result=SimpleNamespace(success=success, error_code=error_code, message=message))


def test_execute_skill_copies_resolved_navigation_goal_and_public_parameters(monkeypatch):
    navigation_goal = ExecuteNavigation.Goal()
    navigation_goal.command_type = ExecuteNavigation.Goal.ABSOLUTE_POSE
    navigation_goal.target_pose.header.frame_id = "map"
    navigation_goal.target_pose.pose.position.x = -1.25
    navigation_goal.target_pose.pose.position.y = 2.5
    navigation_goal.target_pose.pose.orientation.w = 1.0
    captured = {}
    child_goal_handle = _ChildGoalHandle()
    child_goal_handle.result_future.set_result(
        SimpleNamespace(result=SimpleNamespace(success=True, error_code="", message=""))
    )
    node = _make_skill_node(_Future(done=True, result=child_goal_handle))

    def resolve(*_args, **kwargs):
        captured["resolver_kwargs"] = kwargs
        return [PrimitiveSpec("nav_abs_coordinate", navigation_goal=navigation_goal)]

    def send_goal(goal, **_kwargs):
        captured["primitive_goal"] = goal
        primitive_feedback = PrimitiveCommand.Feedback()
        primitive_feedback.state = "running"
        primitive_feedback.detail = "distance_remaining=1.000 estimated_time_remaining_sec=2.000 number_of_recoveries=0"
        _kwargs["feedback_callback"](SimpleNamespace(feedback=primitive_feedback))
        return _Future(done=True, result=child_goal_handle)

    monkeypatch.setattr(rclpy, "ok", lambda: True)
    monkeypatch.setattr(skill_executor_node, "resolve_skill_primitives", resolve)
    node._primitive_client = SimpleNamespace(wait_for_server=lambda **_kwargs: True, send_goal_async=send_goal)
    parent = _NoCancelParentGoalHandle()
    parent.request.x = -1.25
    parent.request.y = 2.5
    parent.request.yaw = 0.0
    parent.request.has_x = True
    parent.request.has_y = True
    parent.request.has_yaw = True

    result = node._execute_skill(parent)

    assert result.success is True
    assert captured["resolver_kwargs"] == {
        "current_joint_positions": {},
        "arm_joint_names": [],
        "direction": "",
        "distance": 0.0,
        "degree": 0.0,
        "x": -1.25,
        "y": 2.5,
        "yaw": 0.0,
    }
    primitive_goal = captured["primitive_goal"]
    assert primitive_goal.navigation_command_type == ExecuteNavigation.Goal.ABSOLUTE_POSE
    assert primitive_goal.navigation_target_pose.header.frame_id == "map"
    assert primitive_goal.navigation_target_pose.pose.position.x == pytest.approx(-1.25)
    assert primitive_goal.navigation_target_pose.pose.position.y == pytest.approx(2.5)
    assert primitive_goal.navigation_value == pytest.approx(0.0)
    assert [(feedback.state, feedback.detail) for feedback in parent.feedback] == [
        ("executing", "step 1 of 1"),
        (
            "running",
            "distance_remaining=1.000 estimated_time_remaining_sec=2.000 number_of_recoveries=0",
        ),
    ]


def test_validate_primitive_copies_navigation_wire_to_safety_request():
    captured = {}
    response = SimpleNamespace(
        allowed=True,
        reason="",
        actual_registry_epoch="",
        actual_registry_generation=0,
        actual_registry_digest="",
    )
    node = object.__new__(SkillExecutorNode)
    node._rpc_timeout = 0.01
    node._validate_primitive_client = SimpleNamespace(
        wait_for_service=lambda **_kwargs: True,
        call_async=lambda request: captured.setdefault("request", request) or _Future(done=True, result=response),
    )
    node._validate_primitive_client.call_async = lambda request: (
        captured.update(request=request) or _Future(done=True, result=response)
    )
    target = skill_executor_node.PoseStamped()
    target.header.frame_id = "map"
    target.pose.position.x = -4.0
    target.pose.orientation.w = 1.0

    allowed, reason = node._validate_primitive(
        "nav_abs_coordinate",
        "",
        0.0,
        navigation_command_type=ExecuteNavigation.Goal.ABSOLUTE_POSE,
        navigation_target_pose=target,
        navigation_value=0.0,
    )

    assert allowed is True
    assert reason == ""
    request = captured["request"]
    assert request.schema_version == 2
    assert request.navigation_command_type == ExecuteNavigation.Goal.ABSOLUTE_POSE
    assert request.navigation_target_pose.header.frame_id == "map"
    assert request.navigation_target_pose.pose.position.x == pytest.approx(-4.0)
    assert request.navigation_value == pytest.approx(0.0)


def test_validate_primitive_copies_manipulation_contract_version_to_safety_request():
    captured = {}
    response = SimpleNamespace(
        allowed=True,
        reason="",
        actual_registry_epoch="",
        actual_registry_generation=0,
        actual_registry_digest="",
    )
    node = object.__new__(SkillExecutorNode)
    node._rpc_timeout = 0.01
    node._validate_primitive_client = SimpleNamespace(
        wait_for_service=lambda **_kwargs: True,
        call_async=lambda request: captured.update(request=request) or _Future(done=True, result=response),
    )

    allowed, reason = node._validate_primitive("open_gripper", "", 1.0)

    assert allowed is True
    assert reason == ""
    assert captured["request"].schema_version == 1


def test_exec_navigation_waits_for_real_result_after_goal_acceptance(monkeypatch):
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    sent = threading.Event()
    child = _ChildGoalHandle()
    client = SimpleNamespace(
        wait_for_server=lambda **_kwargs: True,
        send_goal_async=lambda _goal, **_kwargs: sent.set() or _Future(done=True, result=child),
    )
    node = _navigation_executor_node(client)
    result = []
    thread = threading.Thread(
        target=lambda: result.append(
            node._exec_navigation(_NavigationPrimitiveGoalHandle(), ExecuteNavigation.Goal(), timeout_sec=1.0)
        )
    )

    thread.start()
    assert sent.wait(timeout=1.0)
    thread.join(timeout=0.02)
    assert thread.is_alive()
    child.result_future.set_result(_navigation_result(success=True, error_code=ExecuteNavigation.Result.NONE))
    thread.join(timeout=1.0)

    assert result == [(True, "", "")]


@pytest.mark.parametrize(
    ("error_code", "expected"),
    [
        (ExecuteNavigation.Result.INVALID_GOAL, "NAVIGATION_INVALID_GOAL"),
        (ExecuteNavigation.Result.BUSY, "NAVIGATION_BUSY"),
        (ExecuteNavigation.Result.TF_UNAVAILABLE, "NAVIGATION_TF_UNAVAILABLE"),
        (ExecuteNavigation.Result.NAV2_UNAVAILABLE, "NAVIGATION_NAV2_UNAVAILABLE"),
        (ExecuteNavigation.Result.GOAL_REJECTED, "NAVIGATION_GOAL_REJECTED"),
        (ExecuteNavigation.Result.NAVIGATION_ABORTED, "NAVIGATION_ABORTED"),
        (ExecuteNavigation.Result.NAVIGATION_CANCELED, "NAVIGATION_CANCELED"),
        (ExecuteNavigation.Result.INTERNAL_ERROR, "NAVIGATION_INTERNAL_ERROR"),
    ],
)
def test_exec_navigation_preserves_failure_identity(monkeypatch, error_code, expected):
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    child = _ChildGoalHandle()
    child.result_future.set_result(_navigation_result(success=False, error_code=error_code, message="downstream"))
    node = _navigation_executor_node(
        SimpleNamespace(
            wait_for_server=lambda **_kwargs: True,
            send_goal_async=lambda _goal, **_kwargs: _Future(done=True, result=child),
        )
    )

    result = node._exec_navigation(_NavigationPrimitiveGoalHandle(), ExecuteNavigation.Goal(), timeout_sec=1.0)

    assert result == (False, expected, "downstream")


def test_exec_navigation_maps_stop_timeout_to_cleanup_unknown(monkeypatch):
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    child = _ChildGoalHandle()
    child.result_future.set_result(
        _navigation_result(success=False, error_code=ExecuteNavigation.Result.STOP_TIMEOUT, message="downstream")
    )
    node = _navigation_executor_node(
        SimpleNamespace(
            wait_for_server=lambda **_kwargs: True,
            send_goal_async=lambda _goal, **_kwargs: _Future(done=True, result=child),
        )
    )

    result = node._exec_navigation(_NavigationPrimitiveGoalHandle(), ExecuteNavigation.Goal(), timeout_sec=1.0)

    assert result == (False, "CANCEL_CLEANUP_TIMEOUT", "cancel cleanup timed out: downstream")


def test_exec_navigation_forwards_progress_feedback(monkeypatch):
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    child = _ChildGoalHandle()
    child.result_future.set_result(_navigation_result(success=True, error_code=ExecuteNavigation.Result.NONE))
    navigation_feedback = ExecuteNavigation.Feedback()
    navigation_feedback.state = "running"
    navigation_feedback.distance_remaining = 1.25
    navigation_feedback.estimated_time_remaining.sec = 3
    navigation_feedback.estimated_time_remaining.nanosec = 500_000_000
    navigation_feedback.number_of_recoveries = 2

    def send_goal(_goal, *, feedback_callback):
        feedback_callback(SimpleNamespace(feedback=navigation_feedback))
        return _Future(done=True, result=child)

    node = _navigation_executor_node(SimpleNamespace(wait_for_server=lambda **_kwargs: True, send_goal_async=send_goal))
    goal_handle = _NavigationPrimitiveGoalHandle()

    result = node._exec_navigation(goal_handle, ExecuteNavigation.Goal(), timeout_sec=1.0)

    assert result == (True, "", "")
    assert [(feedback.state, feedback.detail) for feedback in goal_handle.feedback] == [
        (
            "running",
            "distance_remaining=1.250 estimated_time_remaining_sec=3.500 number_of_recoveries=2",
        )
    ]


def test_exec_navigation_cancel_waits_for_downstream_terminal(monkeypatch):
    events = []
    child = _ChildGoalHandle(events)
    goal_handle = _NavigationPrimitiveGoalHandle()
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    monkeypatch.setattr(skill_executor_node.time, "sleep", lambda _seconds: child.complete_cancel_cleanup())

    def send_goal(_goal, **_kwargs):
        goal_handle.cancel_requested = True
        return _Future(done=True, result=child)

    node = _navigation_executor_node(
        SimpleNamespace(
            wait_for_server=lambda **_kwargs: True,
            send_goal_async=send_goal,
        )
    )

    result = node._exec_navigation(goal_handle, ExecuteNavigation.Goal(), timeout_sec=1.0)

    assert result == (False, "NAVIGATION_CANCELED", "navigation canceled")
    assert child.cancel_count == 1
    assert child.result_future.done()
    assert events == ["child_cancel", "child_terminal"]


def test_exec_navigation_reports_unavailable_server():
    node = _navigation_executor_node(SimpleNamespace(wait_for_server=lambda **_kwargs: False))

    result = node._exec_navigation(_NavigationPrimitiveGoalHandle(), ExecuteNavigation.Goal(), timeout_sec=1.0)

    assert result == (
        False,
        "NAVIGATION_SERVER_UNAVAILABLE",
        "navigation action server not available: /navigation/execute",
    )


def test_exec_navigation_reports_action_goal_rejection(monkeypatch):
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    node = _navigation_executor_node(
        SimpleNamespace(
            wait_for_server=lambda **_kwargs: True,
            send_goal_async=lambda _goal, **_kwargs: _Future(
                done=True,
                result=SimpleNamespace(accepted=False),
            ),
        )
    )

    result = node._exec_navigation(_NavigationPrimitiveGoalHandle(), ExecuteNavigation.Goal(), timeout_sec=1.0)

    assert result == (False, "NAVIGATION_GOAL_REJECTED", "navigation goal rejected")


def test_exec_navigation_cancel_after_readiness_prevents_goal_dispatch():
    goal_handle = _NavigationPrimitiveGoalHandle()
    sent = []

    def wait_for_server(**_kwargs):
        goal_handle.cancel_requested = True
        return True

    node = _navigation_executor_node(
        SimpleNamespace(
            wait_for_server=wait_for_server,
            send_goal_async=lambda *_args, **_kwargs: sent.append(True),
        )
    )

    result = node._exec_navigation(goal_handle, ExecuteNavigation.Goal(), timeout_sec=1.0)

    assert result == (False, "NAVIGATION_CANCELED", "navigation canceled before goal dispatch")
    assert sent == []


def test_exec_navigation_timeout_budget_includes_server_readiness(monkeypatch):
    clock = {"now": 0.0}
    sent = []
    child = _ChildGoalHandle()
    child.result_future.set_result(_navigation_result(success=True, error_code=ExecuteNavigation.Result.NONE))

    def wait_for_server(**_kwargs):
        clock["now"] += 2.0
        return True

    node = _navigation_executor_node(
        SimpleNamespace(
            wait_for_server=wait_for_server,
            send_goal_async=lambda *_args, **_kwargs: sent.append(True) or _Future(done=True, result=child),
        )
    )
    node._rpc_timeout = 5.0
    monkeypatch.setattr(skill_executor_node.time, "monotonic", lambda: clock["now"])

    result = node._exec_navigation(
        _NavigationPrimitiveGoalHandle(),
        ExecuteNavigation.Goal(),
        timeout_sec=1.0,
    )

    assert result == (False, "NAVIGATION_TIMEOUT", "timeout waiting for navigation action server")
    assert sent == []


def test_exec_navigation_stop_timeout_retains_cleanup_ownership(monkeypatch):
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    child = _ChildGoalHandle()
    child.result_future.set_result(
        _navigation_result(
            success=False,
            error_code=ExecuteNavigation.Result.STOP_TIMEOUT,
            message="velocity did not stabilize",
        )
    )
    node = _navigation_executor_node(
        SimpleNamespace(
            wait_for_server=lambda **_kwargs: True,
            send_goal_async=lambda *_args, **_kwargs: _Future(done=True, result=child),
        )
    )

    result = node._exec_navigation(
        _NavigationPrimitiveGoalHandle(),
        ExecuteNavigation.Goal(),
        timeout_sec=1.0,
    )

    assert result == (
        False,
        skill_executor_node.PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
        "cancel cleanup timed out: velocity did not stabilize",
    )


def _primitive_digest_goal(*, schema_version: int, primitive_name: str):
    generated_goal = PrimitiveCommand.Goal()
    fields = {name: getattr(generated_goal, name) for name in generated_goal.get_fields_and_field_types()}
    fields.update(schema_version=schema_version, primitive_name=primitive_name)
    return SimpleNamespace(**fields)


def test_navigation_primitive_payload_digest_covers_resolved_goal():
    first = _primitive_digest_goal(schema_version=2, primitive_name="nav_straight")
    first.navigation_command_type = ExecuteNavigation.Goal.FORWARD
    first.navigation_value = 1.0
    second = _primitive_digest_goal(schema_version=2, primitive_name="nav_straight")
    second.navigation_command_type = ExecuteNavigation.Goal.FORWARD
    second.navigation_value = 2.0

    assert SkillExecutorNode._primitive_payload_digest(first) != SkillExecutorNode._primitive_payload_digest(second)


@pytest.mark.parametrize(
    ("primitive_name", "schema_version"),
    [("nav_straight", 1), ("open_gripper", 2)],
)
def test_primitive_payload_canonicalization_uses_submitted_version(monkeypatch, primitive_name, schema_version):
    goal = _primitive_digest_goal(schema_version=schema_version, primitive_name=primitive_name)
    canonical_preimages = []
    canonical_json = skill_executor_node.to_canonical_json

    def capture_preimage(payload):
        canonical_preimages.append(payload)
        return canonical_json(payload)

    monkeypatch.setattr(skill_executor_node, "to_canonical_json", capture_preimage)

    SkillExecutorNode._primitive_payload_digest(goal)

    assert canonical_preimages[0]["schema_version"] == schema_version


def test_primitive_payload_digest_distinguishes_submitted_versions():
    version_1 = _primitive_digest_goal(schema_version=1, primitive_name="nav_straight")
    version_2 = _primitive_digest_goal(schema_version=2, primitive_name="nav_straight")

    assert SkillExecutorNode._primitive_payload_digest(version_1) != SkillExecutorNode._primitive_payload_digest(
        version_2
    )
