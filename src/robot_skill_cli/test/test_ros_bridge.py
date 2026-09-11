import os
import threading
import time
import uuid
from types import SimpleNamespace

import pytest

from embodied_common.skill_request import skill_goal_uuid
from robot_skill_cli.ros_bridge import BridgeError, RosBridge


def _assert_isolated_ros_domain() -> None:
    allocated = os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID", "")
    assert allocated.isdecimal() and 1 <= int(allocated) <= 232
    assert os.environ.get("ROS_DOMAIN_ID") == allocated
    assert os.environ.get("ROS_LOCALHOST_ONLY") == "1"


@pytest.fixture
def bridge_rig():
    _assert_isolated_ros_domain()
    import rclpy
    from rclpy.action import ActionServer, CancelResponse, GoalResponse
    from rclpy.context import Context
    from rclpy.executors import MultiThreadedExecutor

    from ibrobot_msgs.action import SkillCommand
    from ibrobot_msgs.msg import SkillCapabilityStatus
    from ibrobot_msgs.srv import (
        GetSkillGatewayStatus,
        GetVisualGameResult,
        ReloadSkillCatalog,
        StartVisualGame,
        ValidateSkill,
    )

    server_context = Context()
    rclpy.init(context=server_context)
    suffix = f"cli_{os.getpid()}_{uuid.uuid4().hex}"
    names = {
        "status": f"/{suffix}/gateway_status",
        "reload": f"/{suffix}/reload_catalog",
        "validate": f"/{suffix}/validate_skill",
        "action": f"/{suffix}/execute_skill",
        "start_game": f"/{suffix}/start_visual_game",
        "game_result": f"/{suffix}/get_visual_game_result",
    }
    requests = []
    status_control = {"delay_sec": 0.0, "queries": {}}
    action_control = {"reject": False, "goal_ids": [], "goals": [], "delay_sec": 0.0}
    server_node = rclpy.create_node(
        f"robot_skill_cli_test_server_{suffix}",
        context=server_context,
        use_global_arguments=False,
    )

    def get_status(request, response):
        if status_control["delay_sec"]:
            time.sleep(status_control["delay_sec"])
        response.schema_version = 1
        response.robot_name = "test_robot"
        response.motion_authorized = True
        response.active_control_mode = "moveit_planning"
        response.busy = True
        response.active_task_id = "active-task"
        response.default_skill_timeout_sec = 12.0
        response.task_budget_sec = 90.0
        response.rpc_timeout_sec = 2.0
        response.config_digest = "digest-1"
        response.capability_digest = "digest-1"
        response.request_state, response.request_error_code = status_control["queries"].get(
            (request.task_id, request.payload_hash),
            ("active" if request.task_id else "", "DUPLICATE_TASK_ID" if request.payload_hash else ""),
        )
        capability = SkillCapabilityStatus()
        capability.name = "move_relative_ee"
        capability.schema_version = 1
        capability.semantic_level = "skill"
        capability.planner_visible = True
        capability.ready = False
        capability.reason = "SKILL_BUSY"
        capability.required_control_mode = "moveit_planning"
        response.capabilities = [capability]
        return response

    def validate_skill(request, response):
        requests.append(request)
        response.allowed = True
        response.reason = "allowed"
        return response

    server_node.create_service(GetSkillGatewayStatus, names["status"], get_status)

    def reload_catalog(_request, response):
        response.success = True
        response.registry_epoch = "epoch-2"
        response.old_generation = 1
        response.generation = 2
        response.registry_digest = "registry-digest-2"
        response.capability_digest = "capability-digest-2"
        response.source_release_digest = "source-release-2"
        response.provenance_digest = "provenance-2"
        response.message = "reloaded"
        response.changed_skills = ["nod_yes"]
        return response

    server_node.create_service(ReloadSkillCatalog, names["reload"], reload_catalog)
    server_node.create_service(ValidateSkill, names["validate"], validate_skill)

    def start_game(request, response):
        response.accepted = request.game_name == "sorting_hat" and request.expected_config_digest == "game-digest"
        response.duplicate = False
        response.request_id = request.request_id if response.accepted else ""
        response.config_digest = "game-digest"
        response.error_code = "" if response.accepted else "GAME_NOT_ENABLED"
        response.message = "accepted" if response.accepted else "disabled"
        return response

    def get_game_result(request, response):
        response.found = request.request_id == "game-test-1"
        response.terminal = response.found
        response.success = response.found
        response.game_name = "sorting_hat" if response.found else ""
        response.scene_summary = "赫奇帕奇" if response.found else ""
        response.result_json = '{"scene_summary":"赫奇帕奇"}' if response.found else ""
        response.config_digest = "game-digest"
        response.error_code = "" if response.found else "GAME_REQUEST_NOT_FOUND"
        response.message = "completed" if response.found else "missing"
        return response

    server_node.create_service(StartVisualGame, names["start_game"], start_game)
    server_node.create_service(GetVisualGameResult, names["game_result"], get_game_result)

    def goal_callback(_goal_request):
        return GoalResponse.REJECT if action_control["reject"] else GoalResponse.ACCEPT

    def cancel_callback(_goal_handle):
        return CancelResponse.ACCEPT

    def execute_skill(goal_handle):
        action_control["goal_ids"].append(bytes(goal_handle.goal_id.uuid))
        action_control["goals"].append(goal_handle.request)
        feedback = SkillCommand.Feedback()
        feedback.state = "executing"
        feedback.detail = "step 1 of 1"
        goal_handle.publish_feedback(feedback)
        result = SkillCommand.Result()
        deadline = time.monotonic() + action_control["delay_sec"]
        while time.monotonic() < deadline:
            if goal_handle.is_cancel_requested:
                result.success = False
                result.error_code = "SKILL_CANCELLED"
                result.message = "cancelled"
                goal_handle.canceled()
                return result
            time.sleep(0.01)
        result.success = True
        result.error_code = ""
        result.message = "completed"
        result.executed_primitives = ["private_primitive"]
        goal_handle.succeed()
        return result

    action_server = ActionServer(
        server_node,
        SkillCommand,
        names["action"],
        execute_callback=execute_skill,
        goal_callback=goal_callback,
        cancel_callback=cancel_callback,
    )
    executor = MultiThreadedExecutor(num_threads=2, context=server_context)
    executor.add_node(server_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    bridge = RosBridge(
        status_service=names["status"],
        reload_service=names["reload"],
        validate_skill_service=names["validate"],
        skill_action=names["action"],
        start_visual_game_service=names["start_game"],
        get_visual_game_result_service=names["game_result"],
    )
    assert bridge.start() is True

    yield bridge, names, requests, status_control, action_control

    bridge.close()
    action_server.destroy()
    executor.remove_node(server_node)
    executor.shutdown()
    spin_thread.join(timeout=2.0)
    server_node.destroy_node()
    if server_context.ok():
        rclpy.shutdown(context=server_context)


def test_bridge_context_can_be_recreated_in_one_process() -> None:
    _assert_isolated_ros_domain()
    for _ in range(2):
        bridge = RosBridge(
            status_service="/unused/status",
            validate_skill_service="/unused/validate_skill",
            skill_action="/unused/execute_skill",
        )
        assert bridge.start() is True, bridge.start_error
        bridge.close()


def test_status_preserves_gateway_contract_fields(bridge_rig):
    bridge, _names, _requests, _status_control, _action_control = bridge_rig

    status = bridge.get_status(task_id="task-1", payload_hash="hash-1", timeout_sec=1.0)

    assert status == {
        "schema_version": 1,
        "robot_name": "test_robot",
        "motion_authorized": True,
        "active_control_mode": "moveit_planning",
        "busy": True,
        "active_task_id": "active-task",
        "default_skill_timeout_sec": pytest.approx(12.0),
        "task_budget_sec": pytest.approx(90.0),
        "rpc_timeout_sec": pytest.approx(2.0),
        "config_digest": "digest-1",
        "request_state": "active",
        "request_error_code": "DUPLICATE_TASK_ID",
        "capabilities": [
            {
                "name": "move_relative_ee",
                "semantic_level": "skill",
                "planner_visible": True,
                "ready": False,
                "reason": "SKILL_BUSY",
                "required_control_mode": "moveit_planning",
            }
        ],
        "capability_digest": "digest-1",
        "registry_epoch": "",
        "registry_generation": 0,
        "registry_digest": "",
        "primitive_contract_digest": "",
        "source_release_digest": "",
        "provenance_digest": "",
        "control_plane_ready": False,
        "control_plane_state": "",
        "control_plane_error_code": "",
    }


def test_reload_catalog_returns_generation_and_changed_skills(bridge_rig):
    bridge, _names, _requests, _status_control, _action_control = bridge_rig

    result = bridge.reload_skill_catalog(request_id="reload-1", force=True, timeout_sec=1.0)

    assert result["success"] is True
    assert result["old_generation"] == 1
    assert result["generation"] == 2
    assert result["changed_skills"] == ["nod_yes"]


def test_validate_skill_passes_all_fixed_request_fields(bridge_rig):
    bridge, _names, requests, _status_control, _action_control = bridge_rig
    payload = {
        "skill_name": "move_relative_ee",
        "target_name": "banana",
        "container_name": "black bowl",
        "place_name": "home",
        "motion_direction": "forward",
        "motion_distance": 0.03,
        "timeout_sec": 12.0,
    }

    result = bridge.validate_skill(payload, timeout_sec=1.0)

    assert result == {
        "allowed": True,
        "reason": "allowed",
        "error_code": "",
        "actual_registry_epoch": "",
        "actual_registry_generation": 0,
        "actual_registry_digest": "",
        "diagnostics": [],
    }
    assert len(requests) == 1
    request = requests[0]
    assert request.skill_name == "move_relative_ee"
    assert request.target_name == "banana"
    assert request.container_name == "black bowl"
    assert request.place_name == "home"
    assert request.motion_direction == "forward"
    assert request.motion_distance == pytest.approx(0.03)


def test_visual_game_services_preserve_request_and_result_fields(bridge_rig):
    bridge, _names, _requests, _status_control, _action_control = bridge_rig

    started = bridge.start_visual_game(
        "sorting_hat",
        request_id="game-test-1",
        expected_config_digest="game-digest",
        timeout_sec=1.0,
    )
    result = bridge.get_visual_game_result(started["request_id"], timeout_sec=1.0)

    assert started == {
        "accepted": True,
        "duplicate": False,
        "request_id": "game-test-1",
        "config_digest": "game-digest",
        "error_code": "",
        "message": "accepted",
    }
    assert result == {
        "found": True,
        "terminal": True,
        "success": True,
        "game_name": "sorting_hat",
        "scene_summary": "赫奇帕奇",
        "result_json": '{"scene_summary":"赫奇帕奇"}',
        "config_digest": "game-digest",
        "error_code": "",
        "message": "completed",
    }


def test_validate_and_execute_preserve_navigation_fields(bridge_rig):
    bridge, _names, requests, _status_control, action_control = bridge_rig
    payload = {
        "schema_version": 2,
        "skill_name": "nav_abs_coordinate",
        "target_name": "",
        "container_name": "",
        "place_name": "",
        "motion_direction": "",
        "motion_distance": 0.0,
        "direction": "left",
        "distance": 1.25,
        "degree": 90.0,
        "has_x": True,
        "x": 0.0,
        "has_y": True,
        "y": -2.5,
        "has_yaw": True,
        "yaw": 0.0,
        "timeout_sec": 12.0,
    }

    assert bridge.validate_skill(payload, timeout_sec=1.0)["allowed"] is True
    send_future = bridge.send_skill_goal(payload, task_id="nav-task")
    assert bridge.wait_future(send_future, timeout_sec=1.0) is True
    result_future = send_future.result().get_result_async()
    assert bridge.wait_future(result_future, timeout_sec=1.0) is True

    validation = requests[-1]
    goal = action_control["goals"][-1]
    for message in (validation, goal):
        assert message.direction == "left"
        assert message.distance == pytest.approx(1.25)
        assert message.degree == pytest.approx(90.0)
        assert message.has_x is True
        assert message.x == pytest.approx(0.0)
        assert message.has_y is True
        assert message.y == pytest.approx(-2.5)
        assert message.has_yaw is True
        assert message.yaw == pytest.approx(0.0)


def test_navigation_workflow_step_round_trip_preserves_presence():
    from ibrobot_msgs.msg import WorkflowStep

    bridge = RosBridge(status_service="/unused", validate_skill_service="/unused", skill_action="/unused")
    bridge._WorkflowStep = WorkflowStep
    source = {
        "schema_version": 2,
        "skill_name": "nav_abs_coordinate",
        "direction": "",
        "distance": 0.0,
        "degree": 0.0,
        "x": 0.0,
        "y": -1.0,
        "yaw": 0.0,
    }

    message = bridge._workflow_step_message(source)
    restored = bridge._workflow_step_dict(message)

    assert restored["schema_version"] == 2
    assert restored["skill_name"] == "nav_abs_coordinate"
    assert restored["has_x"] is True and restored["x"] == pytest.approx(0.0)
    assert restored["has_y"] is True and restored["y"] == pytest.approx(-1.0)
    assert restored["has_yaw"] is True and restored["yaw"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "source",
    [
        {
            "schema_version": 1,
            "skill_name": "nav_abs_coordinate",
            "has_x": False,
            "x": 0.0,
        },
        {
            "schema_version": 2,
            "skill_name": "open_gripper_skill",
        },
    ],
)
def test_workflow_step_message_preserves_explicit_version_without_domain_inference(source):
    from ibrobot_msgs.msg import WorkflowStep

    bridge = RosBridge(status_service="/unused", validate_skill_service="/unused", skill_action="/unused")
    bridge._WorkflowStep = WorkflowStep

    message = bridge._workflow_step_message(source)

    assert message.schema_version == source["schema_version"]
    assert message.skill_name == source["skill_name"]
    assert message.has_x is False


def test_unavailable_service_maps_to_server_unavailable(bridge_rig):
    _bridge, names, _requests, _status_control, _action_control = bridge_rig
    missing = RosBridge(
        status_service=f"{names['status']}_missing",
        validate_skill_service=f"{names['validate']}_missing",
        skill_action=f"{names['action']}_missing",
    )
    assert missing.start() is True

    try:
        with pytest.raises(BridgeError) as exc_info:
            missing.get_status(timeout_sec=0.05)
        assert exc_info.value.code == "SERVER_UNAVAILABLE"
    finally:
        missing.close()


def test_service_response_timeout_maps_to_result_timeout(bridge_rig):
    bridge, _names, _requests, status_control, _action_control = bridge_rig
    status_control["delay_sec"] = 0.2

    with pytest.raises(BridgeError) as exc_info:
        bridge.get_status(timeout_sec=0.05)

    assert exc_info.value.code == "RESULT_TIMEOUT"


def test_status_ledger_query_matrix(bridge_rig):
    bridge, _names, _requests, status_control, _action_control = bridge_rig
    cases = {
        ("", ""): ("", ""),
        ("active-task", ""): ("active", ""),
        ("terminal-task", ""): ("terminal", ""),
        ("unknown-task", ""): ("", ""),
        ("new-task", "new-hash"): ("", ""),
        ("same-task", "same-hash"): ("terminal", "DUPLICATE_TASK_ID"),
        ("same-task", "different-hash"): ("terminal", "TASK_ID_CONFLICT"),
        ("", "hash-without-task"): ("", "INVALID_ARGUMENT"),
    }
    status_control["queries"] = cases

    actual = {
        query: (
            bridge.get_status(task_id=query[0], payload_hash=query[1], timeout_sec=1.0)["request_state"],
            bridge.get_status(task_id=query[0], payload_hash=query[1], timeout_sec=1.0)["request_error_code"],
        )
        for query in cases
    }

    assert actual == cases


def test_execute_uses_deterministic_goal_uuid_and_public_feedback(bridge_rig):
    bridge, _names, _requests, _status_control, action_control = bridge_rig
    feedback = []
    payload = {
        "skill_name": "move_relative_ee",
        "target_name": "",
        "container_name": "",
        "place_name": "",
        "motion_direction": "forward",
        "motion_distance": 0.03,
        "timeout_sec": 12.0,
    }

    assert bridge.wait_for_skill_server(timeout_sec=1.0) is True
    send_future = bridge.send_skill_goal(payload, task_id="task-uuid", feedback_callback=feedback.append)
    assert bridge.wait_future(send_future, timeout_sec=1.0) is True
    goal_handle = send_future.result()
    assert goal_handle.accepted is True
    result_future = goal_handle.get_result_async()
    assert bridge.wait_future(result_future, timeout_sec=1.0) is True

    assert action_control["goal_ids"] == [skill_goal_uuid("task-uuid").bytes]
    assert feedback == [{"state": "executing", "detail": "step 1 of 1"}]


def test_cancel_task_targets_active_goal_by_deterministic_uuid(bridge_rig):
    bridge, _names, _requests, _status_control, action_control = bridge_rig
    action_control["delay_sec"] = 1.0
    payload = {
        "skill_name": "move_relative_ee",
        "target_name": "",
        "container_name": "",
        "place_name": "",
        "motion_direction": "forward",
        "motion_distance": 0.03,
        "timeout_sec": 12.0,
    }

    assert bridge.wait_for_skill_server(timeout_sec=2.0)
    send_future = bridge.send_skill_goal(payload, task_id="task-cancel", feedback_callback=None)
    assert bridge.wait_future(send_future, timeout_sec=5.0) is True
    goal_handle = send_future.result()
    result_future = goal_handle.get_result_async()

    cancel = bridge.cancel_task("task-cancel", timeout_sec=1.0)

    assert cancel == {"accepted": True, "return_code": 0}
    assert bridge.wait_future(result_future, timeout_sec=1.0) is True
    assert result_future.result().result.error_code == "SKILL_CANCELLED"
    assert action_control["goal_ids"] == [skill_goal_uuid("task-cancel").bytes]


def test_cancel_goal_waits_for_terminal_after_rejected_cancel_response():
    class Future:
        def __init__(self, value, *, done):
            self.value = value
            self.completed = done

        def done(self):
            return self.completed

        def result(self):
            return self.value

    cancel_future = Future(SimpleNamespace(return_code=1, goals_canceling=[]), done=True)
    result_future = Future(None, done=False)
    goal_handle = SimpleNamespace(cancel_goal_async=lambda: cancel_future)

    def complete_result():
        time.sleep(0.02)
        result_future.completed = True

    thread = threading.Thread(target=complete_result)
    thread.start()
    try:
        bridge = RosBridge(status_service="/unused", validate_skill_service="/unused", skill_action="/unused")
        assert bridge.cancel_goal(goal_handle, result_future, timeout_sec=0.2) is True
    finally:
        thread.join(timeout=1.0)


def test_send_skill_goal_exception_maps_to_ros_unavailable(bridge_rig):
    bridge, _names, _requests, _status_control, _action_control = bridge_rig

    def fail_send(*_args, **_kwargs):
        raise RuntimeError("transport failed")

    bridge._skill_client = SimpleNamespace(send_goal_async=fail_send)
    payload = {
        "skill_name": "move_relative_ee",
        "target_name": "",
        "container_name": "",
        "place_name": "",
        "motion_direction": "forward",
        "motion_distance": 0.03,
        "timeout_sec": 12.0,
    }

    with pytest.raises(BridgeError) as exc_info:
        bridge.send_skill_goal(payload, task_id="task-send-error")

    assert exc_info.value.code == "ROS_UNAVAILABLE"
