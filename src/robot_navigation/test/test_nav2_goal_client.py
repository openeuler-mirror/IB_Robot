"""Tests for nav2_goal_client node.

Covers voice command JSON parsing, navigation state management,
goal message construction, timeout logic, navigation-succeeded
callback, and end-to-end topic integration.

Uses unittest.mock to replace ActionClient and ServiceClient.
"""

import json
import math
import time
from unittest.mock import MagicMock

import pytest
import rclpy
from action_msgs.msg import GoalStatus
from std_msgs.msg import String

from robot_navigation.nav2_goal_client import Nav2GoalClient

# ── fixtures ────────────────────────────────────────────────────────────────

# Original bound methods saved before any mocking
_orig = {}


@pytest.fixture(scope="module")
def rclpy_init():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture(scope="module")
def node(rclpy_init):
    """Module-scoped Nav2GoalClient with test topic subscriptions."""
    n = Nav2GoalClient()

    # Save originals before any test can mock them
    _orig["send_goal_async"] = n.nav_to_pose_client.send_goal_async
    _orig["wait_for_service"] = n.evaluation_client.wait_for_service
    _orig["eval_call_async"] = n.evaluation_client.call_async
    _orig["_cancel_navigation"] = n._cancel_navigation
    _orig["_on_navigation_succeeded"] = n._on_navigation_succeeded

    # Re-subscribe to test topics
    n.voice_sub = n.create_subscription(String, "/test/keyword_matched", n.voice_command_callback, 10)
    n.nav_stop_sub = n.create_subscription(String, "/test/nav_stop", n.stop_callback, 10)

    yield n
    n.destroy_node()


@pytest.fixture
def reset_state(node):
    """Reset navigation state AND restore mocked methods before each test."""
    node.is_navigating = False
    node.navigation_succeeded = False
    node.navigation_failed = False
    node.current_task_description = ""
    node.goal_handle = None
    node._nav_start_time = None

    # Restore original methods to prevent cross-test mock pollution
    node.nav_to_pose_client.send_goal_async = _orig["send_goal_async"]
    node.evaluation_client.wait_for_service = _orig["wait_for_service"]
    node.evaluation_client.call_async = _orig["eval_call_async"]
    node._cancel_navigation = _orig["_cancel_navigation"]
    node._on_navigation_succeeded = _orig["_on_navigation_succeeded"]


# ── helpers ─────────────────────────────────────────────────────────────────


def _mock_send_goal_async(node):
    """Replace send_goal_async with a mock that returns a MagicMock future."""
    node.nav_to_pose_client.send_goal_async = MagicMock(return_value=MagicMock())


def _capture_goal(node):
    """Replace send_goal_async to capture the goal message."""
    captured = []

    def capture(goal_msg):
        captured.append(goal_msg)
        future = MagicMock()
        future.add_done_callback = MagicMock()
        return future

    node.nav_to_pose_client.send_goal_async = capture
    return captured


def _make_goal_response(accepted=True):
    response = MagicMock()
    response.accepted = accepted
    return response


def _make_result_future(status):
    result = MagicMock()
    result.status = status
    future = MagicMock()
    future.result.return_value = result
    return future


def _publish_and_collect(node, topic, msg_type, text, pub_topic="/test/keyword_matched", timeout=1.0):
    """Publish text on pub_topic, return first msg received on topic."""
    received = []
    sub = node.create_subscription(msg_type, topic, lambda m: received.append(m), 10)
    pub = node.create_publisher(String, pub_topic, 10)

    deadline = node.get_clock().now() + rclpy.duration.Duration(seconds=0.3)
    while node.get_clock().now() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

    msg = String()
    msg.data = text
    pub.publish(msg)

    deadline = node.get_clock().now() + rclpy.duration.Duration(seconds=timeout)
    while not received and node.get_clock().now() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

    node.destroy_subscription(sub)
    node.destroy_publisher(pub)
    return received[-1] if received else None


# ═══════════════════════════════════════════════════════════════════════════
# 1. TestVoiceCommandParsing — JSON parsing
# ═══════════════════════════════════════════════════════════════════════════


class TestVoiceCommandParsing:
    def test_destination_triggers_send_goal(self, node, reset_state):
        _mock_send_goal_async(node)
        msg = String()
        msg.data = json.dumps(
            {
                "keyword": "去厨房",
                "type": "destination",
                "info": {"x": 1.0, "y": 2.0, "theta": 0.5},
            }
        )
        node.voice_command_callback(msg)
        node.nav_to_pose_client.send_goal_async.assert_called_once()

    def test_destination_coordinates(self, node, reset_state):
        captured = _capture_goal(node)

        msg = String()
        msg.data = json.dumps(
            {
                "keyword": "去厨房",
                "type": "destination",
                "info": {"x": 1.5, "y": 2.0, "theta": 0.5},
            }
        )
        node.voice_command_callback(msg)

        assert len(captured) == 1
        g = captured[0]
        assert g.pose.pose.position.x == pytest.approx(1.5)
        assert g.pose.pose.position.y == pytest.approx(2.0)
        assert g.pose.pose.orientation.z == pytest.approx(math.sin(0.25), abs=1e-6)

    def test_destination_defaults(self, node, reset_state):
        captured = _capture_goal(node)

        msg = String()
        msg.data = json.dumps(
            {
                "keyword": "test",
                "type": "destination",
                "info": {},
            }
        )
        node.voice_command_callback(msg)

        assert len(captured) == 1
        g = captured[0]
        assert g.pose.pose.position.x == pytest.approx(0.0)
        assert g.pose.pose.position.y == pytest.approx(0.0)
        assert g.pose.pose.orientation.w == pytest.approx(1.0)

    def test_action_caches_description(self, node, reset_state):
        msg = String()
        msg.data = json.dumps(
            {
                "keyword": "捡蓝色方块",
                "type": "action",
                "info": {"task_description": "Pick blue"},
            }
        )
        node.voice_command_callback(msg)
        assert node.current_task_description == "Pick blue"

    def test_action_empty_description_ignored(self, node, reset_state):
        node.current_task_description = "existing"
        msg = String()
        msg.data = json.dumps(
            {
                "keyword": "test",
                "type": "action",
                "info": {"task_description": ""},
            }
        )
        node.voice_command_callback(msg)
        assert node.current_task_description == "existing"

    def test_action_missing_key_ignored(self, node, reset_state):
        node.current_task_description = "existing"
        msg = String()
        msg.data = json.dumps(
            {
                "keyword": "test",
                "type": "action",
                "info": {},
            }
        )
        node.voice_command_callback(msg)
        assert node.current_task_description == "existing"

    def test_stop_cancels_navigation(self, node, reset_state):
        node.is_navigating = True
        node._cancel_navigation = MagicMock()
        msg = String()
        msg.data = json.dumps(
            {
                "keyword": "停止",
                "type": "stop",
                "info": {},
            }
        )
        node.voice_command_callback(msg)
        node._cancel_navigation.assert_called_once()

    def test_invalid_json_no_crash(self, node, reset_state):
        msg = String()
        msg.data = "{bad json"
        node.voice_command_callback(msg)
        assert node.is_navigating is False

    def test_missing_type_ignored(self, node, reset_state):
        _mock_send_goal_async(node)
        msg = String()
        msg.data = json.dumps({"keyword": "test"})
        node.voice_command_callback(msg)
        node.nav_to_pose_client.send_goal_async.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════
# 2. TestStateManagement — state flags
# ═══════════════════════════════════════════════════════════════════════════


class TestStateManagement:
    def test_initial_state(self, reset_state, node):
        assert node.is_navigating is False
        assert node.current_task_description == ""
        assert node.goal_handle is None
        assert node.navigation_succeeded is False
        assert node.navigation_failed is False

    def test_send_goal_sets_navigating(self, node, reset_state):
        _mock_send_goal_async(node)
        node.send_goal(1.0, 2.0, 0.5)
        assert node.is_navigating is True
        assert node._nav_start_time is not None

    def test_send_goal_guard(self, node, reset_state):
        _mock_send_goal_async(node)
        node.is_navigating = True
        node.send_goal(1.0, 2.0, 0.5)
        node.nav_to_pose_client.send_goal_async.assert_not_called()

    def test_cancel_resets_when_no_handle(self, node, reset_state):
        node.is_navigating = True
        node.goal_handle = None
        node._cancel_navigation("test")
        assert node.is_navigating is False

    def test_cancel_noop_when_not_navigating(self, node, reset_state):
        node.is_navigating = False
        node._cancel_navigation("test")
        assert node.is_navigating is False

    def test_goal_rejected_resets(self, node, reset_state):
        _mock_send_goal_async(node)
        node.send_goal(1.0, 2.0, 0.0)

        future = MagicMock()
        future.result.return_value = _make_goal_response(accepted=False)
        node._goal_response_callback(future)

        assert node.is_navigating is False

    def test_goal_accepted_stores_handle(self, node, reset_state):
        _mock_send_goal_async(node)
        node.send_goal(1.0, 2.0, 0.0)

        goal_response = _make_goal_response(accepted=True)
        goal_response.get_result_async = MagicMock(return_value=MagicMock())
        future = MagicMock()
        future.result.return_value = goal_response
        node._goal_response_callback(future)

        assert node.goal_handle is not None

    def test_result_succeeded_resets(self, node, reset_state):
        node.is_navigating = True
        node._on_navigation_succeeded = MagicMock()

        future = _make_result_future(GoalStatus.STATUS_SUCCEEDED)
        node._get_result_callback(future)

        assert node.is_navigating is False


# ═══════════════════════════════════════════════════════════════════════════
# 3. TestGoalMessageConstruction
# ═══════════════════════════════════════════════════════════════════════════


class TestGoalMessageConstruction:
    def test_frame_id(self, node, reset_state):
        captured = _capture_goal(node)
        node.send_goal(1.0, 2.0, 0.0)
        assert captured[0].pose.header.frame_id == "map"

    def test_position(self, node, reset_state):
        captured = _capture_goal(node)
        node.send_goal(3.0, 4.0, 0.0)
        assert captured[0].pose.pose.position.x == pytest.approx(3.0)
        assert captured[0].pose.pose.position.y == pytest.approx(4.0)
        assert captured[0].pose.pose.position.z == pytest.approx(0.0)

    def test_orientation_zero(self, node, reset_state):
        captured = _capture_goal(node)
        node.send_goal(1.0, 2.0, 0.0)
        assert captured[0].pose.pose.orientation.z == pytest.approx(0.0, abs=1e-10)
        assert captured[0].pose.pose.orientation.w == pytest.approx(1.0, abs=1e-10)

    def test_orientation_quarter_turn(self, node, reset_state):
        captured = _capture_goal(node)
        theta = math.pi / 2
        node.send_goal(1.0, 2.0, theta)
        assert captured[0].pose.pose.orientation.z == pytest.approx(math.sin(theta / 2), abs=1e-6)
        assert captured[0].pose.pose.orientation.w == pytest.approx(math.cos(theta / 2), abs=1e-6)

    def test_orientation_negative(self, node, reset_state):
        captured = _capture_goal(node)
        theta = -math.pi / 4
        node.send_goal(1.0, 2.0, theta)
        assert captured[0].pose.pose.orientation.z == pytest.approx(math.sin(theta / 2), abs=1e-6)
        assert captured[0].pose.pose.orientation.w == pytest.approx(math.cos(theta / 2), abs=1e-6)

    def test_stamp_set(self, node, reset_state):
        captured = _capture_goal(node)
        node.send_goal(1.0, 2.0, 0.0)
        stamp = captured[0].pose.header.stamp
        assert stamp.sec != 0 or stamp.nanosec != 0


# ═══════════════════════════════════════════════════════════════════════════
# 4. TestTimeoutLogic
# ═══════════════════════════════════════════════════════════════════════════


class TestTimeoutLogic:
    def test_no_timeout_when_not_navigating(self, node, reset_state):
        node._cancel_navigation = MagicMock()
        node._check_timeout()
        node._cancel_navigation.assert_not_called()

    def test_no_timeout_within_limit(self, node, reset_state):
        node.is_navigating = True
        node._nav_start_time = time.monotonic()
        node._cancel_navigation = MagicMock()
        node._check_timeout()
        node._cancel_navigation.assert_not_called()

    def test_timeout_triggers_cancel(self, node, reset_state):
        node.is_navigating = True
        node._nav_start_time = time.monotonic() - 61.0
        node.timeout_sec = 60.0
        node._cancel_navigation = MagicMock()
        node._check_timeout()
        node._cancel_navigation.assert_called_once()
        assert node._nav_start_time is None

    def test_timeout_zero_disables(self, node, reset_state):
        node.is_navigating = True
        node._nav_start_time = time.monotonic() - 999.0
        original_timeout = node.timeout_sec
        node.timeout_sec = 0.0
        node._cancel_navigation = MagicMock()
        node._check_timeout()
        node._cancel_navigation.assert_not_called()
        node.timeout_sec = original_timeout


# ═══════════════════════════════════════════════════════════════════════════
# 5. TestOnNavigationSucceeded
# ═══════════════════════════════════════════════════════════════════════════


class TestOnNavigationSucceeded:
    def test_triggers_evaluation(self, node, reset_state):
        node.current_task_description = "Pick blue"
        node.evaluation_client.wait_for_service = MagicMock(return_value=True)
        node.evaluation_client.call_async = MagicMock(return_value=MagicMock())

        node._on_navigation_succeeded()

        node.evaluation_client.call_async.assert_called_once()

    def test_clears_task_description(self, node, reset_state):
        node.current_task_description = "Pick blue"
        node.evaluation_client.wait_for_service = MagicMock(return_value=True)
        node.evaluation_client.call_async = MagicMock(return_value=MagicMock())

        node._on_navigation_succeeded()

        assert node.current_task_description == ""

    def test_no_evaluation_without_task(self, node, reset_state):
        node.current_task_description = ""
        node.evaluation_client.call_async = MagicMock()

        node._on_navigation_succeeded()

        node.evaluation_client.call_async.assert_not_called()

    def test_service_unavailable_sets_failed(self, node, reset_state):
        node.current_task_description = "Pick blue"
        node.evaluation_client.wait_for_service = MagicMock(return_value=False)
        node.evaluation_client.call_async = MagicMock()

        node._on_navigation_succeeded()

        assert node.navigation_failed is True
        node.evaluation_client.call_async.assert_not_called()

    def test_evaluation_success(self, node, reset_state):
        node.current_task_description = "Pick blue"
        node.evaluation_client.wait_for_service = MagicMock(return_value=True)

        future_mock = MagicMock()
        node.evaluation_client.call_async = MagicMock(return_value=future_mock)

        node._on_navigation_succeeded()

        # Simulate evaluation callback with success
        response = MagicMock()
        response.success = True
        response.message = "Evaluation triggered"
        eval_future = MagicMock()
        eval_future.result.return_value = response

        # Find and call the evaluation_callback that was registered
        assert future_mock.add_done_callback.called
        cb = future_mock.add_done_callback.call_args[0][0]
        cb(eval_future)

        assert node.navigation_succeeded is True


# ═══════════════════════════════════════════════════════════════════════════
# 6. TestEndToEnd — topic integration
# ═══════════════════════════════════════════════════════════════════════════


class TestEndToEnd:
    def test_destination_via_topic(self, node, reset_state):
        _mock_send_goal_async(node)
        _publish_and_collect(
            node,
            "/test/keyword_matched",
            String,
            json.dumps(
                {
                    "keyword": "去厨房",
                    "type": "destination",
                    "info": {"x": 1.0, "y": 2.0, "theta": 0.0},
                }
            ),
        )
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.05)
        node.nav_to_pose_client.send_goal_async.assert_called_once()

    def test_action_via_topic(self, node, reset_state):
        node.current_task_description = ""
        _publish_and_collect(
            node,
            "/test/keyword_matched",
            String,
            json.dumps(
                {
                    "keyword": "捡蓝色方块",
                    "type": "action",
                    "info": {"task_description": "Pick blue"},
                }
            ),
        )
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.05)
        assert node.current_task_description == "Pick blue"

    def test_stop_via_topic_when_navigating(self, node, reset_state):
        node.is_navigating = True
        node._cancel_navigation = MagicMock()
        _publish_and_collect(
            node,
            "/test/nav_stop",
            String,
            "stop",
            pub_topic="/test/nav_stop",
        )
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.05)
        node._cancel_navigation.assert_called()

    def test_stop_via_topic_when_idle(self, node, reset_state):
        node.is_navigating = False
        node._cancel_navigation = MagicMock()
        _publish_and_collect(
            node,
            "/test/nav_stop",
            String,
            "stop",
            pub_topic="/test/nav_stop",
        )
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.05)
        node._cancel_navigation.assert_not_called()

    def test_multiple_commands_sequence(self, node, reset_state):
        _mock_send_goal_async(node)

        # First: destination
        msg1 = String()
        msg1.data = json.dumps(
            {
                "keyword": "去厨房",
                "type": "destination",
                "info": {"x": 1.0, "y": 2.0, "theta": 0.0},
            }
        )
        node.voice_command_callback(msg1)
        assert node.is_navigating is True
        assert node.nav_to_pose_client.send_goal_async.call_count == 1

        # Then: stop — set goal_handle so cancel has something to call
        mock_handle = MagicMock()
        node.goal_handle = mock_handle

        msg2 = String()
        msg2.data = json.dumps({"keyword": "停止", "type": "stop", "info": {}})
        node.voice_command_callback(msg2)

        mock_handle.cancel_goal_async.assert_called_once()

    def test_action_then_destination(self, node, reset_state):
        _mock_send_goal_async(node)

        # First: action
        msg1 = String()
        msg1.data = json.dumps(
            {
                "keyword": "捡蓝色方块",
                "type": "action",
                "info": {"task_description": "Pick blue"},
            }
        )
        node.voice_command_callback(msg1)
        assert node.current_task_description == "Pick blue"

        # Then: destination
        msg2 = String()
        msg2.data = json.dumps(
            {
                "keyword": "去厨房",
                "type": "destination",
                "info": {"x": 1.0, "y": 2.0, "theta": 0.0},
            }
        )
        node.voice_command_callback(msg2)
        # task_description should still be preserved
        assert node.current_task_description == "Pick blue"
        node.nav_to_pose_client.send_goal_async.assert_called_once()
