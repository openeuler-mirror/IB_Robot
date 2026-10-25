"""Layer 1: Pure software E2E tests.

2 real nodes (voice_control + nav2_goal_client) connected via real ROS2
topics, with mock external action/service servers.

Validates the data chain: voice text -> keyword match -> Nav2 goal.
The cmd_vel -> wheel projection and the wheel odometry chain moved to the
LeKiwi runtime (lekiwi_robot/base_node over lekiwi_sdk) when the generic
cmd_vel bridge was removed (openspec lekiwi-application-migration 2.2);
their coverage lives in the lekiwi_sdk tests and the LeKiwi runtime
conformance suite.

Usage:
    colcon test --packages-select robot_navigation --pytest-args -k "test_pipeline_software"
"""

import json
import math
import threading
import time

import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String

from robot_navigation.nav2_goal_client import Nav2GoalClient
from robot_navigation.voice_control import VoiceControl
from test.e2e.mock_servers import (
    MockNavigateToPoseServer,
    MockSetHotwordsServer,
    MockTriggerServer,
)

# ── test data ───────────────────────────────────────────────────────────────

KEYWORDS_JSON = json.dumps(
    {
        "keywords": {
            "捡.*蓝色方块|拿.*蓝色方块|蓝色方块": {
                "type": "action",
                "info": {"task_description": "Pick up the blue square"},
            },
            "去.*a点|到.*a点|a点": {
                "type": "destination",
                "info": {"destination": "point_a"},
            },
            "去.*b点|到.*b点|b点": {
                "type": "destination",
                "info": {"destination": "point_b"},
            },
            "停止|停下": {
                "type": "stop",
                "info": {"task_description": "Stop current action"},
            },
        }
    }
)

DESTINATIONS_JSON = json.dumps(
    {
        "point_a": {"x": 0.0, "y": 0.2, "theta": 1.5708},
        "point_b": {"x": 0.2, "y": 0.0, "theta": 0.0},
    }
)

# Mock action server delay: keeps navigation "in progress" long enough
# for cancel (TC1.3) and double-send (TC1.7) tests to work.
MOCK_NAV_DELAY = 5.0


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def rclpy_init():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture(scope="module")
def pipeline(rclpy_init):
    """Set up the full 3-node pipeline + mock servers with MultiThreadedExecutor."""
    executor = MultiThreadedExecutor()

    # ── Mock servers ──
    mock_nav = MockNavigateToPoseServer(executor, delay_sec=MOCK_NAV_DELAY)
    mock_eval = MockTriggerServer("/action_dispatcher/start_evaluate", executor)
    mock_stop = MockTriggerServer("/action_dispatcher/stop_evaluate", executor)
    mock_hotwords = MockSetHotwordsServer(executor)

    # ── Real nodes ──
    # 1. voice_control: publish to /e2e/keyword_matched, subscribe on /e2e/voice_command
    voice = VoiceControl()
    voice.keywords_json = KEYWORDS_JSON
    voice.keywords = voice._load_keywords()
    voice.destinations = json.loads(DESTINATIONS_JSON)
    # Re-wire publishers/subscribers to isolated e2e topics. The production
    # defaults may receive residual board traffic on a shared ROS domain.
    voice.destroy_publisher(voice.keyword_pub)
    voice.destroy_publisher(voice.nav_stop_pub)
    voice.destroy_subscription(voice.sub)
    voice.destroy_subscription(voice.keyword_sub)
    voice.keyword_pub = voice.create_publisher(String, "/e2e/keyword_matched", 10)
    voice.nav_stop_pub = voice.create_publisher(String, "/e2e/nav_stop", 10)
    voice.sub = voice.create_subscription(String, "/e2e/voice_command", voice._text_callback, 10)
    # Cancel hotword timer (no voice_asr_node in test env)
    if voice._hotword_timer is not None:
        voice.destroy_timer(voice._hotword_timer)
        voice._hotword_timer = None
    executor.add_node(voice)

    # 2. nav2_goal_client: subscribe on e2e topics
    goal_client = Nav2GoalClient()
    goal_client.destroy_subscription(goal_client.voice_sub)
    goal_client.destroy_subscription(goal_client.nav_stop_sub)
    goal_client.voice_sub = goal_client.create_subscription(
        String, "/e2e/keyword_matched", goal_client.voice_command_callback, 10
    )
    goal_client.nav_stop_sub = goal_client.create_subscription(String, "/e2e/nav_stop", goal_client.stop_callback, 10)
    executor.add_node(goal_client)

    # Spin executor in background thread
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # Wait for discovery
    time.sleep(0.5)

    yield {
        "voice": voice,
        "goal_client": goal_client,
        "mock_nav": mock_nav,
        "mock_eval": mock_eval,
        "mock_stop": mock_stop,
        "mock_hotwords": mock_hotwords,
        "executor": executor,
    }

    # Teardown
    executor.shutdown()
    spin_thread.join(timeout=5.0)
    mock_nav.destroy()
    mock_eval.destroy()
    mock_stop.destroy()
    mock_hotwords.destroy()
    goal_client.destroy_node()
    voice.destroy_node()


def _wait_for_nav_complete(gc, timeout=MOCK_NAV_DELAY + 5.0):
    """Wait until goal_client.is_navigating becomes False."""
    deadline = time.monotonic() + timeout
    while gc.is_navigating and time.monotonic() < deadline:
        time.sleep(0.1)


def _wait_until(predicate, timeout=2.0, interval=0.05):
    """Wait until predicate() returns True."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


@pytest.fixture
def reset_pipeline(pipeline):
    """Reset state between tests. Waits for any in-progress navigation first."""
    gc = pipeline["goal_client"]

    # Wait for any navigation to complete naturally
    _wait_for_nav_complete(gc)

    gc.is_navigating = False
    gc.navigation_succeeded = False
    gc.navigation_failed = False
    gc.current_task_description = ""
    gc.goal_handle = None
    gc._nav_start_time = None

    mock_nav = pipeline["mock_nav"]
    with mock_nav._lock:
        mock_nav.received_goals.clear()
        mock_nav.succeeded_goals.clear()

    mock_eval = pipeline["mock_eval"]
    with mock_eval._lock:
        mock_eval.calls.clear()

    mock_stop = pipeline["mock_stop"]
    with mock_stop._lock:
        mock_stop.calls.clear()


# ── helpers ──────────────────────────────────────────────────────────────────


def _publish_text(pipeline, text, timeout=2.0):
    """Publish voice text and wait for processing."""
    voice = pipeline["voice"]
    pub = voice.create_publisher(String, "/e2e/voice_command", 10)
    time.sleep(0.1)  # let pub connect
    msg = String()
    msg.data = text
    pub.publish(msg)
    time.sleep(timeout)
    voice.destroy_publisher(pub)


# ═══════════════════════════════════════════════════════════════════════════
# TC1.1: "去a点" -> Nav2 goal coordinates correct
# ═══════════════════════════════════════════════════════════════════════════


class TestNavGoalCoordinates:
    """TC1.1: Verify voice command resolves to correct Nav2 goal coordinates."""

    def test_destination_a_coordinates(self, pipeline, reset_pipeline):
        _publish_text(pipeline, "去a点", timeout=1.0)
        mock_nav = pipeline["mock_nav"]
        with mock_nav._lock:
            assert len(mock_nav.received_goals) == 1
        goal = mock_nav.received_goals[0]
        assert goal.pose.pose.position.x == pytest.approx(0.0)
        assert goal.pose.pose.position.y == pytest.approx(0.2)
        # theta = 1.5708 => qz = sin(1.5708/2), qw = cos(1.5708/2)
        expected_z = math.sin(1.5708 / 2.0)
        expected_w = math.cos(1.5708 / 2.0)
        assert goal.pose.pose.orientation.z == pytest.approx(expected_z, abs=1e-4)
        assert goal.pose.pose.orientation.w == pytest.approx(expected_w, abs=1e-4)

    def test_destination_b_coordinates(self, pipeline, reset_pipeline):
        _publish_text(pipeline, "去b点", timeout=1.0)
        mock_nav = pipeline["mock_nav"]
        with mock_nav._lock:
            assert len(mock_nav.received_goals) == 1
        goal = mock_nav.received_goals[0]
        assert goal.pose.pose.position.x == pytest.approx(0.2)
        assert goal.pose.pose.position.y == pytest.approx(0.0)
        assert goal.pose.pose.orientation.w == pytest.approx(1.0, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# TC1.2: action + destination -> evaluation triggered after arrival
# ═══════════════════════════════════════════════════════════════════════════


class TestActionDestinationEvaluation:
    """TC1.2: "捡蓝色方块" + "去a点" -> evaluation triggered after nav succeeds."""

    def test_evaluation_triggered_after_nav(self, pipeline, reset_pipeline):
        gc = pipeline["goal_client"]

        # Step 1: send action command to cache task_description
        _publish_text(pipeline, "捡蓝色方块", timeout=0.5)
        assert gc.current_task_description == "Pick up the blue square"

        # Step 2: send destination to trigger navigation
        _publish_text(pipeline, "去a点", timeout=1.0)
        assert gc.is_navigating is True

        # Wait for mock to auto-succeed (delay_sec + buffer)
        _wait_for_nav_complete(gc)

        # After navigation succeeds, evaluation should be triggered
        mock_eval = pipeline["mock_eval"]
        with mock_eval._lock:
            assert len(mock_eval.calls) >= 1, "Evaluation should be triggered after navigation succeeds"


# ═══════════════════════════════════════════════════════════════════════════
# TC1.3: "去b点" navigating then "停止" -> cancel
# ═══════════════════════════════════════════════════════════════════════════


class TestNavigationCancel:
    """TC1.3: Cancel navigation via stop command."""

    def test_stop_cancels_navigation(self, pipeline, reset_pipeline):
        gc = pipeline["goal_client"]

        # Start navigation (mock has 5s delay, so it stays in_progress)
        _publish_text(pipeline, "去b点", timeout=1.0)
        assert gc.is_navigating is True, "Should be navigating after destination command"

        # Send stop while nav is still in progress
        _publish_text(pipeline, "停止", timeout=1.0)

        # Wait for cancel to propagate: cancel_goal_async -> _cancel_callback ->
        # _get_result_callback (STATUS_CANCELED) -> is_navigating = False.
        # The mock execute callback is sleeping for MOCK_NAV_DELAY, so the result
        # callback arrives after the sleep finishes. Wait up to that duration.
        _wait_for_nav_complete(gc, timeout=MOCK_NAV_DELAY + 5.0)
        assert gc.is_navigating is False, "Should stop navigating after stop command"


# ═══════════════════════════════════════════════════════════════════════════
# TC1.4 / TC1.5 (cmd_vel -> wheel IK, joint_states -> odometry) were removed
# with the generic cmd_vel bridge (openspec lekiwi-application-migration
# 2.2). The projection and odometry behavior now lives in the LeKiwi
# runtime: equivalence was recorded in that change's baseline replay, and
# runtime coverage runs in the lekiwi_sdk tests and the LeKiwi conformance
# suite (base velocity, odometry, staleness, navigation gate).
# ═══════════════════════════════════════════════════════════════════════════
# TC1.6: Invalid voice text -> no goal sent
# ═══════════════════════════════════════════════════════════════════════════


class TestInvalidVoiceText:
    """TC1.6: Unrecognized voice text produces no Nav2 goal."""

    def test_no_goal_for_unknown_text(self, pipeline, reset_pipeline):
        _publish_text(pipeline, "今天天气真好", timeout=1.0)
        mock_nav = pipeline["mock_nav"]
        with mock_nav._lock:
            assert len(mock_nav.received_goals) == 0


# ═══════════════════════════════════════════════════════════════════════════
# TC1.7: Rapid consecutive destinations -> only first executes
# ═══════════════════════════════════════════════════════════════════════════


class TestRapidDestinations:
    """TC1.7: Two destinations in quick succession -> only first goal is sent."""

    def test_only_first_destination_executes(self, pipeline, reset_pipeline):
        # Send two destinations rapidly
        voice = pipeline["voice"]
        pub = voice.create_publisher(String, "/e2e/voice_command", 10)
        time.sleep(0.1)

        msg1 = String()
        msg1.data = "去a点"
        pub.publish(msg1)
        mock_nav = pipeline["mock_nav"]
        assert _wait_until(lambda: len(mock_nav.received_goals) >= 1, timeout=2.0)
        with mock_nav._lock:
            goals_after_first = len(mock_nav.received_goals)

        msg2 = String()
        msg2.data = "去b点"
        pub.publish(msg2)
        time.sleep(1.0)

        with mock_nav._lock:
            assert len(mock_nav.received_goals) == goals_after_first, (
                "Second destination should be blocked while navigating"
            )
        # Verify it's point_a (first destination)
        goal = mock_nav.received_goals[0]
        assert goal.pose.pose.position.y == pytest.approx(0.2)  # point_a y

        voice.destroy_publisher(pub)
