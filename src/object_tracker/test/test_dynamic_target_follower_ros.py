import threading
import time
from types import SimpleNamespace

import pytest
import rclpy
from geometry_msgs.msg import TransformStamped
from nav2_msgs.action import ComputePathToPose, FollowPath
from nav_msgs.msg import Path
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_srvs.srv import Trigger

from ibrobot_msgs.msg import TrackState
from object_tracker.dynamic_target_follower_node import DynamicTargetFollowerNode


class MockPlanner:
    def __init__(self, executor, action_name):
        self.node = Node("mock_object_follow_planner")
        self.goals = []
        self.server = ActionServer(
            self.node,
            ComputePathToPose,
            action_name,
            goal_callback=lambda goal: self._accept(goal),
            execute_callback=self._execute,
        )
        executor.add_node(self.node)

    def _accept(self, goal):
        self.goals.append(goal)
        return GoalResponse.ACCEPT

    def _execute(self, handle):
        result = ComputePathToPose.Result()
        result.path = Path()
        result.path.header = handle.request.goal.header
        result.path.poses = [handle.request.goal]
        handle.succeed()
        return result

    def destroy(self):
        self.server.destroy()
        self.node.destroy_node()


class MockController:
    def __init__(self, executor, action_name):
        self.node = Node("mock_object_follow_controller")
        self.events = []
        self.server = ActionServer(
            self.node,
            FollowPath,
            action_name,
            goal_callback=self._accept,
            cancel_callback=self._cancel,
            execute_callback=self._execute,
        )
        executor.add_node(self.node)

    def _accept(self, _goal):
        self.events.append(f"follow-{len([event for event in self.events if event.startswith('follow-')]) + 1}")
        return GoalResponse.ACCEPT

    def _cancel(self, _handle):
        active = len([event for event in self.events if event.startswith("follow-")])
        self.events.append(f"cancel-{active}")
        return CancelResponse.ACCEPT

    @staticmethod
    def _execute(handle):
        # Must terminate on shutdown, not only on cancellation. rclpy runs an
        # action execute_callback on a ThreadPoolExecutor worker, and
        # concurrent.futures joins those workers from an atexit hook with no
        # timeout. A goal left uncancelled by a test therefore keeps a worker
        # spinning, and the pytest process hangs at interpreter exit AFTER
        # reporting every test as passed - which reads as colcon never finishing
        # the package rather than as a test failure.
        deadline = time.monotonic() + 10.0
        while not handle.is_cancel_requested:
            if not rclpy.ok() or time.monotonic() > deadline:
                handle.abort()
                return FollowPath.Result()
            time.sleep(0.01)
        handle.canceled()
        return FollowPath.Result()

    def destroy(self):
        self.server.destroy()
        self.node.destroy_node()


def _wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not satisfied before timeout")


def _track(node, x):
    state = TrackState()
    state.header.frame_id = "odom"
    state.header.stamp = node.get_clock().now().to_msg()
    state.session_id = "session-1"
    state.object_id = "banana-1"
    state.lifecycle_state = TrackState.TRACKING
    state.pose.pose.position.x = x
    state.pose.pose.orientation.w = 1.0
    state.pose.covariance[0] = 0.01
    state.pose.covariance[7] = 0.01
    state.measured = True
    state.actionable = True
    state.confidence = 0.9
    return state


@pytest.fixture
def ros_context():
    rclpy.init()
    executor = MultiThreadedExecutor(num_threads=4)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    yield executor
    executor.shutdown()
    thread.join(timeout=2.0)
    rclpy.shutdown()


def test_target_motion_cancels_active_path_before_replacement(ros_context):
    planner_name = "/object_tracker_test/compute_path"
    controller_name = "/object_tracker_test/follow_path"
    planner = MockPlanner(ros_context, planner_name)
    controller = MockController(ros_context, controller_name)
    follower = DynamicTargetFollowerNode(
        parameter_overrides=[
            Parameter("enabled", value=True),
            Parameter("compute_path_action", value=planner_name),
            Parameter("follow_path_action", value=controller_name),
            Parameter("minimum_replan_interval_s", value=0.0),
            Parameter("replan_displacement_m", value=0.1),
        ]
    )
    follower._target_in_map = lambda state: (state.pose.pose.position.x, 0.0)
    follower._robot_position = lambda: (0.0, 0.0)
    ros_context.add_node(follower)

    try:
        follower._on_track_state(_track(follower, 2.0))
        follower._tick()
        _wait_for(lambda: controller.events == ["follow-1"])

        follower._on_track_state(_track(follower, 2.5))
        follower._tick()
        _wait_for(lambda: controller.events == ["follow-1", "cancel-1", "follow-2"])

        assert len(planner.goals) == 2
        assert planner.goals[0].goal.pose.position.x == pytest.approx(1.2)
        assert planner.goals[1].goal.pose.position.x == pytest.approx(1.7)
    finally:
        follower._stop_following("test complete")
        _wait_for(lambda: controller.events[-1:] == ["cancel-2"])
        ros_context.remove_node(follower)
        follower.destroy_node()
        controller.destroy()
        planner.destroy()


class MockReadiness:
    def __init__(self, executor, name, ready):
        self.node = Node(f"mock_{name.replace('/', '_').strip('_')}")
        self.ready = ready
        self.server = self.node.create_service(Trigger, name, self._handle)
        executor.add_node(self.node)

    def _handle(self, _request, response):
        response.success = self.ready
        response.message = "ready" if self.ready else "not ready"
        return response

    def destroy(self):
        self.node.destroy_service(self.server)
        self.node.destroy_node()


def test_readiness_gate_fails_closed_until_services_are_ready(ros_context):
    planner_name = "/object_tracker_test/readiness_compute_path"
    controller_name = "/object_tracker_test/readiness_follow_path"
    slam = MockReadiness(ros_context, "/object_tracker_test/slam_ready", False)
    navigation = MockReadiness(ros_context, "/object_tracker_test/navigation_ready", True)
    planner = MockPlanner(ros_context, planner_name)
    controller = MockController(ros_context, controller_name)
    follower = DynamicTargetFollowerNode(
        parameter_overrides=[
            Parameter("enabled", value=True),
            Parameter("require_slam_readiness", value=True),
            Parameter("require_navigation_readiness", value=True),
            Parameter("slam_readiness_service", value="/object_tracker_test/slam_ready"),
            Parameter("navigation_readiness_service", value="/object_tracker_test/navigation_ready"),
            Parameter("compute_path_action", value=planner_name),
            Parameter("follow_path_action", value=controller_name),
            Parameter("readiness_poll_interval_s", value=0.0),
            Parameter("minimum_replan_interval_s", value=0.0),
        ]
    )
    follower._target_in_map = lambda state: (state.pose.pose.position.x, 0.0)
    follower._robot_position = lambda: (0.0, 0.0)
    ros_context.add_node(follower)

    try:
        follower._on_track_state(_track(follower, 2.0))
        follower._tick()
        _wait_for(lambda: not follower._readiness_pending)
        assert planner.goals == []
        assert not follower._slam_ready

        slam.ready = True
        follower._tick()
        _wait_for(lambda: not follower._readiness_pending)
        follower._tick()
        _wait_for(lambda: controller.events == ["follow-1"])
    finally:
        follower._stop_following("test complete")
        _wait_for(lambda: controller.events[-1:] == ["cancel-1"])
        ros_context.remove_node(follower)
        follower.destroy_node()
        controller.destroy()
        planner.destroy()
        navigation.destroy()
        slam.destroy()


def test_target_transform_uses_humble_pose_api(ros_context):
    follower = DynamicTargetFollowerNode()
    transform = TransformStamped()
    transform.header.frame_id = "map"
    transform.child_frame_id = "odom"
    transform.transform.translation.x = 1.0
    transform.transform.translation.y = 2.0
    transform.transform.rotation.w = 1.0
    follower._tf = SimpleNamespace(lookup_transform=lambda *_args: transform)

    try:
        x, y = follower._target_in_map(_track(follower, 2.0))
        assert x == pytest.approx(3.0)
        assert y == pytest.approx(2.0)
    finally:
        follower.destroy_node()


def test_mock_controller_goal_releases_its_worker_when_the_context_shuts_down(monkeypatch):
    """The execute callback must not outlive the ROS context.

    rclpy runs an action execute_callback on a ThreadPoolExecutor worker, and
    concurrent.futures joins those workers from an atexit hook with no timeout.
    A callback that only ends on cancellation pins a worker once its server is
    destroyed, because an orphaned goal handle never reports one - and the
    pytest process then hangs at interpreter exit with every test reported as
    passed. Assert the loop honours shutdown so that cannot regress silently.

    rclpy.ok is patched rather than actually shut down: the global context is
    shared with the other ROS test module in this package.
    """

    class _OrphanedHandle:
        """A goal handle whose server is gone: it never reports a cancellation."""

        is_cancel_requested = False
        aborted = False

        def abort(self):
            self.aborted = True

        def canceled(self):  # pragma: no cover - reaching this means the premise broke
            raise AssertionError("handle reported a cancellation it cannot receive")

    handle = _OrphanedHandle()
    context_ok = True
    monkeypatch.setattr(rclpy, "ok", lambda *args, **kwargs: context_ok)

    finished = threading.Event()
    worker = threading.Thread(target=lambda: (MockController._execute(handle), finished.set()), daemon=True)
    worker.start()

    assert not finished.wait(timeout=0.2), "callback returned before shutdown"

    context_ok = False
    assert finished.wait(timeout=5.0), "callback ignored shutdown and would hang the process at exit"
    assert handle.aborted
