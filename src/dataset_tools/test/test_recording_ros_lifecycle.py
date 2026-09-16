"""Real DDS/action executor tests with a mock recording server; no hardware."""

import json
import os
import signal
import threading
import time
from types import SimpleNamespace

import pytest
import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

from dataset_tools.episode_recorder import EpisodeRecorderServer
from dataset_tools.record_cli import RecordCLI
from ibrobot_msgs.action import RecordEpisode
from ibrobot_msgs.msg import RuntimeStatus
from ibrobot_msgs.srv import SetRuntimeMode


@pytest.mark.parametrize("pending_acceptance", [False, True])
def test_real_sigint_cancels_owned_goal_including_pending_acceptance(pending_acceptance):
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    server_node = Node("mock_episode_server")
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(server_node)
    accepting = threading.Event()
    release = threading.Event()
    cancellations = []
    if not pending_acceptance:
        release.set()

    def accept(request):
        accepting.set()
        release.wait(2.0)
        return GoalResponse.ACCEPT

    def cancel(goal):
        cancellations.append(bytes(goal.goal_id.uuid))
        return CancelResponse.ACCEPT

    def execute(goal):
        deadline = time.monotonic() + 3.0
        while not goal.is_cancel_requested and time.monotonic() < deadline:
            time.sleep(0.01)
        goal.succeed()
        return RecordEpisode.Result(success=True, message="Stopped early (Saved)")

    server = ActionServer(
        server_node,
        RecordEpisode,
        "record_episode",
        execute_callback=execute,
        goal_callback=accept,
        cancel_callback=cancel,
        callback_group=ReentrantCallbackGroup(),
    )
    thread = threading.Thread(target=executor.spin)
    thread.start()
    cli = None
    other = None
    try:
        cli = RecordCLI()
        executor.add_node(cli)
        cli.send_goal("own goal")
        assert accepting.wait(2.0)
        if not pending_acceptance:
            assert cli._goal_started_evt.wait(2.0)
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGINT)
        assert rclpy.ok(), "SIGINT must not destroy the ROS context before cancel"
        timer = threading.Timer(0.05, release.set)
        timer.start()
        cli.finish_before_shutdown()
        timer.join()
        assert cli._episode_finished_evt.is_set()
        assert cli._last_result_success
        assert len(cancellations) == 1

        # No owned goal: cleanup must not cancel another client's recording.
        other = ActionClient(server_node, RecordEpisode, "record_episode")
        future = other.send_goal_async(RecordEpisode.Goal(prompt="other client"))
        assert cli._wait_for_future(future, 2.0)
        handle = future.result()
        assert handle.accepted
        cli.finish_before_shutdown()
        assert len(cancellations) == 1
        result = handle.get_result_async()
        cancel_future = handle.cancel_goal_async()
        assert cli._wait_for_future(cancel_future, 2.0)
        assert cli._wait_for_future(result, 2.0)
    finally:
        release.set()
        executor.shutdown(timeout_sec=5.0)
        thread.join(timeout=2.0)
        if other:
            other.destroy()
        if cli:
            cli.destroy_node()
        server.destroy()
        server_node.destroy_node()
        rclpy.shutdown()


@pytest.mark.parametrize("scenario", ["silent", "static", "loss", "slow_finalize", "shutdown"])
def test_real_recorder_discovery_admission_and_action_integrity(tmp_path, monkeypatch, scenario):
    from robot_config import loader

    contract = SimpleNamespace(
        name="test",
        robot_type="mock",
        observations=[],
        tasks=[],
        max_duration_s=10.0,
        recording={"storage": "sqlite3"},
        actions=[
            SimpleNamespace(publish_topic=topic, type="std_msgs/msg/Float64MultiArray", publish_qos={})
            for topic in ("/test_arm", "/test_gripper")
        ],
    )
    monkeypatch.setattr(loader, "load_robot_config_dict", lambda _: {})
    monkeypatch.setattr(loader, "build_contract_from_robot_config_dict", lambda _: contract)
    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            "robot_config_path:=/tmp/mock_recording.yaml",
            "-p",
            f"bag_base_dir:={tmp_path}",
            "-p",
            "control_mode:=teleop",
            "-p",
            "require_action_stream:=true",
            "-p",
            "action_stream_start_timeout_sec:=0.3",
            "-p",
            "action_stream_gap_timeout_sec:=0.15",
            "-p",
            "runtime_set_mode_service:=/test_mode",
            "-p",
            "teleop_rearm_service:=/test_rearm",
            "-p",
            "teleop_stop_service:=/test_stop",
            "-p",
            "runtime_status_topic:=/test_status",
        ],
        signal_handler_options=SignalHandlerOptions.NO,
    )
    runtime = Node("mock_runtime", use_global_arguments=False)
    status_pub = runtime.create_publisher(RuntimeStatus, "/test_status", 1)
    publishers = [runtime.create_publisher(Float64MultiArray, topic, 10) for topic in ("/test_arm", "/test_gripper")]
    state = {"mode": "stream", "publish": scenario != "silent"}
    calls = []
    finalizing = threading.Event()
    release_finalization = threading.Event()

    def mode(request, response):
        calls.append(request.mode)
        state["mode"] = request.mode
        response.success = True
        return response

    def rearm(request, response):
        calls.append("rearm")
        assert state["mode"] == "idle"
        state["mode"] = "stream"
        response.success = True
        return response

    def stop_session(_request, response):
        calls.append("stop")
        state["mode"] = "idle"
        response.success = True
        release_finalization.set()
        return response

    runtime.create_service(SetRuntimeMode, "/test_mode", mode)
    runtime.create_service(Trigger, "/test_rearm", rearm)
    runtime.create_service(Trigger, "/test_stop", stop_session)

    def publish():
        status_pub.publish(
            RuntimeStatus(
                lifecycle="ACTIVE",
                active_mode=state["mode"],
                stop_latched=False,
                stamp=runtime.get_clock().now().to_msg(),
            )
        )
        if state["mode"] == "stream" and state["publish"]:
            for pub in publishers:
                pub.publish(Float64MultiArray(data=[1.0]))

    runtime.create_timer(0.02, publish)
    recorder = EpisodeRecorderServer()
    if scenario == "slow_finalize":
        finalize = recorder._finalize_episode

        def slow_finalize(*args):
            finalizing.set()
            assert release_finalization.wait(5.0), "teleop stop must not wait for bag finalization"
            finalize(*args)

        monkeypatch.setattr(recorder, "_finalize_episode", slow_finalize)
    prior = recorder._episodes_dir / "episode_000001"
    prior.mkdir()
    (prior / "keep").write_text("prior valid recording")
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(runtime)
    executor.add_node(recorder)
    thread = threading.Thread(target=executor.spin)
    thread.start()
    cli = None
    try:
        # CLI uses only the defaults, just as a standalone ros2 run invocation.
        monkeypatch.setattr(RecordCLI, "__init__", _cli_without_global_overrides(RecordCLI.__init__))
        cli = RecordCLI()
        executor.add_node(cli)
        info = cli._info_client.call_async(Trigger.Request())
        assert cli._wait_for_future(info, 2.0)
        cli.configure_from_recorder_info(json.loads(info.result().message))
        cli.send_goal("test static arm")
        assert calls == ["idle", "rearm"]
        assert cli._goal_started_evt.wait(2.0)
        if scenario != "silent":
            deadline = time.monotonic() + 2.0
            while recorder._get_total_messages_written() < 8 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert recorder._get_total_messages_written() >= 8
        if scenario in {"static", "slow_finalize"}:
            cli.cancel_recording()
        elif scenario == "loss":
            state["publish"] = False
        if scenario == "slow_finalize":
            assert finalizing.wait(2.0)
        if scenario in {"slow_finalize", "shutdown"}:
            cli.set_parameters([Parameter("shutdown_timeout_sec", value=2.0)])
            cli.finish_before_shutdown()
            assert not cli._teleop_session_owned
            assert cli._episode_finished_evt.is_set()
        assert cli._episode_finished_evt.wait(3.0)
        assert cli._last_result_success == (scenario in {"static", "slow_finalize", "shutdown"})
        assert prior.exists()
        if scenario in {"static", "slow_finalize", "shutdown"}:
            assert recorder._last_episode_dir.exists()
            assert recorder._last_episode_messages >= 8
        else:
            assert "Required action stream" in cli._last_result_message
            assert list(recorder._episodes_dir.iterdir()) == [prior]
            assert recorder._last_episode_dir is None
        # Exiting the CLI stops the teleop session it armed and restores idle.
        cli.finish_before_shutdown()
        assert calls == ["idle", "rearm", "stop", "idle"]
        assert not cli._teleop_session_owned
    finally:
        release_finalization.set()
        if cli:
            cli.finish_before_shutdown()
        executor.shutdown(timeout_sec=5.0)
        thread.join(timeout=2.0)
        if cli:
            cli.destroy_node()
        recorder.destroy_node()
        runtime.destroy_node()
        rclpy.shutdown()


def _cli_without_global_overrides(init):
    def initialize(node):
        # Only the recorder gets launch parameters; default CLI discovers via DDS.
        original = Node.__init__
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(Node, "__init__", lambda self, name: original(self, name, use_global_arguments=False))
            init(node)

    return initialize


def test_required_action_stream_rejects_empty_contract(monkeypatch):
    from robot_config import loader

    monkeypatch.setattr(loader, "load_robot_config_dict", lambda _: {})
    monkeypatch.setattr(loader, "build_contract_from_robot_config_dict", lambda _: SimpleNamespace(actions=[]))
    rclpy.init(
        args=["--ros-args", "-p", "robot_config_path:=/tmp/mock_recording.yaml", "-p", "require_action_stream:=true"],
        signal_handler_options=SignalHandlerOptions.NO,
    )
    node = object.__new__(EpisodeRecorderServer)
    try:
        with pytest.raises(ValueError, match="at least one contract action"):
            node.__init__()
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_real_writer_finalization_failure_allows_next_goal(tmp_path, monkeypatch):
    from robot_config import loader

    contract = SimpleNamespace(
        name="test",
        robot_type="mock",
        observations=[],
        tasks=[],
        actions=[],
        max_duration_s=10.0,
        recording={"storage": "sqlite3"},
    )
    monkeypatch.setattr(loader, "load_robot_config_dict", lambda _: {})
    monkeypatch.setattr(loader, "build_contract_from_robot_config_dict", lambda _: contract)
    rclpy.init(
        args=["--ros-args", "-p", "robot_config_path:=/tmp/mock_recording.yaml", "-p", f"bag_base_dir:={tmp_path}"],
        signal_handler_options=SignalHandlerOptions.NO,
    )
    node = None
    try:
        node = EpisodeRecorderServer()
        transitions = []
        goal = SimpleNamespace(
            request=RecordEpisode.Goal(prompt="finalize failure"),
            is_cancel_requested=True,
            is_active=True,
            abort=lambda: transitions.append("abort"),
            succeed=lambda: transitions.append("succeed"),
        )

        def fail_finalize(*args):
            assert list(node._episodes_dir.glob("episode_*/*.db3"))
            assert node._ws.writer is None
            assert node.goal_callback(goal.request) == GoalResponse.REJECT
            assert transitions == []
            raise OSError("metadata disk failure")

        monkeypatch.setattr(node, "_finalize_episode", fail_finalize)
        assert node.goal_callback(goal.request) == GoalResponse.ACCEPT
        with pytest.raises(OSError, match="metadata disk failure"):
            node.execute_callback(goal)
        assert transitions == []
        assert node._current_goal_handle is None
        assert not node._flags.is_recording
        assert not node._goal_reserved
        assert node.goal_callback(RecordEpisode.Goal(prompt="retry")) == GoalResponse.ACCEPT
    finally:
        # Failed finalization leaves timers for the recorder's shutdown hook.
        rclpy.shutdown()
        if node is not None:
            node.destroy_node()
