"""Required-action validation, episode isolation, and close/write race tests."""

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rclpy.action import GoalResponse
from rclpy.time import Time

from dataset_tools import episode_recorder
from dataset_tools.episode_recorder import (
    EpisodeRecorderServer,
    Flags,
    WriterState,
    _ActionStreamWatchdog,
    _TopicCounter,
)


def test_watchdog_checks_each_action_channel_and_immediate_finalization():
    watchdog = _ActionStreamWatchdog({"/arm", "/gripper"}, 2.0, 1.0, 0.0)
    watchdog.observe("/arm", 0.1)
    assert not watchdog.check(0.2)
    assert "/gripper: no commands" in watchdog.check(0.2, final=True)


def test_watchdog_checks_grace_and_latches_resumed_gap():
    watchdog = _ActionStreamWatchdog({"/arm"}, 2.0, 1.0, 0.0)
    assert not watchdog.check(1.9)
    assert "no commands" in watchdog.check(2.01)
    watchdog = _ActionStreamWatchdog({"/arm"}, 2.0, 1.0, 0.0)
    watchdog.observe("/arm", 0.1)
    watchdog.observe("/arm", 1.11)
    assert "command gap" in watchdog.check(1.11)
    watchdog.observe("/arm", 1.12)
    assert "command gap" in watchdog.check(1.12, final=True)


def test_watchdog_accepts_static_but_continuously_publishing_arm():
    watchdog = _ActionStreamWatchdog({"/arm", "/gripper"}, 2.0, 1.0, 0.0)
    for now in (0.1, 0.9, 1.8, 2.6, 3.5):
        watchdog.observe("/arm", now)
        watchdog.observe("/gripper", now)
    assert not watchdog.check(3.6, final=True)


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    node = object.__new__(EpisodeRecorderServer)
    node._flags = Flags()
    node._ws = WriterState(counts={"/arm": _TopicCounter(), "/gripper": _TopicCounter()})
    node._goal_reserved = False
    node._current_goal_handle = None
    node._require_action_stream = True
    node._action_start_timeout = 2.0
    node._action_gap_timeout = 1.0
    node._action_watchdog = None
    node._episode_done_evt = threading.Event()
    node._feedback_timer = None
    node._timeout_timer = None
    node._video_recording_coordinator = None
    node._cbg = None
    node._contract = SimpleNamespace(
        recording={}, max_duration_s=300.0, actions=[SimpleNamespace(publish_topic=t) for t in node._ws.counts]
    )
    node._topics = [(t, "std_msgs/msg/Float64MultiArray", {}) for t in node._ws.counts]
    node._episodes_dir = tmp_path
    prior = tmp_path / "episode_000001"
    prior.mkdir()
    (prior / "keep").write_text("previous episode")
    node._last_episode_dir = prior
    node._last_episode_index = 1
    node._last_episode_messages = 10
    node._last_episode_prompt = "previous"
    node.get_logger = lambda: Mock()
    node.get_clock = lambda: SimpleNamespace(now=lambda: Time(nanoseconds=100))
    node._write_episode_metadata = Mock()
    node._write_dataset_metadata = Mock()
    node._start_feedback_timer = Mock()
    writer = Mock()

    def open_writer(path, storage):
        Path(path).mkdir()
        return writer

    node._open_writer = open_writer
    node.create_subscription = lambda cls, topic, cb, qos, **kwargs: cb
    callbacks = {t: node._make_sub(t, typ, qos) for t, typ, qos in node._topics}
    now = [0.0]
    monkeypatch.setattr(episode_recorder.time, "monotonic", lambda: now[0])
    goal = SimpleNamespace(
        request=SimpleNamespace(prompt="new"), is_cancel_requested=False, is_active=True, abort=Mock(), succeed=Mock()
    )
    return node, callbacks, now, goal, writer, prior


@pytest.mark.parametrize("scenario", ["none", "partial", "lost", "resumed", "late_start"])
def test_invalid_episode_discard_preserves_prior_episode(recorder, scenario):
    node, callbacks, now, goal, writer, prior = recorder

    def record(_duration):
        if scenario != "none":
            if scenario == "late_start":
                now[0] = 2.1
            callbacks["/arm"](b"same position")
        if scenario not in {"none", "partial"}:
            callbacks["/gripper"](b"same position")
        if scenario in {"lost", "resumed"}:
            now[0] = 1.1
        if scenario == "resumed":
            callbacks["/arm"](b"same position")
            callbacks["/gripper"](b"same position")
        node._flags.stop_requested = True

    node._start_timeout_timer = record
    result = node.execute_callback(goal)
    assert not result.success
    assert "Required action stream" in result.message
    assert "discarded" in result.message
    assert prior.exists()
    assert list(prior.parent.iterdir()) == [prior]
    assert node._last_episode_dir is None
    assert node._ws.writer is None
    goal.abort.assert_called_once()
    goal.succeed.assert_not_called()


@pytest.mark.parametrize("stream_required", [False, True])
def test_normal_stop_keeps_valid_static_recording_and_sparse_inference(recorder, stream_required):
    node, callbacks, now, goal, writer, prior = recorder
    node._require_action_stream = stream_required

    def record(_duration):
        if stream_required:
            for tick in (0.1, 0.9, 1.8):
                now[0] = tick
                for cb in callbacks.values():
                    cb(b"unchanged position")
        else:
            now[0] = 100.0
        node._flags.stop_requested = True

    node._start_timeout_timer = record
    result = node.execute_callback(goal)
    assert result.success
    assert node._last_episode_dir.exists()
    assert node._last_episode_dir != prior
    goal.succeed.assert_called_once()
    goal.abort.assert_not_called()


def test_late_callback_during_finalization_cannot_repair_missing_stream(recorder):
    node, callbacks, now, goal, writer, prior = recorder
    node._start_timeout_timer = lambda _: setattr(node._flags, "stop_requested", True)

    def late_callback():
        for cb in callbacks.values():
            cb(b"too late")
        assert node.goal_callback(None) == GoalResponse.REJECT
        return True

    node._video_recording_coordinator = SimpleNamespace(
        start_episode=lambda _: None, is_recording=lambda: True, stop_episode=late_callback
    )
    result = node.execute_callback(goal)
    assert not result.success
    writer.write.assert_not_called()
    assert all(c.seen == 0 for c in node._ws.counts.values())
    assert prior.exists()


def test_write_failure_during_final_callback_cannot_be_reported_success(recorder):
    node, callbacks, now, goal, writer, prior = recorder
    writer.write.side_effect = OSError("disk failure")

    def record(_duration):
        for cb in callbacks.values():
            cb(b"command")

    node._start_timeout_timer = record
    result = node.execute_callback(goal)
    assert not result.success
    assert result.message == "Writer error"
    assert node._last_episode_dir is None


def test_goal_reserved_before_execute_prevents_two_writers(recorder):
    node, *_ = recorder
    assert node.goal_callback(None) == GoalResponse.ACCEPT
    assert node.goal_callback(None) == GoalResponse.REJECT


@pytest.mark.parametrize("failure_stage", ["finalize", "video_stop", "remove"])
def test_finalization_failure_releases_goal_admission(recorder, monkeypatch, failure_stage):
    node, _, _, goal, _, prior = recorder
    node._start_timeout_timer = lambda _: setattr(node._flags, "stop_requested", True)
    assert node.goal_callback(None) == GoalResponse.ACCEPT

    def fail(*args):
        assert node._flags.is_recording
        assert node._goal_reserved
        assert node._current_goal_handle is goal
        goal.abort.assert_not_called()
        goal.succeed.assert_not_called()
        raise OSError("finalization failed")

    if failure_stage == "finalize":
        monkeypatch.setattr(node, "_finalize_episode", fail)
    elif failure_stage == "video_stop":
        node._video_recording_coordinator = SimpleNamespace(
            start_episode=lambda _: None, is_recording=lambda: True, stop_episode=fail
        )
    else:
        monkeypatch.setattr(episode_recorder.shutil, "rmtree", fail)

    if failure_stage == "remove":
        result = node.execute_callback(goal)
        assert not result.success
        assert "failed to remove invalid episode" in result.message
        goal.abort.assert_called_once()
    else:
        with pytest.raises(OSError, match="finalization failed"):
            node.execute_callback(goal)
        goal.abort.assert_not_called()
    goal.succeed.assert_not_called()
    assert node._current_goal_handle is None
    assert node._ws.writer is None
    assert not node._flags.is_recording
    assert not node._goal_reserved
    assert prior.exists()
    assert node.goal_callback(None) == GoalResponse.ACCEPT


def test_monitor_aborts_silent_stream_without_user_stop(recorder):
    node, callbacks, now, goal, writer, prior = recorder
    node._start_timeout_timer = lambda _: now.__setitem__(0, 2.1)
    result = node.execute_callback(goal)
    assert not result.success
    assert "no commands" in result.message
    assert prior.exists()


def test_queued_old_timeout_cannot_stop_next_episode(recorder):
    node, *_ = recorder
    callbacks = []
    node.create_timer = lambda duration, cb, **kwargs: callbacks.append(cb) or SimpleNamespace(cancel=Mock())
    node._current_goal_handle = object()
    node._start_timeout_timer(1.0)
    node._current_goal_handle = object()
    node._flags.is_recording = True
    callbacks[0]()
    assert not node._flags.stop_requested
