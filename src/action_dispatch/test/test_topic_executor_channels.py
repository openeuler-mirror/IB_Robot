from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from action_dispatch.executors import topic as topic_executor_module
from action_dispatch.topic_executor import TopicExecutor


def test_execute_channel_publishes_only_selected_contract_topic(monkeypatch) -> None:
    assert TopicExecutor is topic_executor_module.TopicExecutor
    publishers: dict[str, MagicMock] = {}
    node = MagicMock()
    trace = MagicMock()
    monkeypatch.setattr(topic_executor_module, "trace", trace)

    def create_publisher(_message_type, topic, _qos):
        publisher = MagicMock()
        publishers[topic] = publisher
        return publisher

    node.create_publisher.side_effect = create_publisher
    specs = [
        SimpleNamespace(
            topic="/arm_controller/commands",
            ros_type="std_msgs/msg/Float64MultiArray",
            names=["action.0", "action.1"],
        ),
        SimpleNamespace(
            topic="/base_controller/commands",
            ros_type="std_msgs/msg/Float64MultiArray",
            names=["action.2", "action.3"],
        ),
    ]
    executor = TopicExecutor(node, {"action_specs": specs})
    assert executor.initialize()

    executor.execute_channel("/base_controller/commands", np.array([0.0, 0.0]))

    publishers["/arm_controller/commands"].publish.assert_not_called()
    message = publishers["/base_controller/commands"].publish.call_args.args[0]
    assert list(message.data) == [0.0, 0.0]
    trace.event.assert_called_once_with(
        "safe_stop_topic_publish",
        timestamp_ns=None,
        origin="built-in",
        topic="/base_controller/commands",
        values=2,
    )
    assert trace.event.call_args.args[0] != "action_topic_publish"


@pytest.mark.parametrize("trace_step", [False, True])
@pytest.mark.parametrize("tracing", [False, True])
def test_execute_publishes_once_and_traces_only_selected_steps(monkeypatch, trace_step, tracing) -> None:
    node = MagicMock()
    publisher = node.create_publisher.return_value
    trace = MagicMock()
    trace.enabled = tracing
    monkeypatch.setattr(topic_executor_module, "trace", trace)
    spec = SimpleNamespace(
        topic="/arm_controller/commands",
        ros_type="std_msgs/msg/Float64MultiArray",
        names=["action.0", "action.1"],
    )
    executor = TopicExecutor(node, {"action_specs": [spec]})
    assert executor.initialize()

    metadata = {"request_id": "req-1", "execute_index": 7, "queue_size": 2}
    with topic_executor_module.execution_trace("req-1", 0, 7, 2) if trace_step else nullcontext():
        assert executor.execute(np.array([1.0, 2.0]), metadata)
    assert metadata == {"request_id": "req-1", "execute_index": 7, "queue_size": 2}

    publisher.publish.assert_called_once()
    assert list(publisher.publish.call_args.args[0].data) == [1.0, 2.0]
    if tracing and trace_step:
        trace.event.assert_called_once_with(
            "action_topic_publish",
            timestamp_ns=None,
            origin="built-in",
            trace_id="req-1",
            execute_index=7,
            consumed_index=0,
            topic=spec.topic,
            values=2,
            queue_size=2,
        )
    else:
        trace.event.assert_not_called()


def test_execute_channel_rejects_wrong_vector_size() -> None:
    node = MagicMock()
    node.create_publisher.return_value = MagicMock()
    spec = SimpleNamespace(
        topic="/arm_controller/commands",
        ros_type="std_msgs/msg/Float64MultiArray",
        names=["action.0", "action.1"],
    )
    executor = TopicExecutor(node, {"action_specs": [spec]})
    executor.initialize()

    try:
        executor.execute_channel("/arm_controller/commands", np.array([0.0]))
    except ValueError as exc:
        assert "expects 2 values" in str(exc)
    else:
        raise AssertionError("execute_channel accepted a partial safety command")


def test_capture_does_not_rewrite_or_replace_business_exception():
    class BusinessError(Exception):
        def __setattr__(self, name, value):
            if name == "__traceback__":
                raise AssertionError("tracing must not rewrite business traceback")
            super().__setattr__(name, value)

    error = BusinessError("original")
    sink = MagicMock(enabled=True)
    sink.event.side_effect = ValueError("failed trace sink")
    with (
        pytest.raises(BusinessError) as caught,
        topic_executor_module.capture_traces(True),
        topic_executor_module.execution_trace("r", 0, 0, 1),
    ):
        topic_executor_module.capture_event(sink, "point", timestamp_ns=1, value=2)
        raise error
    assert caught.value is error
    sink.event.assert_called_once()
    assert topic_executor_module._trace_records.get() is None
    assert topic_executor_module._execution_trace.get() is None


def test_capture_is_bounded_and_does_not_format_objects():
    class Unsupported:
        def __str__(self):
            raise AssertionError("must not format business objects")

    sink = MagicMock(enabled=True)
    with topic_executor_module.capture_traces(True), topic_executor_module.capture_traces(True):
        for index in range(topic_executor_module._MAX_CAPTURE_RECORDS + 5):
            topic_executor_module.capture_event(sink, "point", timestamp_ns=index, value=Unsupported())
        sink.event.assert_not_called()
    assert sink.event.call_count == topic_executor_module._MAX_CAPTURE_RECORDS
    assert all("value" not in call.kwargs for call in sink.event.call_args_list)


def test_capture_preserves_late_id_and_drops_oversized_id():
    sink = MagicMock(enabled=True)
    fields = {f"extra_{i}": "x" for i in range(40)}
    fields["trace_id"] = "original"
    with topic_executor_module.capture_traces(True):
        topic_executor_module.capture_event(sink, "good", **fields)
        topic_executor_module.capture_event(sink, "bad", trace_id="prefix" * 200)
    sink.event.assert_called_once()
    assert sink.event.call_args.kwargs["trace_id"] == "original"


def test_disabled_capture_does_not_construct_scope_or_read_clock(monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("disabled capture must not initialize tracing")

    monkeypatch.setattr(topic_executor_module, "_TraceScope", unexpected)
    monkeypatch.setattr(topic_executor_module.time, "time_ns", unexpected)
    sink = MagicMock(enabled=False)
    with topic_executor_module.capture_traces(False):
        topic_executor_module.capture_event(sink, "ignored")
    sink.event.assert_not_called()


@pytest.mark.parametrize("dispatch_enabled,execute_enabled", [(True, False), (False, True), (True, True)])
def test_capture_considers_both_producers(monkeypatch, dispatch_enabled, execute_enabled):
    execute_sink = MagicMock(enabled=execute_enabled)
    dispatch_sink = MagicMock(enabled=dispatch_enabled)
    monkeypatch.setattr(topic_executor_module, "trace", execute_sink)
    with topic_executor_module.capture_traces(dispatch_enabled):
        topic_executor_module.capture_event(execute_sink, "publish")
        topic_executor_module.capture_event(dispatch_sink, "dispatch")
        execute_sink.event.assert_not_called()
        dispatch_sink.event.assert_not_called()
    assert execute_sink.event.call_count == int(execute_enabled)
    assert dispatch_sink.event.call_count == int(dispatch_enabled)


@pytest.mark.parametrize("error_type", [MemoryError, SystemError])
def test_capture_fatal_errors_propagate_outside_business_handler(monkeypatch, error_type):
    original = error_type("trace failure")
    sink = MagicMock(enabled=True)
    sink.event.side_effect = original
    classified = []
    with pytest.raises(error_type) as caught, topic_executor_module.capture_traces(True):
        try:
            topic_executor_module.capture_event(sink, "publish")
        except Exception:
            classified.append("business failed")
    assert caught.value is original and classified == []


@pytest.mark.parametrize("enabled", [True, False])
def test_submit_preserves_receipt_and_single_publication(monkeypatch, enabled):
    from action_dispatch.executors.base import ExecutionContext

    node = MagicMock()
    sink = MagicMock(enabled=enabled)
    monkeypatch.setattr(topic_executor_module, "trace", sink)
    spec = SimpleNamespace(topic="/commands", ros_type="std_msgs/msg/Float64MultiArray", names=["a", "b"])
    executor = TopicExecutor(node, {"action_specs": [spec]})
    executor.initialize()
    receipt = executor.submit(np.array([1.0, 2.0]), ExecutionContext(correlation_id="r"))
    assert receipt.accepted and receipt.correlation_id == "r" and receipt.immediate_completion is not None
    node.create_publisher.return_value.publish.assert_called_once()
    assert sink.event.call_count == int(enabled)
