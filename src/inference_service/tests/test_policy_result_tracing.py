from __future__ import annotations

import threading
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from ibrobot_msgs.action import DispatchInfer
from ibrobot_msgs.msg import VariantsList
from ibrobot_tracing.definitions import BUILTIN_TRACEPOINT_REGISTRY
from inference_service import pipeline_policy_node
from inference_service.pipeline_policy_node import PipelinePolicyNode
from tests.test_cpu_smoke_bundle import _TraceRecorder


@pytest.mark.parametrize("failure", [None, "admission", "inference", "canceled"])
@pytest.mark.parametrize("enabled", [False, True])
def test_legacy_result_send_is_paired_once_for_success_and_failure(monkeypatch, failure, enabled):
    recorder = _TraceRecorder(enabled=enabled)
    monkeypatch.setattr(pipeline_policy_node, "trace", recorder)
    terminals = []
    request = DispatchInfer.Goal()
    request.inference_id = "legacy-result"
    request.obs_timestamp.sec = 1
    goal = SimpleNamespace(
        request=request,
        is_cancel_requested=failure == "canceled",
        abort=lambda: terminals.append("abort"),
        succeed=lambda: terminals.append("succeed"),
        canceled=lambda: terminals.append("canceled"),
    )

    def infer(*args, **kwargs):
        if failure == "inference":
            raise RuntimeError("backend failed")
        return SimpleNamespace(
            action=np.zeros((1, 6), dtype=np.float32),
            actual_chunk_size=1,
            metadata={},
            backend_latency_ms=1.0,
            total_latency_ms=2.0,
        )

    def commit(goal, *_args):
        goal.succeed()
        return VariantsList()

    logger = SimpleNamespace(info=lambda *a, **kw: None, warning=lambda *a, **kw: None, error=lambda *a, **kw: None)
    node = SimpleNamespace(
        _config=SimpleNamespace(pipeline_id="policy", execution_mode="monolithic", request_timeout=1.0),
        _manifest=SimpleNamespace(fingerprint="sha256:test"),
        _goal_request_ids={},
        _goal_state_lock=threading.Lock(),
        _operation_lock=threading.Lock(),
        _reset_pending=threading.Event(),
        _cancel_requested_goals=set(),
        _cancel_confirmed_goals=set(),
        _completed_goals=set(),
        _inference_count=0,
        _goal_deadline=lambda deadline: None,
        _raise_if_deadline_expired=PipelinePolicyNode._raise_if_deadline_expired,
        _sample_observations=lambda *a, **kw: {},
        _to_policy_inputs=lambda inputs: inputs,
        _require_manager=lambda: SimpleNamespace(infer=infer),
        _commit_action=commit,
        _fail_distributed_after_deadline=lambda *args: None,
        get_logger=lambda: logger,
    )
    for name in (
        "_acquire_operation",
        "_goal_cancel_requested",
        "_goal_cancel_confirmed",
        "_finish_canceled_goal",
        "_execute_inference_request",
    ):
        setattr(node, name, MethodType(getattr(PipelinePolicyNode, name), node))
    if failure == "admission":
        node._reset_pending.set()

    result = PipelinePolicyNode._dispatch_infer_callback(node, goal)

    assert result.success is (failure is None)
    assert terminals == ["succeed" if failure is None else "canceled" if failure == "canceled" else "abort"]
    assert not node._goal_request_ids
    assert not node._operation_lock.locked()
    if not enabled:
        assert recorder.events == []
        return
    sends = [
        (name, fields) for kind, name, fields in recorder.events if kind == "flow_send" and name == "result_to_decode"
    ]
    assert len(sends) == 1
    assert sends[0][1] == {
        "flow_id": "legacy-result",
        "edge_id": "result_to_decode",
        "trace_id": "legacy-result",
        "request_id": "legacy-result",
        "component_id": "policy.postprocess",
        "pipeline_id": "policy",
    }
    results = [fields for kind, name, fields in recorder.events if kind == "dispatch_result"]
    assert len(results) == 1
    assert results[0]["success"] is result.success
    for kind, name, fields in recorder.events:
        if kind in {"span", "dispatch_result"}:
            identity_kind = "span" if kind == "span" else "event"
            assert (
                identity_kind,
                fields["component_id"],
                name,
                fields.get("origin", "built-in"),
            ) in BUILTIN_TRACEPOINT_REGISTRY
