from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from embodied_common.workflow_contracts import CanonicalWorkflowStep
from ibrobot_agent.contracts import Presentation, RequestKey, TaskRef
from ibrobot_agent.presentation import PresentationError, PresentationGate


def value():
    return Presentation(
        TaskRef("task", "plan", "pdig", "epoch", 1, "rdig", 1),
        1,
        (CanonicalWorkflowStep(1, "nod_yes"),),
        "immediate_after_presentation",
        30.0,
        "nod",
    )


def test_gate_requires_matching_display_receipt_and_rejects_stale_receipts():
    gate = PresentationGate(1.0)
    key = RequestKey("robot", "channel", "operator", "request")
    published, ready = {}, threading.Event()

    def publish(detail):
        published.update(detail)
        ready.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(gate.present, key, value(), publish=publish, cancel_token=threading.Event())
        assert ready.wait(0.5)
        assert not future.done()
        digest, token = published["presentation_digest"], published["receipt_token"]
        assert not gate.acknowledge(replace(key, request_id="other"), digest=digest, token=token)
        assert not gate.acknowledge(key, digest="wrong-plan", token=token)
        assert not gate.acknowledge(key, digest=digest, token="wrong-token")
        assert not future.done()
        assert gate.acknowledge(key, digest=digest, token=token)
        future.result(timeout=0.5)
        assert not gate.acknowledge(key, digest=digest, token=token)


def test_missing_client_times_out_and_cannot_be_released_late():
    gate = PresentationGate(0.02)
    key = RequestKey("robot", "channel", "operator", "request")
    published = {}
    with pytest.raises(PresentationError, match="timed out"):
        gate.present(key, value(), publish=published.update, cancel_token=threading.Event())
    assert not gate.acknowledge(key, digest=published["presentation_digest"], token=published["receipt_token"])


def test_stop_wins_over_display_receipt():
    gate = PresentationGate(1.0)
    key = RequestKey("robot", "channel", "operator", "request")
    cancel = threading.Event()

    def publish(detail):
        cancel.set()
        assert not gate.acknowledge(key, digest=detail["presentation_digest"], token=detail["receipt_token"])

    with pytest.raises(PresentationError, match="stopped"):
        gate.present(key, value(), publish=publish, cancel_token=cancel)


def test_transport_failure_does_not_leave_a_pending_presentation():
    gate = PresentationGate(1.0)
    key = RequestKey("robot", "channel", "operator", "request")

    def fail(detail):
        raise RuntimeError("transport unavailable")

    with pytest.raises(RuntimeError, match="transport unavailable"):
        gate.present(key, value(), publish=fail, cancel_token=threading.Event())
    assert not gate.acknowledge(key, digest="anything", token="anything")
