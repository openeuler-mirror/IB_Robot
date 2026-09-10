import pytest

from inference_service.unified_runtime import ModelResult, OutcomeEvidence, RuntimeLatency
from manipulation_service.graspgen_wrapper import _backend_latency_ms


def _result(latency):
    return ModelResult(
        outputs={},
        latency=latency,
        evidence=OutcomeEvidence.completed("graspgen"),
    )


def test_backend_latency_uses_unified_runtime_latency_object():
    assert _backend_latency_ms(_result(RuntimeLatency(total_ms=20.0, backend_ms=12.5))) == pytest.approx(12.5)


def test_backend_latency_falls_back_to_total_latency_number():
    assert _backend_latency_ms(_result(20.0)) == pytest.approx(20.0)


def test_backend_latency_falls_back_to_total_when_backend_ms_missing():
    assert _backend_latency_ms(_result(RuntimeLatency(total_ms=20.0))) == pytest.approx(20.0)
