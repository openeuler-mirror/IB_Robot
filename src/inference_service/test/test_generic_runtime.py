from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from inference_service.unified_runtime import (
    ExecutionContext,
    ModelRequest,
    ModelResult,
    OutcomeEvidence,
    RuntimeLatency,
)


def test_native_request_and_result_snapshot_mappings() -> None:
    inputs = {"image": np.zeros((1, 3, 8, 8), dtype=np.float32)}
    metadata = {"source": "test"}
    request = ModelRequest(inputs, metadata)
    inputs["other"] = object()
    metadata["changed"] = True

    assert isinstance(request.inputs, MappingProxyType)
    assert "other" not in request.inputs
    assert request.metadata == {"source": "test"}
    with pytest.raises(TypeError):
        request.inputs["new"] = object()  # type: ignore[index]

    result = ModelResult(
        outputs={"tag_logits": np.ones((1, 4), dtype=np.float32), "boxes": []},
        latency=RuntimeLatency(total_ms=3.0, backend_ms=2.0),
        evidence=OutcomeEvidence.completed("backend"),
        metadata={"model_type": "ram_plus"},
    )
    assert set(result.outputs) == {"tag_logits", "boxes"}
    assert result.successful


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ModelRequest("invalid", {}),
        lambda: ModelRequest({}, {"bad": object()}),
        lambda: RuntimeLatency(total_ms=-1.0, backend_ms=0.0),
        lambda: ExecutionContext(""),
    ],
)
def test_native_contract_validation(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()
