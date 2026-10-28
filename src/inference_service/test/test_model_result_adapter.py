from types import SimpleNamespace

import pytest

from inference_service.pipeline import ModelResultAdapter


def test_model_result_adapter_returns_stored_result() -> None:
    result = object()
    frame = SimpleNamespace(values={"_model_result": result})

    assert ModelResultAdapter().adapt(frame) is result


def test_model_result_adapter_rejects_missing_result() -> None:
    with pytest.raises(RuntimeError, match="missing model result"):
        ModelResultAdapter().adapt(SimpleNamespace(values={}))


def test_model_result_adapter_reraises_original_cause() -> None:
    cause = ValueError("invalid model output")

    with pytest.raises(ValueError, match="invalid model output"):
        ModelResultAdapter().adapt_error(SimpleNamespace(cause=cause, message="wrapped"))


def test_model_result_adapter_uses_execution_message_without_cause() -> None:
    with pytest.raises(RuntimeError, match="model execution failed"):
        ModelResultAdapter().adapt_error(SimpleNamespace(cause=None, message="model execution failed"))
