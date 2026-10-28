from types import SimpleNamespace

import numpy as np
import pytest

from inference_service.backends.errors import BackendLoadError
from inference_service.model_sessions.lerobot_torch import LeRobotTorchModelSession


@pytest.mark.parametrize("device", ["npu", "cpu"])
def test_preprocessor_packs_host_views_only_for_npu(device):
    import torch

    session = LeRobotTorchModelSession(device)
    session._torch = torch
    session._preprocessor = lambda batch: batch
    image = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3).transpose(2, 0, 1)
    tensor = torch.from_numpy(image)
    contiguous = np.ones((1, 6), dtype=np.float32)
    scalar = np.array(1.0, dtype=np.float32)
    inputs = {"image": image, "tensor": tensor, "state": contiguous, "scalar": scalar, "task": "pick"}

    result = session.preprocess(inputs)

    np.testing.assert_array_equal(result["image"], image)
    torch.testing.assert_close(result["tensor"], tensor)
    assert result["image"].flags.c_contiguous == (device == "npu")
    assert result["tensor"].is_contiguous() == (device == "npu")
    assert result["state"] is contiguous
    assert result["scalar"] is scalar
    assert result["task"] == "pick"
    assert inputs["image"] is image and not image.flags.c_contiguous


def test_npu_device_name_uses_physical_sku() -> None:
    torch_module = SimpleNamespace(npu=SimpleNamespace(get_device_name=lambda index: f"Ascend310P{index + 1}"))

    assert LeRobotTorchModelSession._npu_device_name(torch_module) == "Ascend310P1"


def test_model_owned_input_hook_runs_before_lerobot_processor():
    session = LeRobotTorchModelSession("cpu")
    config = object()
    calls = []
    session._policy_config = config

    def transform(inputs, *, config):
        calls.append(("provider", config))
        return {**inputs, "resized": True}

    def processor(inputs):
        calls.append(("processor", inputs["resized"]))
        return inputs

    session._policy_provider = SimpleNamespace(preprocess_inputs=transform)
    session._preprocessor = processor
    source = {"task": "pick"}
    result = session.preprocess(source)

    assert result == {"task": "pick", "resized": True}
    assert source == {"task": "pick"}
    assert calls == [("provider", config), ("processor", True)]


def test_npu_device_name_fails_closed_without_api() -> None:
    with pytest.raises(BackendLoadError, match="does not expose get_device_name"):
        LeRobotTorchModelSession._npu_device_name(SimpleNamespace(npu=SimpleNamespace()))


def test_extra_parameters_are_passed_to_provider_without_mutating_manifest():
    original = {"arbitrary_future_model": {"values": [1, 2], "enabled": True}}
    config = object()
    token = object()
    expected_config = config

    def resolve(*, config, extra_parameters):
        assert config is expected_config
        assert extra_parameters == original
        extra_parameters["arbitrary_future_model"]["values"].append(3)
        return {"inference_config": token}

    options = LeRobotTorchModelSession._provider_load_options(
        SimpleNamespace(resolve_load_options=resolve), config, original
    )
    assert options == {"inference_config": token}
    assert original["arbitrary_future_model"]["values"] == [1, 2]


def test_extra_parameters_require_a_model_consumer():
    assert LeRobotTorchModelSession._provider_load_options(None, object(), {}) == {}
    with pytest.raises(BackendLoadError, match="does not support"):
        LeRobotTorchModelSession._provider_load_options(None, object(), {"solver": "ab2"})
    with pytest.raises(BackendLoadError, match="does not support"):
        LeRobotTorchModelSession._provider_load_options(SimpleNamespace(), object(), {"anything": 1})


def test_provider_extra_parameters_cannot_disable_strict_weight_loading():
    provider = SimpleNamespace(resolve_load_options=lambda **kwargs: {"strict": False})
    with pytest.raises(ValueError, match="cannot replace"):
        LeRobotTorchModelSession._provider_load_options(provider, object(), {"anything": 1})
