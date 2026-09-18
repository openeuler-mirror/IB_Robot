from types import SimpleNamespace

import pytest

from inference_service.backends.errors import BackendLoadError
from inference_service.model_sessions.lerobot_torch import LeRobotTorchModelSession


def test_npu_device_name_uses_physical_sku() -> None:
    torch_module = SimpleNamespace(npu=SimpleNamespace(get_device_name=lambda index: f"Ascend310P{index + 1}"))

    assert LeRobotTorchModelSession._npu_device_name(torch_module) == "Ascend310P1"


def test_npu_device_name_fails_closed_without_api() -> None:
    with pytest.raises(BackendLoadError, match="does not expose get_device_name"):
        LeRobotTorchModelSession._npu_device_name(SimpleNamespace(npu=SimpleNamespace()))
