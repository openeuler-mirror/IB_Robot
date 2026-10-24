"""Unpackaged models must fail before runtime or plugin construction."""

import os
from unittest.mock import Mock

import pytest
import rclpy

from ibrobot_msgs.srv import EchoModel
from inference_manifest import ManifestValidationError
from inference_service.core.pure_inference_engine import PureInferenceEngine
from inference_service.model_service_node import ModelServiceNode
from inference_service.runtime_composition import build_runtime_dependencies


@pytest.fixture
def runtime_dependencies():
    dependencies = build_runtime_dependencies()
    try:
        yield dependencies
    finally:
        dependencies.providers.close()


@pytest.fixture
def unpackaged_model(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"unpackaged weights")
    return tmp_path


def test_unbundled_model_never_constructs_a_pipeline(unpackaged_model, monkeypatch, runtime_dependencies):
    dependencies = runtime_dependencies
    create_manager = Mock(side_effect=AssertionError("Unpackaged model reached pipeline construction"))
    monkeypatch.setattr("inference_service.core.pure_inference_engine.create_pipeline_manager", create_manager)

    with pytest.raises(ManifestValidationError, match="bundle_manifest_required"):
        PureInferenceEngine(
            unpackaged_model,
            "torch_cpu",
            registry_set=dependencies.registry_set,
            providers=dependencies.providers,
        )
    create_manager.assert_not_called()


@pytest.mark.parametrize("required", [False, True])
def test_unbundled_model_service_blocks_plugin_and_reports_failure(
    unpackaged_model, monkeypatch, required, runtime_dependencies
):
    dependencies = runtime_dependencies
    load_plugin = Mock(side_effect=AssertionError("Unpackaged model reached plugin import"))
    monkeypatch.setattr("inference_service.model_service_node._load_class", load_plugin)
    assert os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID"), "ROS node tests require an isolated test domain"
    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            f"bundle_path:={unpackaged_model}",
            "-p",
            "deployment:=torch_cpu",
            "-p",
            "adapter_class:=unused:Plugin",
            "-p",
            "service_type:=ibrobot_msgs/srv/EchoModel",
            "-p",
            "service_endpoint:=/bundle_startup_test/echo",
            "-p",
            f"required:={str(required).lower()}",
        ]
    )
    node = None
    try:
        if required:
            with pytest.raises(ManifestValidationError, match="bundle_manifest_required"):
                ModelServiceNode(registry_set=dependencies.registry_set, providers=dependencies.providers)
        else:
            node = ModelServiceNode(registry_set=dependencies.registry_set, providers=dependencies.providers)
            info = node.runtime_info()
            assert not info.ready
            assert info.runtime_state == "failed"
            assert "bundle_manifest_required" in info.failure_reason
            response = node._dispatch(EchoModel.Request(), EchoModel.Response())
            assert not response.success
            assert "bundle_manifest_required" in response.message
            assert not response.model.ready
        load_plugin.assert_not_called()
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
