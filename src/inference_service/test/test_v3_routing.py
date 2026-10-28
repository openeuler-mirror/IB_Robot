from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from inference_manifest import (
    AscendRuntimeProfile,
    Deployment,
    DeploymentTarget,
    ExecutionContract,
    ModelIdentity,
    RoleRuntimeProfile,
    TorchRuntimeProfile,
)
from inference_service.backends import STATIC_BACKEND_DESCRIPTORS, BackendRegistry
from inference_service.backends.registry import _validate_ascend
from inference_service.runtime_composition import build_model_service_runtime_dependencies
from inference_service.unified_runtime import SessionBuilderKey
from inference_service.unified_runtime.factory import RuntimeFactoryError, _canonical_target_runtime

_SRC_ROOT = Path(__file__).resolve().parents[2]
_STATIC_BACKEND_REGISTRY = BackendRegistry(STATIC_BACKEND_DESCRIPTORS)
for _package in ("perception_service", "voice_asr_service", "voice_tts_service"):
    sys.path.insert(0, str(_SRC_ROOT / _package))
try:
    importlib.import_module("cv2")
except ModuleNotFoundError:
    # The registration module only needs the name at import time; image
    # execution tests provide the real OpenCV dependency separately.
    sys.modules["cv2"] = ModuleType("cv2")


def test_production_session_builders_use_v3_keys() -> None:
    dependencies = build_model_service_runtime_dependencies()
    try:
        keys = set(dependencies.registry_set.session_builder_registry.keys)
        assert keys
        assert all(key.interface in {"policy", "tensor_model"} for key in keys)
        assert all(key.operation for key in keys)
        assert all(key.operation == "predict" for key in keys if key.interface == "policy")
        assert SessionBuilderKey("tensor_model", "zipvoice", "synthesize", "torch") in keys
        assert SessionBuilderKey("tensor_model", "fullsubnet", "enhance", "ascend") in keys
        assert SessionBuilderKey("tensor_model", "silero_vad", "vad", "ascend") in keys
        assert SessionBuilderKey("tensor_model", "graspgen", "generate_grasps", "ascend") in keys
        assert SessionBuilderKey("tensor_model", "fullsubnet", "enhance", "onnx") in keys
        assert SessionBuilderKey("tensor_model", "silero_vad", "vad", "onnx") in keys
        assert not any(key.interface in {"perception", "generic"} for key in keys)
    finally:
        dependencies.providers.close()


def test_backend_descriptors_publish_concrete_tensor_model_evidence() -> None:
    for descriptor in _STATIC_BACKEND_REGISTRY.descriptors.values():
        assert descriptor.supported_identities
        assert all(identity[0] in {"policy", "tensor_model"} for identity in descriptor.supported_identities)
        assert all(identity[2] for identity in descriptor.supported_identities)
        assert all(identity[0] in {"policy", "tensor_model"} for identity in descriptor.evidence_identities)

    assert ("tensor_model", "grounding_dino", "detect") in _STATIC_BACKEND_REGISTRY.descriptor(
        "torch"
    ).supported_identities
    assert ("tensor_model", "zipvoice", "synthesize") in _STATIC_BACKEND_REGISTRY.descriptor(
        "ascend"
    ).supported_identities
    assert ("tensor_model", "fullsubnet", "enhance") in _STATIC_BACKEND_REGISTRY.descriptor(
        "ascend"
    ).supported_identities
    assert ("tensor_model", "silero_vad", "vad") in _STATIC_BACKEND_REGISTRY.descriptor("onnx").supported_identities
    assert ("tensor_model", "speech_direction", "enhance_and_vad") in _STATIC_BACKEND_REGISTRY.descriptor(
        "onnx"
    ).supported_identities

    onnx_evidence = _STATIC_BACKEND_REGISTRY.descriptor("onnx").conformance_evidence
    assert any(item.session_type == "OnnxRuntimeModelSession" for item in onnx_evidence)
    assert all(item.target_runtimes == frozenset({"onnx"}) for item in onnx_evidence)

    torch_evidence = _STATIC_BACKEND_REGISTRY.descriptor("torch").conformance_evidence
    assert any(
        item.interface == "policy" and item.session_type == "LeRobotTorchModelSession" for item in torch_evidence
    )
    assert any(item.interface == "tensor_model" and item.session_type == "TorchModelSession" for item in torch_evidence)


def test_ascend_target_family_is_canonical_and_abi_is_separate() -> None:
    with pytest.raises(ValueError, match="canonical runtime family"):
        DeploymentTarget(runtime="acl-v2")
    with pytest.raises(ValueError, match="canonical runtime family"):
        DeploymentTarget(runtime="raw_acl")
    with pytest.raises(ValueError, match="backend='ascend'"):
        RoleRuntimeProfile(
            backend="ascend",
            target={"runtime": "ascend", "runtime_abi": "cann-8.0"},
            profile={"device_id": 0},
        )

    assert _canonical_target_runtime("acl", "ascend") == "acl"
    with pytest.raises(RuntimeFactoryError, match="target.runtime_abi"):
        _canonical_target_runtime("acl-v2", "ascend")
    assert "exactly 'acl'" in (_validate_ascend(SimpleNamespace(target=SimpleNamespace(runtime="acl-v2"))) or "")


def test_runtime_context_exposes_typed_profile_without_sdk_import() -> None:
    profile = RoleRuntimeProfile(
        backend="ascend",
        target={"soc": "Ascend310P", "runtime": "acl", "runtime_abi": "cann-8.0"},
        profile={"device_id": 3},
    )
    deployment = Deployment(
        execution_contract=ExecutionContract(
            state_scope="request",
            execution_structure="direct",
            cancellation_granularity="request_boundary",
        ),
        runtime_profile=profile,
    )
    identity = ModelIdentity(interface="tensor_model", model_type="silero_vad", operation="vad")
    validated = SimpleNamespace(
        deployment=deployment,
        deployment_name="ascend_310p",
        bundle_root=Path("/tmp"),
        role_runtime_profiles={},
        role_identities={},
        top_level_identity=identity,
        manifest=SimpleNamespace(model=identity),
    )

    from inference_service.backends import RuntimeContext

    context = RuntimeContext(validated)
    assert context.backend == "ascend"
    assert context.target_runtime == "acl"
    assert context.runtime_abi == "cann-8.0"
    assert context.device_id == 3
    assert isinstance(context.backend_profile, AscendRuntimeProfile)
    assert _STATIC_BACKEND_REGISTRY.validate(context).name == "ascend"

    torch_profile = TorchRuntimeProfile(device="cpu")
    assert torch_profile.backend == "torch"
