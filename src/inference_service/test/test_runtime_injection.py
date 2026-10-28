from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from inference_service.distributed import (
    CloudSession,
    EdgeSession,
    UnsupportedDistributedRuntimeError,
    build_pipeline_identity,
)
from inference_service.distributed.runtime import CloudBackendRuntime, EdgeProcessorRuntime
from inference_service.runtime_composition import build_policy_runtime_dependencies
from inference_service.unified_runtime import (
    ModelRuntimeFactory,
    RuntimeDependencyError,
    RuntimeFactoryError,
    RuntimeProviders,
)


def _policy_manifest() -> SimpleNamespace:
    return SimpleNamespace(
        manifest=SimpleNamespace(model=SimpleNamespace(interface="policy", model_type="act", operation="predict"))
    )


def _tensor_manifest() -> SimpleNamespace:
    return SimpleNamespace(
        manifest=SimpleNamespace(model=SimpleNamespace(interface="tensor_model", model_type="sam2", operation="prompt"))
    )


def test_policy_bootstrap_is_explicit_and_does_not_mutate_module_registries() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import inference_service.model_sessions as sessions; "
                "import inference_service.pipeline.factory as factory; "
                "assert not hasattr(sessions, 'MODEL_SESSION_BUILDER_REGISTRY'); "
                "assert not hasattr(factory, 'MODEL_SESSION_FACTORY_REGISTRY')"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr

    dependencies = build_policy_runtime_dependencies()
    try:
        assert dependencies.registry_set.frozen
        assert len(dependencies.registry_set.session_builder_registry.keys) == 11
        keys = dependencies.registry_set.runtime_assembler_registry.runtime_keys
        assert len(keys) == 15
        for model_type, backend in (("pi05", "ascend"), ("pi05", "hmm"), ("smolvla", "hmm"), ("smolvla", "rknn")):
            assert any(
                key.model_type == model_type
                and key.backend == backend
                and key.execution_contract == "request-iterative"
                and key.orchestration_visibility == "executor"
                for key in keys
            )
            assert any(
                key.model_type == model_type and key.backend == backend and key.execution_contract == "request-direct"
                for key in keys
            )
    finally:
        dependencies.providers.close()


def test_unified_factory_reports_missing_construction_dependencies() -> None:
    spec = SimpleNamespace()
    providers = RuntimeProviders.create(object(), object())

    with pytest.raises(RuntimeFactoryError) as missing_registry:
        ModelRuntimeFactory.create(spec, None, providers)
    assert missing_registry.value.code == "registry_set_required"

    with pytest.raises(RuntimeFactoryError) as missing_providers:
        ModelRuntimeFactory.create(spec, object(), None)
    assert missing_providers.value.code == "runtime_providers_required"


def test_public_legacy_construction_names_and_pipeline_fallback_are_absent() -> None:
    import inference_service.backends as backends
    import inference_service.model_sessions as model_sessions
    import inference_service.pipeline as pipeline

    assert not hasattr(backends, "InferenceBackend")
    assert not hasattr(backends, "BackendResult")
    assert not hasattr(model_sessions, "MODEL_SESSION_BUILDER_REGISTRY")
    assert not hasattr(pipeline, "ModelSessionFactoryRegistry")
    assert not hasattr(pipeline, "MODEL_SESSION_FACTORY_REGISTRY")

    from inference_service.pipeline import create_inference_pipeline

    with pytest.raises(RuntimeDependencyError) as error:
        create_inference_pipeline("missing-deps", object())
    assert error.value.code == "registry_set_required"


def test_distributed_entry_points_reject_tensor_model_before_wire_construction() -> None:
    tensor_manifest = _tensor_manifest()

    with pytest.raises(UnsupportedDistributedRuntimeError) as identity_error:
        build_pipeline_identity("tensor", tensor_manifest)
    assert identity_error.value.code == "distributed_tensor_model_unsupported"

    with pytest.raises(UnsupportedDistributedRuntimeError) as edge_error:
        EdgeProcessorRuntime("tensor", tensor_manifest)
    assert edge_error.value.code == "distributed_tensor_model_unsupported"

    with pytest.raises(UnsupportedDistributedRuntimeError) as cloud_error:
        CloudBackendRuntime("tensor", tensor_manifest)
    assert cloud_error.value.code == "distributed_tensor_model_unsupported"


def test_distributed_sessions_reject_tensor_model_without_changing_wire_types() -> None:
    identity = SimpleNamespace()
    with pytest.raises(UnsupportedDistributedRuntimeError):
        EdgeSession(identity, runtime_interface="tensor_model", runtime_model_type="sam2")
    with pytest.raises(UnsupportedDistributedRuntimeError):
        CloudSession(identity, runtime_interface="tensor_model", runtime_model_type="sam2")


def test_cloud_runtime_requires_both_injected_dependencies_for_policy_manifests() -> None:
    with pytest.raises(Exception) as error:
        CloudBackendRuntime("policy", _policy_manifest())
    assert getattr(error.value, "code", None) == "registry_set_required"
