from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from inference_manifest import (
    DeploymentTarget,
    ExecutionContract,
    RoleRuntimeProfile,
    TensorBinding,
    TorchDeployment,
    TorchRuntimeProfile,
    load_inference_manifest,
)
from model_utils.acl_abi_inspection import write_acl_om_abi
from model_utils.inference_manifest_export import (
    RuntimeABI,
    RuntimeTensor,
    artifact_bindings,
    compiled_deployment,
    copy_policy_metadata_bundle,
    package_deployment_artifact,
    read_runtime_abi,
    read_tcim_abi,
    refresh_bundle_revision,
    upsert_deployment,
)
from model_utils.package_compiled_deployment import package_compiled_deployment
from model_utils.package_torch_deployment import package_torch_deployments


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _create_bundle(root: Path, policy_type: str = "act") -> None:
    _write_json(
        root / "config.json",
        {
            "type": policy_type,
            "input_features": {
                "observation.state": {"type": "STATE", "shape": [6]},
                "observation.images.top": {"type": "VISUAL", "shape": [3, 16, 24]},
            },
            "output_features": {"action": {"type": "ACTION", "shape": [6]}},
        },
    )
    _write_json(root / "policy_preprocessor.json", {"name": "pre", "steps": []})
    _write_json(root / "policy_postprocessor.json", {"name": "post", "steps": []})


def _act_abi() -> RuntimeABI:
    return RuntimeABI(
        inputs=(
            RuntimeTensor("state", 0, "float32", (1, 6)),
            RuntimeTensor("image", 1, "float32", (1, 3, 16, 24)),
        ),
        outputs=(RuntimeTensor("action", 0, "float32", (1, 4, 6)),),
    )


def _act_bindings():
    return artifact_bindings(
        _act_abi(),
        input_semantics={"state": "observation.state", "image": "observation.images.top"},
        output_semantics={"action": "action"},
        image_layouts={"observation.images.top": "NCHW"},
    )


def _torch_deployment(device: str = "cpu") -> TorchDeployment:
    return TorchDeployment(
        execution_contract=ExecutionContract(
            state_scope="request",
            execution_structure="direct",
            cancellation_granularity="request_boundary",
        ),
        runtime_profile=RoleRuntimeProfile(
            backend="torch",
            target=DeploymentTarget(runtime="torch"),
            profile=TorchRuntimeProfile(device=device),
        ),
    )


def test_upsert_deployment_preserves_existing_deployments_and_structural_identity(tmp_path):
    _create_bundle(tmp_path)
    artifact = tmp_path / "artifacts" / "policy.rknn"
    artifact.parent.mkdir()
    artifact.write_bytes(b"rknn")
    torch_deployment = _torch_deployment()
    with pytest.raises(ValueError, match="model.safetensors"):
        upsert_deployment(tmp_path, "cpu", torch_deployment)

    (tmp_path / "model.safetensors").write_bytes(b"weights")
    upsert_deployment(tmp_path, "cpu", torch_deployment)
    first = load_inference_manifest(tmp_path, "cpu")
    deployment = compiled_deployment(
        tmp_path,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (artifact, "rknn")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )
    validated = upsert_deployment(tmp_path, "rknn", deployment)

    assert set(validated.manifest.deployments) == {"cpu", "rknn"}
    assert "model.safetensors" in {entry.path for entry in validated.manifest.bundle.files}
    assert load_inference_manifest(tmp_path, "cpu").deployment.backend == "torch"
    assert validated.manifest.bundle.uuid == first.manifest.bundle.uuid
    assert validated.manifest.bundle.revision == first.manifest.bundle.revision
    assert validated.deployment.uuid != first.deployment.uuid


def test_upsert_deployment_preserves_uuid_and_automatically_revises_structural_changes(tmp_path):
    _create_bundle(tmp_path)
    artifact = tmp_path / "policy.rknn"
    artifact.write_bytes(b"rknn-v1")
    deployment = compiled_deployment(
        tmp_path,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (artifact, "rknn")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )

    first = upsert_deployment(tmp_path, "rknn", deployment)
    no_op = upsert_deployment(tmp_path, "rknn", deployment)
    changed_profile = deployment.runtime_profile.model_copy(
        update={"target": deployment.runtime_profile.target.model_copy(update={"runtime": "rknn-lite2-2.3"})}
    )
    changed = upsert_deployment(tmp_path, "rknn", deployment.model_copy(update={"runtime_profile": changed_profile}))

    assert no_op.deployment.uuid == first.deployment.uuid
    assert no_op.deployment.revision == first.deployment.revision
    assert changed.deployment.uuid == first.deployment.uuid
    assert changed.deployment.revision == first.deployment.revision + 1
    assert changed.fingerprint != first.fingerprint


def test_refresh_bundle_revision_preserves_uuid_and_changes_deployment_fingerprint(tmp_path):
    _create_bundle(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    first = upsert_deployment(tmp_path, "cpu", _torch_deployment())
    config_path = tmp_path / "config.json"
    config_path.write_text(config_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    refreshed = refresh_bundle_revision(tmp_path)
    second = load_inference_manifest(tmp_path, "cpu")

    assert refreshed.bundle.uuid == first.manifest.bundle.uuid
    assert refreshed.bundle.revision == first.manifest.bundle.revision + 1
    assert refreshed.bundle.digest.value != first.manifest.bundle.digest.value
    assert second.deployment.uuid == first.deployment.uuid
    assert second.deployment.revision == first.deployment.revision
    assert second.fingerprint != first.fingerprint


def test_package_torch_deployments_generates_cpu_and_cuda_by_default(tmp_path):
    _create_bundle(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    validated = package_torch_deployments(tmp_path)

    assert [item.deployment_name for item in validated] == ["torch-cpu", "torch-cuda"]
    assert [item.deployment.device for item in validated] == ["cpu", "cuda"]
    manifest = load_inference_manifest(tmp_path, "torch-cpu").manifest
    assert set(manifest.deployments) == {"torch-cpu", "torch-cuda"}


def test_package_torch_deployments_supports_device_selection_and_prefix(tmp_path):
    _create_bundle(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    validated = package_torch_deployments(tmp_path, devices=("cpu",), deployment_prefix="native")

    assert validated[0].deployment_name == "native-cpu"
    assert validated[0].deployment.device == "cpu"


@pytest.mark.parametrize("policy_type", ["act", "pi05"])
def test_native_devices_share_one_model_descriptor_without_selector(tmp_path, policy_type):
    _create_bundle(tmp_path, policy_type=policy_type)
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    package_torch_deployments(tmp_path, devices=("cpu", "cuda"))
    result = package_torch_deployments(tmp_path, devices=("npu",))[0]

    assert set(result.manifest.deployments) == {"torch-cpu", "torch-cuda", "torch-npu"}
    assert result.manifest.model.model_type == policy_type
    assert result.manifest.model.architecture_class is None


def test_package_deployment_artifact_reuses_identical_immutable_generation(tmp_path):
    source = tmp_path / "policy.rknn"
    source.write_bytes(b"rknn")

    packaged = package_deployment_artifact(
        tmp_path,
        source,
        backend="rknn",
        deployment_name="rk3588",
        role="policy",
        force_copy=True,
    )

    assert packaged != source
    assert packaged.read_bytes() == b"rknn"
    assert packaged.relative_to(tmp_path).parts[:4] == ("artifacts", "rknn", "rk3588", "generations")
    assert packaged.name == "policy.rknn"
    repackaged = package_deployment_artifact(
        tmp_path,
        packaged,
        backend="rknn",
        deployment_name="rk3588",
        role="policy",
    )
    assert repackaged == packaged

    source.chmod(0o755)
    executable = package_deployment_artifact(
        tmp_path,
        source,
        backend="rknn",
        deployment_name="rk3588-executable",
        role="policy",
    )
    assert executable != packaged
    assert executable.stat().st_mode & 0o111


def test_package_deployment_artifact_prefers_target_deployments_current_generation(tmp_path):
    _create_bundle(tmp_path)
    older = tmp_path / "artifacts/rknn/older/generations/000/policy.rknn"
    current = tmp_path / "artifacts/rknn/target/generations/999/policy.rknn"
    source = tmp_path / "source.rknn"
    for path in (older, current, source):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"same")
    deployment = compiled_deployment(
        tmp_path,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (current, "rknn")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )
    upsert_deployment(tmp_path, "target", deployment)

    packaged = package_deployment_artifact(
        tmp_path,
        source,
        backend="rknn",
        deployment_name="target",
        role="policy",
    )

    assert packaged == current


def test_artifact_binding_requires_complete_semantic_and_layout_mapping():
    with pytest.raises(ValueError, match="No semantic mapping"):
        artifact_bindings(
            _act_abi(),
            input_semantics={"state": "observation.state"},
            output_semantics={"action": "action"},
        )
    with pytest.raises(ValueError, match="explicit runtime layout"):
        artifact_bindings(
            _act_abi(),
            input_semantics={"state": "observation.state", "image": "observation.images.top"},
            output_semantics={"action": "action"},
        )


def test_read_runtime_abi_rejects_noncontiguous_indices(tmp_path):
    metadata = tmp_path / "abi.json"
    _write_json(
        metadata,
        {
            "inputs": [{"name": "state", "index": 1, "dtype": "float32", "shape": [1, 6]}],
            "outputs": [{"name": "action", "index": 0, "dtype": "float32", "shape": [1, 4, 6]}],
        },
    )

    with pytest.raises(ValueError, match="contiguous from zero"):
        read_runtime_abi(metadata)


def test_read_runtime_abi_accepts_sparse_output_indices(tmp_path):
    metadata = tmp_path / "abi.json"
    _write_json(
        metadata,
        {
            "inputs": [{"name": "state", "index": 0, "dtype": "float32", "shape": [1, 6]}],
            "outputs": [{"name": "action", "index": 3, "dtype": "float32", "shape": [1, 4, 6]}],
        },
    )

    abi = read_runtime_abi(metadata)

    assert abi.outputs[0].index == 3


def test_write_acl_om_abi_lazily_uses_runtime_descriptor(tmp_path, monkeypatch):
    model = tmp_path / "model.om"
    model.write_bytes(b"om")
    calls = []

    class FakeRuntime:
        @staticmethod
        def set_device(device_id):
            calls.append(("set_device", device_id))
            return 0

        @staticmethod
        def reset_device(device_id):
            calls.append(("reset_device", device_id))
            return 0

        @staticmethod
        def create_context(device_id):
            calls.append(("create_context", device_id))
            return "context", 0

        @staticmethod
        def set_context(context):
            calls.append(("set_context", context))
            return 0

        @staticmethod
        def destroy_context(context):
            calls.append(("destroy_context", context))
            return 0

    class FakeModel:
        names = {"input": ["image"], "output": ["action"]}
        shapes = {"input": [[1, 3, 16, 24]], "output": [[1, 2, 8]]}
        dtypes = {"input": [1], "output": [0]}

        @staticmethod
        def load_from_file(path):
            calls.append(("load", path))
            return 7, 0

        @staticmethod
        def create_desc():
            return object()

        @staticmethod
        def get_desc(desc, model_id):
            return 0

        @classmethod
        def get_num_inputs(cls, desc):
            return 1

        @classmethod
        def get_num_outputs(cls, desc):
            return 1

        @classmethod
        def get_input_name_by_index(cls, desc, index):
            return cls.names["input"][index]

        @classmethod
        def get_output_name_by_index(cls, desc, index):
            return cls.names["output"][index]

        @classmethod
        def get_input_dims(cls, desc, index):
            return {"dims": cls.shapes["input"][index]}, 0

        @classmethod
        def get_output_dims(cls, desc, index):
            return {"dims": cls.shapes["output"][index]}, 0

        @classmethod
        def get_input_data_type(cls, desc, index):
            return cls.dtypes["input"][index]

        @classmethod
        def get_output_data_type(cls, desc, index):
            return cls.dtypes["output"][index]

        @staticmethod
        def destroy_desc(desc):
            calls.append(("destroy_desc",))

        @staticmethod
        def unload(model_id):
            calls.append(("unload", model_id))
            return 0

    class FakeACL:
        rt = FakeRuntime()
        mdl = FakeModel()

        @staticmethod
        def init(config_path=None):
            calls.append(("init", config_path))
            return 0

        @staticmethod
        def finalize():
            calls.append(("finalize",))
            return 0

    monkeypatch.setattr("model_utils.acl_abi_inspection.importlib.import_module", lambda name: FakeACL())

    output = write_acl_om_abi(model, tmp_path / "model.om.abi.json")
    abi = read_runtime_abi(output)

    assert abi.inputs == (RuntimeTensor("image", 0, "float16", (1, 3, 16, 24)),)
    assert abi.outputs == (RuntimeTensor("action", 0, "float32", (1, 2, 8)),)
    assert calls[0] == ("init", None)
    assert ("create_context", 0) in calls
    assert ("destroy_context", "context") in calls
    assert calls[-1] == ("finalize",)


def test_acl_config_path_is_limited_to_offline_abi_inspection(tmp_path, monkeypatch):
    calls = []

    class FakeACL:
        class rt:
            @staticmethod
            def set_device(_device_id):
                return 0

            @staticmethod
            def reset_device(_device_id):
                return 0

            @staticmethod
            def create_context(_device_id):
                return "context", 0

            @staticmethod
            def set_context(_context):
                return 0

            @staticmethod
            def destroy_context(_context):
                return 0

        class mdl:
            @staticmethod
            def load_from_file(_path):
                return 1, 0

            @staticmethod
            def create_desc():
                return object()

            @staticmethod
            def get_desc(_descriptor, _model_id):
                return 0

            @staticmethod
            def get_num_inputs(_descriptor):
                return 1

            @staticmethod
            def get_num_outputs(_descriptor):
                return 1

            @staticmethod
            def get_input_name_by_index(_descriptor, _index):
                return "input"

            @staticmethod
            def get_output_name_by_index(_descriptor, _index):
                return "output"

            @staticmethod
            def get_input_dims(_descriptor, _index):
                return {"dims": [1]}, 0

            @staticmethod
            def get_output_dims(_descriptor, _index):
                return {"dims": [1]}, 0

            @staticmethod
            def get_input_data_type(_descriptor, _index):
                return 0

            @staticmethod
            def get_output_data_type(_descriptor, _index):
                return 0

            @staticmethod
            def destroy_desc(_descriptor):
                return 0

            @staticmethod
            def unload(_model_id):
                return 0

        @staticmethod
        def init(config_path=None):
            calls.append(config_path)
            return 0

        @staticmethod
        def finalize():
            return 0

    model = tmp_path / "model.om"
    model.write_bytes(b"om")
    monkeypatch.setattr("model_utils.acl_abi_inspection.importlib.import_module", lambda _name: FakeACL)

    write_acl_om_abi(model, tmp_path / "abi.json", acl_config_path="inspection.json")

    assert calls == ["inspection.json"]


def test_write_acl_om_abi_explains_missing_acl_runtime(tmp_path, monkeypatch):
    om_path = tmp_path / "policy.om"
    om_path.write_bytes(b"om")

    def missing_acl(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("model_utils.acl_abi_inspection.importlib.import_module", missing_acl)

    with pytest.raises(RuntimeError, match="source the CANN environment or provide a pre-generated"):
        write_acl_om_abi(om_path, tmp_path / "policy.om.abi.json")


def test_runtime_image_layout_is_authoritative():
    abi = RuntimeABI(
        inputs=(RuntimeTensor("image", 0, "float32", (1, 16, 24, 3), "NHWC"),),
        outputs=(RuntimeTensor("action", 0, "float32", (1, 4, 6)),),
    )

    bindings = artifact_bindings(
        abi,
        input_semantics={"image": "observation.images.top"},
        output_semantics={"action": "action"},
    )

    assert bindings.inputs[0].layout == "NHWC"
    with pytest.raises(ValueError, match="runtime ABI reports NHWC"):
        artifact_bindings(
            abi,
            input_semantics={"image": "observation.images.top"},
            output_semantics={"action": "action"},
            image_layouts={"observation.images.top": "NCHW"},
        )


def test_upsert_deployment_restores_previous_manifest_on_strict_validation_failure(tmp_path):
    _create_bundle(tmp_path)
    artifact = tmp_path / "policy.rknn"
    artifact.write_bytes(b"rknn")
    valid = compiled_deployment(
        tmp_path,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (artifact, "rknn")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )
    upsert_deployment(tmp_path, "rknn", valid)
    original = (tmp_path / "inference_manifest.json").read_bytes()
    invalid_bindings = artifact_bindings(
        RuntimeABI(
            inputs=(RuntimeTensor("unknown", 0, "float32", (1, 6)),),
            outputs=(RuntimeTensor("action", 0, "float32", (1, 4, 6)),),
        ),
        input_semantics={"unknown": "observation.unknown"},
        output_semantics={"action": "action"},
    )
    invalid = compiled_deployment(
        tmp_path,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (artifact, "rknn")},
        execution=("policy",),
        bindings={"policy": invalid_bindings},
    )

    with pytest.raises(ValueError, match="unknown LeRobot input feature"):
        upsert_deployment(tmp_path, "rknn", invalid)

    assert (tmp_path / "inference_manifest.json").read_bytes() == original


def test_manifest_writer_preserves_existing_mode_and_uses_readable_default(tmp_path):
    _create_bundle(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    first = upsert_deployment(tmp_path, "cpu", _torch_deployment())
    path = first.manifest_path
    assert stat.S_IMODE(path.stat().st_mode) == 0o644

    path.chmod(0o640)
    upsert_deployment(tmp_path, "cuda", _torch_deployment("cuda"))
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_upsert_deployment_does_not_rehash_preserved_deployment_artifacts(tmp_path):
    _create_bundle(tmp_path)
    first_artifact = tmp_path / "first.rknn"
    first_artifact.write_bytes(b"first")
    first = compiled_deployment(
        tmp_path,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (first_artifact, "rknn")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )
    upsert_deployment(tmp_path, "first", first)
    first_artifact.write_bytes(b"changed")

    second_artifact = tmp_path / "second.om"
    second_artifact.write_bytes(b"second")
    second = compiled_deployment(
        tmp_path,
        backend="ascend",
        target_soc="Ascend310P3",
        target_runtime="acl",
        artifacts={"policy": (second_artifact, "om")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )

    validated = upsert_deployment(tmp_path, "second", second)
    assert set(validated.manifest.deployments) == {"first", "second"}


def test_upsert_deployment_rejects_external_semantic_dependencies(tmp_path):
    _create_bundle(tmp_path, "smolvla")
    preprocessor = json.loads((tmp_path / "policy_preprocessor.json").read_text(encoding="utf-8"))
    preprocessor["steps"] = [
        {
            "registry_name": "tokenizer_processor",
            "config": {"tokenizer_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"},
        }
    ]
    _write_json(tmp_path / "policy_preprocessor.json", preprocessor)
    artifact = tmp_path / "policy.rknn"
    artifact.write_bytes(b"rknn")
    deployment = compiled_deployment(
        tmp_path,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (artifact, "rknn")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )

    with pytest.raises(ValueError, match="all semantic dependencies to be local bundle assets"):
        upsert_deployment(tmp_path, "rknn", deployment)

    assert not (tmp_path / "inference_manifest.json").exists()


def test_package_hisilicon_requires_complete_runtime_abi_and_executable_worker(tmp_path):
    _create_bundle(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    model = artifacts / "policy.om"
    worker = artifacts / "worker"
    model.write_bytes(b"om")
    worker.write_text("#!/bin/sh\n", encoding="utf-8")
    worker.chmod(0o755)
    _write_json(
        tmp_path / "policy_abi.json",
        {
            "inputs": [
                {"name": "state", "index": 0, "dtype": "float32", "shape": [1, 6]},
                {"name": "image", "index": 1, "dtype": "float32", "shape": [1, 3, 16, 24]},
            ],
            "outputs": [{"name": "action", "index": 0, "dtype": "float32", "shape": [1, 4, 6]}],
        },
    )
    spec = tmp_path / "hisilicon.json"
    _write_json(
        spec,
        {
            "execution": ["policy"],
            "roles": {
                "policy": {
                    "artifact": "artifacts/policy.om",
                    "format": "om",
                    "abi": "policy_abi.json",
                    "input_semantics": {"state": "observation.state", "image": "observation.images.top"},
                    "output_semantics": {"action": "action"},
                    "image_layouts": {"observation.images.top": "NCHW"},
                }
            },
            "artifacts": {"worker": {"path": "artifacts/worker", "format": "executable"}},
        },
    )

    validated = package_compiled_deployment(
        bundle_root=tmp_path,
        deployment_name="sd3403",
        backend="hisilicon",
        target_soc="sd3403",
        target_runtime="hisilicon-worker",
        spec_path=spec,
    )

    assert validated.deployment.backend == "hisilicon"
    assert set(validated.deployment.artifacts) == {"policy", "worker"}
    assert load_inference_manifest(tmp_path, "sd3403").deployment.target.soc == "sd3403"

    worker.chmod(0o644)
    with pytest.raises(ValueError, match="not executable"):
        package_compiled_deployment(
            bundle_root=tmp_path,
            deployment_name="sd3403",
            backend="hisilicon",
            target_soc="sd3403",
            target_runtime="hisilicon-worker",
            spec_path=spec,
        )


def test_package_compiled_deployment_rejects_backend_target_mismatch(tmp_path):
    _create_bundle(tmp_path)
    artifact = tmp_path / "policy.om"
    artifact.write_bytes(b"om")
    _write_json(
        tmp_path / "policy_abi.json",
        {
            "inputs": [
                {"name": "state", "index": 0, "dtype": "float32", "shape": [1, 6]},
                {
                    "name": "image",
                    "index": 1,
                    "dtype": "float32",
                    "shape": [1, 3, 16, 24],
                    "layout": "NCHW",
                },
            ],
            "outputs": [{"name": "action", "index": 0, "dtype": "float32", "shape": [1, 4, 6]}],
        },
    )
    spec = tmp_path / "package.json"
    _write_json(
        spec,
        {
            "execution": ["policy"],
            "roles": {
                "policy": {
                    "artifact": "policy.om",
                    "format": "om",
                    "abi": "policy_abi.json",
                    "input_semantics": {"state": "observation.state", "image": "observation.images.top"},
                    "output_semantics": {"action": "action"},
                }
            },
        },
    )

    with pytest.raises(ValueError, match="RKNN deployment requires"):
        package_compiled_deployment(
            bundle_root=tmp_path,
            deployment_name="bad",
            backend="rknn",
            target_soc="rk3588",
            target_runtime="acl",
            spec_path=spec,
        )


def test_tensor_binding_type_is_strict():
    with pytest.raises(ValueError):
        TensorBinding(semantic="action", runtime_name="action", index=0, dtype="float32", shape=(1, 0))


def test_read_tcim_abi_uses_model_descriptors(tmp_path):
    metadata = tmp_path / "model.json"
    _write_json(
        metadata,
        {
            "Golden": {"inputs": [], "outputs": []},
            "Model": {
                "inputs": [{"name": "x_t", "shape": [1, 2, 8], "dtype": {"code": "float", "bits": 16}}],
                "outputs": [{"name": "v_t", "shape": [1, 2, 8], "dtype": {"code": "float", "bits": 16}}],
            },
        },
    )

    abi = read_tcim_abi(metadata)

    assert abi.inputs[0].dtype == "float16"
    assert abi.outputs[0].shape == (1, 2, 8)


def test_copy_policy_metadata_bundle_copies_only_required_semantic_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _create_bundle(source)
    (source / "model.safetensors").write_bytes(b"native")
    destination = tmp_path / "compiled"

    copied = copy_policy_metadata_bundle(source, destination)

    assert set(copied) == {"config.json", "policy_preprocessor.json", "policy_postprocessor.json"}
    assert (destination / "config.json").is_file()
    assert not (destination / "model.safetensors").exists()


def test_copy_policy_metadata_bundle_only_revises_changed_content(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "compiled"
    source.mkdir()
    destination.mkdir()
    _create_bundle(source)
    _create_bundle(destination)
    artifact = destination / "policy.rknn"
    artifact.write_bytes(b"rknn")
    deployment = compiled_deployment(
        destination,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (artifact, "rknn")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )
    first = upsert_deployment(destination, "rknn", deployment)

    copy_policy_metadata_bundle(source, destination)
    unchanged = load_inference_manifest(destination, "rknn")
    assert unchanged.manifest.bundle.revision == first.manifest.bundle.revision

    _write_json(source / "policy_preprocessor.json", {"name": "updated", "steps": []})
    copy_policy_metadata_bundle(source, destination)
    changed = load_inference_manifest(destination, "rknn")
    assert changed.manifest.bundle.revision == first.manifest.bundle.revision + 1
    assert json.loads((destination / "policy_preprocessor.json").read_text())["name"] == "updated"


def test_copy_policy_metadata_bundle_rolls_back_files_and_manifest_on_validation_failure(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "compiled"
    source.mkdir()
    destination.mkdir()
    _create_bundle(source)
    _create_bundle(destination)
    artifact = destination / "policy.rknn"
    artifact.write_bytes(b"rknn")
    deployment = compiled_deployment(
        destination,
        backend="rknn",
        target_soc="rk3588",
        target_runtime="rknn-lite2",
        artifacts={"policy": (artifact, "rknn")},
        execution=("policy",),
        bindings={"policy": _act_bindings()},
    )
    upsert_deployment(destination, "rknn", deployment)
    manifest_before = (destination / "inference_manifest.json").read_bytes()
    config_before = (destination / "config.json").read_bytes()
    config = json.loads((source / "config.json").read_text())
    config["input_features"].pop("observation.state")
    _write_json(source / "config.json", config)

    with pytest.raises(ValueError, match="unknown LeRobot input feature"):
        copy_policy_metadata_bundle(source, destination)

    assert (destination / "config.json").read_bytes() == config_before
    assert (destination / "inference_manifest.json").read_bytes() == manifest_before
