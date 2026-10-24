"""Manifest identity contracts extracted from the later robot_config integration tests."""

from types import SimpleNamespace

import pytest

from inference_manifest import (
    BundleFile,
    RoleRuntimeProfile,
    canonical_bundle_digest,
    load_inference_manifest,
    write_inference_manifest,
)
from inference_manifest.runtime_identity import DEVICE_RESOLVERS, physical_device, runtime_composition_fingerprint


def _profile(device):
    return RoleRuntimeProfile.model_validate(
        {
            "backend": "torch",
            "target": {"runtime": "torch"},
            "profile": {"device": device},
        }
    )


def test_composite_placement_uses_one_accelerator_and_rejects_multiple_devices():
    validated = SimpleNamespace(
        runtime_profile=None, role_runtime_profiles={"helper": _profile("cpu"), "model": _profile("cuda")}
    )
    assert physical_device(validated, torch_device_index=0) == "local/cuda:0"
    validated.role_runtime_profiles["other"] = _profile("npu")
    with pytest.raises(ValueError, match="exactly one physical device"):
        physical_device(validated, torch_device_index=0)


@pytest.mark.parametrize("device", ["cuda", "npu"])
def test_torch_identity_requires_the_callers_resolved_device_index(device):
    validated = SimpleNamespace(runtime_profile=_profile(device), role_runtime_profiles={})
    for index in (None, -1, True, "1"):
        with pytest.raises(ValueError, match="explicit resolved torch_device_index"):
            physical_device(validated, torch_device_index=index)
    assert physical_device(validated, torch_device_index=0) == f"local/{device}:0"
    assert physical_device(validated, torch_device_index=1) == f"local/{device}:1"


@pytest.mark.parametrize(
    "profiles,message",
    [({}, "missing runtime profile"), ({"model": SimpleNamespace(backend="unknown")}, "unsupported.*unknown")],
)
def test_missing_or_unsupported_profiles_have_distinct_neutral_diagnostics(profiles, message):
    from inference_manifest import validate_runtime_backend_support

    for check in (physical_device, validate_runtime_backend_support):
        with pytest.raises(ValueError, match=message) as error:
            check(SimpleNamespace(runtime_profile=None, role_runtime_profiles=profiles))
        assert "scheduling" not in str(error.value)


def test_composition_identity_tracks_adapter_options_and_stage_but_not_key_order():
    options = {"precision": "fp16", "device_id": 0}
    identity = runtime_composition_fingerprint("example:Adapter", options, stage="encoder")
    assert (
        runtime_composition_fingerprint("example:Adapter", dict(reversed(options.items())), stage="encoder") == identity
    )
    assert runtime_composition_fingerprint("example:Other", options, stage="encoder") != identity
    assert runtime_composition_fingerprint("example:Adapter", {**options, "device_id": 1}, stage="encoder") != identity
    assert runtime_composition_fingerprint("example:Adapter", options, stage="decoder") != identity


def test_composition_v1_has_a_fixed_unicode_nested_payload_digest():
    from inference_manifest.runtime_identity import RUNTIME_COMPOSITION_FINGERPRINT_VERSION

    # Format v1 includes ASCII Unicode escapes, sorted nested keys and a stage.
    # Changing this vector requires an explicit format version decision.
    assert RUNTIME_COMPOSITION_FINGERPRINT_VERSION == 1
    options = {"nested": {"scale": 0.5, "labels": ["中文", None], "enabled": True}, "device_id": 2}
    assert runtime_composition_fingerprint("example:Adapter/v1", options, stage="编码") == (
        "fb1693e2bea40cb90536d4e1208a993170cf231fc4aba72f0a02701ed69f4dc0"
    )


@pytest.mark.parametrize("backend", DEVICE_RESOLVERS)
def test_every_resolver_produces_a_valid_physical_device(monkeypatch, backend):
    from inference_manifest import validate_runtime_backend_support, validate_runtime_device_visibility

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    cases = {
        "torch": [
            ({"device": device}, f"local/{device}" + (":2" if device in {"cuda", "npu"} else ""))
            for device in ("cpu", "mps", "cuda", "npu")
        ],
        "onnx": [({"provider": "cpu"}, "local/cpu"), ({"provider": "cuda"}, "local/cuda:0")],
        "ascend": [({"device_id": 2}, "local/npu:2")],
        "rknn": [({}, "local/rknn:0")],
        "hmm": [({"device_id": 2}, "local/hmm:2")],
        "hisilicon": [({}, "local/hisilicon:0")],
    }
    for settings, expected in cases[backend]:
        runtime = "acl" if backend == "ascend" else backend
        profile = RoleRuntimeProfile(backend=backend, target={"runtime": runtime}, profile=settings)
        placement = SimpleNamespace(runtime_profile=profile, role_runtime_profiles={})
        validate_runtime_backend_support(placement)
        device = physical_device(placement, torch_device_index=2)
        assert device == expected
        validate_runtime_device_visibility(device, "test/resolver")


@pytest.mark.parametrize("name,device", [("CUDA_VISIBLE_DEVICES", "cuda:0"), ("ASCEND_RT_VISIBLE_DEVICES", "npu:0")])
@pytest.mark.parametrize("value", ["", "0"])
def test_managed_device_visibility_rejects_even_empty_remapping(monkeypatch, name, device, value):
    from inference_manifest.runtime_identity import validate_runtime_device_visibility

    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=f"perception/demo.*{name}"):
        validate_runtime_device_visibility(f"local/{device}", "perception/demo")
    validate_runtime_device_visibility("local/cpu", "perception/cpu")


@pytest.mark.parametrize("device", ["cpu", "mps", "hmm:0", "rknn:0", "hisilicon:0", "cuda:0", "npu:0"])
def test_unrelated_visibility_variables_do_not_block_other_devices(monkeypatch, device):
    from inference_manifest.runtime_identity import validate_runtime_device_visibility

    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    for name in ("CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    if not device.startswith("cuda"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    if not device.startswith("npu"):
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")
    validate_runtime_device_visibility(f"local/{device}", "test")


@pytest.mark.parametrize(
    "device,provider,expected",
    [
        ("cpu", None, "local/cpu"),
        ("cuda", "CUDAExecutionProvider", "local/cuda:0"),
        (None, "CUDAExecutionProvider", "local/cuda:0"),
        (None, "cuda", "local/cuda:0"),
        (None, "cpu", "local/cpu"),
        ("cpu", "CUDAExecutionProvider", "local/cuda:0"),
        ("cuda", "cpu", "local/cpu"),
        ("cuda:1", None, None),
    ],
)
def test_onnx_physical_device_mapping(device, provider, expected):
    from inference_manifest.runtime_identity import physical_device

    profile = SimpleNamespace(
        backend="onnx", profile=SimpleNamespace(model_dump=lambda **kwargs: {"device": device, "provider": provider})
    )
    validated = SimpleNamespace(runtime_profile=profile, role_runtime_profiles={})
    if expected is None:
        with pytest.raises(ValueError, match="unambiguous"):
            physical_device(validated)
    else:
        assert physical_device(validated) == expected


@pytest.mark.parametrize(
    "declared,options,expected",
    [(None, None, 0), (None, {}, 0), (None, {"device_id": 2}, 2), (2, None, 2), (2, {}, 2), (2, {"device_id": 2}, 2)],
)
def test_hmm_effective_placement_includes_runtime_options(declared, options, expected):
    from inference_manifest.models import RoleRuntimeProfile
    from inference_manifest.runtime_identity import physical_device

    profile = RoleRuntimeProfile.model_validate(
        {
            "backend": "hmm",
            "target": {"runtime": "hmm"},
            "profile": {} if declared is None else {"device_id": declared},
        }
    )
    validated = SimpleNamespace(runtime_profile=profile, role_runtime_profiles={})
    assert physical_device(validated, options) == f"local/hmm:{expected}"


@pytest.mark.parametrize("option", [0, -1, True, "2"])
def test_hmm_identity_resolution_rejects_conflicting_or_invalid_device_options(option):
    from inference_manifest.models import RoleRuntimeProfile
    from inference_manifest.runtime_identity import physical_device

    profile = RoleRuntimeProfile.model_validate(
        {
            "backend": "hmm",
            "target": {"runtime": "hmm"},
            "profile": {"device_id": 2},
        }
    )
    with pytest.raises(ValueError, match="HMM device_id"):
        physical_device(SimpleNamespace(runtime_profile=profile, role_runtime_profiles={}), {"device_id": option})


@pytest.mark.parametrize("value", [b"x", {"x"}, object(), float("nan"), float("inf"), -float("inf"), {1: "x"}])
def test_composition_rejects_values_without_a_stable_json_identity(value):
    with pytest.raises(ValueError):
        runtime_composition_fingerprint("example:Adapter", {"nested": [value]})


def test_composition_rejects_paths_instead_of_colliding_with_strings():
    from pathlib import Path

    with pytest.raises(ValueError, match="JSON-safe"):
        runtime_composition_fingerprint("example:Adapter", {"path": Path("model")})
    assert runtime_composition_fingerprint("example:Adapter", {"path": "model"})


def test_mapping_support_is_distinct_from_single_device_placement():
    from inference_manifest.runtime_identity import validate_runtime_backend_support

    profiles = SimpleNamespace(
        runtime_profile=None, role_runtime_profiles={"a": _profile("cuda"), "b": _profile("npu")}
    )
    validate_runtime_backend_support(profiles)
    with pytest.raises(ValueError, match="exactly one physical device"):
        physical_device(profiles, torch_device_index=0)


@pytest.mark.parametrize("adapter", [None, 1, "", " ", "\t\n", "\u3000"])
def test_composition_rejects_missing_adapter_identity(adapter):
    with pytest.raises(ValueError, match="adapter"):
        runtime_composition_fingerprint(adapter, {})


def test_physical_device_accepts_property_based_profiles():
    class Placement:
        @property
        def runtime_profile(self):
            return RoleRuntimeProfile(backend="torch", target={"runtime": "torch"}, profile={"device": "cpu"})

        @property
        def role_runtime_profiles(self):
            return {}

    placement = Placement()
    assert physical_device(placement) == "local/cpu"


def test_device_apis_are_available_from_package():
    from inference_manifest import (
        physical_device,
        runtime_composition_fingerprint,
        validate_runtime_backend_support,
        validate_runtime_device_visibility,
    )

    assert all(
        callable(api)
        for api in (
            physical_device,
            runtime_composition_fingerprint,
            validate_runtime_backend_support,
            validate_runtime_device_visibility,
        )
    )


@pytest.mark.parametrize("order", [None, "", "FASTEST_FIRST", "PCI_BUS_ID"])
@pytest.mark.parametrize("backend", ["torch", "onnx"])
def test_cuda_identity_requires_uniform_enumeration_order(monkeypatch, order, backend):
    from inference_manifest import validate_runtime_device_visibility

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    if order is None:
        monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    else:
        monkeypatch.setenv("CUDA_DEVICE_ORDER", order)
    profile = RoleRuntimeProfile.model_validate(
        {
            "backend": backend,
            "target": {"runtime": "torch" if backend == "torch" else "onnxruntime"},
            "profile": {"device": "cuda"},
        }
    )
    device = physical_device(SimpleNamespace(runtime_profile=profile, role_runtime_profiles={}), torch_device_index=0)
    if order == "PCI_BUS_ID":
        validate_runtime_device_visibility(device, "test/cuda")
    else:
        with pytest.raises(ValueError, match="CUDA_DEVICE_ORDER=PCI_BUS_ID"):
            validate_runtime_device_visibility(device, "test/cuda")
    validate_runtime_device_visibility("local/cpu", "test/cpu")


@pytest.mark.parametrize("settings", [{}, {"device_id": 0}, {"device": "rknn:0", "device_id": 0}])
def test_rknn_identity_accepts_only_the_local_default_device(settings):
    profile = RoleRuntimeProfile(backend="rknn", target={"runtime": "rknn"}, profile=settings)
    placement = SimpleNamespace(
        runtime_profile=None,
        role_runtime_profiles={"helper": _profile("cpu"), "model": profile},
    )
    assert physical_device(placement) == "local/rknn:0"


@pytest.mark.parametrize("settings", [{"device_id": 1}, {"device": "rknn:1"}, {"device": "usb:0"}])
@pytest.mark.parametrize("composite", [False, True])
def test_rknn_identity_rejects_unsupported_device_declarations(settings, composite):
    profile = RoleRuntimeProfile(backend="rknn", target={"runtime": "rknn"}, profile=settings)
    default = RoleRuntimeProfile(backend="rknn", target={"runtime": "rknn"}, profile={})
    placement = SimpleNamespace(
        runtime_profile=None if composite else profile,
        role_runtime_profiles={"model": profile, "other": default} if composite else {},
    )
    with pytest.raises(ValueError, match="RKNN.*local.*device"):
        physical_device(placement)


@pytest.mark.parametrize("options", [[], "", [1], "device_id", 0, False])
def test_hmm_identity_rejects_non_mapping_runtime_options(options):
    profile = RoleRuntimeProfile(backend="hmm", target={"runtime": "hmm"}, profile={})
    placement = SimpleNamespace(runtime_profile=profile, role_runtime_profiles={})
    with pytest.raises(ValueError, match="runtime_options must be a mapping"):
        physical_device(placement, options)


@pytest.mark.parametrize(
    "device", ["local/cdua:0", "cuda:0", "local/cuda", "local/cuda:-1", "local/cuda:x", "local/cpu:0", "", None]
)
def test_visibility_rejects_invalid_device_names_before_environment_checks(monkeypatch, device):
    from inference_manifest import validate_runtime_device_visibility

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="runtime target demo.*invalid physical device"):
        validate_runtime_device_visibility(device, "demo")


def _load_runtime_placement(root, *, runtime_profile=None, role_runtime_profiles=None):
    root.mkdir(parents=True)
    (root / "payload.json").write_text("{}", encoding="utf-8")
    model = {
        "interface": "tensor_model",
        "model_type": "speech_direction",
        "operation": "enhance_and_vad",
        "inputs": [{"semantic": "audio", "dtype": "float32", "shape": [1, 4]}],
        "outputs": [{"semantic": "audio", "dtype": "float32", "shape": [1, 4]}],
    }
    tensor_binding = {"semantic": "audio", "index": 0, "dtype": "float32", "shape": [1, 4]}
    if role_runtime_profiles is None:
        (root / "model.onnx").write_bytes(b"model")
        deployment = {
            "uuid": "123e4567-e89b-42d3-a456-426614174001",
            "revision": 1,
            "execution_contract": {
                "state_scope": "request",
                "execution_structure": "direct",
                "cancellation_granularity": "request_boundary",
            },
            "runtime_profile": runtime_profile.model_dump(mode="json", exclude_none=True),
            "artifacts": {"model": {"path": "model.onnx", "format": "onnx"}},
            "execution": ["model"],
            "bindings": {"model": {"inputs": [tensor_binding], "outputs": [tensor_binding]}},
        }
    else:
        roles = ("enhancer", "vad")
        artifact_names = {"enhancer": "enhancer.onnx", "vad": "vad.onnx"}
        for path in artifact_names.values():
            (root / path).write_bytes(b"model")
        deployment = {
            "uuid": "123e4567-e89b-42d3-a456-426614174001",
            "revision": 1,
            "execution_contract": {
                "state_scope": "request",
                "execution_structure": "direct",
                "cancellation_granularity": "request_boundary",
            },
            "role_identities": {
                "enhancer": {"interface": "tensor_model", "model_type": "fullsubnet", "operation": "enhance"},
                "vad": {"interface": "tensor_model", "model_type": "silero_vad", "operation": "vad"},
            },
            "role_runtime_profiles": {
                role: role_runtime_profiles[role].model_dump(mode="json", exclude_none=True) for role in roles
            },
            "artifacts": {role: {"path": artifact_names[role], "format": "onnx"} for role in roles},
            "execution": list(roles),
            "bindings": {role: {"inputs": [tensor_binding], "outputs": [tensor_binding]} for role in roles},
        }

    bundle_files = (BundleFile(path="payload.json"),)
    bundle_uuid = "123e4567-e89b-42d3-a456-426614174000"
    bundle_name = root.name
    raw = {
        "schema_version": 3,
        "bundle": {
            "uuid": bundle_uuid,
            "revision": 1,
            "name": bundle_name,
            "files": [entry.model_dump(mode="json") for entry in bundle_files],
            "digest": {
                "algorithm": "sha256",
                "scope": "structure",
                "value": canonical_bundle_digest(bundle_uuid, 1, bundle_name, bundle_files),
            },
        },
        "model": model,
        "deployments": {"test": deployment},
    }
    write_inference_manifest(root / "inference_manifest.json", raw)
    return load_inference_manifest(root, "test")


def test_writer_loader_physical_device_integration_for_single_and_composite_deployments(tmp_path):
    from inference_manifest import validate_runtime_backend_support
    from inference_manifest.runtime_identity import physical_device

    single = _load_runtime_placement(tmp_path / "single", runtime_profile=_profile("cuda"))
    validate_runtime_backend_support(single)
    assert physical_device(single, torch_device_index=1) == "local/cuda:1"

    composite = _load_runtime_placement(
        tmp_path / "composite",
        role_runtime_profiles={"enhancer": _profile("cpu"), "vad": _profile("cuda")},
    )
    validate_runtime_backend_support(composite)
    assert composite.runtime_profile is None
    assert physical_device(composite, torch_device_index=0) == "local/cuda:0"

    multi_device = _load_runtime_placement(
        tmp_path / "multi-device",
        role_runtime_profiles={"enhancer": _profile("cuda"), "vad": _profile("npu")},
    )
    validate_runtime_backend_support(multi_device)
    with pytest.raises(ValueError, match="exactly one physical device"):
        physical_device(multi_device, torch_device_index=0)

    unsupported = _load_runtime_placement(
        tmp_path / "unsupported",
        role_runtime_profiles={
            "enhancer": _profile("cpu"),
            "vad": RoleRuntimeProfile(backend="tensorrt", target={"runtime": "tensorrt"}, profile={}),
        },
    )
    for check in (validate_runtime_backend_support, physical_device):
        with pytest.raises(ValueError, match="unsupported runtime backend.*tensorrt"):
            check(unsupported)
