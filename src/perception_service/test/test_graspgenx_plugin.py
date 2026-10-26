# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""GraspGenX served as a typed ``GenerateGrasps`` plugin over a packaged bundle.

The Torch device is faked so the plugin, the adapter, the session factory and the
``PointCloud2`` -> ``GraspCandidateArray`` translation run without a GPU. The final test
drives the official sampler on a real bundle when ``IBROBOT_GRASPGENX_BUNDLE`` names one.
"""

from __future__ import annotations

import importlib.util
import os
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import FakeAclRuntimeManager
from sensor_msgs.msg import PointCloud2, PointField

from inference_manifest import TorchRuntimeProfile, load_inference_manifest
from inference_service.backends import STATIC_BACKEND_DESCRIPTORS, BackendRegistry, ResourceDomainAdmissions
from inference_service.backends.types import RuntimeContext
from inference_service.model_sessions import TorchModelSession
from inference_service.runtime_composition import RuntimeDependencies
from inference_service.unified_runtime import (
    RegistrySet,
    RuntimeAssemblerRegistry,
    RuntimeProviders,
    SessionBuilderRegistry,
)
from perception_service.graspgenx_adapter import GraspGenXAdapter
from perception_service.model_service_plugins import GraspGenXGenerateGraspsPlugin
from perception_service.model_session_builders import build_graspgenx_session, register_perception_session_builders
from perception_service.package_graspgenx_torch_bundle import package_graspgenx_torch_bundle
from perception_service.torch_model_loaders import _GraspGenXModule

FRAME = "rgbd_head_front_color_optical_frame"


class _FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def empty_cache():
        return None


class _FakeTorch:
    """Just enough of ``torch`` for ``TorchModelSession`` to load and run on host arrays."""

    __version__ = "fake-2.7"
    cuda = _FakeCuda()

    @staticmethod
    def device(name):
        return name

    @staticmethod
    def is_tensor(_value):
        return False

    @staticmethod
    def as_tensor(value):
        return np.asarray(value)


class _FakeSampler:
    """Return grasps around the cloud centroid, the way the official sampler frames them."""

    def __init__(self):
        self.calls = []

    def __call__(self, inputs):
        points = np.asarray(inputs["observation.object_points"])
        count = int(np.asarray(inputs["observation.grasp_sample_count"]).reshape(-1)[0])
        self.calls.append((points, count))
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
        poses[:, :3, 3] = points.mean(axis=0)
        return {"grasp.poses": poses, "grasp.confidence": np.linspace(0.0, 1.0, count, dtype=np.float32)}


@pytest.fixture
def bundle(tmp_path) -> Path:
    release = tmp_path / "release"
    for network in ("gen", "dis"):
        (release / network).mkdir(parents=True)
        # data.num_points is the object-cloud budget the packager copies into
        # adapter.json, so a bundle cannot be built without it.
        (release / network / "config.yaml").write_text("data:\n  num_points: 3500\n", encoding="utf-8")
        (release / network / "epoch_1.pth").write_bytes(network.encode())
    gripper = tmp_path / "omnihand_left_x2"
    gripper.mkdir()
    (gripper / "config.json").write_text("{}", encoding="utf-8")
    root = tmp_path / "graspgenx_omnihand_left_x2"
    package_graspgenx_torch_bundle(root, checkpoint_root=release, gripper_dir=gripper)
    return root


@pytest.fixture
def dependencies():
    session_registry = SessionBuilderRegistry()
    register_perception_session_builders(session_registry)
    registry_set = RegistrySet(
        BackendRegistry(STATIC_BACKEND_DESCRIPTORS), session_registry, RuntimeAssemblerRegistry()
    ).freeze()
    providers = RuntimeProviders.create(FakeAclRuntimeManager(), ResourceDomainAdmissions())
    yield RuntimeDependencies(registry_set, providers)
    providers.close()


@pytest.fixture
def sampler():
    return _FakeSampler()


@pytest.fixture
def plugin(bundle, dependencies, sampler):
    class _Plugin(GraspGenXGenerateGraspsPlugin):
        _session_factory = staticmethod(
            lambda *_args: TorchModelSession(lambda _context: sampler, torch_loader=lambda: _FakeTorch)
        )

    instance = _Plugin(
        SimpleNamespace(bridge=None),
        load_inference_manifest(bundle, "torch_cuda"),
        {},
        registry_set=dependencies.registry_set,
        providers=dependencies.providers,
    )
    yield instance
    instance.close()


def _cloud(points: np.ndarray, frame_id: str = FRAME) -> PointCloud2:
    fields = [
        PointField(name=axis, offset=4 * index, datatype=PointField.FLOAT32, count=1)
        for index, axis in enumerate("xyz")
    ]
    data = b"".join(struct.pack("<3f", *point) for point in np.asarray(points, dtype=np.float32))
    cloud = PointCloud2(
        height=1,
        width=len(points),
        fields=fields,
        is_bigendian=False,
        point_step=12,
        row_step=12 * len(points),
        data=data,
        is_dense=True,
    )
    cloud.header.frame_id = frame_id
    return cloud


def _bottle(count: int = 600) -> np.ndarray:
    rng = np.random.default_rng(3)
    theta, height = rng.uniform(0, np.pi, count), rng.uniform(-0.1, 0.1, count)
    return np.stack([0.035 * np.cos(theta), height, 0.6 - 0.035 * np.sin(theta)], axis=1).astype(np.float32)


def _request(points=None, *, max_grasps=0, min_confidence=0.0, frame_id=FRAME):
    cloud = _cloud(_bottle() if points is None else points, frame_id)
    return SimpleNamespace(object_points=cloud, max_grasps=max_grasps, min_confidence=min_confidence)


def test_a_request_returns_the_top_k_source_frame_grasps_in_confidence_order(plugin, sampler):
    assert plugin.runtime_status().ready
    response = SimpleNamespace(grasps=None)

    message = plugin.handle(_request(), response)

    grasps = response.grasps.grasps
    assert message == "generated 400 grasps" and len(grasps) == 400
    assert sampler.calls[-1][1] == 400
    assert response.grasps.header.frame_id == FRAME and all(g.header.frame_id == FRAME for g in grasps)
    confidences = [g.confidence for g in grasps]
    assert confidences == sorted(confidences, reverse=True)
    translation = np.asarray(grasps[0].pose_matrix).reshape(4, 4)[:3, 3]
    np.testing.assert_allclose(translation, sampler.calls[-1][0].mean(axis=0), atol=1e-5)
    assert grasps[0].target_width_m == 0.0 and grasps[0].collision_free is False


def test_request_limits_and_confidence_floor_reach_the_sampler_and_the_decoder(plugin, sampler):
    larger, floored = SimpleNamespace(grasps=None), SimpleNamespace(grasps=None)

    # Above the bundle's own draw, so the request is what sets the sample count.
    plugin.handle(_request(max_grasps=700), larger)
    assert sampler.calls[-1][1] == 700 and len(larger.grasps.grasps) == 700

    plugin.handle(_request(max_grasps=5, min_confidence=0.99), floored)
    assert 0 < len(floored.grasps.grasps) <= 5
    assert min(g.confidence for g in floored.grasps.grasps) >= 0.99


def test_a_tiny_cloud_answers_with_no_grasps_without_running_the_model(plugin, sampler):
    response = SimpleNamespace(grasps=None)

    assert plugin.handle(_request(np.zeros((5, 3))), response) == "generated 0 grasps"
    assert response.grasps.grasps == [] and sampler.calls == []


def test_a_cloud_without_a_frame_is_refused_before_the_model(plugin, sampler):
    with pytest.raises(ValueError, match="frame_id is required"):
        plugin.handle(_request(frame_id=""), SimpleNamespace(grasps=None))
    assert sampler.calls == []


def test_the_session_factory_requires_a_cuda_torch_deployment_without_options(bundle):
    validated = load_inference_manifest(bundle, "torch_cuda")
    adapter = GraspGenXAdapter.from_bundle(bundle)

    assert isinstance(build_graspgenx_session(RuntimeContext(validated), adapter=adapter), TorchModelSession)
    with pytest.raises(RuntimeError, match="requires a typed cuda profile"):
        build_graspgenx_session(
            RuntimeContext(validated, runtime_profile=TorchRuntimeProfile(device="cpu")), adapter=adapter
        )
    with pytest.raises(ValueError, match="does not accept runtime options"):
        build_graspgenx_session(RuntimeContext(validated, runtime_options={"device_id": 0}), adapter=adapter)


def test_only_the_torch_backend_claims_the_graspgenx_identity(bundle):
    """GraspGenX is a canonical model type: Torch hosts it, no other backend claims it.

    Both its devices live under that one backend, so this says nothing about
    cuda versus npu -- the registry separates those through the evidence's
    device set, which test_graspgenx_npu_launch_environment covers.
    """
    from inference_manifest import CANONICAL_MODEL_MAPPING

    registry = BackendRegistry(STATIC_BACKEND_DESCRIPTORS)
    identity = ("tensor_model", "graspgenx", "generate_grasps")

    assert CANONICAL_MODEL_MAPPING["graspgenx"]["operations"] == ("generate_grasps",)
    registry.validate(RuntimeContext(load_inference_manifest(bundle, "torch_cuda")))
    assert [name for name, d in registry.descriptors.items() if identity in d.supported_identities] == ["torch"]


def test_the_module_draws_the_requested_count_without_the_samplers_own_outlier_pass():
    class Tensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def __len__(self):
            return len(self.value)

        def to(self, dtype):
            return self.value.astype(dtype)

        def reshape(self, *shape):
            return Tensor(self.value.reshape(*shape))

    calls = []

    def sample(points, **options):
        calls.append(options)
        return Tensor(np.empty((0, 4, 4))), Tensor(np.empty((0,))), None

    torch = SimpleNamespace(float32=np.float32, empty=lambda shape, dtype: Tensor(np.empty(shape, dtype=dtype)))
    module = _GraspGenXModule(SimpleNamespace(sample=sample), torch)

    outputs = module({"observation.object_points": np.ones((32, 3)), "observation.grasp_sample_count": [250]})

    assert calls == [{"threshold": -1.0, "num_grasps": 250, "remove_outliers": False}]
    assert outputs["grasp.poses"].shape == (0, 4, 4) and outputs["grasp.confidence"].shape == (0,)


_REAL_BUNDLE = os.environ.get("IBROBOT_GRASPGENX_BUNDLE", "")


@pytest.mark.skipif(
    not _REAL_BUNDLE or importlib.util.find_spec("graspgenx") is None,
    reason="set IBROBOT_GRASPGENX_BUNDLE to a packaged bundle and install graspgenx to run on the GPU",
)
def test_the_official_sampler_serves_a_real_bundle_on_cuda(dependencies):
    plugin = GraspGenXGenerateGraspsPlugin(
        SimpleNamespace(bridge=None),
        load_inference_manifest(Path(_REAL_BUNDLE), "torch_cuda"),
        {},
        registry_set=dependencies.registry_set,
        providers=dependencies.providers,
    )
    try:
        response = SimpleNamespace(grasps=None)
        plugin.handle(_request(_bottle(4000)), response)
        grasps = response.grasps.grasps
        assert len(grasps) == 100
        matrices = np.asarray([g.pose_matrix for g in grasps]).reshape(-1, 4, 4)
        assert np.isfinite(matrices).all()
        rotations = matrices[:, :3, :3]
        np.testing.assert_allclose(
            np.einsum("nji,njk->nik", rotations, rotations), np.eye(3)[None].repeat(100, 0), atol=2e-3
        )
        assert max(g.confidence for g in grasps) > 0.5
    finally:
        plugin.close()
