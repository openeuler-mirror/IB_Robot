# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Shared GraspGen fixtures: one compiled export, one packaged bundle, one fake ACL.

The eight OM ABIs are transcribed from a real ``atc`` run - including the mangled
``PartitionedCall_...:0:features`` runtime names, which are what forced the manifest to
carry runtime names separately from semantics in the first place. The packager tests, the
session tests and the plugin tests all need them, and they need to agree, so they are
declared once here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from model_utils.graspgen_contract import GRASPGEN_CONTRACT_VERSION, GRASPGEN_EXECUTION, graspgen_geometry
from model_utils.inference_manifest_export import RuntimeABI, RuntimeTensor


def _bundle_snapshot(root):
    snapshot = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_symlink():
            snapshot[relative] = ("symlink", str(path.readlink()))
        elif path.is_file():
            snapshot[relative] = ("file", path.read_bytes())
        elif path.is_dir():
            snapshot[relative] = ("directory",)
        else:
            snapshot[relative] = ("special", path.lstat().st_mode)
    return snapshot


GRASPGEN_BATCH = 1000
GRASPGEN_POINTS = 2048
GRASPGEN_DEPLOYMENT = "ascend_310p"

GRASPGEN_ROLE_ABI: dict[str, RuntimeABI] = {
    "generator_sa1": RuntimeABI(
        inputs=(RuntimeTensor("grouped_features", 0, "float32", (1, 3, 256, 64)),),
        outputs=(RuntimeTensor("PartitionedCall_/ReduceMax_ReduceMax_1:0:features", 0, "float32", (1, 128, 256)),),
    ),
    "generator_sa2": RuntimeABI(
        inputs=(RuntimeTensor("grouped_features", 0, "float32", (1, 131, 64, 128)),),
        outputs=(RuntimeTensor("features", 0, "float32", (1, 256, 64)),),
    ),
    "generator_encoder_head": RuntimeABI(
        inputs=(RuntimeTensor("grouped_features", 0, "float32", (1, 259, 1, 64)),),
        outputs=(
            RuntimeTensor(
                "PartitionedCall_/prediction_head/prediction_head.4/Gemm_add_17:0:object_embedding",
                0,
                "float32",
                (1, 512),
            ),
        ),
    ),
    "discriminator_sa1": RuntimeABI(
        inputs=(RuntimeTensor("grouped_features", 0, "float32", (1, 3, 256, 64)),),
        outputs=(RuntimeTensor("features", 0, "float32", (1, 128, 256)),),
    ),
    "discriminator_sa2": RuntimeABI(
        inputs=(RuntimeTensor("grouped_features", 0, "float32", (1, 131, 64, 128)),),
        outputs=(RuntimeTensor("features", 0, "float32", (1, 256, 64)),),
    ),
    "discriminator_encoder_head": RuntimeABI(
        inputs=(RuntimeTensor("grouped_features", 0, "float32", (1, 259, 1, 64)),),
        outputs=(RuntimeTensor("object_embedding", 0, "float32", (1, 512)),),
    ),
    "denoiser": RuntimeABI(
        inputs=(
            RuntimeTensor("object_embedding", 0, "float32", (1, 512)),
            RuntimeTensor("sample", 1, "float32", (GRASPGEN_BATCH, 6)),
            RuntimeTensor("timestep", 2, "float32", (1,)),
        ),
        outputs=(
            RuntimeTensor(
                "PartitionedCall_/prediction_head/prediction_head.4/Gemm_add_130:0:predicted_noise",
                0,
                "float32",
                (GRASPGEN_BATCH, 6),
            ),
        ),
    ),
    "discriminator_head": RuntimeABI(
        inputs=(
            RuntimeTensor("object_embedding", 0, "float32", (1, 512)),
            RuntimeTensor("grasp_rt", 1, "float32", (GRASPGEN_BATCH, 6)),
        ),
        outputs=(
            RuntimeTensor(
                "PartitionedCall_/prediction_head/prediction_head.4/Gemm_add_24:0:logits",
                0,
                "float32",
                (GRASPGEN_BATCH, 1),
            ),
            RuntimeTensor("/Sigmoid:0:confidence", 1, "float32", (GRASPGEN_BATCH, 1)),
        ),
    ),
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def runtime_abi_json(abi: RuntimeABI) -> dict:
    def tensors(values):
        return [
            {"name": tensor.name, "index": tensor.index, "dtype": tensor.dtype, "shape": list(tensor.shape)}
            for tensor in values
        ]

    return {"inputs": tensors(abi.inputs), "outputs": tensors(abi.outputs)}


@dataclass
class GraspGenExport:
    """Everything ``graspgen-onnx-to-om`` leaves behind, plus an empty bundle root."""

    bundle: Path
    onnx_manifest: dict
    om_dir: Path
    om_abi_dir: Path


@pytest.fixture
def graspgen_export(tmp_path: Path) -> GraspGenExport:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    om_dir = tmp_path / "om"
    om_abi_dir = tmp_path / "om_abi"
    om_dir.mkdir()
    om_abi_dir.mkdir()
    for role in GRASPGEN_EXECUTION:
        (om_dir / f"{role}.om").write_bytes(f"{role}-om".encode())
        write_json(om_abi_dir / f"{role}.om.abi.json", runtime_abi_json(GRASPGEN_ROLE_ABI[role]))
    config_path = tmp_path / "graspgen_config.yml"
    generator_checkpoint = tmp_path / "generator_checkpoint.pth"
    discriminator_checkpoint = tmp_path / "discriminator_checkpoint.pth"
    config_path.write_text("data:\n  gripper_name: test_gripper\n  num_points: 2048\n", encoding="utf-8")
    generator_checkpoint.write_bytes(b"generator")
    discriminator_checkpoint.write_bytes(b"discriminator")
    return GraspGenExport(
        bundle=bundle,
        onnx_manifest={
            "schema_version": 1,
            "contract_version": GRASPGEN_CONTRACT_VERSION,
            "model_type": "graspgen",
            "backend": "onnx",
            "source": {
                "config": str(config_path),
                "generator_checkpoint": str(generator_checkpoint),
                "discriminator_checkpoint": str(discriminator_checkpoint),
            },
            "artifacts": {role: {"onnx": f"{role}.onnx"} for role in GRASPGEN_EXECUTION},
            "execution": list(GRASPGEN_EXECUTION),
            "backend_config": {
                "grasp_batch_size": GRASPGEN_BATCH,
                "point_count": GRASPGEN_POINTS,
                "grasp_repr": "r3_so3",
                "kappa": 2.02217,
                "diffusion_steps": 10,
                "compositional_scheduler": True,
                "geometry": graspgen_geometry(include_head_stage=True),
            },
        },
        om_dir=om_dir,
        om_abi_dir=om_abi_dir,
    )


def package_graspgen_export(export: GraspGenExport, **overrides) -> Path:
    """Run the real packager over a seeded export, so tests read a real manifest."""
    from perception_service.package_graspgen_ascend_bundle import write_graspgen_ascend_bundle

    kwargs = {
        "deployment_name": GRASPGEN_DEPLOYMENT,
        "om_dir": export.om_dir,
        "om_abi_dir": export.om_abi_dir,
        "soc_version": "Ascend310P3",
        "onnx_manifest": export.onnx_manifest,
        "grasp_batch_size": GRASPGEN_BATCH,
        "point_count": GRASPGEN_POINTS,
    }
    kwargs.update(overrides)
    return write_graspgen_ascend_bundle(export.bundle, **kwargs)


@pytest.fixture
def graspgen_bundle(graspgen_export: GraspGenExport) -> Path:
    package_graspgen_export(graspgen_export)
    return graspgen_export.bundle


class FakeAclLease:
    def __init__(self) -> None:
        self.close_calls = 0
        # The session reports its runtime version off the leased ACL module.
        self.acl = SimpleNamespace(__version__="fake-acl-1.0")

    def close(self) -> None:
        self.close_calls += 1


class FakeAclRuntimeManager:
    def __init__(self) -> None:
        self.lease = FakeAclLease()
        self.acquire_calls: list[int] = []

    def acquire(self, device_id: int):
        self.acquire_calls.append(device_id)
        return self.lease


class FakeAclModel:
    """One compiled role: records what it was fed, returns its declared output shapes.

    There is no 310P in CI, so the ACL layer is the seam every GraspGen test cuts at. What
    stays real above it is the manifest - each role's bindings come from the packager - so
    a wrong index, dtype or shape still fails, just without a device.
    """

    instances: dict[str, FakeAclModel] = {}
    order: list[str] = []

    def __init__(self, _lease, role, path, bindings) -> None:
        self.role = role
        self.path = path
        self.bindings = bindings
        self.calls: list[dict[int, np.ndarray]] = []
        self.read_outputs: set[int] = set()
        self.input_overrides: dict[int, object] = {}
        self.close_calls = 0
        self.__class__.instances[role] = self

    def load_descriptor(self) -> None:
        def descriptor(binding):
            return SimpleNamespace(size=int(np.prod(binding.shape)) * np.dtype(binding.dtype).itemsize)

        self.input_descriptors = tuple(descriptor(binding) for binding in self.bindings.inputs)
        self.output_descriptors = tuple(descriptor(binding) for binding in self.bindings.outputs)
        self.output_buffers: list[object] = []

    def prepare_datasets(self, *, input_overrides=None) -> None:
        self.input_overrides = input_overrides or {}
        self.output_buffers = [
            SimpleNamespace(pointer=(self.role, index), size=descriptor.size)
            for index, descriptor in enumerate(self.output_descriptors)
        ]

    def output_buffer(self, index):
        return self.output_buffers[index]

    def execute(self, inputs, *, read_outputs=None):
        self.__class__.order.append(self.role)
        self.calls.append({index: np.array(value, copy=True) for index, value in inputs.items()})
        selected = set(range(len(self.bindings.outputs))) if read_outputs is None else set(read_outputs)
        self.read_outputs = selected
        return {
            int(binding.index): np.full(binding.shape, 0.25, dtype=np.dtype(binding.dtype))
            for binding in self.bindings.outputs
            if binding.index is not None and int(binding.index) in selected
        }

    def close(self) -> None:
        self.close_calls += 1


@pytest.fixture
def fake_acl():
    """Hand out ``FakeAclModel`` with a clean instance registry and call order."""
    FakeAclModel.instances = {}
    FakeAclModel.order = []
    return FakeAclModel
