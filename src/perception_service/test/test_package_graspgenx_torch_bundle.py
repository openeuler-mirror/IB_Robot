# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""A GraspGenX bundle binds the official release checkpoints to one gripper description."""

from __future__ import annotations

import json

import pytest

from inference_manifest import load_inference_manifest
from model_utils.graspgenx_export.binding import generator_binding
from perception_service.graspgenx_adapter import GRASPGENX_SAMPLE_COUNT_SEMANTIC, GraspGenXAdapter
from perception_service.package_graspgenx_torch_bundle import package_graspgenx_torch_bundle


def _sources(tmp_path):
    release = tmp_path / "release"
    for network, epochs in (("gen", (90, 736)), ("dis", (1056,))):
        (release / network).mkdir(parents=True)
        (release / network / "config.yaml").write_text(
            f"network: {network}\ndata:\n  num_points: 3500\n"
            "diffusion:\n  num_diffusion_iters: 20\n  num_diffusion_iters_eval: 20\n",
            encoding="utf-8",
        )
        for epoch in epochs:
            (release / network / f"epoch_{epoch}.pth").write_bytes(f"{network}-{epoch}".encode())
    gripper = tmp_path / "omnihand_left_x2"
    (gripper / "meshes").mkdir(parents=True)
    (gripper / "config.json").write_text("{}", encoding="utf-8")
    (gripper / "gripper.urdf").write_text("<robot/>", encoding="utf-8")
    (gripper / "meshes" / "palm.obj").write_text("v 0 0 0\n", encoding="utf-8")
    return release, gripper


def test_the_packager_writes_a_cuda_bundle_with_the_newest_checkpoints_and_the_gripper(tmp_path):
    release, gripper = _sources(tmp_path)
    root = tmp_path / "graspgenx_omnihand_left_x2"

    package_graspgenx_torch_bundle(root, checkpoint_root=release, gripper_dir=gripper)

    validated = load_inference_manifest(root, "torch_cuda")
    model = validated.manifest.model
    assert (model.interface, model.model_type, model.operation) == ("tensor_model", "graspgenx", "generate_grasps")
    assert [tensor.semantic for tensor in model.inputs][1] == GRASPGENX_SAMPLE_COUNT_SEMANTIC
    assert validated.deployment.device == "cuda"
    assert (root / "assets/graspgenx/gen/generator.pth").read_bytes() == b"gen-736"
    assert (root / "assets/graspgenx/dis/discriminator.pth").read_bytes() == b"dis-1056"
    files = {item.path for item in validated.manifest.bundle.files}
    assert {
        "assets/adapter.json",
        "assets/x_grippers/omnihand_left_x2/config.json",
        "assets/x_grippers/omnihand_left_x2/meshes/palm.obj",
    } <= files
    adapter = json.loads((root / "assets/adapter.json").read_text(encoding="utf-8"))
    assert adapter["torch_module_loader"] == "perception_service.torch_model_loaders:load_graspgenx"
    assert GraspGenXAdapter.from_bundle(root).config.gripper_name == "omnihand_left_x2"


def test_repackaging_in_place_keeps_the_revision_and_a_new_checkpoint_bumps_it(tmp_path):
    release, gripper = _sources(tmp_path)
    root = tmp_path / "bundle"
    package_graspgenx_torch_bundle(root, checkpoint_root=release, gripper_dir=gripper)
    first = load_inference_manifest(root, "torch_cuda").manifest.bundle

    package_graspgenx_torch_bundle(root)
    assert load_inference_manifest(root, "torch_cuda").manifest.bundle.revision == first.revision

    (release / "gen" / "epoch_736.pth").write_bytes(b"retrained")
    package_graspgenx_torch_bundle(root, checkpoint_root=release, gripper_dir=gripper)
    changed = load_inference_manifest(root, "torch_cuda").manifest.bundle
    assert (changed.uuid, changed.revision) == (first.uuid, first.revision + 1)

    # bundle.files and the digest record paths only; a changed mesh must still bump it.
    (gripper / "meshes" / "palm.obj").write_text("v 1 0 0\n", encoding="utf-8")
    package_graspgenx_torch_bundle(root, checkpoint_root=release, gripper_dir=gripper)
    assert load_inference_manifest(root, "torch_cuda").manifest.bundle.revision == first.revision + 2


def test_the_packager_fails_closed_on_missing_sources(tmp_path):
    release, gripper = _sources(tmp_path)
    (release / "dis" / "config.yaml").unlink()
    with pytest.raises(FileNotFoundError, match="dis config is missing"):
        package_graspgenx_torch_bundle(tmp_path / "a", checkpoint_root=release, gripper_dir=gripper)

    release, gripper = _sources(tmp_path / "b")
    (gripper / "config.json").unlink()
    with pytest.raises(FileNotFoundError, match="no config.json"):
        package_graspgenx_torch_bundle(tmp_path / "c", checkpoint_root=release, gripper_dir=gripper)

    with pytest.raises(FileNotFoundError, match="provide --checkpoint-root"):
        package_graspgenx_torch_bundle(tmp_path / "empty")
    with pytest.raises(ValueError, match="pass both"):
        package_graspgenx_torch_bundle(tmp_path / "d", checkpoint_root=release)


def test_repackaging_under_another_gripper_name_keeps_one_gripper(tmp_path):
    release, gripper = _sources(tmp_path)
    root = tmp_path / "bundle"
    package_graspgenx_torch_bundle(root, checkpoint_root=release, gripper_dir=gripper)
    package_graspgenx_torch_bundle(root, checkpoint_root=release, gripper_dir=gripper, gripper_name="omnihand_right_x2")

    files = {item.path for item in load_inference_manifest(root, "torch_cuda").manifest.bundle.files}
    assert not [path for path in files if "omnihand_left_x2" in path]
    assert "assets/x_grippers/omnihand_right_x2/config.json" in files


def _loop(tmp_path, release, **overrides):
    path = tmp_path / "loop.om"
    path.write_bytes(b"compiled loop")
    metadata = {
        "grasps": 400,
        "steps": 20,
        "generator_binding": generator_binding(
            release / "gen/epoch_736.pth",
            release / "gen/config.yaml",
        ),
        **overrides,
    }
    path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    return path


def test_matching_loop_is_packaged_and_repackaged_without_changing_revision(tmp_path):
    release, gripper = _sources(tmp_path)
    root = tmp_path / "bundle"
    loop = _loop(tmp_path, release)
    package_graspgenx_torch_bundle(
        root,
        checkpoint_root=release,
        gripper_dir=gripper,
        deployment_name="torch_npu",
        device="npu",
        diffusion_loop=loop,
    )
    first = load_inference_manifest(root, "torch_npu").manifest
    package_graspgenx_torch_bundle(
        root, deployment_name="torch_npu", device="npu", diffusion_loop=root / "artifacts/ascend_310p/diffusion_loop.om"
    )
    assert load_inference_manifest(root, "torch_npu").manifest.bundle.revision == first.bundle.revision


def test_a_config_without_a_point_budget_is_refused_before_touching_the_bundle(tmp_path):
    release, gripper = _sources(tmp_path)
    root = tmp_path / "bundle"
    package_graspgenx_torch_bundle(root, checkpoint_root=release, gripper_dir=gripper)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}

    (release / "gen" / "epoch_800.pth").write_bytes(b"gen-800")
    config = release / "gen" / "config.yaml"
    config.write_text(config.read_text(encoding="utf-8").replace("  num_points: 3500\n", ""), encoding="utf-8")
    with pytest.raises(ValueError, match="data.num_points"):
        package_graspgenx_torch_bundle(root, checkpoint_root=release, gripper_dir=gripper)

    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("change", ["checkpoint", "config", "steps", "grasps", "missing_binding"])
def test_incompatible_incoming_loop_is_rejected_before_materializing_assets(tmp_path, change):
    release, gripper = _sources(tmp_path)
    overrides = {"steps": 10} if change == "steps" else {"grasps": 200} if change == "grasps" else {}
    loop = _loop(tmp_path, release, **overrides)
    if change in {"checkpoint", "config"}:
        path = release / "gen" / ("epoch_736.pth" if change == "checkpoint" else "config.yaml")
        path.write_bytes(path.read_bytes() + b"\n# changed\n")
    if change == "missing_binding":
        loop.with_suffix(".json").write_text('{"grasps":400,"steps":20}', encoding="utf-8")
    root = tmp_path / "bundle"
    with pytest.raises(ValueError, match="binding|must cover"):
        package_graspgenx_torch_bundle(
            root,
            checkpoint_root=release,
            gripper_dir=gripper,
            deployment_name="torch_npu",
            device="npu",
            diffusion_loop=loop,
        )
    assert not (root / "assets").exists()


@pytest.mark.parametrize("changed_file", ["epoch_736.pth", "config.yaml"])
def test_cuda_update_cannot_preserve_a_loop_from_an_old_generator(tmp_path, changed_file):
    release, gripper = _sources(tmp_path)
    root = tmp_path / "bundle"
    loop = _loop(tmp_path, release)
    package_graspgenx_torch_bundle(
        root,
        checkpoint_root=release,
        gripper_dir=gripper,
        deployment_name="torch_npu",
        device="npu",
        diffusion_loop=loop,
    )
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    path = release / "gen" / changed_file
    path.write_bytes(path.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="generator binding"):
        package_graspgenx_torch_bundle(root, checkpoint_root=release, gripper_dir=gripper)
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    load_inference_manifest(root, "torch_npu")


def test_runtime_rejects_unbound_loop_before_assembly_or_acl(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch_models.graspgenx_ascend_310p as ascend
    from perception_service.torch_model_loaders import _graspgenx_ascend_model

    def unexpected(*args, **kwargs):
        pytest.fail("incompatible metadata reached NPU assembly or ACL")

    monkeypatch.setitem(ascend.__dict__, "assemble", unexpected)
    monkeypatch.setitem(ascend.__dict__, "DiffusionLoopOM", unexpected)
    release, _ = _sources(tmp_path)
    metadata = tmp_path / "loop.json"
    metadata.write_text('{"grasps":400,"steps":20}', encoding="utf-8")
    context = SimpleNamespace(
        validated_manifest=SimpleNamespace(
            resolved_artifacts={
                "diffusion_loop": tmp_path / "loop.om",
                "diffusion_loop_metadata": metadata,
            }
        )
    )
    with pytest.raises(ValueError, match="generator binding"):
        _graspgenx_ascend_model(
            context,
            release,
            {
                "generator_checkpoint": "gen/epoch_736.pth",
                "checkpoints_dir": ".",
                "num_grasps": 400,
            },
        )
