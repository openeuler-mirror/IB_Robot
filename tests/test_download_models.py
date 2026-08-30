import hashlib
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import download_models as dm  # noqa: E402 - scripts dir is not a package


def _sam_like_manifest() -> dict:
    """sam2.1-style: two SoC targets + shared PyTorch checkpoint in bundle.files."""
    enc = hashlib.sha256(b"om-enc").hexdigest()
    dec = hashlib.sha256(b"om-dec").hexdigest()
    enc_b = hashlib.sha256(b"om-b-enc").hexdigest()
    return {
        "schema_version": 3,
        "bundle": {
            "files": [
                {"path": "assets/adapter.json"},
                {"path": "assets/sam2.1_hiera_tiny.pt"},
            ]
        },
        "deployments": {
            "ascend_310p": {
                "runtime_profile": {"backend": "ascend", "target": {"soc": "Ascend310P1"}},
                "artifacts": {
                    "encoder": {"path": "artifacts/ascend_310p/encoder.om", "sha256": enc},
                    "decoder": {"path": "artifacts/ascend_310p/decoder.om", "sha256": dec},
                },
            },
            "ascend_310b": {
                "runtime_profile": {"backend": "ascend", "target": {"soc": "Ascend310B1"}},
                "artifacts": {
                    "encoder": {"path": "artifacts/ascend_310b/encoder.om", "sha256": enc_b},
                },
            },
        },
    }


def _pi05_like_manifest() -> dict:
    """pi05-style: torch deployments without artifacts, weights only in bundle.files."""
    return {
        "schema_version": 3,
        "bundle": {
            "files": [
                {"path": "model.safetensors"},
                {"path": "bert-base-uncased/tokenizer.json"},
            ]
        },
        "deployments": {
            "torch-cpu": {"runtime_profile": {"backend": "torch", "target": {}}},
            "torch-cuda": {"runtime_profile": {"backend": "torch", "target": {}}},
        },
    }


def test_target_keyword_matches_soc_aliases():
    plan = build_plan_default(_sam_like_manifest(), targets=["310p"])
    assert plan.matched_deployments == ["ascend_310p"]
    assert "artifacts/ascend_310b/encoder.om" not in plan.patterns
    assert "artifacts/ascend_310p/encoder.om" in plan.patterns
    assert "artifacts/ascend_310p/decoder.om" in plan.patterns


def build_plan_default(manifest, targets=(), deployments=()):
    return dm.build_plan("some_bundle", "openEuler", manifest, list(targets), list(deployments))


def test_shared_files_always_included_for_any_filter():
    sam_plan = build_plan_default(_sam_like_manifest(), targets=["310p"])
    pi05_plan = build_plan_default(_pi05_like_manifest(), targets=["cuda"])
    for plan in (sam_plan, pi05_plan):
        assert "inference_manifest.json" in plan.patterns
    assert "assets/sam2.1_hiera_tiny.pt" in sam_plan.patterns
    assert "model.safetensors" in pi05_plan.patterns
    assert pi05_plan.matched_deployments == ["torch-cuda"]


def test_torch_deployment_matches_name_and_backend_keywords():
    for keyword in ("torch", "cpu", "cuda"):
        _, had_filter = dm.filter_deployments(dm.collect_deployments(_pi05_like_manifest()), [keyword])
        assert had_filter
    matched, _ = dm.filter_deployments(dm.collect_deployments(_pi05_like_manifest()), ["cpu"])
    assert [info.name for info in matched] == ["torch-cpu"]


def test_no_filter_downloads_every_artifact():
    plan = build_plan_default(_sam_like_manifest())
    assert set(plan.matched_deployments) == {"ascend_310p", "ascend_310b"}
    assert len(plan.verify_map) == 3


def test_unmatched_target_lists_available_deployments():
    with pytest.raises(dm.DownloadError) as excinfo:
        build_plan_default(_sam_like_manifest(), targets=["rk3588"])
    message = str(excinfo.value)
    assert "ascend_310p" in message and "Ascend310P1" in message and "rk3588" in message


def test_exact_deployment_filter_narrows_within_and_beyond_target():
    plan = build_plan_default(_sam_like_manifest(), targets=["ascend"], deployments=["ascend_310b"])
    assert plan.matched_deployments == ["ascend_310b"]
    assert plan.verify_map == {"artifacts/ascend_310b/encoder.om": hashlib.sha256(b"om-b-enc").hexdigest()}


def test_verify_downloads_passes_on_matching_files(tmp_path):
    root = tmp_path / "bundle"
    payload = b"artifact-bytes"
    digest = hashlib.sha256(payload).hexdigest()
    (root / "artifacts" / "x").mkdir(parents=True)
    (root / "artifacts" / "x" / "model.om").write_bytes(payload)
    dm.verify_downloads(root, {"artifacts/x/model.om": digest})


def test_verify_downloads_rejects_corruption_and_missing(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    good = hashlib.sha256(b"data").hexdigest()

    corrupted_dir = root / "c"
    corrupted_dir.mkdir()
    target = corrupted_dir / "f.om"
    target.write_bytes(b"tampered")
    with pytest.raises(dm.DownloadError, match="sha256 mismatch"):
        dm.verify_downloads(corrupted_dir, {"f.om": good})

    with pytest.raises(dm.DownloadError, match="missing after download"):
        dm.verify_downloads(root / "void", {"g.om": good})


def test_load_manifest_accepts_v2_v3_and_rejects_others(tmp_path):
    path = tmp_path / dm.MANIFEST_FILENAME
    path.write_text(json.dumps({"schema_version": 3, "bundle": {"files": []}, "deployments": {}}))
    assert dm.load_manifest(path)["schema_version"] == 3
    path.write_text(json.dumps({"schema_version": 2}))
    assert dm.load_manifest(path)["schema_version"] == 2
    path.write_text(json.dumps({"schema_version": 99}))
    with pytest.raises(dm.DownloadError, match="schema_version"):
        dm.load_manifest(path)
    path.write_text("{broken")
    with pytest.raises(dm.DownloadError, match="not valid JSON"):
        dm.load_manifest(path)


def test_prune_hf_cache_removes_transfer_metadata(tmp_path):
    cache = tmp_path / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "x.metadata").touch()
    dm.prune_hf_cache(tmp_path)
    assert not (tmp_path / ".cache").exists()


def test_end_to_end_download_with_fake_hub(monkeypatch, tmp_path):
    manifest = _sam_like_manifest()
    stored: dict[str, bytes] = {}
    for relative, content in {
        "inference_manifest.json": json.dumps(manifest).encode(),
        "assets/sam2.1_hiera_tiny.pt": b"pt-weights",
        "artifacts/ascend_310p/encoder.om": b"om-enc",
        "artifacts/ascend_310p/decoder.om": b"om-dec",
    }.items():
        stored[relative] = content

    def fake_fetch(org, name, dest_root):
        bundle_dir = dest_root / name
        bundle_dir.mkdir(parents=True, exist_ok=True)
        (bundle_dir / dm.MANIFEST_FILENAME).write_bytes(stored[dm.MANIFEST_FILENAME])
        return dm.load_manifest(bundle_dir / dm.MANIFEST_FILENAME)

    def fake_snapshot(repo_id, local_dir, allow_patterns):
        local = Path(local_dir)
        for relative in allow_patterns:
            if relative in stored:
                target = local / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(stored[relative])
        return str(local)

    monkeypatch.setattr(dm, "fetch_manifest", fake_fetch)
    monkeypatch.setattr(dm, "snapshot_download", fake_snapshot)

    code = dm.main(["--models", "sam2.1_hiera_tiny", "--target", "310p", "--dest", str(tmp_path)])
    assert code == 0

    bundle = tmp_path / "sam2.1_hiera_tiny"
    assert (bundle / "artifacts" / "ascend_310p" / "encoder.om").read_bytes() == b"om-enc"
    assert not (bundle / "artifacts" / "ascend_310b").exists()
    expected_digest = hashlib.sha256(b"om-enc").hexdigest()
    from_file = dm.sha256_of(bundle / "artifacts" / "ascend_310p" / "encoder.om")
    assert from_file == expected_digest


def test_split_csv_and_resolve_names():
    assert dm.split_csv(["310p, cpu", "cuda"]) == ["310p", "cpu", "cuda"]
    assert dm.resolve_names("pi05, fullsubnet") == ["pi05", "fullsubnet"]
    names = dm.resolve_names("totally_new_bundle")
    assert names == ["totally_new_bundle"]


def test_repository_aliases_and_runtime_directories():
    assert (
        dm.repository_for_name("ACT_1arm_2cam_banana_pick_v1_step_160000_distill_20260515")
        == "IB_Robot_ACT_banana_pick_distill"
    )
    assert dm.runtime_directory("graspgen", "grasp") == "graspgen"
    assert dm.runtime_directory("fullsubnet", "fullsubnet") == "voice_asr"
    assert (
        dm.runtime_directory("grounding_dino_swint_seq8_1280x720", "grounding_dino_swint_seq8_1280x720")
        == "grounded_sam2_swint_ogc"
    )
    assert dm.resolve_names("all", ["pi05", "zipvoice"]) == ["pi05", "zipvoice"]


def test_build_plan_separates_repository_from_local_directory():
    plan = dm.build_plan(
        "ACT_1arm_2cam_banana_pick_v1_step_160000_distill_20260515",
        "openEuler",
        _pi05_like_manifest(),
        [],
        [],
        repo_name="IB_Robot_ACT_banana_pick_distill",
    )
    assert plan.name == "ACT_1arm_2cam_banana_pick_v1_step_160000_distill_20260515"
    assert plan.repo_id == "openEuler/IB_Robot_ACT_banana_pick_distill"


def test_fullsubnet_runtime_aliases(tmp_path):
    source_root = tmp_path / "voice_asr"
    assets = source_root / "assets"
    assets.mkdir(parents=True)
    checkpoint = assets / "cum_fullsubnet_best_model_218epochs.tar"
    manifest = assets / "cum_fullsubnet_best_model_218epochs.manifest.json"
    checkpoint.write_bytes(b"checkpoint")
    manifest.write_text("{}")

    dm.materialize_runtime_aliases("fullsubnet", source_root)

    torch_alias = source_root / "artifacts/torch/fullsubnet/cum_fullsubnet_best_model_218epochs.tar"
    ascend_alias = source_root / "artifacts/ascend/fullsubnet/cum_fullsubnet_best_model_218epochs.manifest.json"
    assert torch_alias.is_symlink() and torch_alias.read_bytes() == b"checkpoint"
    assert ascend_alias.is_symlink() and ascend_alias.read_text() == "{}"


def test_legacy_download_uses_filtered_snapshot(monkeypatch, tmp_path):
    captured = {}

    def fake_snapshot(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(dm, "snapshot_download", fake_snapshot)
    target = dm.download_legacy_repo("witty-tune-model", tmp_path)
    assert target == tmp_path / "witty-tune-model"
    assert captured["repo_id"] == "openEuler/witty-tune-model"
    assert "*.mp4" in captured["ignore_patterns"]


def test_main_reports_failure_per_bundle_without_aborting(monkeypatch, tmp_path):
    def boom(org, name, dest_root):
        raise RuntimeError(f"repo not found: {name}")

    calls = []

    def ok_fetch(org, name, dest_root):
        calls.append(name)
        bundle_dir = dest_root / name
        bundle_dir.mkdir(parents=True, exist_ok=True)
        (bundle_dir / dm.MANIFEST_FILENAME).write_text(json.dumps({"schema_version": 3}))
        return {"schema_version": 3, "bundle": {"files": []}, "deployments": {}}

    state = {"first": True}

    def fetch(org, name, dest_root):
        if state["first"]:
            state["first"] = False
            raise RuntimeError("network down")
        return ok_fetch(org, name, dest_root)

    monkeypatch.setattr(dm, "fetch_manifest", fetch)
    code = dm.main(["--models", "badone,pi05", "--dest", str(tmp_path)])
    assert code == 1
    assert calls == ["pi05"]
