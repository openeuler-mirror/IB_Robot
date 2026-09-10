"""Tests for the FullSubNet standalone bundle packager."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voice_asr_service.package_fullsubnet_bundle import (
    _FB_OM_310B_REL,
    _FB_OM_REL,
    _SB_OM_310B_REL,
    _SB_OM_REL,
    package_fullsubnet_bundle,
)


def _write_asset(bundle: Path, rel: str, payload: bytes) -> None:
    path = bundle / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


@pytest.fixture()
def bundle(tmp_path: Path) -> Path:
    root = tmp_path / "models" / "fullsubnet"
    _write_asset(root, "assets/adapter.json", b"{}")
    _write_asset(root, "assets/cum_fullsubnet_best_model_218epochs.tar", b"fake-checkpoint")
    _write_asset(root, "assets/cum_fullsubnet_best_model_218epochs.manifest.json", b"{}")
    return root


def test_torch_only_bundle_degrades_without_ascend_oms(bundle: Path) -> None:
    manifest_path = package_fullsubnet_bundle(bundle)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["deployments"]) == {"torch_cpu", "torch_cuda"}
    for name, device in (("torch_cpu", "cpu"), ("torch_cuda", "cuda")):
        deployment = manifest["deployments"][name]
        assert deployment["runtime_profile"]["backend"] == "torch"
        assert deployment["runtime_profile"]["profile"]["device"] == device


def test_ascend_om_pairs_are_picked_up_and_bundle_identity_stays_stable(bundle: Path) -> None:
    first = json.loads(package_fullsubnet_bundle(bundle).read_text(encoding="utf-8"))
    _write_asset(bundle, _FB_OM_REL, b"fake-310p-fb-om")
    _write_asset(bundle, _SB_OM_REL, b"fake-310p-sb-om")
    _write_asset(bundle, _FB_OM_310B_REL, b"fake-310b-fb-om")
    _write_asset(bundle, _SB_OM_310B_REL, b"fake-310b-sb-om")

    second = json.loads(package_fullsubnet_bundle(bundle).read_text(encoding="utf-8"))

    assert set(second["deployments"]) == {"ascend_310p", "ascend_310b", "torch_cpu", "torch_cuda"}
    assert second["bundle"]["uuid"] == first["bundle"]["uuid"]
    assert second["bundle"]["revision"] == first["bundle"]["revision"]
    assert second["deployments"]["ascend_310b"]["runtime_profile"]["target"]["runtime_abi"] == "cann-8.3.RC1"
    # torch deployments keep their identity across the repack
    for name in ("torch_cpu", "torch_cuda"):
        assert second["deployments"][name]["uuid"] == first["deployments"][name]["uuid"]
        assert second["deployments"][name]["revision"] == first["deployments"][name]["revision"]


def test_partial_om_pair_is_rejected_fail_closed(bundle: Path) -> None:
    _write_asset(bundle, _FB_OM_REL, b"fake-310p-fb-om")

    with pytest.raises(FileNotFoundError, match="incomplete"):
        package_fullsubnet_bundle(bundle)
