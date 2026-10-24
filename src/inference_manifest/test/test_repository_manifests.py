from __future__ import annotations

import json
from pathlib import Path

from inference_manifest import load_inference_manifest_metadata, validate_manifest_schema


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _released_manifest_paths(root: Path) -> tuple[Path, ...]:
    excluded = {"build", "install", "log", ".git", "libs", "third_party", "tmp", "venv", ".venv", "_work"}
    paths = []
    for path in root.rglob("inference_manifest.json"):
        relative = path.relative_to(root)
        if any(part in excluded for part in relative.parts):
            continue
        paths.append(path)
    return tuple(sorted(paths))


def test_inventory_excludes_temporary_and_conversion_workspaces(tmp_path):
    published = tmp_path / "models/released/inference_manifest.json"
    for path in (
        published,
        tmp_path / "tmp/upload/inference_manifest.json",
        tmp_path / "models/_work/model/exporter/inference_manifest.json",
        tmp_path / "venv/share/example/inference_manifest.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    assert _released_manifest_paths(tmp_path) == (published,)


def test_released_inference_manifests_load_as_v3_metadata() -> None:
    root = _repository_root()
    paths = _released_manifest_paths(root)
    assert paths, "repository inventory should contain at least one inference manifest"

    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        assert value["schema_version"] == 3, path
        validate_manifest_schema(value, str(path))
        model = value["model"]
        assert set(("interface", "model_type", "operation")) <= set(model), path
        assert "kind" not in model and "family" not in model, path
        if model["interface"] == "policy":
            assert model["operation"] == "predict", path
        for deployment_name in value["deployments"]:
            deployment = value["deployments"][deployment_name]
            serialized = json.dumps(deployment, sort_keys=True)
            assert '"kind"' not in serialized and '"family"' not in serialized, path
            assert "acl_config_path" not in serialized and "acl_config" not in serialized, path
            assert all(
                profile["target"]["runtime"] not in {"raw_acl", "stateful_raw_acl", "stateful_om"}
                and not profile["target"]["runtime"].startswith("acl-")
                for profile in (
                    ([deployment["runtime_profile"]] if "runtime_profile" in deployment else [])
                    + list(deployment.get("role_runtime_profiles", {}).values())
                )
            ), path
            load_inference_manifest_metadata(path.parent, deployment_name)
