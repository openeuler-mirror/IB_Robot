#!/usr/bin/env python3
"""Prepare workspace-owned configuration and assets for hf-libero.

The ``hf-libero`` wheel intentionally omits the approximately 409 MiB MuJoCo
asset tree. Its runtime resolves assets through ``get_assets_path()``, which
first checks ``<site-packages>/libero/libero/assets`` and otherwise downloads
to a user-home cache. Setup makes this deterministic and non-interactive by:

1. locating the installed ``hf-libero`` distribution without importing it;
2. downloading or reusing ``lerobot/libero-assets`` under the workspace venv;
3. validating the expected 585 business asset files;
4. linking the provider's package-local ``assets`` path to that cache; and
5. atomically writing ``${WORKSPACE}/venv/ibrobot_libero/config.yaml``.

This script never imports ``libero`` before the config exists and never reads
or modifies the legacy ``libs/libero`` checkout.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import sys
import tempfile
from pathlib import Path

from packaging.specifiers import SpecifierSet

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required but not installed", file=sys.stderr)
    sys.exit(1)

HF_LIBERO_SPEC = SpecifierSet(">=0.1.4,<0.2.0")
HF_ASSETS_REPO_ID = "lerobot/libero-assets"
HF_ASSETS_REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
EXPECTED_ASSET_FILE_COUNT = 585
EXPECTED_ASSET_TREE_SHA256 = "ac6d9c70ae4de9b8e4781d0b85958b0bf6f5e5378bdcf74846e79c3dba9b5b37"
EXPECTED_ASSET_DIRS = (
    "articulated_objects",
    "scenes",
    "stable_hope_objects",
    "stable_scanned_objects",
    "textures",
    "turbosquid_objects",
)


def _find_provider_root() -> tuple[Path, str]:
    """Return the installed hf-libero runtime root without importing it."""
    try:
        distribution = importlib.metadata.distribution("hf-libero")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("hf-libero is not installed; install local lerobot[libero] first") from exc
    version = distribution.version
    if version not in HF_LIBERO_SPEC:
        raise RuntimeError(f"hf-libero version {version!r} does not satisfy {HF_LIBERO_SPEC}")
    provider_root = Path(distribution.locate_file("libero/libero")).resolve()
    if not provider_root.is_dir():
        raise RuntimeError(f"hf-libero runtime root does not exist: {provider_root}")
    for dirname in ("bddl_files", "init_files"):
        if not (provider_root / dirname).is_dir():
            raise RuntimeError(f"hf-libero runtime is missing {dirname}: {provider_root / dirname}")
    return provider_root, version


def _business_asset_files(root: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in root.rglob("*")
        if path.is_file() and ".cache" not in path.parts and path.name != ".gitattributes"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_tree_sha256(assets_dir: Path, files: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files):
        relative = str(path.relative_to(assets_dir)).encode()
        digest.update(relative + b"\0" + _file_sha256(path).encode() + b"\n")
    return digest.hexdigest()


def _validate_assets(assets_dir: Path) -> tuple[int, str]:
    if not assets_dir.is_dir():
        raise RuntimeError(f"LIBERO assets directory does not exist: {assets_dir}")
    missing = [dirname for dirname in EXPECTED_ASSET_DIRS if not (assets_dir / dirname).is_dir()]
    if missing:
        raise RuntimeError(f"LIBERO assets are missing required directories: {missing}")
    files = _business_asset_files(assets_dir)
    count = len(files)
    if count != EXPECTED_ASSET_FILE_COUNT:
        raise RuntimeError(f"LIBERO assets contain {count} business files; expected {EXPECTED_ASSET_FILE_COUNT}")
    tree_sha256 = _asset_tree_sha256(assets_dir, files)
    if tree_sha256 != EXPECTED_ASSET_TREE_SHA256:
        raise RuntimeError(f"LIBERO asset tree SHA-256 is {tree_sha256}; expected {EXPECTED_ASSET_TREE_SHA256}")
    return count, tree_sha256


def _prepare_assets(config_dir: Path, provider_root: Path) -> tuple[Path, int, str]:
    """Download/reuse assets and make hf-libero's package lookup deterministic."""
    package_assets = provider_root / "assets"
    if package_assets.exists() and not package_assets.is_symlink():
        count, tree_sha256 = _validate_assets(package_assets)
        return package_assets.resolve(), count, tree_sha256

    assets_dir = config_dir / "assets"
    try:
        count, tree_sha256 = _validate_assets(assets_dir)
    except RuntimeError:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is required to download LIBERO assets") from exc
        snapshot_download(
            repo_id=HF_ASSETS_REPO_ID,
            repo_type="dataset",
            revision=HF_ASSETS_REVISION,
            local_dir=str(assets_dir),
        )
        count, tree_sha256 = _validate_assets(assets_dir)

    if package_assets.is_symlink():
        if package_assets.resolve() != assets_dir.resolve():
            package_assets.unlink()
            package_assets.symlink_to(assets_dir, target_is_directory=True)
    elif package_assets.exists():
        raise RuntimeError(f"refusing to replace unexpected hf-libero assets path: {package_assets}")
    else:
        package_assets.symlink_to(assets_dir, target_is_directory=True)

    if package_assets.resolve() != assets_dir.resolve():
        raise RuntimeError(f"hf-libero assets link does not resolve to workspace cache: {package_assets}")
    return assets_dir.resolve(), count, tree_sha256


def _resolve_paths(workspace: str, provider_root: Path, assets_dir: Path) -> dict[str, str]:
    datasets_dir = Path(workspace) / "datasets" / "libero"
    return {
        "benchmark_root": str(provider_root),
        "bddl_files": str(provider_root / "bddl_files"),
        "init_states": str(provider_root / "init_files"),
        "assets": str(assets_dir),
        "datasets": str(datasets_dir),
    }


def generate_config(workspace: str, config_dir: str | None = None) -> tuple[str, str, str, int, str]:
    """Prepare hf-libero resources and atomically generate config.yaml."""
    workspace_path = Path(workspace).resolve()
    target_dir = Path(config_dir).resolve() if config_dir else workspace_path / "venv" / "ibrobot_libero"
    target_dir.mkdir(parents=True, exist_ok=True)

    provider_root, version = _find_provider_root()
    assets_dir, asset_count, asset_tree_sha256 = _prepare_assets(target_dir, provider_root)
    paths = _resolve_paths(str(workspace_path), provider_root, assets_dir)
    Path(paths["datasets"]).mkdir(parents=True, exist_ok=True)

    config_path = target_dir / "config.yaml"
    fd, tmp_path = tempfile.mkstemp(dir=target_dir, suffix=".tmp", prefix="config_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            yaml.safe_dump(paths, stream, default_flow_style=False, sort_keys=True)
        os.replace(tmp_path, config_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return str(config_path), version, str(assets_dir), asset_count, asset_tree_sha256


def main() -> int:
    workspace = os.path.abspath(os.environ.get("WORKSPACE", os.getcwd()))
    config_dir = os.path.join(workspace, "venv", "ibrobot_libero")
    try:
        config_path, version, assets_dir, asset_count, asset_tree_sha256 = generate_config(workspace, config_dir)
        provider_root, _ = _find_provider_root()
        paths = _resolve_paths(workspace, provider_root, Path(assets_dir))
    except Exception as exc:
        print(f"ERROR: Failed to prepare hf-libero: {exc}", file=sys.stderr)
        return 1

    print(f"LIBERO_CONFIG_PATH={config_dir}")
    print(f"config_file={config_path}")
    print(f"hf_libero_version={version}")
    print(f"asset_business_file_count={asset_count}")
    print(f"asset_tree_sha256={asset_tree_sha256}")
    for key, value in sorted(paths.items()):
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
