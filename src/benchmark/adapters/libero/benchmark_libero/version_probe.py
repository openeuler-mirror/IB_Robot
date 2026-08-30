"""Fail-closed hf-libero provider identity probe.

The Benchmark adapter consumes the provider installed by the local LeRobot
``libero`` extra. The legacy ``libs/libero`` submodule is no longer an import
provider once the migration is enabled; this probe prevents an accidental
mixed-provider environment.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from packaging.specifiers import SpecifierSet


class ProviderProbeError(RuntimeError):
    """Raised when the hf-libero provider identity cannot be verified."""


EXPECTED_HF_LIBERO_SPEC = SpecifierSet(">=0.1.4,<0.2.0")
EXPECTED_ASSET_DIRS = (
    "articulated_objects",
    "scenes",
    "stable_hope_objects",
    "stable_scanned_objects",
    "textures",
    "turbosquid_objects",
)


@dataclass(frozen=True, slots=True)
class ProviderIdentity:
    """Snapshot of the verified hf-libero provider identity."""

    module_path: str
    distribution_name: str
    distribution_version: str
    provider_root: str
    assets_path: str
    apis_present: tuple[str, ...]


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _locations(module: object) -> tuple[Path, ...]:
    locations: list[Path] = []
    module_file = getattr(module, "__file__", None)
    if isinstance(module_file, str) and module_file:
        locations.append(Path(module_file).resolve())
    for item in getattr(module, "__path__", ()):
        if isinstance(item, str) and item:
            locations.append(Path(item).resolve())
    return tuple(locations)


def _provider_distribution() -> tuple[importlib.metadata.Distribution, Path]:
    try:
        distribution = importlib.metadata.distribution("hf-libero")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ProviderProbeError("hf-libero distribution is not installed") from exc
    if distribution.version not in EXPECTED_HF_LIBERO_SPEC:
        raise ProviderProbeError(f"hf-libero version is {distribution.version!r}; expected {EXPECTED_HF_LIBERO_SPEC}")
    provider_root = Path(distribution.locate_file("libero/libero")).resolve()
    if not provider_root.is_dir():
        raise ProviderProbeError(f"hf-libero provider root does not exist: {provider_root}")
    return distribution, provider_root


def _verify_provider_import(
    distribution: importlib.metadata.Distribution, provider_root: Path
) -> tuple[str, tuple[str, ...], object, object, object]:
    package_root = Path(distribution.locate_file("libero")).resolve()
    try:
        top_module = importlib.import_module("libero")
        runtime_module = importlib.import_module("libero.libero")
        benchmark_module = importlib.import_module("libero.libero.benchmark")
        envs_module = importlib.import_module("libero.libero.envs")
    except Exception as exc:  # noqa: BLE001
        raise ProviderProbeError(f"failed to import hf-libero provider: {exc}") from exc

    all_locations = {
        name: _locations(module)
        for name, module in (
            ("libero", top_module),
            ("libero.libero", runtime_module),
            ("libero.libero.benchmark", benchmark_module),
            ("libero.libero.envs", envs_module),
        )
    }
    workspace = os.environ.get("WORKSPACE", "")
    legacy_provider = (Path(workspace).resolve() / "libs" / "libero") if workspace else None
    legacy_sys_path_entries = tuple(
        Path(entry).resolve()
        for entry in sys.path
        if entry and legacy_provider is not None and _is_within(Path(entry).resolve(), legacy_provider)
    )
    if legacy_sys_path_entries:
        raise ProviderProbeError(f"legacy libs/libero is exposed on sys.path: {legacy_sys_path_entries}")
    for name, locations in all_locations.items():
        if not locations:
            raise ProviderProbeError(f"{name} exposes no filesystem location")
        for location in locations:
            if not _is_within(location, package_root):
                raise ProviderProbeError(
                    f"{name} resolves outside hf-libero package root: {location}; expected under {package_root}"
                )
            if legacy_provider is not None and _is_within(location, legacy_provider):
                raise ProviderProbeError(f"{name} resolves from legacy libs/libero: {location}")

    try:
        importlib.metadata.distribution("libero")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise ProviderProbeError("legacy 'libero' distribution is installed alongside hf-libero")

    required = (
        (benchmark_module, "get_benchmark_dict"),
        (runtime_module, "get_libero_path"),
        (runtime_module, "get_assets_path"),
        (envs_module, "OffScreenRenderEnv"),
    )
    for module, attribute in required:
        if not hasattr(module, attribute):
            raise ProviderProbeError(f"{module.__name__} is missing required API {attribute}")
    if not isinstance(envs_module.OffScreenRenderEnv, type):
        raise ProviderProbeError("OffScreenRenderEnv is not a class")

    return (
        str(top_module.__file__ or next(iter(top_module.__path__))),
        ("get_benchmark_dict", "get_libero_path", "get_assets_path", "OffScreenRenderEnv"),
        runtime_module,
        benchmark_module,
        envs_module,
    )


def _verify_config_and_assets(runtime_module: object, provider_root: Path) -> Path:
    config_value = os.environ.get("LIBERO_CONFIG_PATH", "")
    if not config_value:
        raise ProviderProbeError("LIBERO_CONFIG_PATH is not set; workspace-owned hf-libero config is required")
    config_file = Path(config_value).resolve() / "config.yaml"
    if not config_file.is_file():
        raise ProviderProbeError(f"workspace-owned LIBERO config is missing: {config_file}")

    try:
        configured_root = Path(runtime_module.get_libero_path("benchmark_root")).resolve()
        assets_path = Path(runtime_module.get_assets_path()).resolve()
        bddl_path = Path(runtime_module.get_libero_path("bddl_files")).resolve()
        init_path = Path(runtime_module.get_libero_path("init_states")).resolve()
    except Exception as exc:  # noqa: BLE001
        raise ProviderProbeError(f"hf-libero resource path resolution failed: {exc}") from exc
    if configured_root != provider_root:
        raise ProviderProbeError(f"configured benchmark_root {configured_root} != provider root {provider_root}")
    if not _is_within(bddl_path, provider_root) or not bddl_path.is_dir():
        raise ProviderProbeError(f"invalid hf-libero bddl_files path: {bddl_path}")
    if not _is_within(init_path, provider_root) or not init_path.is_dir():
        raise ProviderProbeError(f"invalid hf-libero init_states path: {init_path}")
    if not assets_path.is_dir():
        raise ProviderProbeError(f"hf-libero assets path does not exist: {assets_path}")
    missing = [name for name in EXPECTED_ASSET_DIRS if not (assets_path / name).is_dir()]
    if missing:
        raise ProviderProbeError(f"hf-libero assets are missing directories: {missing}")
    return assets_path


def probe_libero_provider() -> ProviderIdentity:
    """Verify and return the sole hf-libero provider identity."""
    distribution, provider_root = _provider_distribution()
    module_path, apis, runtime_module, _benchmark_module, _envs_module = _verify_provider_import(
        distribution, provider_root
    )
    assets_path = _verify_config_and_assets(runtime_module, provider_root)
    return ProviderIdentity(
        module_path=module_path,
        distribution_name="hf-libero",
        distribution_version=distribution.version,
        provider_root=str(provider_root),
        assets_path=str(assets_path),
        apis_present=apis,
    )
