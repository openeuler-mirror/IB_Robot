#!/usr/bin/env python3
"""Unified downloader for IB-Robot inference model bundles published on HuggingFace.

Bundles are v3 inference manifests published under the openEuler org
(e.g. openEuler/sam2.1_hiera_tiny).  The download plan is driven by the
manifest itself:

- ``bundle.files`` are shared base files (assets, tokenizers) and are always
  downloaded together with ``inference_manifest.json``.
- Each entry in ``deployments`` declares its compiled artifacts (OM / RKNN /
  ONNX) with an inline sha256.  ``--target`` filters deployments by keyword so
  only the artifacts for the requested accelerator (310P, 310B, RK3588, torch,
  ...) are fetched.

Downloaded artifact files are verified against the manifest sha256 digests and
the huggingface transfer cache residue is pruned afterwards.

The manifest is parsed with plain JSON access instead of the repo's
``inference_manifest`` loader: partial downloads cannot satisfy full integrity
validation anyway, and this keeps the script working across schema-v2/v3 tool
generations (v2 manifests expose the same deployment topology).

Examples:
    ./scripts/download_models.py --list
    ./scripts/download_models.py --models sam2.1_hiera_tiny --target 310p
    ./scripts/download_models.py --models pi05,fullsubnet
    ./scripts/download_models.py --models ACT_1arm_2cam_banana_pick_v1_step_160000_distill_20260515 \
        --target rk3588 --dest /data/models

Set ``HF_ENDPOINT=https://hf-mirror.com`` to route through a mirror, and
``HF_TOKEN`` when a repository requires authentication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
    raise SystemExit(
        "Error: huggingface_hub is required. Install the project environment first with ./scripts/setup.sh."
    ) from exc

DEFAULT_ORG = "openEuler"
MANIFEST_FILENAME = "inference_manifest.json"


@dataclass(frozen=True)
class BundleSource:
    """Allowlisted Hugging Face bundle and its local runtime identity."""

    name: str
    repository: str
    directory: str
    interface: str
    model_type: str
    operation: str


# Allowlisted model bundles are the trusted registry of this downloader: an
# entry pins the Hugging Face repository, the local runtime directory and the
# schema-v3 model identity, and is the sole source for ``--models all`` and
# ``--list``. Repositories in the openEuler organization are not automatically
# trusted as IB-Robot runtime bundles; add an entry only after its schema-v3
# identity is wired in-tree. Manually requested repositories outside this
# registry are still downloadable with a warning (no identity to validate).
MODEL_BUNDLE_ALLOWLIST = (
    BundleSource(
        "ACT_1arm_2cam_banana_pick_v1_step_160000_distill_20260515",
        "IB_Robot_ACT_banana_pick_distill",
        "ACT_1arm_2cam_banana_pick_v1_step_160000_distill_20260515",
        "policy",
        "act",
        "predict",
    ),
    BundleSource("pi05", "pi05", "pi05", "policy", "pi05", "predict"),
    BundleSource("smolvla", "smolvla", "smolvla", "policy", "smolvla", "predict"),
    BundleSource(
        "ram_plus_swin_large_14m",
        "ram_plus_swin_large_14m",
        "ram_plus_swin_large_14m",
        "tensor_model",
        "ram_plus",
        "recognize_tags",
    ),
    BundleSource(
        "sam2.1_hiera_tiny",
        "sam2.1_hiera_tiny",
        "sam2.1_hiera_tiny",
        "tensor_model",
        "sam2",
        "automatic",
    ),
    BundleSource(
        "sam2.1_hiera_tiny_prompt_ascend",
        "sam2.1_hiera_tiny_prompt_ascend",
        "sam2.1_hiera_tiny_prompt_ascend",
        "tensor_model",
        "sam2",
        "prompt",
    ),
    BundleSource(
        "siglip2_so400m_patch14_384",
        "siglip2_so400m_patch14_384",
        "siglip2_so400m_patch14_384",
        "tensor_model",
        "siglip2",
        "encode",
    ),
    BundleSource(
        "grounding_dino_swint_seq8_1280x720",
        "grounding_dino_swint_seq8_1280x720",
        "grounding_dino_swint_seq8_1280x720",
        "tensor_model",
        "grounding_dino",
        "detect",
    ),
    BundleSource("graspgen", "graspgen", "graspgen", "tensor_model", "graspgen", "generate_grasps"),
    BundleSource("zipvoice", "zipvoice", "zipvoice", "tensor_model", "zipvoice", "synthesize"),
    BundleSource("fullsubnet", "fullsubnet", "fullsubnet", "tensor_model", "fullsubnet", "enhance"),
    BundleSource("silero-vad", "silero-vad", "silero-vad", "tensor_model", "silero_vad", "vad"),
    BundleSource(
        "pear_parameter_network",
        "pear_parameter_network",
        "pear_parameter_network",
        "tensor_model",
        "pear_parameter_network",
        "predict_parameters",
    ),
    BundleSource("yolox_x_640", "yolox_x_640", "yolox_x_640", "tensor_model", "yolox_person", "detect"),
)

for _attr in ("name", "repository", "directory"):
    _values = [getattr(source, _attr) for source in MODEL_BUNDLE_ALLOWLIST]
    _duplicates = sorted({value for value in _values if _values.count(value) > 1})
    if _duplicates:
        raise RuntimeError(f"duplicate bundle {_attr} in MODEL_BUNDLE_ALLOWLIST: {_duplicates}")
del _attr, _values, _duplicates

_BUNDLE_BY_NAME = {source.name: source for source in MODEL_BUNDLE_ALLOWLIST}
_BUNDLE_BY_REPOSITORY = {source.repository: source for source in MODEL_BUNDLE_ALLOWLIST}

LEGACY_REPOSITORIES = {
    "IB_Robot_ACT_banana_pick": "IB_Robot_ACT_banana_pick",
    "IB_Robot_ACT_dual_arm_banana_pick": "IB_Robot_ACT_dual_arm_banana_pick",
    "witty-tune-model": "witty-tune-model",
}


class DownloadError(RuntimeError):
    """Raised for bundle selection, planning or verification failures."""


class NoMatchingDeploymentError(DownloadError):
    """Raised when an allowlisted bundle has no requested deployment."""


@dataclass
class BundlePlan:
    """Resolved download plan for one bundle."""

    name: str
    repo_id: str
    patterns: list[str]
    artifact_paths: list[str]
    verify_map: dict[str, str]
    matched_deployments: list[str]

    @property
    def artifact_count(self) -> int:
        return len(self.artifact_paths)


@dataclass
class ResolvedBundle:
    """A bundle resolved against the local directory map, ready to download."""

    name: str
    local_name: str
    plan: BundlePlan | None = None
    conflict_with: str | None = None
    duplicate: bool = False


@dataclass
class DeploymentInfo:
    """Summary of a deployment as seen by target filtering."""

    name: str
    backend: str
    soc: str
    match_terms: list[str] = field(default_factory=list)
    artifact_paths: list[str] = field(default_factory=list)

    @property
    def haystack(self) -> str:
        return " ".join((self.name, self.backend, self.soc, *self.match_terms)).lower()

    def describe(self) -> str:
        detail = f"backend={self.backend}, soc={self.soc}"
        count = len(self.artifact_paths)
        return f"{self.name} ({detail}, {count} artifact{'s' if count != 1 else ''})"


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DownloadError(f"{MANIFEST_FILENAME} in not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise DownloadError(f"{MANIFEST_FILENAME} must be a JSON object")
    if raw.get("schema_version") not in (2, 3):
        raise DownloadError(f"unsupported schema_version {raw.get('schema_version')!r}; expected 2 or 3")
    return raw


def collect_deployments(manifest: dict[str, Any]) -> list[DeploymentInfo]:
    infos = []
    for name, deployment in manifest.get("deployments", {}).items():
        profile = deployment.get("runtime_profile", {}) if isinstance(deployment, dict) else {}
        target = profile.get("target", {}) or {}
        backend = str(profile.get("backend") or target.get("runtime") or "")
        soc = str(target.get("soc") or "")
        profiles = [profile, *(deployment.get("role_runtime_profiles", {}) or {}).values()]
        match_terms = []
        for runtime_profile in profiles:
            if not isinstance(runtime_profile, dict):
                continue
            runtime_target = runtime_profile.get("target", {}) or {}
            runtime_options = runtime_profile.get("profile", {}) or {}
            match_terms.extend(
                str(value)
                for value in (
                    runtime_profile.get("backend"),
                    runtime_target.get("runtime"),
                    runtime_target.get("runtime_abi"),
                    runtime_target.get("soc"),
                    runtime_options.get("device"),
                    runtime_options.get("target_name"),
                )
                if value not in (None, "")
            )
        artifacts = [
            spec["path"]
            for spec in (deployment.get("artifacts", {}) or {}).values()
            if isinstance(spec, dict) and spec.get("path")
        ]
        infos.append(
            DeploymentInfo(
                name=name,
                backend=backend,
                soc=soc,
                match_terms=match_terms,
                artifact_paths=artifacts,
            )
        )
    return infos


def filter_deployments(infos: list[DeploymentInfo], keywords: list[str]) -> tuple[list[DeploymentInfo], bool]:
    """Return matching deployments plus whether any keyword was checked."""
    lowered = [keyword.lower() for keyword in keywords]
    matches = [info for info in infos if any(keyword in info.haystack for keyword in lowered)]
    return matches, bool(lowered)


def build_plan(
    name: str,
    org: str,
    manifest: dict[str, Any],
    targets: list[str],
    deployments: list[str],
    repo_name: str | None = None,
) -> BundlePlan:
    shared_paths = sorted(
        {
            entry.get("path", "")
            for entry in manifest.get("bundle", {}).get("files", [])
            if isinstance(entry, dict) and entry.get("path")
        }
    )
    blank = [path for path in shared_paths if not path]
    if blank:
        raise DownloadError("manifest contains bundle.files entries without a path")

    infos = collect_deployments(manifest)
    exact = {dep.lower() for dep in deployments}
    selected, had_target_filter = filter_deployments(infos, targets)
    if not had_target_filter and not exact:
        selected = list(infos)
    if exact:
        selected = [info for info in selected if info.name.lower() in exact] or [
            info for info in infos if info.name.lower() in exact
        ]

    if (had_target_filter or exact) and not selected:
        available = "; ".join(info.describe() for info in infos)
        wanted = ", ".join(targets + deployments)
        raise NoMatchingDeploymentError(f"no deployment of '{name}' matches [{wanted}]. Available: {available}")

    verify_map: dict[str, str] = {}
    patterns: set[str] = {MANIFEST_FILENAME, *shared_paths}
    artifact_paths: set[str] = set()
    for info in selected:
        for artifact_path in info.artifact_paths:
            patterns.add(artifact_path)
            artifact_paths.add(artifact_path)
            digest = _artifact_digest(manifest, info.name, artifact_path)
            if digest:
                verify_map[artifact_path] = digest.lower()
    unmatched_verifies = set(verify_map) - patterns
    if unmatched_verifies:
        raise DownloadError(f"internal error: verification paths outside plan: {sorted(unmatched_verifies)}")

    return BundlePlan(
        name=name,
        repo_id=f"{org}/{repo_name or name}",
        patterns=sorted(patterns),
        artifact_paths=sorted(artifact_paths),
        verify_map=verify_map,
        matched_deployments=[info.name for info in selected],
    )


def _artifact_digest(manifest: dict[str, Any], deployment_name: str, artifact_path: str) -> str | None:
    for name, deployment in manifest.get("deployments", {}).items():
        if name != deployment_name:
            continue
        for spec in (deployment.get("artifacts", {}) or {}).values():
            if isinstance(spec, dict) and spec.get("path") == artifact_path:
                return spec.get("sha256") or spec.get("digest")
    return None


def sha256_of(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_downloads(bundle_dir: Path, verify_map: dict[str, str], required_paths: list[str] | None = None) -> None:
    for relative_path in sorted(required_paths or []):
        if not (bundle_dir / relative_path).is_file():
            raise DownloadError(f"missing after download: {relative_path}")
    for relative_path, expected in sorted(verify_map.items()):
        candidate = bundle_dir / relative_path
        if not candidate.is_file():
            raise DownloadError(f"missing after download: {relative_path}")
        actual = sha256_of(candidate)
        if actual != expected:
            raise DownloadError(
                f"sha256 mismatch for {relative_path}: actual={actual} expected={expected}; delete the file and retry"
            )


def prune_hf_cache(bundle_dir: Path) -> None:
    residue = bundle_dir / ".cache" / "huggingface"
    if residue.is_dir():
        shutil.rmtree(residue.parent)
        print(f"[clean] removed snapshot transfer metadata: {bundle_dir / '.cache'}")


def bundle_source(name: str) -> BundleSource | None:
    return _BUNDLE_BY_NAME.get(name) or _BUNDLE_BY_REPOSITORY.get(name)


def runtime_directory(repo_name: str, bundle_name: str | None = None) -> str:
    source = bundle_source(repo_name)
    if source is not None:
        return source.directory
    return bundle_name or repo_name


def repository_for_name(name: str) -> str:
    source = bundle_source(name)
    return source.repository if source is not None else name


def validate_manifest_identity(source: BundleSource, manifest: dict[str, Any]) -> None:
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise DownloadError(f"{source.repository} manifest has no model identity")
    actual = (model.get("interface"), model.get("model_type"), model.get("operation"))
    expected = (source.interface, source.model_type, source.operation)
    if actual != expected:
        raise DownloadError(
            f"{source.repository} manifest identity is {'/'.join(str(value) for value in actual)}; "
            f"allowlist expects {'/'.join(expected)}"
        )


def materialize_runtime_aliases(repo_name: str, bundle_dir: Path) -> None:
    """Expose legacy runtime paths while preserving manifest-relative files.

    The FullSubNet aliases pointed at the retired ``artifacts/{torch,ascend}``
    layout of the pre-migration composite bundle; the standalone
    ``models/fullsubnet`` bundle consumes ``assets/`` files directly, so no
    alias remains necessary.
    """
    return


def download_bundle(plan: BundlePlan, dest_root: Path, dry_run: bool = False, token: str | None = None) -> Path:
    bundle_dir = dest_root / plan.name
    if dry_run:
        return bundle_dir
    bundle_dir.mkdir(parents=True, exist_ok=True)
    print(f"[plan] {plan.repo_id}: {len(plan.patterns)} paths (deployments: {', '.join(plan.matched_deployments)})")
    for pattern in plan.patterns:
        print(f"       - {pattern}")
    snapshot_download(repo_id=plan.repo_id, local_dir=str(bundle_dir), allow_patterns=plan.patterns, token=token)
    verify_downloads(bundle_dir, plan.verify_map, plan.patterns)
    prune_hf_cache(bundle_dir)
    materialize_runtime_aliases(plan.repo_id.rsplit("/", 1)[-1], bundle_dir)
    return bundle_dir


def download_legacy_repo(repo_name: str, dest_root: Path, dry_run: bool = False, token: str | None = None) -> Path:
    """Download a published repository that predates the manifest contract."""
    bundle_dir = dest_root / LEGACY_REPOSITORIES.get(repo_name, repo_name)
    if dry_run:
        print(f"[dry-run] {DEFAULT_ORG}/{repo_name}: legacy repository -> {bundle_dir}")
        return bundle_dir
    bundle_dir.mkdir(parents=True, exist_ok=True)
    print(f"[legacy] {DEFAULT_ORG}/{repo_name} -> {bundle_dir}")
    snapshot_download(
        repo_id=f"{DEFAULT_ORG}/{repo_name}",
        local_dir=str(bundle_dir),
        ignore_patterns=[".gitattributes", "README*", "*.mp4"],
        token=token,
    )
    prune_hf_cache(bundle_dir)
    return bundle_dir


def fetch_manifest(org: str, name: str, token: str | None = None) -> dict[str, Any]:
    """Fetch a manifest through the HF cache without mutating the destination."""
    local = hf_hub_download(repo_id=f"{org}/{name}", filename=MANIFEST_FILENAME, token=token)
    return load_manifest(Path(local))


def list_bundles() -> None:
    print(f"Allowlisted IB-Robot model bundles (org={DEFAULT_ORG}):")
    for source in MODEL_BUNDLE_ALLOWLIST:
        identity = f"{source.interface}/{source.model_type}/{source.operation}"
        print(f"  {source.name} -> {DEFAULT_ORG}/{source.repository} -> models/{source.directory} [{identity}]")


def resolve_names(requested: str) -> list[str]:
    names = [item.strip() for item in requested.split(",") if item.strip()]
    if not names:
        raise DownloadError("--models requires at least one bundle name (see --list)")
    if len(names) == 1 and names[0].lower() == "all":
        return [source.repository for source in MODEL_BUNDLE_ALLOWLIST]
    resolved = [repository_for_name(name) for name in names]
    unknown = [name for name in names if bundle_source(name) is None]
    if unknown:
        preview = ", ".join(unknown)
        print(f"[warn] not in the allowlist registry; assuming repo id {DEFAULT_ORG}/<name>: {preview}")
    return resolved


def split_csv(values: list[str]) -> list[str]:
    parts: list[str] = []
    for value in values:
        parts.extend(item.strip() for item in value.split(",") if item.strip())
    return parts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download IB-Robot model bundles from HuggingFace with per-target artifact filtering.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Targets match case-insensitively against deployment name, backend and SoC,\n"
            "e.g. 310p selects ascend_310p/Ascend310P1, rk3588 selects rknn_rk3588.\n"
            f"HF_ENDPOINT can point to a mirror; default org is {DEFAULT_ORG}."
        ),
    )
    parser.add_argument("--list", action="store_true", help="list allowlisted IB-Robot model bundles and exit")
    parser.add_argument("--models", help="comma-separated repository/bundle names, or 'all'")
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="deployment filter keyword (repeatable): 310p, 310b, rk3588, torch, cpu, cuda, npu, hmm",
    )
    parser.add_argument(
        "--deployment",
        action="append",
        default=[],
        help="exact deployment-name filter inside the already selected targets",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "models",
        help="target models root directory",
    )
    parser.add_argument("--dry-run", action="store_true", help="resolve and print plans without downloading")
    parser.add_argument("--token", default=None, help="HuggingFace token (defaults to HF_TOKEN)")
    return parser.parse_args(argv)


def resolve_bundles(
    names: list[str],
    dest_root: Path,
    targets: list[str],
    deployments: list[str],
    token: str | None = None,
    skip_unmatched: bool = False,
) -> tuple[list[ResolvedBundle], int]:
    """Resolve every requested bundle before touching the destination.

    Resolving up front lets conflicting local directories fail loudly instead
    of letting a later download silently overwrite an earlier one (issue #132),
    and validates the declared model identity of allowlisted bundles.
    """
    resolved: list[ResolvedBundle] = []
    failures = 0
    for name in names:
        try:
            source = bundle_source(name)
            if name in LEGACY_REPOSITORIES and source is None:
                resolved.append(ResolvedBundle(name=name, local_name=LEGACY_REPOSITORIES.get(name, name)))
                continue
            manifest = fetch_manifest(DEFAULT_ORG, name, token)
            if source is not None:
                validate_manifest_identity(source, manifest)
            bundle_name = manifest.get("bundle", {}).get("name") or name
            local_name = runtime_directory(name, bundle_name)
            plan = build_plan(local_name, DEFAULT_ORG, manifest, targets, deployments, repo_name=name)
            resolved.append(ResolvedBundle(name=name, local_name=local_name, plan=plan))
        except NoMatchingDeploymentError as exc:
            if skip_unmatched:
                print(f"[skip] {name}: {exc}")
                continue
            failures += 1
            print(f"[error] {name}: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - report per-bundle, keep going
            failures += 1
            print(f"[error] {name}: {exc}", file=sys.stderr)

    owners: dict[str, str] = {}
    seen: set[str] = set()
    for record in resolved:
        if record.name in seen:
            record.duplicate = True
            continue
        seen.add(record.name)
        previous = owners.setdefault(record.local_name, record.name)
        if previous == record.name:
            continue
        record.conflict_with = previous
        failures += 1
        print(
            f"[error] {record.name}: local directory '{record.local_name}' is already taken by "
            f"'{previous}'; pin an explicit directory in MODEL_BUNDLE_ALLOWLIST so each "
            f"repository downloads into its own directory",
            file=sys.stderr,
        )
    return resolved, failures


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        list_bundles()
        return 0
    if not args.models:
        print("Error: --models is required (or use --list)", file=sys.stderr)
        return 2

    try:
        names = resolve_names(args.models)
    except DownloadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    download_all = args.models.strip().lower() == "all"
    targets = split_csv(args.target)
    deployments = split_csv(args.deployment)
    records, failures = resolve_bundles(names, args.dest, targets, deployments, args.token, skip_unmatched=download_all)
    for record in records:
        if record.conflict_with or record.duplicate:
            continue
        name = record.name
        try:
            if record.plan is None:
                download_legacy_repo(name, args.dest, args.dry_run, args.token)
                continue
            if args.dry_run:
                artifacts = record.plan.artifact_count
                print(
                    f"[dry-run] {record.plan.repo_id}: {artifacts} artifact file(s), "
                    f"{len(record.plan.patterns) - artifacts} shared file(s), deployments: "
                    f"{', '.join(record.plan.matched_deployments)}"
                )
                continue
            download_bundle(record.plan, args.dest, token=args.token)
            print(f"[done] {record.plan.repo_id} -> {args.dest / record.local_name}")
        except Exception as exc:  # noqa: BLE001 - report per-bundle, keep going
            failures += 1
            print(f"[error] {name}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
