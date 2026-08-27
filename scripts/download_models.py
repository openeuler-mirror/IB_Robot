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
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
    raise SystemExit(
        "Error: huggingface_hub is required. Install the project environment first with ./scripts/setup.sh."
    ) from exc

DEFAULT_ORG = "openEuler"
MANIFEST_FILENAME = "inference_manifest.json"

KNOWN_BUNDLES = {
    "ACT_1arm_2cam_banana_pick_v1_step_160000_distill_20260515",
    "pi05",
    "smolvla",
    "sam2.1_hiera_tiny",
    "ram_plus_swin_large_14m",
    "siglip2_so400m_patch14_384",
    "grounding_dino_swint_seq8_1280x720",
    "graspgen",
    "fullsubnet",
}


class DownloadError(RuntimeError):
    """Raised for bundle selection, planning or verification failures."""


@dataclass
class BundlePlan:
    """Resolved download plan for one bundle."""

    name: str
    repo_id: str
    patterns: list[str]
    verify_map: dict[str, str]
    matched_deployments: list[str]

    @property
    def artifact_count(self) -> int:
        return len(self.patterns) - 1


@dataclass
class DeploymentInfo:
    """Summary of a deployment as seen by target filtering."""

    name: str
    backend: str
    soc: str
    artifact_paths: list[str] = field(default_factory=list)

    @property
    def haystack(self) -> str:
        return f"{self.name} {self.backend} {self.soc}".lower()

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
        artifacts = [
            spec["path"]
            for spec in (deployment.get("artifacts", {}) or {}).values()
            if isinstance(spec, dict) and spec.get("path")
        ]
        infos.append(DeploymentInfo(name=name, backend=backend, soc=soc, artifact_paths=artifacts))
    return infos


def filter_deployments(infos: list[DeploymentInfo], keywords: list[str]) -> tuple[list[DeploymentInfo], bool]:
    """Return matching deployments plus whether any keyword was checked."""
    lowered = [keyword.lower() for keyword in keywords]
    matches = [info for info in infos if any(keyword in info.haystack for keyword in lowered)]
    return matches, bool(lowered)


def build_plan(name: str, org: str, manifest: dict[str, Any], targets: list[str], deployments: list[str]) -> BundlePlan:
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
        raise DownloadError(f"no deployment of '{name}' matches [{wanted}]. Available: {available}")

    verify_map: dict[str, str] = {}
    patterns: set[str] = {MANIFEST_FILENAME, *shared_paths}
    for info in selected:
        for artifact_path in info.artifact_paths:
            patterns.add(artifact_path)
            digest = _artifact_digest(manifest, info.name, artifact_path)
            if digest:
                verify_map[artifact_path] = digest.lower()
    unmatched_verifies = set(verify_map) - patterns
    if unmatched_verifies:
        raise DownloadError(f"internal error: verification paths outside plan: {sorted(unmatched_verifies)}")

    return BundlePlan(
        name=name,
        repo_id=f"{org}/{name}",
        patterns=sorted(patterns),
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


def verify_downloads(bundle_dir: Path, verify_map: dict[str, str]) -> None:
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


def download_bundle(plan: BundlePlan, dest_root: Path, dry_run: bool = False) -> Path:
    bundle_dir = dest_root / plan.name
    if dry_run:
        return bundle_dir
    bundle_dir.mkdir(parents=True, exist_ok=True)
    print(f"[plan] {plan.repo_id}: {len(plan.patterns)} paths (deployments: {', '.join(plan.matched_deployments)})")
    for pattern in plan.patterns:
        print(f"       - {pattern}")
    snapshot_download(repo_id=plan.repo_id, local_dir=str(bundle_dir), allow_patterns=plan.patterns)
    verify_downloads(bundle_dir, plan.verify_map)
    prune_hf_cache(bundle_dir)
    return bundle_dir


def fetch_manifest(org: str, name: str, dest_root: Path) -> dict[str, Any]:
    """Download just the manifest into <dest_root>/<name>/ and parse it."""
    bundle_dir = dest_root / name
    local = hf_hub_download(repo_id=f"{org}/{name}", filename=MANIFEST_FILENAME, local_dir=str(bundle_dir))
    return load_manifest(Path(local))


def list_bundles(api_token: str | None) -> None:
    api = HfApi(token=api_token)
    known = ", ".join(sorted(KNOWN_BUNDLES))
    print(f"Known bundles (org={DEFAULT_ORG}):")
    print(f"  {known}")
    try:
        remote = api.list_repo_files(repo_id=f"{DEFAULT_ORG}/{next(iter(sorted(KNOWN_BUNDLES)))}")
    except Exception:
        print("\n(note: could not probe remote; pass --token or check network)")
        return
    del remote


def resolve_names(requested: str) -> list[str]:
    names = [item.strip() for item in requested.split(",") if item.strip()]
    if not names:
        raise DownloadError("--models requires at least one bundle name (see --list)")
    unknown = [name for name in names if name not in KNOWN_BUNDLES]
    if unknown:
        preview = ", ".join(unknown)
        print(f"[warn] not in the known-bundle registry; assuming repo id {DEFAULT_ORG}/<name>: {preview}")
    return names


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
    parser.add_argument("--list", action="store_true", help="list known bundles and exit")
    parser.add_argument("--models", help="comma-separated bundle names, e.g. pi05,sam2.1_hiera_tiny")
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        list_bundles(args.token)
        return 0
    if not args.models:
        print("Error: --models is required (or use --list)", file=sys.stderr)
        return 2

    names = resolve_names(args.models)
    targets = split_csv(args.target)
    deployments = split_csv(args.deployment)
    failures = 0
    for name in names:
        try:
            manifest = fetch_manifest(DEFAULT_ORG, name, args.dest)
            plan = build_plan(name, DEFAULT_ORG, manifest, targets, deployments)
            if args.dry_run:
                artifacts = len(plan.patterns) - 1
                print(
                    f"[dry-run] {plan.repo_id}: {artifacts} artifact file(s), "
                    f"{len(plan.patterns) - artifacts} shared file(s), deployments: "
                    f"{', '.join(plan.matched_deployments)}"
                )
                continue
            download_bundle(plan, args.dest)
            print(f"[done] {plan.repo_id} -> {args.dest / name}")
        except Exception as exc:  # noqa: BLE001 - report per-bundle, keep going
            failures += 1
            print(f"[error] {name}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
