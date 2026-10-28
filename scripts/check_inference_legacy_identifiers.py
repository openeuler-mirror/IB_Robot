#!/usr/bin/env python3
"""Reject removed inference architecture identifiers in active source and documentation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ACTIVE_ROOTS = ("src", "scripts")
ACTIVE_SUFFIXES = frozenset({".py", ".sh", ".yaml", ".yml", ".json", ".xml", ".md"})
EXCLUDED_DIRECTORIES = frozenset({"__pycache__", "build", "install", "log"})
DOCUMENTATION_ROOT = "docs"
EXCLUDED_DOCUMENTATION_DIRECTORIES = frozenset({"migration", "official", "source", "yocto-meta-openeuler"})
SELF_PATH = "scripts/check_inference_legacy_identifiers.py"

PATTERNS = {
    "per_backend_manifest": re.compile(r"(?<![A-Za-z0-9_])config\.(?:om|rknn|hmm)\.json(?![A-Za-z0-9_])"),
    "per_backend_manifest_option": re.compile(
        r"(?<![A-Za-z0-9_])(?:skip_om_manifest|--skip-om-manifest|--om-manifest-dir)(?![A-Za-z0-9_])"
    ),
    "removed_backend_alias": re.compile(r"(?<![A-Za-z0-9_])(?:ascend_om(?:_3403)?|3403)(?![A-Za-z0-9_])"),
    "legacy_backend_field": re.compile(
        r"(?<![A-Za-z0-9_])(?:is_rknn_enabled|is_hmm_enabled|is_ascend_om_enabled)(?![A-Za-z0-9_])"
    ),
    "legacy_artifact_override": re.compile(r"(?<![A-Za-z0-9_])HMM_MODEL_PATH(?![A-Za-z0-9_])"),
    "legacy_runtime_surface": re.compile(
        r"(?<![A-Za-z0-9_])(?:lerobot_policy_node|InferenceCoordinator|CompiledPolicyWrapper)(?![A-Za-z0-9_])"
        r"|inference_service\.core\.ascend_om"
    ),
    "legacy_ros_surface": re.compile(r"/act_inference_node(?:/|\b)|/(?:[A-Za-z0-9_]+/)*reset_policy_state(?:\b|/)"),
    "legacy_launch_argument": re.compile(r"(?<![A-Za-z0-9_])(?:execution_mode|cloud_local|policy_path):="),
    "legacy_distributed_topic": re.compile(r"/preprocessed/batch(?:\b|/)|/inference/action(?:\b|/)"),
    "removed_native_migration_surface": re.compile(
        r"inference_service\.(?:_legacy_named_tensor|generic_runtime|backends\.lifecycle)"
        r"|(?<![A-Za-z0-9_])(?:GenericModelPipeline|NamedTensorRequest|NamedTensorResult)(?![A-Za-z0-9_])"
    ),
}

# These tests intentionally prove that removed backend aliases are rejected.
ALLOWLIST = {
    "src/model_utils/test/test_pi05_export_manifest.py": frozenset({"per_backend_manifest_option"}),
    "src/inference_service/test/test_backend_contract.py": frozenset({"removed_backend_alias"}),
    "src/inference_service/test/test_inference_manifest.py": frozenset({"removed_backend_alias"}),
    "src/inference_service/test/test_legacy_identifier_guard.py": frozenset(
        {"legacy_launch_argument", "legacy_runtime_surface"}
    ),
    "docs/Houmo_HMM_Conversion.md": frozenset({"per_backend_manifest", "legacy_artifact_override"}),
}


def find_violations(repository_root: Path) -> list[str]:
    violations: list[str] = []
    for relative_path, path in _active_files(repository_root):
        allowed = ALLOWLIST.get(relative_path, frozenset())
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for category, pattern in PATTERNS.items():
                if category in allowed or not pattern.search(line):
                    continue
                violations.append(f"{relative_path}:{line_number}: {category}: {line.strip()}")
    return violations


def _active_files(repository_root: Path):
    yielded: set[str] = set()
    for root_name in ACTIVE_ROOTS:
        root = repository_root / root_name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in ACTIVE_SUFFIXES:
                continue
            relative_path = path.relative_to(repository_root).as_posix()
            if relative_path == SELF_PATH or EXCLUDED_DIRECTORIES.intersection(path.relative_to(root).parts):
                continue
            yielded.add(relative_path)
            yield relative_path, path

    documentation_root = repository_root / DOCUMENTATION_ROOT
    if documentation_root.is_dir():
        for path in sorted(documentation_root.rglob("*.md")):
            relative_to_docs = path.relative_to(documentation_root)
            if EXCLUDED_DOCUMENTATION_DIRECTORIES.intersection(relative_to_docs.parts):
                continue
            relative_path = path.relative_to(repository_root).as_posix()
            if relative_path not in yielded:
                yielded.add(relative_path)
                yield relative_path, path

    for path in sorted(repository_root.glob("README*.md")):
        relative_path = path.relative_to(repository_root).as_posix()
        if path.is_file() and relative_path not in yielded:
            yield relative_path, path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    repository_root = args.root.expanduser().resolve(strict=True)
    violations = find_violations(repository_root)
    if violations:
        print("Removed inference identifiers found in active repository files:")
        for violation in violations:
            print(f"  {violation}")
        return 1
    print("Inference legacy identifier check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
