"""Package compiler-produced artifacts into a unified inference deployment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from inference_manifest import DeviceLink
from model_utils.inference_manifest_export import (
    artifact_bindings,
    compiled_deployment,
    load_published_artifact_index,
    package_deployment_artifact,
    read_runtime_abi,
    read_tcim_abi,
    upsert_deployment,
)


def package_compiled_deployment(
    *,
    bundle_root: str | Path,
    deployment_name: str,
    backend: str,
    target_soc: str,
    target_runtime: str,
    spec_path: str | Path,
):
    """Package one complete compiler spec and validate it with the runtime loader."""

    root = Path(bundle_root).expanduser().resolve(strict=True)
    spec = _load_spec(spec_path)
    execution = _string_list(spec.get("execution"), "execution")
    roles = spec.get("roles")
    if not isinstance(roles, dict):
        raise ValueError("Packaging spec requires a roles object")

    published = load_published_artifact_index(root)
    artifacts: dict[str, tuple[str | Path, str]] = {}
    bindings = {}
    for role in execution:
        role_spec = roles.get(role)
        if not isinstance(role_spec, dict):
            raise ValueError(f"Packaging spec is missing execution role {role!r}")
        artifact = _non_empty_string(role_spec.get("artifact"), f"roles.{role}.artifact")
        artifact_format = _non_empty_string(role_spec.get("format"), f"roles.{role}.format")
        abi_path = _non_empty_string(role_spec.get("abi"), f"roles.{role}.abi")
        abi_format = role_spec.get("abi_format", "runtime")
        if abi_format not in {"runtime", "tcim"}:
            raise ValueError(f"roles.{role}.abi_format must be 'runtime' or 'tcim'")
        input_semantics = _string_mapping(role_spec.get("input_semantics"), f"roles.{role}.input_semantics")
        output_semantics = _string_mapping(role_spec.get("output_semantics"), f"roles.{role}.output_semantics")
        image_layouts = _string_mapping(role_spec.get("image_layouts", {}), f"roles.{role}.image_layouts")
        artifacts[role] = (
            package_deployment_artifact(
                root,
                _resolve_spec_path(spec_path, artifact),
                backend=backend,
                deployment_name=deployment_name,
                role=role,
                published=published,
            ),
            artifact_format,
        )
        abi = (
            read_tcim_abi(_resolve_spec_path(spec_path, abi_path))
            if abi_format == "tcim"
            else read_runtime_abi(_resolve_spec_path(spec_path, abi_path))
        )
        bindings[role] = artifact_bindings(
            abi,
            input_semantics=input_semantics,
            output_semantics=output_semantics,
            image_layouts=image_layouts,
        )

    extra_artifacts = spec.get("artifacts", {})
    if not isinstance(extra_artifacts, dict):
        raise ValueError("Packaging spec artifacts must be an object")
    for role, artifact_spec in extra_artifacts.items():
        if role in artifacts:
            raise ValueError(f"Packaging spec defines duplicate artifact role {role!r}")
        if not isinstance(artifact_spec, dict):
            raise ValueError(f"Packaging spec artifact {role!r} must be an object")
        path = _non_empty_string(artifact_spec.get("path"), f"artifacts.{role}.path")
        artifact_format = _non_empty_string(artifact_spec.get("format"), f"artifacts.{role}.format")
        artifacts[role] = (
            package_deployment_artifact(
                root,
                _resolve_spec_path(spec_path, path),
                backend=backend,
                deployment_name=deployment_name,
                role=role,
                published=published,
            ),
            artifact_format,
        )

    links = spec.get("device_links", [])
    if not isinstance(links, list):
        raise ValueError("Packaging spec device_links must be a list")
    deployment = compiled_deployment(
        root,
        backend=backend,
        target_soc=target_soc,
        target_runtime=target_runtime,
        artifacts=artifacts,
        published=published,
        execution=execution,
        bindings=bindings,
        device_links=tuple(DeviceLink.model_validate(link) for link in links),
    )
    _validate_backend_package(deployment, root)
    return upsert_deployment(root, deployment_name, deployment)


def _validate_backend_package(deployment, root: Path) -> None:
    execution_formats = {deployment.artifacts[role].format for role in deployment.execution}
    runtime = deployment.target.runtime
    if deployment.backend == "hisilicon":
        _validate_hisilicon_package(deployment, root)
    elif deployment.backend == "ascend":
        if execution_formats != {"om"} or not (runtime.startswith("acl") or runtime.startswith("ascend")):
            raise ValueError("Ascend deployment requires OM execution artifacts and an ACL runtime target")
    elif deployment.backend == "rknn":
        invalid = {
            role: deployment.artifacts[role].format
            for role in deployment.execution
            if role != "embedding" and deployment.artifacts[role].format != "rknn"
        }
        if invalid or not runtime.startswith("rknn"):
            raise ValueError("RKNN deployment requires RKNN execution artifacts and an RKNN runtime target")
    elif deployment.backend == "hmm":
        invalid = {
            role: deployment.artifacts[role].format
            for role in deployment.execution
            if role != "embedding" and deployment.artifacts[role].format != "hmm"
        }
        if invalid or not (runtime.startswith("hmm") or runtime.startswith("tcim")):
            raise ValueError("HMM deployment requires HMM execution artifacts and a TCIM/HMM runtime target")


def _validate_hisilicon_package(deployment, root: Path) -> None:
    if deployment.execution != ("policy",):
        raise ValueError("Hisilicon deployment requires execution ['policy']")
    if deployment.target.soc != "sd3403" or deployment.target.runtime != "hisilicon-worker":
        raise ValueError("Hisilicon deployment requires target sd3403/hisilicon-worker")
    if set(deployment.artifacts) != {"policy", "worker"}:
        raise ValueError("Hisilicon deployment requires exactly policy and worker artifacts")
    if deployment.artifacts["policy"].format != "om" or deployment.artifacts["worker"].format != "executable":
        raise ValueError("Hisilicon policy/worker formats must be om/executable")
    worker = root.joinpath(*deployment.artifacts["worker"].path.split("/"))
    if not os.access(worker, os.X_OK):
        raise ValueError(f"Hisilicon worker is not executable: {worker}")


def _load_spec(path: str | Path) -> dict:
    spec_path = Path(path).expanduser().resolve(strict=True)
    with spec_path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Packaging spec must be a JSON object: {spec_path}")
    return value


def _resolve_spec_path(spec_path: str | Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(spec_path).expanduser().resolve(strict=True).parent / path
    return path.resolve(strict=True)


def _non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _string_list(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must be a non-empty string list")
    return tuple(value)


def _string_mapping(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not key or not isinstance(item, str) or not item for key, item in value.items()
    ):
        raise ValueError(f"{name} must be a string-to-string object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--backend", choices=("ascend", "hisilicon", "rknn", "hmm"), required=True)
    parser.add_argument("--target-soc", required=True)
    parser.add_argument("--target-runtime", required=True)
    parser.add_argument("--spec", required=True, help="Compiler packaging JSON with complete runtime ABI mappings")
    args = parser.parse_args()
    validated = package_compiled_deployment(
        bundle_root=args.bundle_root,
        deployment_name=args.deployment,
        backend=args.backend,
        target_soc=args.target_soc,
        target_runtime=args.target_runtime,
        spec_path=args.spec,
    )
    print(validated.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
