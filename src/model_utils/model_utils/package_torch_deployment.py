"""Generate native Torch deployments for a LeRobot policy bundle."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from inference_manifest import (
    DeploymentTarget,
    ExecutionContract,
    RoleRuntimeProfile,
    TorchDeployment,
    TorchRuntimeProfile,
    ValidatedManifest,
)
from model_utils.inference_manifest_export import upsert_deployment

_TORCH_DEVICES = ("cpu", "cuda", "mps", "npu")


def package_torch_deployments(
    bundle_root: str | Path,
    *,
    devices: Sequence[str] = ("cpu", "cuda"),
    deployment_prefix: str = "torch",
) -> tuple[ValidatedManifest, ...]:
    """Add one named native Torch deployment for each requested device."""

    selected_devices = tuple(devices)
    if not selected_devices:
        raise ValueError("at least one Torch device is required")
    if len(selected_devices) != len(set(selected_devices)):
        raise ValueError("Torch devices must not contain duplicates")
    unsupported = sorted(set(selected_devices) - set(_TORCH_DEVICES))
    if unsupported:
        raise ValueError(f"unsupported Torch devices: {unsupported}")
    if not deployment_prefix:
        raise ValueError("deployment prefix must not be empty")

    return tuple(
        upsert_deployment(
            bundle_root,
            f"{deployment_prefix}-{device}",
            TorchDeployment(
                execution_contract=ExecutionContract(
                    state_scope="request",
                    execution_structure="direct",
                    cancellation_granularity="request_boundary",
                ),
                runtime_profile=RoleRuntimeProfile(
                    backend="torch",
                    target=DeploymentTarget(runtime="torch"),
                    profile=TorchRuntimeProfile(device=device),
                ),
            ),
        )
        for device in selected_devices
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True, help="LeRobot save_pretrained policy directory")
    parser.add_argument(
        "--devices",
        nargs="+",
        choices=_TORCH_DEVICES,
        default=("cpu", "cuda"),
        help="Torch devices to register (default: cpu cuda)",
    )
    parser.add_argument(
        "--deployment-prefix",
        default="torch",
        help="Deployment name prefix (default: torch)",
    )
    args = parser.parse_args()

    validated = package_torch_deployments(
        args.bundle_root,
        devices=args.devices,
        deployment_prefix=args.deployment_prefix,
    )
    print(validated[-1].manifest_path)
    for item in validated:
        print(f"{item.deployment_name}: {item.deployment.device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
