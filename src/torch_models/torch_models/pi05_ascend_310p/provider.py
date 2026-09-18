"""Runtime provider for PI0.5 native Torch inference on Ascend310P."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

from packaging.version import Version

from torch_models.pi05_ascend_310p.modeling_pi05_ascend_310p import (
    PI05Ascend310PPolicy,
    configure_pi05_ascend_310p_config,
)
from torch_models.policy_provider import PolicyProvider

PI05_ASCEND_310P_LOAD_OPTIONS = {"skip_weight_init": True}


def validate_pi05_ascend_310p(*, config, bundle_root: Path, tokenizer_path: str | None, device_name: str) -> None:
    """Validate the platform-specific contract before allocating model weights."""

    if "Ascend310P" not in device_name:
        raise ValueError(f"PI05 Ascend310P provider requires Ascend310P, got {device_name!r}")
    if tokenizer_path is None:
        raise ValueError("PI05 Ascend310P provider requires a bundled tokenizer")
    if not (bundle_root / "model.safetensors").is_file():
        raise ValueError("PI05 Ascend310P provider requires bundled model.safetensors")
    try:
        transformers_version = Version(package_version("transformers"))
    except (PackageNotFoundError, ValueError) as exc:
        raise ValueError(f"unable to identify Transformers: {exc}") from exc
    if transformers_version.base_version != "5.3.0":
        raise ValueError(f"PI05 Ascend310P provider requires Transformers 5.3.0, got {transformers_version}")


def prepare_pi05_ascend_310p(*, policy, deployment_fingerprint, torch_module, torch_npu_module, device_name):
    return policy.prepare_for_inference(
        deployment_fingerprint=deployment_fingerprint,
        torch_module=torch_module,
        torch_npu_module=torch_npu_module,
        device_name=device_name,
    )


def _execution_metadata(policy) -> dict[str, object]:
    records = policy.model.get_action_fused_stage_timing_records()
    return {"pi05_stage_timing": records[-1]} if records else {}


def create_provider() -> PolicyProvider:
    return PolicyProvider(
        policy_class=PI05Ascend310PPolicy,
        configure_config=configure_pi05_ascend_310p_config,
        validate=validate_pi05_ascend_310p,
        prepare=prepare_pi05_ascend_310p,
        load_options=PI05_ASCEND_310P_LOAD_OPTIONS,
        execution_metadata=_execution_metadata,
    )


__all__ = [
    "PI05_ASCEND_310P_LOAD_OPTIONS",
    "PI05Ascend310PPolicy",
    "configure_pi05_ascend_310p_config",
    "create_provider",
    "prepare_pi05_ascend_310p",
    "validate_pi05_ascend_310p",
]
