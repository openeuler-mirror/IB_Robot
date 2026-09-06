"""Latency-relevant inference runtime options shared by config and runtime.

`robot_config` folds these effective values into the per-pipeline
profile-compatibility fingerprint, and `inference_service` loads Torch
sessions with the same defaults. Keeping the table in one place means a
runtime option that changes model execution latency — for example enabling
AutoHorizon attention collection, which adds per-layer GPU→CPU attention
copies inside the policy forward pass — always changes the calibrated
latency-profile identity and forces recalibration instead of silently
reusing measurements taken under a different execution configuration.

This module is intentionally stdlib-only so both packages can import it
without ROS or numpy dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

TORCH_RUNTIME_MODEL_DTYPE_DEFAULT = "native"

AUTO_HORIZON_RUNTIME_OPTION_DEFAULTS: Mapping[str, object] = MappingProxyType(
    {
        "auto_horizon_enabled": False,
        "auto_horizon_hold_threshold": 0.3,
        "auto_horizon_entropy_quantile": 0.9,
        "auto_horizon_run_length": 1,
        "auto_horizon_sampling_step": 3,
    }
)


def effective_latency_runtime_options(options: Mapping[str, object]) -> dict[str, object]:
    """Return the effective latency-relevant runtime options for identity.

    Known Torch options missing from ``options`` are filled with their
    effective defaults so a config that spells out a default explicitly
    keeps the same identity as one that omits it. Unknown options pass
    through unchanged: any future option still changes the identity and
    therefore requires recalibrated profiles (fail-closed direction).
    """
    effective = dict(options)
    effective.setdefault("model_dtype", TORCH_RUNTIME_MODEL_DTYPE_DEFAULT)
    for name, default in AUTO_HORIZON_RUNTIME_OPTION_DEFAULTS.items():
        effective.setdefault(name, default)
    return effective


__all__ = [
    "AUTO_HORIZON_RUNTIME_OPTION_DEFAULTS",
    "TORCH_RUNTIME_MODEL_DTYPE_DEFAULT",
    "effective_latency_runtime_options",
]
