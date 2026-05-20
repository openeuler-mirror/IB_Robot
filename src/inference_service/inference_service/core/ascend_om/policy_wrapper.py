"""Thin PolicyWrapper adapters for the migrated Ascend OM ACT wrappers."""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from inference_service.core.ascend_om._common import (
    as_action_tensor,
    candidate_paths,
    chunk_size_from_config,
    load_policy_config,
    normalize_device_name,
    policy_type_from_path,
    resolve_first_existing,
)
from inference_service.core.ascend_om.ACTWrapper_3403 import (
    DEFAULT_OM_BASENAME,
    _guess_worker_path_from_model,
    _is_om_path,
)
from inference_service.core.pure_inference_engine import PolicyWrapper

ACTWrapper = None
ACT3403Policy = None


def __getattr__(name: str) -> Any:
    if name == "ACTWrapper":
        value = import_module("inference_service.core.ascend_om.ACTWrapper").ACTWrapper
    elif name == "ACT3403Policy":
        value = import_module("inference_service.core.ascend_om.ACTWrapper_3403").ACT3403Policy
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def _resolve_runtime_attr(name: str) -> Any:
    value = globals().get(name)
    if value is not None:
        return value
    return __getattr__(name)


# ---------------------------------------------------------------------------
# ACT-specific config view
# ---------------------------------------------------------------------------


class _FeatureConfig:
    def __init__(self, shape: Any):
        self.shape = list(shape) if isinstance(shape, list | tuple) else [int(shape)]


class _ACTConfigView:
    """Config view with the fields consumed by the migrated ACTWrapper."""

    def __init__(self, raw_config: dict[str, Any]):
        self.chunk_size = chunk_size_from_config(raw_config)
        self.input_features = raw_config.get("input_features") or {}
        self.output_features = {
            "action": _FeatureConfig(_action_shape_from_config(raw_config)),
        }


def _shape_from_feature(feature: Any) -> list[int]:
    if isinstance(feature, dict):
        shape = feature.get("shape")
    else:
        shape = getattr(feature, "shape", None)
    if shape is None:
        return []
    if isinstance(shape, int):
        return [shape]
    return [int(dim) for dim in shape]


def _action_shape_from_config(config: dict[str, Any]) -> list[int]:
    output_features = config.get("output_features") or {}
    if isinstance(output_features, dict) and "action" in output_features:
        shape = _shape_from_feature(output_features["action"])
        if shape:
            return shape
    action_dim = config.get("action_dim")
    if action_dim is not None:
        return [int(action_dim)]
    return [6]


# ---------------------------------------------------------------------------
# OM model + SD3403 worker path resolution
# ---------------------------------------------------------------------------


def resolve_om_model_path(path: str, config: dict[str, Any] | None = None) -> Path:
    """Resolve an OM model from policy config, environment, or policy layout."""
    config = config if config is not None else load_policy_config(path)
    raw_path = Path(path).expanduser()

    candidates = candidate_paths(
        path,
        config,
        env_names=("ASCEND_OM_MODEL_PATH", "OM_MODEL_PATH", "SVP_MODEL_PATH"),
        config_key="om_model_path",
        basename_candidates=("model.om", DEFAULT_OM_BASENAME),
        extra_dir_globs=("*.om",),
    )
    # Allow ``path`` itself to be the .om file.
    if raw_path.is_file() and _is_om_path(str(raw_path)):
        candidates.insert(0, raw_path)

    return resolve_first_existing(
        candidates,
        "Ascend OM model file",
        predicate=lambda p: _is_om_path(str(p)),
    )


def resolve_3403_worker_path(path: str, model_path: Path, config: dict[str, Any]) -> Path:
    """Resolve the SD3403 worker executable from config, env, or common layout."""
    raw_path = Path(path).expanduser()

    candidates = candidate_paths(
        path,
        config,
        env_names=("SVP_WORKER_EXECUTABLE", "SVP_CPP_EXECUTABLE"),
        config_key="cpp_executable",
        basename_candidates=("main", "out/main", "build/main"),
    )
    # Allow ``path`` itself to be the executable.
    if raw_path.is_file() and not _is_om_path(str(raw_path)):
        candidates.insert(0, raw_path)
    # Last-resort heuristic relative to the model file.
    candidates.append(Path(_guess_worker_path_from_model(str(model_path))).expanduser())

    return resolve_first_existing(
        candidates,
        "Ascend 3403 worker executable",
        predicate=lambda p: os.access(p, os.X_OK),
    )


# ---------------------------------------------------------------------------
# PolicyWrapper adapters
# ---------------------------------------------------------------------------


class AscendOMPolicyWrapper(PolicyWrapper):
    """PolicyWrapper adapter for the migrated generic ACTWrapper."""

    def __init__(self) -> None:
        self._impl: Any | None = None
        self._device = torch.device("cpu")
        self._chunk_size = 1
        self._policy_type = "ascend_om"

    def load(self, path: str, device: torch.device) -> None:
        self._device = device
        config = load_policy_config(path)
        config_view = _ACTConfigView(config)
        self._chunk_size = config_view.chunk_size
        model_path = resolve_om_model_path(path, config)
        act_wrapper_cls = _resolve_runtime_attr("ACTWrapper")
        self._impl = act_wrapper_cls(str(model_path), config_view)

    def infer(self, batch: dict[str, Tensor]) -> Tensor:
        if self._impl is None:
            raise RuntimeError("AscendOMPolicyWrapper is not loaded")
        output = self._impl.predict(batch)
        if not output:
            raise RuntimeError("Ascend OM model returned no outputs")
        return as_action_tensor(output[0], self._device)

    def get_chunk_size(self) -> int:
        return self._chunk_size

    @property
    def policy_type(self) -> str:
        return self._policy_type

    def close(self) -> None:
        self._impl = None


class AscendOM3403PolicyWrapper(PolicyWrapper):
    """PolicyWrapper adapter for the migrated SD3403 ACT wrapper."""

    def __init__(self) -> None:
        self._impl: Any | None = None
        self._device = torch.device("cpu")
        self._chunk_size = int(os.environ.get("SVP_ACTION_CHUNK_SIZE", "1"))
        self._policy_type = "ascend_om_3403"

    def load(self, path: str, device: torch.device) -> None:
        self._device = device
        config = load_policy_config(path)
        self._chunk_size = chunk_size_from_config(config)
        model_path = resolve_om_model_path(path, config)
        worker_path = resolve_3403_worker_path(path, model_path, config)
        act_3403_cls = _resolve_runtime_attr("ACT3403Policy")
        self._impl = act_3403_cls(str(worker_path), str(model_path))

    def infer(self, batch: dict[str, Tensor]) -> Tensor:
        if self._impl is None:
            raise RuntimeError("AscendOM3403PolicyWrapper is not loaded")
        output = self._impl.predict(batch)
        if not output:
            raise RuntimeError("Ascend 3403 OM model returned no outputs")
        action = as_action_tensor(output[0], self._device)
        if action.ndim >= 2:
            self._chunk_size = int(action.shape[0])
        return action

    def get_chunk_size(self) -> int:
        return self._chunk_size

    @property
    def policy_type(self) -> str:
        return self._policy_type

    def close(self) -> None:
        if self._impl is not None:
            self._impl.close()
            self._impl = None


# ---------------------------------------------------------------------------
# Top-level dispatcher (used by PureInferenceEngine)
# ---------------------------------------------------------------------------


def create_ascend_om_policy_wrapper(device: str, path: str | None = None) -> PolicyWrapper:
    normalized = normalize_device_name(device)
    if normalized == "ascend_om_3403":
        # SD3403 currently only supports ACT.
        return AscendOM3403PolicyWrapper()
    if normalized == "ascend_om":
        policy_type = policy_type_from_path(path) if path else ""
        if policy_type == "pi05":
            from inference_service.core.ascend_om.pi05 import (
                create_ascend_om_pi05_policy_wrapper,
            )

            return create_ascend_om_pi05_policy_wrapper()
        # Default / ACT fallback.
        return AscendOMPolicyWrapper()
    raise ValueError(f"Unsupported Ascend OM inference device: {device}")
