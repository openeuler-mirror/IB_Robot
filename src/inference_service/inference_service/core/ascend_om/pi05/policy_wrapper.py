"""PolicyWrapper adapter for the PI05 Ascend OM inference backend."""

from __future__ import annotations

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
    resolve_first_existing,
)
from inference_service.core.pure_inference_engine import PolicyWrapper

# Lazy-loaded heavy backend (depends on ACL on NPU hosts).
PI05Wrapper = None


def __getattr__(name: str) -> Any:
    if name == "PI05Wrapper":
        value = import_module("inference_service.core.ascend_om.pi05.PI05Wrapper").PI05Wrapper
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------


class _FeatureSpec:
    def __init__(self, feature_type: str, shape: list[int]):
        self.type = _FeatureType(feature_type)
        self.shape = list(shape)


class _FeatureType:
    def __init__(self, name: str):
        self.name = (name or "").upper()

    def __eq__(self, other: object) -> bool:  # pragma: no cover - trivial
        if isinstance(other, _FeatureType):
            return self.name == other.name
        return False


def _feature_type(feature: Any) -> str:
    if isinstance(feature, dict):
        return str(feature.get("type", "")).upper()
    ftype = getattr(feature, "type", None)
    if ftype is None:
        return ""
    return str(getattr(ftype, "name", ftype)).upper()


def _feature_shape(feature: Any) -> list[int]:
    if isinstance(feature, dict):
        shape = feature.get("shape") or []
    else:
        shape = getattr(feature, "shape", None) or []
    if isinstance(shape, int):
        return [shape]
    return [int(dim) for dim in shape]


def _ordered_image_features(raw_config: dict[str, Any]) -> dict[str, _FeatureSpec]:
    """Return image features as an ordered dict mirroring the saved config."""
    inputs = raw_config.get("input_features") or {}
    if not isinstance(inputs, dict):
        return {}
    result: dict[str, _FeatureSpec] = {}
    for key, value in inputs.items():
        ftype = _feature_type(value)
        if ftype == "VISUAL":
            result[key] = _FeatureSpec(ftype, _feature_shape(value))
    return result


# ---------------------------------------------------------------------------
# PI05 config view (duck-typed)
# ---------------------------------------------------------------------------


class _PI05ConfigView:
    """Minimal config exposing only the fields the PI05Wrapper consumes."""

    def __init__(self, raw_config: dict[str, Any]):
        self.chunk_size = chunk_size_from_config(raw_config)
        self.max_action_dim = int(raw_config.get("max_action_dim", 32))
        self.num_inference_steps = int(raw_config.get("num_inference_steps", 10))
        self.image_features = _ordered_image_features(raw_config)


# ---------------------------------------------------------------------------
# OM path resolution (VLM + Action Expert)
# ---------------------------------------------------------------------------


def _is_om_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() == ".om"


def resolve_pi05_om_paths(path: str, config: dict[str, Any] | None = None) -> tuple[Path, Path]:
    """Resolve (VLM, action expert) OM files for PI05."""
    config = config if config is not None else load_policy_config(path)

    vlm_candidates = candidate_paths(
        path,
        config,
        env_names=("PI05_OM_VLM_PATH", "PI05_VLM_OM_PATH"),
        config_key="om_vlm_model_path",
        basename_candidates=("vlm.om", "pi05_vlm.om"),
        extra_dir_globs=("*vlm*.om",),
    )
    ae_candidates = candidate_paths(
        path,
        config,
        env_names=("PI05_OM_AE_PATH", "PI05_OM_ACTION_EXPERT_PATH"),
        config_key="om_action_expert_model_path",
        basename_candidates=("action_expert.om", "pi05_action_expert.om"),
        extra_dir_globs=("*action_expert*.om", "*ae*.om"),
    )

    vlm = resolve_first_existing(
        vlm_candidates,
        "PI05 VLM OM model file",
        predicate=_is_om_file,
    )
    ae = resolve_first_existing(
        ae_candidates,
        "PI05 action-expert OM model file",
        predicate=_is_om_file,
    )
    return vlm, ae


# ---------------------------------------------------------------------------
# PolicyWrapper adapter
# ---------------------------------------------------------------------------


class AscendOMPi05PolicyWrapper(PolicyWrapper):
    """PolicyWrapper adapter for the PI05 fused OM backend."""

    def __init__(self) -> None:
        self._impl: Any | None = None
        self._device = torch.device("cpu")
        self._chunk_size = 50
        self._policy_type = "pi05"

    def load(self, path: str, device: torch.device) -> None:
        self._device = device
        config = load_policy_config(path)
        config_view = _PI05ConfigView(config)
        self._chunk_size = config_view.chunk_size
        vlm_path, ae_path = resolve_pi05_om_paths(path, config)
        wrapper_cls = __getattr__("PI05Wrapper") if PI05Wrapper is None else PI05Wrapper
        self._impl = wrapper_cls(
            vlm_model_path=str(vlm_path),
            action_expert_model_path=str(ae_path),
            config=config_view,
        )

    def infer(self, batch: dict[str, Tensor]) -> Tensor:
        if self._impl is None:
            raise RuntimeError("AscendOMPi05PolicyWrapper is not loaded")
        output = self._impl.predict(batch)
        if not output:
            raise RuntimeError("PI05 OM model returned no outputs")
        return as_action_tensor(output[0], self._device)

    def get_chunk_size(self) -> int:
        return self._chunk_size

    @property
    def policy_type(self) -> str:
        return self._policy_type

    def close(self) -> None:
        if self._impl is not None and hasattr(self._impl, "close"):
            self._impl.close()
        self._impl = None


def create_ascend_om_pi05_policy_wrapper() -> PolicyWrapper:
    return AscendOMPi05PolicyWrapper()
