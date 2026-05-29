from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor

from inference_service.core._policy_config import override_runtime_policy_device
from inference_service.core.pure_inference_engine import PolicyWrapper

COMPILED_MANIFEST_BASENAME = "config.om.json"


@dataclass(frozen=True)
class CompiledManifest:
    artifacts: dict[str, Path]
    execution: list[str]

    def require_artifact(self, role: str, *, suffix: str | None = None) -> Path:
        try:
            artifact = self.artifacts[role]
        except KeyError as exc:
            roles = ", ".join(sorted(self.artifacts)) or "<none>"
            raise KeyError(f"Compiled manifest is missing artifact role {role!r}; available roles: {roles}") from exc
        if suffix is not None and artifact.suffix.lower() != suffix:
            raise ValueError(f"Compiled artifact {role!r} must be a {suffix} file: {artifact}")
        if not artifact.is_file():
            raise FileNotFoundError(f"Compiled artifact {role!r} does not exist: {artifact}")
        return artifact.resolve()

    def require_execution(self, expected: list[str]) -> None:
        if not self.execution:
            return
        if self.execution != expected:
            raise ValueError(f"Compiled manifest execution must be {expected}, got {self.execution}")


class CompiledModelAdapter(Protocol):
    @classmethod
    def from_config(cls, config: dict[str, Any], backend: str) -> CompiledModelAdapter: ...

    def prepare_inputs(self, batch: dict[str, Tensor]) -> Any: ...

    def decode_outputs(self, raw: Any, device: torch.device) -> Tensor: ...

    def get_chunk_size(self) -> int: ...

    @property
    def policy_type(self) -> str: ...

    @property
    def uses_action_chunking(self) -> bool: ...


class RuntimeSession(Protocol):
    def load(self, policy_path: str, config: dict[str, Any], device: torch.device) -> None: ...

    def execute(self, inputs: Any) -> Any: ...

    def release(self) -> None: ...


def normalize_backend_name(device: str) -> str:
    return str(device).lower().strip().replace("-", "_")


def _policy_config_path(path: str) -> Path | None:
    candidate = Path(path).expanduser()
    if candidate.is_file() and candidate.name == "config.json":
        return candidate
    if candidate.is_dir():
        config_path = candidate / "config.json"
        if config_path.is_file():
            return config_path
    return None


def _manifest_config_path(path: str) -> Path | None:
    candidate = Path(path).expanduser()
    if candidate.is_file() and candidate.name == COMPILED_MANIFEST_BASENAME:
        return candidate
    if candidate.is_dir():
        manifest_path = candidate / COMPILED_MANIFEST_BASENAME
        if manifest_path.is_file():
            return manifest_path
    return None


def load_compiled_policy_config(
    path: str,
    backend: str,
    runtime_device: Any | None = None,
) -> dict[str, Any]:
    config_path = _policy_config_path(path)
    if config_path is None:
        raise FileNotFoundError(
            f"Compiled backend {backend} requires policy_path/config.json with policy type metadata"
        )
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Compiled backend {backend} policy config must be a JSON object: {config_path}")
    return override_runtime_policy_device(data, runtime_device)


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON config must be an object: {path}")
    return data


def _resolve_manifest_artifact_path(base_dir: Path, value: Any, role: str) -> Path:
    if isinstance(value, str):
        artifact_path = Path(value).expanduser()
    elif isinstance(value, dict) and isinstance(value.get("path"), str):
        artifact_path = Path(value["path"]).expanduser()
    else:
        raise ValueError(f"Compiled manifest artifact {role!r} must be a path string or object with path")
    if not artifact_path.is_absolute():
        artifact_path = base_dir / artifact_path
    return artifact_path


def load_compiled_manifest(path: str, backend: str, policy_type: str | None = None) -> CompiledManifest | None:
    manifest_path = _manifest_config_path(path)
    if manifest_path is None:
        raise FileNotFoundError(
            f"Compiled backend {backend} requires {COMPILED_MANIFEST_BASENAME} under policy_path {path}"
        )
    data = _read_json_object(manifest_path)
    manifest_backend = str(data.get("backend", "")).lower().strip()
    if manifest_backend and normalize_backend_name(manifest_backend) != normalize_backend_name(backend):
        raise ValueError(
            f"Compiled manifest backend {manifest_backend!r} does not match requested backend {backend!r}: {manifest_path}"
        )
    manifest_policy = str(data.get("policy_type", "")).lower().strip()
    if policy_type and manifest_policy and manifest_policy != policy_type:
        raise ValueError(
            f"Compiled manifest policy_type {manifest_policy!r} does not match config type {policy_type!r}: {manifest_path}"
        )

    artifact_dir = data.get("artifact_dir", "")
    base_dir = manifest_path.parent
    if isinstance(artifact_dir, str) and artifact_dir:
        artifact_base = Path(artifact_dir).expanduser()
        if not artifact_base.is_absolute():
            artifact_base = base_dir / artifact_base
        base_dir = artifact_base

    raw_artifacts = data.get("artifacts")
    if not isinstance(raw_artifacts, dict) or not raw_artifacts:
        raise ValueError(f"Compiled manifest must define non-empty artifacts map: {manifest_path}")
    artifacts = {
        str(role): _resolve_manifest_artifact_path(base_dir, value, str(role)) for role, value in raw_artifacts.items()
    }

    raw_execution = data.get("execution", [])
    if raw_execution is None:
        execution: list[str] = []
    elif isinstance(raw_execution, list):
        execution = [str(role) for role in raw_execution]
    else:
        raise ValueError(f"Compiled manifest execution must be a list of artifact roles: {manifest_path}")
    return CompiledManifest(artifacts=artifacts, execution=execution)


def _shape_from_feature(feature: Any) -> list[int]:
    if isinstance(feature, dict):
        shape = feature.get("shape")
    elif isinstance(feature, list | tuple):
        shape = feature
    else:
        shape = getattr(feature, "shape", None)
    if shape is None:
        return []
    if isinstance(shape, int):
        return [shape]
    return [int(dim) for dim in shape]


def _feature_type(feature: Any) -> str:
    if isinstance(feature, dict):
        return str(feature.get("type", "")).upper()
    feature_type = getattr(feature, "type", None)
    if feature_type is None:
        return ""
    return str(getattr(feature_type, "name", feature_type)).upper()


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


def _chunk_size_from_config(config: dict[str, Any]) -> int:
    for key in ("chunk_size", "n_action_steps", "action_chunk_size"):
        value = config.get(key)
        if value is not None:
            return int(value)
    action_shape = _action_shape_from_config(config)
    if len(action_shape) >= 2:
        return int(action_shape[-2])
    return 1


def _action_dim_from_config(config: dict[str, Any]) -> int:
    action_shape = _action_shape_from_config(config)
    if action_shape:
        return int(action_shape[-1])
    return 6


def _real_action_dim_from_config(config: dict[str, Any], fallback: int) -> int:
    output_features = config.get("output_features") or {}
    if isinstance(output_features, dict) and "action" in output_features:
        action_shape = _shape_from_feature(output_features["action"])
        if action_shape:
            return int(action_shape[-1])
    action_dim = config.get("action_dim")
    if action_dim is not None:
        return int(action_dim)
    return int(fallback)


def _input_order_from_config(config: dict[str, Any]) -> list[str]:
    for key in ("compiled_runtime_input_order", "runtime_input_order", "input_order"):
        value = config.get(key)
        if isinstance(value, list) and value:
            return [str(item) for item in value]

    input_features = config.get("input_features") or {}
    if not isinstance(input_features, dict) or not input_features:
        raise ValueError("compiled policy config must define non-empty input_features")

    input_keys = [key for key in input_features if key == "observation.state" or key.startswith("observation.images.")]
    if not input_keys:
        raise ValueError("compiled policy config does not expose supported state/image input features")
    return input_keys


def _to_numpy_float32(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value.astype(np.float32, copy=False))
    if isinstance(value, Tensor):
        if value.device.type == "cpu" and value.dtype == torch.float32 and value.is_contiguous():
            return value.detach().numpy()
        return np.ascontiguousarray(value.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.ascontiguousarray(np.asarray(value, dtype=np.float32))


def _to_numpy_int64(value: Any) -> np.ndarray:
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=np.int64))


def _to_numpy_bool(value: Any) -> np.ndarray:
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=np.bool_))


def _to_numpy_optional(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value))


def _as_action_tensor(output: Any, device: torch.device) -> Tensor:
    tensor = output if isinstance(output, Tensor) else torch.as_tensor(output, dtype=torch.float32)
    if tensor.ndim >= 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor.to(device)


@dataclass
class PI05RuntimeInputs:
    images: list[np.ndarray]
    tokens: np.ndarray
    masks: np.ndarray
    noise: np.ndarray | None = None


class _FeatureTypeView:
    def __init__(self, name: str):
        self.name = (name or "").upper()


class _FeatureSpecView:
    def __init__(self, feature_type: str, shape: list[int]):
        self.type = _FeatureTypeView(feature_type)
        self.shape = list(shape)


def _ordered_pi05_image_features(config: dict[str, Any]) -> dict[str, _FeatureSpecView]:
    input_features = config.get("input_features") or {}
    if not isinstance(input_features, dict):
        return {}
    result: dict[str, _FeatureSpecView] = {}
    for key, value in input_features.items():
        feature_type = _feature_type(value)
        if feature_type == "VISUAL" or key.startswith("observation.images."):
            result[key] = _FeatureSpecView(feature_type or "VISUAL", _shape_from_feature(value))
    return result


class _PI05ConfigView:
    def __init__(self, config: dict[str, Any]):
        self.chunk_size = _chunk_size_from_config(config)
        self.max_action_dim = int(config.get("max_action_dim", 32))
        self.num_inference_steps = int(config.get("num_inference_steps", 10))
        self.image_features = _ordered_pi05_image_features(config)


class ACTCompiledAdapter:
    def __init__(self, config: dict[str, Any], backend: str):
        self._config = config
        self._backend = backend
        self._input_features = config.get("input_features") or {}
        self._input_keys = _input_order_from_config(config)
        self._chunk_size = _chunk_size_from_config(config)
        self._action_dim = _action_dim_from_config(config)

    @classmethod
    def from_config(cls, config: dict[str, Any], backend: str) -> ACTCompiledAdapter:
        policy_type = str(config.get("type", "")).lower().strip()
        if not policy_type:
            raise ValueError(f"Compiled backend {backend} policy config is missing required type metadata")
        if policy_type != "act":
            raise ValueError(f"Compiled backend {backend} does not support policy type {policy_type!r}")
        return cls(config, backend)

    @property
    def policy_type(self) -> str:
        return "act"

    @property
    def uses_action_chunking(self) -> bool:
        return True

    def get_chunk_size(self) -> int:
        return self._chunk_size

    def prepare_inputs(self, batch: dict[str, Tensor]) -> list[np.ndarray]:
        inputs: list[np.ndarray] = []
        for key in self._input_keys:
            if key not in batch:
                raise KeyError(f"Missing compiled policy input tensor for {self._backend}: {key}")
            tensor = batch[key]
            if not isinstance(tensor, Tensor):
                tensor = torch.as_tensor(tensor)
            if key.startswith("observation.images."):
                tensor = self._prepare_image_tensor(key, tensor)
            elif tensor.ndim == 1:
                tensor = tensor.reshape(1, -1)
            inputs.append(_to_numpy_float32(tensor))
        return inputs

    def _prepare_image_tensor(self, key: str, tensor: Tensor) -> Tensor:
        if tensor.dtype != torch.float32:
            tensor = tensor.to(dtype=torch.float32)
        if tensor.ndim == 3:
            tensor = tensor.reshape(1, *tensor.shape)
        if tensor.ndim != 4:
            raise RuntimeError(f"{key} must be NCHW tensor, got shape={tuple(tensor.shape)}")

        target_hw = self._image_target_hw(key)
        if target_hw is not None and tuple(tensor.shape[-2:]) != target_hw:
            tensor = functional.interpolate(
                tensor,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        return tensor.contiguous()

    def _image_target_hw(self, key: str) -> tuple[int, int] | None:
        feature_shape = _shape_from_feature(self._input_features.get(key))
        if len(feature_shape) >= 3:
            return int(feature_shape[-2]), int(feature_shape[-1])
        if self._backend == "ascend_om_3403":
            return (
                int(os.environ.get("SVP_IMAGE_HEIGHT", "240")),
                int(os.environ.get("SVP_IMAGE_WIDTH", "320")),
            )
        return None

    def decode_outputs(self, raw: list[np.ndarray], device: torch.device) -> Tensor:
        if not raw:
            raise RuntimeError(f"Compiled backend {self._backend} returned no outputs")
        if self._backend == "ascend_om_3403":
            return self._decode_sd3403_output(raw[0], device)
        if self._backend == "ascend_om":
            return self._decode_om_output(raw[0], device)
        return self._decode_first_action_output(raw[0], device)

    def _decode_om_output(self, output: Any, device: torch.device) -> Tensor:
        action = np.asarray(output, dtype=np.float32)
        if action.ndim == 1:
            expected_size = self._chunk_size * self._action_dim
            if action.size == expected_size:
                action = action.reshape(1, self._chunk_size, self._action_dim)
            elif action.size % self._action_dim == 0:
                action = action.reshape(1, -1, self._action_dim)
            else:
                raise RuntimeError(
                    f"unexpected ACT OM action tensor size={action.size}, "
                    f"not divisible by action_dim={self._action_dim}"
                )
        return _as_action_tensor(action, device)

    def _decode_sd3403_output(self, output: Any, device: torch.device) -> Tensor:
        flat = np.asarray(output, dtype=np.float32).reshape(-1)
        stride = int(self._config.get("sd3403_action_stride") or os.environ.get("SVP_ACTION_STRIDE", "8"))
        if flat.size % stride != 0:
            raise RuntimeError(f"unexpected action tensor size={flat.size}, not divisible by {stride}")
        action = flat.reshape(-1, stride)[:, : self._action_dim]
        self._chunk_size = int(action.shape[0])
        return torch.from_numpy(np.ascontiguousarray(action)).to(device)

    def _decode_first_action_output(self, output: Any, device: torch.device) -> Tensor:
        action = np.asarray(output, dtype=np.float32)
        if action.ndim == 1 and action.size % self._action_dim == 0:
            action = action.reshape(-1, self._action_dim)
        return _as_action_tensor(action, device)


class PI05CompiledAdapter:
    def __init__(self, config: dict[str, Any], backend: str):
        self._config = config
        self._backend = backend
        self._chunk_size = _chunk_size_from_config(config)
        self._max_action_dim = int(config.get("max_action_dim", 32))
        self._action_dim = _real_action_dim_from_config(config, self._max_action_dim)
        self._image_features = _ordered_pi05_image_features(config)

    @classmethod
    def from_config(cls, config: dict[str, Any], backend: str) -> PI05CompiledAdapter:
        policy_type = str(config.get("type", "")).lower().strip()
        if not policy_type:
            raise ValueError(f"Compiled backend {backend} policy config is missing required type metadata")
        if policy_type != "pi05":
            raise ValueError(f"Compiled backend {backend} does not support policy type {policy_type!r}")
        if normalize_backend_name(backend) != "ascend_om":
            raise ValueError(f"Compiled backend {backend} does not support PI05 OM policy")
        return cls(config, backend)

    @property
    def policy_type(self) -> str:
        return "pi05"

    @property
    def uses_action_chunking(self) -> bool:
        return True

    def get_chunk_size(self) -> int:
        return self._chunk_size

    def prepare_inputs(self, batch: dict[str, Tensor]) -> PI05RuntimeInputs:
        images: list[np.ndarray] = []
        for key in self._image_features:
            if key not in batch:
                raise KeyError(f"Missing PI05 image tensor for {self._backend}: {key}")
            images.append(_to_numpy_float32(batch[key]))
        if not images:
            raise ValueError("PI05 compiled policy config must define at least one VISUAL input feature")

        tokens = batch.get("observation.language.tokens", batch.get("lang_tokens"))
        masks = batch.get("observation.language.attention_mask", batch.get("lang_masks"))
        if tokens is None or masks is None:
            raise KeyError("Missing PI05 language tokens or attention masks")

        return PI05RuntimeInputs(
            images=images,
            tokens=_to_numpy_int64(tokens),
            masks=_to_numpy_bool(masks),
            noise=_to_numpy_optional(batch.get("_noise")),
        )

    def decode_outputs(self, raw: Any, device: torch.device) -> Tensor:
        if isinstance(raw, list):
            if not raw:
                raise RuntimeError(f"Compiled backend {self._backend} returned no outputs")
            raw = raw[0]
        if raw is None:
            raise RuntimeError(f"Compiled backend {self._backend} returned no outputs")
        if getattr(raw, "shape", None) is not None and raw.shape[-1] > self._action_dim:
            raw = raw[..., : self._action_dim]
        action = _as_action_tensor(raw, device)
        if action.ndim >= 2:
            self._chunk_size = int(action.shape[-2])
        return action


ADAPTER_REGISTRY: dict[str, type[CompiledModelAdapter]] = {
    "act": ACTCompiledAdapter,
    "pi05": PI05CompiledAdapter,
}


def create_compiled_model_adapter(config: dict[str, Any], backend: str) -> CompiledModelAdapter:
    policy_type = str(config.get("type", "")).lower().strip()
    if not policy_type:
        raise ValueError(f"Compiled backend {backend} policy config is missing required type metadata")
    adapter_cls = ADAPTER_REGISTRY.get(policy_type)
    if adapter_cls is None:
        raise ValueError(f"Compiled backend {backend} does not support policy type {policy_type!r}")
    return adapter_cls.from_config(config, backend)


def resolve_om_model_path(
    path: str,
    config: dict[str, Any] | None = None,
    manifest: CompiledManifest | None = None,
) -> Path:
    del config
    if manifest is None:
        manifest = load_compiled_manifest(path, "ascend_om")
    return manifest.require_artifact("policy", suffix=".om")


def resolve_pi05_om_paths(
    path: str,
    config: dict[str, Any] | None = None,
    manifest: CompiledManifest | None = None,
) -> tuple[Path, Path]:
    del config
    if manifest is None:
        manifest = load_compiled_manifest(path, "ascend_om", "pi05")
    manifest.require_execution(["vlm", "action_expert"])
    return (
        manifest.require_artifact("vlm", suffix=".om"),
        manifest.require_artifact("action_expert", suffix=".om"),
    )


def resolve_rknn_model_path(path: str) -> Path:
    raw_path = Path(path).expanduser()
    candidates: list[Path] = []

    if raw_path.is_file() and raw_path.suffix == ".rknn":
        candidates.append(raw_path)
    if raw_path.is_dir():
        candidates.extend([raw_path / "model.rknn", raw_path / f"{raw_path.name}.rknn"])
        candidates.extend(sorted(raw_path.glob("*.rknn")))

    checked: list[str] = []
    for candidate in candidates:
        candidate = candidate.expanduser()
        checked.append(str(candidate))
        if candidate.is_file() and candidate.suffix == ".rknn":
            return candidate.resolve()
    raise FileNotFoundError("RKNN model file not found under policy_path. Checked: " + ", ".join(checked))


class OMRuntimeSession:
    def __init__(self) -> None:
        self._model: Any = None

    def load(self, policy_path: str, config: dict[str, Any], device: torch.device) -> None:
        del device
        manifest = load_compiled_manifest(policy_path, "ascend_om", str(config.get("type", "")).lower().strip())
        manifest.require_execution(["policy"])
        model_path = resolve_om_model_path(policy_path, config, manifest)
        from inference_service.core.ascend_om.OMmodel import OMmodel

        self._model = OMmodel(str(model_path))

    def execute(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        if self._model is None:
            raise RuntimeError("OMRuntimeSession is not loaded")
        return list(self._model.forward(inputs))

    def release(self) -> None:
        if self._model is not None:
            close = getattr(self._model, "close", None)
            if callable(close):
                close()
            self._model = None


class PI05OMRuntimeSession:
    def __init__(self) -> None:
        self._model: Any = None

    def load(self, policy_path: str, config: dict[str, Any], device: torch.device) -> None:
        del device
        manifest = load_compiled_manifest(policy_path, "ascend_om", str(config.get("type", "")).lower().strip())
        vlm_path, action_expert_path = resolve_pi05_om_paths(policy_path, config, manifest)
        from inference_service.core.ascend_om.pi05.PI05OMModel import PI05OMModel

        self._model = PI05OMModel(str(vlm_path), str(action_expert_path), _PI05ConfigView(config))

    def execute(self, inputs: PI05RuntimeInputs) -> Tensor:
        if self._model is None:
            raise RuntimeError("PI05OMRuntimeSession is not loaded")
        if not isinstance(inputs, PI05RuntimeInputs):
            raise TypeError("PI05OMRuntimeSession expects PI05RuntimeInputs")
        from inference_service.core.ascend_om.pi05.prefix_mask_utils import (
            build_prefix_att_2d_masks_4d_np,
        )

        prefix_mask = build_prefix_att_2d_masks_4d_np(
            num_cameras=len(inputs.images),
            lang_masks=inputs.masks,
            prefix_seq_len=self._model.prefix_seq_len,
        )
        return self._model.forward(
            inputs.images,
            inputs.tokens,
            inputs.masks,
            prefix_mask,
            noise=inputs.noise,
        )

    def release(self) -> None:
        if self._model is not None:
            close = getattr(self._model, "close", None)
            if callable(close):
                close()
            self._model = None


class SD3403RuntimeSession:
    def __init__(self) -> None:
        self._worker: Any = None

    def load(self, policy_path: str, config: dict[str, Any], device: torch.device) -> None:
        del device
        manifest = load_compiled_manifest(policy_path, "ascend_om_3403", str(config.get("type", "")).lower().strip())
        manifest.require_execution(["policy", "worker"])
        model_path = resolve_om_model_path(policy_path, config, manifest)
        worker_path = manifest.require_artifact("worker")
        if not os.access(worker_path, os.X_OK):
            raise FileNotFoundError(f"Compiled artifact 'worker' is not executable: {worker_path}")
        from inference_service.core.ascend_om.ACTWrapper_3403 import ACT3403Policy

        self._worker = ACT3403Policy(str(worker_path), str(model_path))

    def execute(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        if self._worker is None:
            raise RuntimeError("SD3403RuntimeSession is not loaded")
        execute_arrays = getattr(self._worker, "execute_arrays", None)
        if execute_arrays is None:
            raise RuntimeError("SD3403 worker does not expose execute_arrays")
        return [execute_arrays(inputs)]

    def release(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None


class RKNNRuntimeSession:
    def __init__(self) -> None:
        self._rknn: Any = None

    def load(self, policy_path: str, config: dict[str, Any], device: torch.device) -> None:
        del config, device
        model_path = resolve_rknn_model_path(policy_path)
        from rknnlite.api import RKNNLite

        self._rknn = RKNNLite()
        ret = self._rknn.load_rknn(str(model_path))
        if ret != 0:
            raise RuntimeError(f"RKNN load_rknn failed with ret={ret}")
        ret = self._rknn.init_runtime(target=None)
        if ret != 0:
            raise RuntimeError(f"RKNN init_runtime failed with ret={ret}")

    def execute(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        if self._rknn is None:
            raise RuntimeError("RKNNRuntimeSession is not loaded")
        outputs = self._rknn.inference(inputs=inputs)
        if outputs is None or len(outputs) == 0:
            raise RuntimeError("RKNN inference returned no outputs")
        return list(outputs)

    def release(self) -> None:
        if self._rknn is not None:
            self._rknn.release()
            self._rknn = None


def create_runtime_session(backend: str, config: dict[str, Any] | None = None) -> RuntimeSession:
    normalized = normalize_backend_name(backend)
    if normalized == "ascend_om":
        policy_type = str((config or {}).get("type", "")).lower().strip()
        if policy_type == "pi05":
            return PI05OMRuntimeSession()
        return OMRuntimeSession()
    if normalized == "ascend_om_3403":
        return SD3403RuntimeSession()
    if normalized == "rknn":
        return RKNNRuntimeSession()
    raise ValueError(f"Unsupported compiled inference backend: {backend}")


class CompiledPolicyWrapper(PolicyWrapper):
    def __init__(self, backend: str, runtime_session: RuntimeSession | None = None) -> None:
        self._backend = normalize_backend_name(backend)
        self._runtime_session = runtime_session
        self._adapter: CompiledModelAdapter | None = None
        self._device = torch.device("cpu")

    def load(self, path: str, device: torch.device) -> None:
        self._device = device
        config = load_compiled_policy_config(path, self._backend, runtime_device=device)
        self._adapter = create_compiled_model_adapter(config, self._backend)
        if self._runtime_session is None:
            self._runtime_session = create_runtime_session(self._backend, config)
        self._runtime_session.load(path, config, device)

    def infer(self, batch: dict[str, Tensor]) -> Tensor:
        if self._adapter is None or self._runtime_session is None:
            raise RuntimeError(f"CompiledPolicyWrapper for {self._backend} is not loaded")
        inputs = self._adapter.prepare_inputs(batch)
        outputs = self._runtime_session.execute(inputs)
        return self._adapter.decode_outputs(outputs, self._device)

    def get_chunk_size(self) -> int:
        if self._adapter is None:
            return 1
        return self._adapter.get_chunk_size()

    @property
    def policy_type(self) -> str:
        if self._adapter is None:
            return ""
        return self._adapter.policy_type

    @property
    def backend_type(self) -> str:
        return self._backend

    @property
    def uses_action_chunking(self) -> bool:
        return bool(self._adapter is not None and self._adapter.uses_action_chunking)

    def close(self) -> None:
        if self._runtime_session is not None:
            self._runtime_session.release()
