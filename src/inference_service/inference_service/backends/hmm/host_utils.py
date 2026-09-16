"""Shared host-side tensor and artifact helpers for compiled HMM families."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from inference_manifest import CompiledDeployment, TensorBinding, is_image_semantic
from inference_service.backends.errors import BackendInferenceError, BackendLoadError
from inference_service.backends.types import RuntimeContext

NOISE_SEMANTICS = frozenset({"noise", "action.noise", "_noise"})


def numpy_dtype(dtype: str) -> np.dtype:
    if dtype != "bfloat16":
        return np.dtype(dtype)
    try:
        return np.dtype(dtype)
    except TypeError:
        try:
            extension = importlib.import_module("ml_dtypes")
        except ImportError as exc:
            raise BackendLoadError(
                "HMM bfloat16 bindings require NumPy bfloat16 support or ml_dtypes",
                code="unsupported_runtime_dtype",
            ) from exc
        return np.dtype(extension.bfloat16)


def binding_for_semantics(
    bindings: Sequence[TensorBinding], semantics: set[str] | frozenset[str], description: str
) -> TensorBinding:
    matches = [binding for binding in bindings if binding.semantic in semantics]
    if len(matches) != 1:
        raise BackendLoadError(
            f"HMM deployment requires exactly one {description} binding",
            code="invalid_bindings",
        )
    return matches[0]


def binding_for_semantic(bindings: Sequence[TensorBinding], semantic: str, description: str) -> TensorBinding:
    return binding_for_semantics(bindings, {semantic}, description)


def static_shape(binding: TensorBinding) -> tuple[int, ...]:
    if any(dimension < 1 for dimension in binding.shape):
        raise BackendInferenceError(
            f"HMM runtime-generated input {binding.semantic!r} requires a static shape, got {binding.shape}",
            code="dynamic_runtime_input",
        )
    return binding.shape


def compatible_shape(expected: tuple[int, ...], actual: tuple[int, ...]) -> bool:
    return len(expected) == len(actual) and all(
        declared == -1 or declared == observed for declared, observed in zip(expected, actual, strict=True)
    )


def convert_runtime_value(binding: TensorBinding, value: object, role: str, direction: str) -> np.ndarray:
    try:
        converted = np.ascontiguousarray(np.asarray(value, dtype=numpy_dtype(binding.dtype)))
    except (TypeError, ValueError) as exc:
        raise BackendInferenceError(
            f"HMM role {role!r} {direction} {binding.semantic!r} cannot convert to {binding.dtype}",
            code=f"runtime_{direction}_dtype_mismatch",
        ) from exc
    if not compatible_shape(binding.shape, converted.shape):
        raise BackendInferenceError(
            f"HMM role {role!r} {direction} {binding.semantic!r} shape {converted.shape} "
            f"does not match manifest shape {binding.shape}",
            code=f"runtime_{direction}_shape_mismatch",
        )
    return converted


def convert_semantic_outputs(
    bindings: Sequence[TensorBinding], values: Mapping[str, object], role: str
) -> dict[str, np.ndarray]:
    outputs: dict[str, np.ndarray] = {}
    for binding in bindings:
        try:
            value = values[binding.semantic]
        except KeyError as exc:
            raise BackendInferenceError(
                f"HMM CPU role {role!r} did not generate {binding.semantic!r}",
                code="missing_runtime_output",
            ) from exc
        outputs[binding.semantic] = convert_runtime_value(binding, value, role, "output")
    return outputs


def pad_axis_one(value: np.ndarray, shape: tuple[int, ...], pad_value: object) -> np.ndarray:
    if len(shape) != value.ndim or shape[0] not in {-1, value.shape[0]}:
        raise BackendInferenceError(
            f"HMM prefix tensor shape {value.shape} is incompatible with manifest shape {shape}",
            code="invalid_prefix_shape",
        )
    target_length = shape[1]
    if target_length < value.shape[1] or any(
        expected != -1 and expected != actual for expected, actual in zip(shape[2:], value.shape[2:], strict=True)
    ):
        raise BackendInferenceError(
            f"HMM prefix tensor shape {value.shape} is incompatible with manifest shape {shape}",
            code="invalid_prefix_shape",
        )
    if target_length == value.shape[1]:
        return value
    pad_shape = list(value.shape)
    pad_shape[1] = target_length - value.shape[1]
    return np.concatenate((value, np.full(pad_shape, pad_value, dtype=value.dtype)), axis=1)


def to_additive_attention(mask: np.ndarray, dtype: str) -> np.ndarray:
    target_dtype = numpy_dtype(dtype)
    return np.where(mask, 0.0, np.finfo(target_dtype).min).astype(target_dtype)


def validate_token_ids(tokens: np.ndarray, vocabulary_size: int) -> None:
    if tokens.min(initial=0) < 0 or tokens.max(initial=0) >= vocabulary_size:
        raise BackendInferenceError("HMM token id is outside the embedding table", code="invalid_token_id")


def require_positive_config(config: Mapping[str, object], key: str, policy: str) -> None:
    value = config.get(key)
    if type(value) is not int or value < 1:
        raise BackendLoadError(
            f"HMM {policy} requires positive integer {key!r} in LeRobot config",
            code="invalid_policy_config",
        )


def require_artifact(context: RuntimeContext, role: str) -> Path:
    try:
        path = context.resolved_artifacts[role]
    except KeyError as exc:
        raise BackendLoadError(
            f"HMM deployment is missing artifact role {role!r}", code="missing_artifact_role"
        ) from exc
    if not path.is_file():
        raise BackendLoadError(f"HMM artifact {role!r} is not a regular file: {path}", code="invalid_artifact")
    return path


def load_torch_mapping(path: Path, description: str) -> Mapping[str, object]:
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        raise BackendLoadError(
            f"HMM {description} requires PyTorch to load {path}: {exc}",
            code="missing_dependency",
        ) from exc
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise BackendLoadError(
            f"Unable to load HMM {description} artifact {path}: {exc}", code="invalid_embedding"
        ) from exc
    if not isinstance(value, Mapping):
        raise BackendLoadError(f"HMM {description} artifact must contain a tensor mapping", code="invalid_embedding")
    return value


def to_numpy_weight(value: object, path: Path, name: str) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    if str(getattr(value, "dtype", "")) == "torch.bfloat16":
        value = value.float()
    to_numpy = getattr(value, "numpy", None)
    if callable(to_numpy):
        value = to_numpy()
    try:
        return np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    except (TypeError, ValueError) as exc:
        raise BackendLoadError(
            f"HMM artifact {path} contains invalid tensor {name!r}",
            code="invalid_embedding",
        ) from exc


def validate_pi05_plan(
    deployment: CompiledDeployment, policy_config: Mapping[str, object], token_weight: np.ndarray
) -> None:
    suffix = ("embedding", "prefill", "action_in_proj", "time_mlp", "decode", "action_out_proj")
    vision_roles = deployment.execution[: -len(suffix)]
    if (
        not vision_roles
        or deployment.execution[-len(suffix) :] != suffix
        or any(role != "vision" and not role.startswith("vision_") for role in vision_roles)
    ):
        raise BackendLoadError(
            "HMM PI0.5 requires vision role(s) followed by embedding, prefill, action_in_proj, "
            "time_mlp, decode, and action_out_proj",
            code="invalid_execution_plan",
        )
    for key in ("chunk_size", "max_action_dim", "num_inference_steps"):
        require_positive_config(policy_config, key, "PI0.5")
    if token_weight.ndim != 2:
        raise BackendLoadError("HMM PI0.5 token embedding must be rank 2", code="invalid_embedding")
    if not deployment.device_links or any(
        link.producer != "prefill" or link.consumer != "decode" or link.producer_binding != "output"
        for link in deployment.device_links
    ):
        raise BackendLoadError(
            "HMM PI0.5 requires prefill output to decode input device links",
            code="invalid_device_links",
        )
    noise = binding_for_semantics(deployment.bindings["action_in_proj"].inputs, NOISE_SEMANTICS, "noise")
    action = binding_for_semantics(deployment.bindings["action_out_proj"].outputs, {"action"}, "action output")
    expected = (1, int(policy_config["chunk_size"]), int(policy_config["max_action_dim"]))
    if noise.shape != expected or action.shape != expected:
        raise BackendLoadError(
            f"HMM PI0.5 noise and action bindings must use shape {expected}",
            code="invalid_bindings",
        )
    image_outputs: list[str] = []
    for role in vision_roles:
        bindings = deployment.bindings[role]
        if len(bindings.inputs) != 1 or len(bindings.outputs) != 1:
            raise BackendLoadError(
                f"HMM vision role {role!r} requires exactly one input and one output",
                code="invalid_bindings",
            )
        if not is_image_semantic(bindings.inputs[0].semantic) or not bindings.outputs[0].semantic.startswith(
            "internal.image_embedding."
        ):
            raise BackendLoadError(
                f"HMM vision role {role!r} must map one image to one internal image embedding",
                code="invalid_bindings",
            )
        image_outputs.append(bindings.outputs[0].semantic)
    embedding_images = [
        binding.semantic
        for binding in deployment.bindings["embedding"].inputs
        if binding.semantic.startswith("internal.image_embedding.")
    ]
    if image_outputs != embedding_images:
        raise BackendLoadError("HMM embedding image inputs must match vision execution order", code="invalid_bindings")
