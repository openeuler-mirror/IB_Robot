"""Manifest-driven conversion between semantic tensors and runtime bindings."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from importlib import import_module

import numpy as np

from inference_manifest import ArtifactBindings, TensorBinding, is_image_semantic
from inference_service.codecs.types import (
    BoundInputs,
    BoundTensor,
    CodecRequest,
    CodecResult,
    RuntimeOutputs,
)


class BindingError(ValueError):
    """Raised when a manifest binding cannot be satisfied safely."""


class MissingSemanticTensorError(BindingError, KeyError):
    """Raised when the canonical batch does not contain a required semantic."""


def _validate_direction(bindings: tuple[TensorBinding, ...], direction: str) -> None:
    semantics = [binding.semantic for binding in bindings]
    if len(semantics) != len(set(semantics)):
        raise BindingError(f"{direction} bindings contain duplicate semantic values")

    runtime_names = [binding.runtime_name for binding in bindings if binding.runtime_name is not None]
    if len(runtime_names) != len(set(runtime_names)):
        raise BindingError(f"{direction} bindings contain duplicate runtime names")

    indices = [binding.index for binding in bindings if binding.index is not None]
    if len(indices) != len(set(indices)):
        raise BindingError(f"{direction} bindings contain duplicate runtime indices")
    if indices and len(indices) != len(bindings):
        raise BindingError(f"{direction} bindings must either all declare indices or all omit them")
    if direction == "input" and indices and sorted(indices) != list(range(len(indices))):
        raise BindingError(f"{direction} runtime indices must be contiguous and start at zero")

    for binding in bindings:
        needs_layout = len(binding.shape) == 4 and is_image_semantic(binding.semantic)
        if needs_layout and binding.layout not in {"NCHW", "NHWC"}:
            raise BindingError(f"rank-4 image binding {binding.semantic!r} requires NCHW or NHWC layout")
        if len(binding.shape) != 4 and binding.layout is not None:
            raise BindingError(f"non-rank-4 binding {binding.semantic!r} must not declare layout")


def validate_artifact_bindings(bindings: ArtifactBindings) -> None:
    """Validate slot identity and ordering independently of model construction."""

    _validate_direction(bindings.inputs, "input")
    _validate_direction(bindings.outputs, "output")


def _numpy_dtype(dtype: str) -> np.dtype:
    if dtype != "bfloat16":
        return np.dtype(dtype)
    try:
        return np.dtype(dtype)
    except TypeError:
        try:
            extension = import_module("ml_dtypes")
        except ImportError as exc:
            raise BindingError("bfloat16 conversion requires NumPy bfloat16 support or the ml_dtypes package") from exc
        return np.dtype(extension.bfloat16)


def _as_numpy(value: object, dtype: str) -> np.ndarray:
    target_dtype = _numpy_dtype(dtype)
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    to_numpy = getattr(candidate, "numpy", None)
    if callable(to_numpy):
        try:
            candidate = to_numpy()
        except (RuntimeError, TypeError):
            to_float = getattr(candidate, "float", None)
            if not callable(to_float):
                raise
            candidate = to_float().numpy()
    try:
        return np.ascontiguousarray(np.asarray(candidate, dtype=target_dtype))
    except (TypeError, ValueError) as exc:
        raise BindingError(f"cannot convert tensor value to declared dtype {dtype!r}") from exc


def _validate_shape(binding: TensorBinding, value: np.ndarray) -> None:
    if value.ndim != len(binding.shape):
        raise BindingError(
            f"binding {binding.semantic!r} expects rank {len(binding.shape)}, got rank {value.ndim} with shape {value.shape}"
        )
    mismatches = [
        (axis, expected, actual)
        for axis, (expected, actual) in enumerate(zip(binding.shape, value.shape, strict=True))
        if expected != -1 and expected != actual
    ]
    if mismatches:
        details = ", ".join(f"axis {axis}: expected {expected}, got {actual}" for axis, expected, actual in mismatches)
        raise BindingError(f"binding {binding.semantic!r} shape mismatch ({details})")


def convert_input(binding: TensorBinding, value: object) -> np.ndarray:
    """Convert one canonical tensor to its declared runtime ABI."""

    converted = _as_numpy(value, binding.dtype)
    if is_image_semantic(binding.semantic) and binding.layout == "NHWC":
        if converted.ndim != 4:
            raise BindingError(
                f"image binding {binding.semantic!r} requires rank-4 canonical NCHW input for NHWC conversion"
            )
        converted = np.ascontiguousarray(np.transpose(converted, (0, 2, 3, 1)))
    _validate_shape(binding, converted)
    return converted


def bind_inputs(request: CodecRequest, bindings: ArtifactBindings) -> BoundInputs:
    """Resolve semantic inputs and convert them without backend-specific logic."""

    validate_artifact_bindings(bindings)
    bound: list[BoundTensor] = []
    for binding in bindings.inputs:
        try:
            value = request.semantic_tensors[binding.semantic]
        except KeyError as exc:
            raise MissingSemanticTensorError(
                f"missing semantic tensor {binding.semantic!r}; available semantics: {sorted(request.semantic_tensors)}"
            ) from exc
        bound.append(
            BoundTensor(
                semantic=binding.semantic,
                runtime_name=binding.runtime_name,
                index=binding.index,
                value=convert_input(binding, value),
            )
        )
    return BoundInputs(tuple(bound))


def _resolve_output_value(
    outputs: RuntimeOutputs,
    binding: TensorBinding,
    output_count: int,
) -> object:
    if isinstance(outputs, Mapping):
        if binding.runtime_name is not None and binding.runtime_name in outputs:
            return outputs[binding.runtime_name]
        if binding.index is not None and binding.index in outputs:
            return outputs[binding.index]
        slots = sorted(str(slot) for slot in outputs)
        raise BindingError(
            f"runtime output for semantic {binding.semantic!r} is missing; available runtime slots: {slots}"
        )

    if isinstance(outputs, Sequence) and not isinstance(outputs, str | bytes | bytearray | np.ndarray):
        if binding.index is None:
            raise BindingError(f"sequence runtime outputs require an index for semantic {binding.semantic!r}")
        if binding.index >= len(outputs):
            raise BindingError(
                f"runtime output index {binding.index} for semantic {binding.semantic!r} exceeds {len(outputs)} outputs"
            )
        return outputs[binding.index]

    if output_count != 1:
        raise BindingError(f"a direct runtime output is ambiguous for {output_count} declared output bindings")
    return outputs


def decode_bound_outputs(outputs: RuntimeOutputs, bindings: ArtifactBindings) -> CodecResult:
    """Decode declared runtime outputs and select the semantic action tensor."""

    return decode_bound_outputs_with_transforms(outputs, bindings)


def decode_bound_outputs_with_transforms(
    outputs: RuntimeOutputs,
    bindings: ArtifactBindings,
    raw_transforms: Mapping[str, Callable[[object, TensorBinding], object]] | None = None,
) -> CodecResult:
    """Decode outputs after optional semantic transforms needed before ABI validation."""

    validate_artifact_bindings(bindings)
    semantic_outputs: dict[str, np.ndarray] = {}
    for binding in bindings.outputs:
        raw_value = _resolve_output_value(outputs, binding, len(bindings.outputs))
        if raw_transforms is not None and binding.semantic in raw_transforms:
            raw_value = raw_transforms[binding.semantic](raw_value, binding)
        converted = _as_numpy(raw_value, binding.dtype)
        _validate_shape(binding, converted)
        semantic_outputs[binding.semantic] = converted

    try:
        action = semantic_outputs["action"]
    except KeyError as exc:
        raise BindingError("output bindings do not declare the required semantic 'action'") from exc
    return CodecResult(action=action, semantic_tensors=semantic_outputs)


class BindingPolicyCodec:
    """Generic manifest-bound codec used before policy-specific codecs migrate."""

    def encode_inputs(self, request: CodecRequest, bindings: ArtifactBindings) -> BoundInputs:
        return bind_inputs(request, bindings)

    def decode_outputs(self, outputs: RuntimeOutputs, bindings: ArtifactBindings) -> CodecResult:
        return decode_bound_outputs(outputs, bindings)
