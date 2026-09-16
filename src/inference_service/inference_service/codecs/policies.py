"""Policy-family semantic preparation layered on manifest binding conversion."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from inference_manifest import ArtifactBindings, PolicyMetadata, TensorBinding
from inference_service.codecs.bindings import (
    BindingError,
    MissingSemanticTensorError,
    bind_inputs,
    convert_input,
    decode_bound_outputs,
    decode_bound_outputs_with_transforms,
)
from inference_service.codecs.execution import ExecutionPlan
from inference_service.codecs.types import BoundInputs, BoundTensor, CodecRequest, CodecResult, RuntimeOutputs

_LANGUAGE_ALIASES = {
    "observation.language.tokens": ("lang_tokens", "lang_token"),
    "observation.language.attention_mask": ("lang_masks", "lang_mask"),
}
_NOISE_SEMANTICS = frozenset({"noise", "action.noise", "_noise"})
_TIME_SEMANTICS = frozenset({"time", "timestep", "action.time", "_time"})
_PREFIX_MASK_SEMANTICS = frozenset({"prefix_att_2d_masks_4d", "observation.prefix_att_2d_masks_4d"})
_OPENPI_ATTENTION_MASK_VALUE = np.float32(-2.3819763e38)


def _to_numpy(value: object) -> np.ndarray:
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    to_numpy = getattr(candidate, "numpy", None)
    if callable(to_numpy):
        candidate = to_numpy()
    return np.asarray(candidate)


def _static_target_image_hw(binding: TensorBinding) -> tuple[int, int] | None:
    if len(binding.shape) != 4:
        return None
    if binding.layout == "NCHW":
        height, width = binding.shape[-2:]
    else:
        height, width = binding.shape[-3:-1]
    if height == -1 or width == -1:
        return None
    return int(height), int(width)


def _resize_bilinear_nchw(image: np.ndarray, height: int, width: int) -> np.ndarray:
    source_height, source_width = image.shape[-2:]
    if (source_height, source_width) == (height, width):
        return np.ascontiguousarray(image, dtype=np.float32)

    y = (np.arange(height, dtype=np.float32) + 0.5) * (source_height / height) - 0.5
    x = (np.arange(width, dtype=np.float32) + 0.5) * (source_width / width) - 0.5
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, source_height - 1)
    x1 = np.clip(x0 + 1, 0, source_width - 1)
    y_weight = (y - y0)[None, None, :, None]
    x_weight = (x - x0)[None, None, None, :]
    y0 = np.clip(y0, 0, source_height - 1)
    x0 = np.clip(x0, 0, source_width - 1)

    source = np.asarray(image, dtype=np.float32)
    vertical = source[:, :, y0, :] * (1.0 - y_weight) + source[:, :, y1, :] * y_weight
    resized = vertical[:, :, :, x0] * (1.0 - x_weight) + vertical[:, :, :, x1] * x_weight
    return np.ascontiguousarray(resized, dtype=np.float32)


def _resize_with_pad_nchw(
    image: np.ndarray,
    height: int,
    width: int,
    *,
    alignment: str = "center",
) -> np.ndarray:
    source_height, source_width = image.shape[-2:]
    ratio = max(source_width / width, source_height / height)
    resized_height = max(1, int(source_height / ratio))
    resized_width = max(1, int(source_width / ratio))
    resized = _resize_bilinear_nchw(image, resized_height, resized_width)
    if alignment == "left_top":
        pad_top = height - resized_height
        pad_left = width - resized_width
    else:
        pad_top = (height - resized_height) // 2
        pad_left = (width - resized_width) // 2
    result = np.zeros((*image.shape[:-2], height, width), dtype=np.float32)
    result[..., pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized
    return np.ascontiguousarray(result)


class _ManifestPolicyCodec:
    image_resize_mode = "stretch"
    image_pad_alignment = "center"
    pad_state = False
    crop_action = False

    def __init__(self, metadata: PolicyMetadata) -> None:
        self._metadata = metadata
        self._action_dim = int(metadata.output_features["action"].shape[-1])

    def encode_inputs(self, request: CodecRequest, bindings: ArtifactBindings) -> BoundInputs:
        prepared: dict[str, object] = {}
        for binding in bindings.inputs:
            value = self._resolve_semantic(request.semantic_tensors, binding.semantic)
            prepared[binding.semantic] = self._prepare_value(binding, value)
        return bind_inputs(CodecRequest(prepared), bindings)

    def decode_outputs(self, outputs: RuntimeOutputs, bindings: ArtifactBindings) -> CodecResult:
        result = decode_bound_outputs(outputs, bindings)
        if not self.crop_action or result.action.shape[-1] <= self._action_dim:
            return result
        action = np.ascontiguousarray(result.action[..., : self._action_dim])
        semantic_tensors = dict(result.semantic_tensors)
        semantic_tensors["action"] = action
        return CodecResult(action=action, semantic_tensors=semantic_tensors)

    def _resolve_semantic(self, tensors: Mapping[str, object], semantic: str) -> object:
        if semantic in tensors:
            return tensors[semantic]

        aliases = _LANGUAGE_ALIASES.get(semantic, ())
        if semantic in _NOISE_SEMANTICS:
            aliases = tuple(_NOISE_SEMANTICS - {semantic})
        elif semantic in _TIME_SEMANTICS:
            aliases = tuple(_TIME_SEMANTICS - {semantic})
        for alias in aliases:
            if alias in tensors:
                return tensors[alias]
        raise MissingSemanticTensorError(
            f"missing semantic tensor {semantic!r}; available semantics: {sorted(tensors)}"
        )

    def _prepare_value(self, binding: TensorBinding, value: object) -> object:
        semantic = binding.semantic
        if semantic == "observation.state":
            return self._prepare_state(binding, value)
        if (
            semantic == "observation.image"
            or semantic.startswith("observation.image.")
            or semantic.startswith("observation.images.")
        ):
            return self._prepare_image(binding, value)
        if semantic in _LANGUAGE_ALIASES:
            return self._prepare_batched(value, binding, use_last_observation=False)
        if semantic in _NOISE_SEMANTICS:
            return self._prepare_batched(value, binding, use_last_observation=False)
        if semantic in _TIME_SEMANTICS:
            return self._prepare_batched(value, binding, use_last_observation=False)
        return value

    def _prepare_state(self, binding: TensorBinding, value: object) -> np.ndarray:
        state = self._prepare_batched(value, binding, use_last_observation=True)
        target_dim = binding.shape[-1]
        if self.pad_state and target_dim != -1 and state.shape[-1] != target_dim:
            if state.shape[-1] > target_dim:
                raise BindingError(f"state dimension {state.shape[-1]} exceeds declared runtime dimension {target_dim}")
            padded = np.zeros((*state.shape[:-1], target_dim), dtype=state.dtype)
            padded[..., : state.shape[-1]] = state
            state = padded
        return np.ascontiguousarray(state)

    def _prepare_image(self, binding: TensorBinding, value: object) -> np.ndarray:
        image = _to_numpy(value)
        if image.ndim == 3:
            image = image[None, ...]
        elif image.ndim == 5:
            image = image[:, -1, ...]
        if image.ndim != 4:
            raise BindingError(
                f"image semantic {binding.semantic!r} requires a canonical NCHW tensor, got shape {image.shape}"
            )
        image = np.ascontiguousarray(image, dtype=np.float32)
        target_hw = _static_target_image_hw(binding)
        if target_hw is None or image.shape[-2:] == target_hw:
            return image
        if self.image_resize_mode == "pad":
            return _resize_with_pad_nchw(image, *target_hw, alignment=self.image_pad_alignment)
        return _resize_bilinear_nchw(image, *target_hw)

    @staticmethod
    def _prepare_batched(value: object, binding: TensorBinding, *, use_last_observation: bool) -> np.ndarray:
        array = _to_numpy(value)
        expected_rank = len(binding.shape)
        if use_last_observation and array.ndim == expected_rank + 1:
            array = array[:, -1, ...]
        if array.ndim == expected_rank - 1:
            array = array[None, ...]
        return np.ascontiguousarray(array)


class ACTPolicyCodec(_ManifestPolicyCodec):
    """ACT state/image preparation and manifest-selected action decoding."""

    def decode_outputs(self, outputs: RuntimeOutputs, bindings: ArtifactBindings) -> CodecResult:
        return decode_bound_outputs_with_transforms(outputs, bindings, {"action": self._reshape_action})

    def _reshape_action(self, value: object, binding: TensorBinding) -> np.ndarray:
        action = _to_numpy(value)
        if action.ndim == 1:
            static_size = int(np.prod(binding.shape)) if all(dimension > 0 for dimension in binding.shape) else None
            if static_size == action.size:
                return action.reshape(binding.shape)
            if action.size % self._action_dim == 0:
                chunk_size = action.size // self._action_dim
                if len(binding.shape) == 3:
                    return action.reshape(1, chunk_size, self._action_dim)
                if len(binding.shape) == 2:
                    return action.reshape(chunk_size, self._action_dim)
        if action.ndim == 2 and len(binding.shape) == 3 and binding.shape[0] in {-1, 1}:
            return action[None, ...]
        return action


class PI05PolicyCodec(_ManifestPolicyCodec):
    """PI0.5 state, image, language, noise, time, and action semantics."""

    image_resize_mode = "pad"
    pad_state = True
    crop_action = True

    @staticmethod
    def validate_staged_deployment(deployment) -> None:
        """Validate the PI0.5 prefix-cache ABI before reuse across requests.

        The prefix depends only on images and language, never action state,
        noise, time or a previous request. The executor enforces prompt,
        freshness and generation matching on each cached prefix.
        """
        if deployment.execution != ("vlm", "action_expert") or deployment.execution_contract.state_scope != "request":
            raise ValueError("independent execution requires the request-scoped PI0.5 split ABI")
        producer = deployment.bindings["vlm"]
        terminal = deployment.bindings["action_expert"]
        prefix = {"internal.past_kv", "internal.prefix_pad_masks"}
        allowed = {
            "observation.language.tokens",
            "observation.language.attention_mask",
            "prefix_att_2d_masks_4d",
            "observation.prefix_att_2d_masks_4d",
        }
        if any(b.semantic not in allowed and not b.semantic.startswith("observation.images.") for b in producer.inputs):
            raise ValueError("PI0.5 prefix reuse forbids state, iterative and hidden input dependencies")
        if (
            {b.semantic for b in producer.outputs} != prefix
            or {b.semantic for b in terminal.inputs} != prefix | {"time", "noise"}
            or {b.semantic for b in terminal.outputs} != {"action"}
        ):
            raise ValueError("PI0.5 prefix reuse requires the validated cache and denoising bindings")
        for output in producer.outputs:
            target = next(b for b in terminal.inputs if b.semantic == output.semantic)
            if output.dtype != target.dtype or output.shape != target.shape:
                raise ValueError("PI0.5 prefix cache input/output ABI mismatch")
        if any(
            link.producer != "vlm"
            or link.consumer != "action_expert"
            or link.producer_binding != "output"
            or link.semantic not in prefix
            for link in deployment.device_links
        ):
            raise ValueError("PI0.5 prefix reuse forbids reverse or input-sourced device links")

    def encode_execution(self, request: CodecRequest, plan: ExecutionPlan) -> Mapping[str, BoundInputs]:
        """Bind external tensors while leaving runtime-owned links and denoising inputs unmaterialized."""

        role_inputs: dict[str, BoundInputs] = {}
        for role in plan.roles:
            bound: list[BoundTensor] = []
            for binding in role.bindings.inputs:
                semantic = binding.semantic
                if semantic.startswith("internal.") or semantic in _TIME_SEMANTICS:
                    continue
                if semantic in _PREFIX_MASK_SEMANTICS:
                    value = self._build_prefix_mask(request.semantic_tensors, role.bindings, binding)
                else:
                    try:
                        value = self._resolve_semantic(request.semantic_tensors, semantic)
                    except MissingSemanticTensorError:
                        if semantic in _NOISE_SEMANTICS:
                            continue
                        raise
                    value = self._prepare_value(binding, value)
                bound.append(
                    BoundTensor(
                        semantic=semantic,
                        runtime_name=binding.runtime_name,
                        index=binding.index,
                        value=convert_input(binding, value),
                    )
                )
            role_inputs[role.name] = BoundInputs(tuple(bound))
        return MappingProxyType(role_inputs)

    def decode_execution(self, outputs: RuntimeOutputs, plan: ExecutionPlan) -> CodecResult:
        action_roles = [
            role for role in plan.roles if any(binding.semantic == "action" for binding in role.bindings.outputs)
        ]
        if len(action_roles) != 1:
            raise BindingError(
                f"PI0.5 execution plan requires exactly one action-producing role, got {len(action_roles)}"
            )
        action_role = action_roles[0]
        role_outputs = outputs
        if isinstance(outputs, Mapping) and action_role.name in outputs:
            role_outputs = outputs[action_role.name]
        return self.decode_outputs(role_outputs, action_role.bindings)

    def _build_prefix_mask(
        self,
        tensors: Mapping[str, object],
        bindings: ArtifactBindings,
        binding: TensorBinding,
    ) -> np.ndarray:
        language_mask = _to_numpy(self._resolve_semantic(tensors, "observation.language.attention_mask"))
        if language_mask.ndim == 1:
            language_mask = language_mask[None, ...]
        if language_mask.ndim != 2:
            raise BindingError("PI0.5 language attention mask must be rank 1 or 2 to derive prefix_att_2d_masks_4d")
        if len(binding.shape) != 4 or binding.shape[-1] < 1 or binding.shape[-2] != binding.shape[-1]:
            raise BindingError(
                f"PI0.5 prefix attention binding requires static shape (B, 1, S, S), got {binding.shape}"
            )

        camera_count = sum(
            feature.type.upper() == "VISUAL"
            for input_binding in bindings.inputs
            if (feature := self._metadata.input_features.get(input_binding.semantic)) is not None
        )
        prefix_length = int(binding.shape[-1])
        image_tokens = prefix_length - int(language_mask.shape[-1])
        if camera_count < 1 or image_tokens <= 0 or image_tokens % camera_count != 0:
            raise BindingError(
                "PI0.5 prefix attention shape is incompatible with the declared camera and language bindings"
            )

        image_mask = np.ones((language_mask.shape[0], image_tokens), dtype=bool)
        pad_mask = np.concatenate((image_mask, language_mask.astype(bool, copy=False)), axis=1)
        attention = pad_mask[:, None, :] & pad_mask[:, :, None]
        return np.where(attention[:, None, :, :], np.float32(0.0), _OPENPI_ATTENTION_MASK_VALUE)


class SmolVLAPolicyCodec(_ManifestPolicyCodec):
    """SmolVLA state, image, language, noise, time, and action semantics."""

    image_resize_mode = "pad"
    image_pad_alignment = "left_top"
    pad_state = True
    crop_action = True

    def encode_execution(self, request: CodecRequest, plan: ExecutionPlan) -> Mapping[str, BoundInputs]:
        """Bind only external policy tensors; internal prefix/KV tensors remain executor-owned."""

        role_inputs: dict[str, BoundInputs] = {}
        for role in plan.roles:
            bound: list[BoundTensor] = []
            for binding in role.bindings.inputs:
                semantic = binding.semantic
                if semantic.startswith("internal.") or semantic in _TIME_SEMANTICS:
                    continue
                try:
                    value = self._resolve_semantic(request.semantic_tensors, semantic)
                except MissingSemanticTensorError:
                    if semantic in _NOISE_SEMANTICS:
                        continue
                    raise
                value = self._prepare_value(binding, value)
                bound.append(
                    BoundTensor(
                        semantic=semantic,
                        runtime_name=binding.runtime_name,
                        index=binding.index,
                        value=convert_input(binding, value),
                    )
                )
            role_inputs[role.name] = BoundInputs(tuple(bound))
        return MappingProxyType(role_inputs)

    def decode_execution(self, outputs: RuntimeOutputs, plan: ExecutionPlan) -> CodecResult:
        action_roles = [
            role for role in plan.roles if any(binding.semantic == "action" for binding in role.bindings.outputs)
        ]
        if len(action_roles) != 1:
            raise BindingError(
                f"SmolVLA execution plan requires exactly one action-producing role, got {len(action_roles)}"
            )
        action_role = action_roles[0]
        role_outputs = outputs
        if isinstance(outputs, Mapping) and action_role.name in outputs:
            role_outputs = outputs[action_role.name]
        return self.decode_outputs(role_outputs, action_role.bindings)


class PolicyCodecRegistry:
    """Immutable policy-family selection with no runtime/backend participation."""

    def __init__(self, codecs: Mapping[str, type[_ManifestPolicyCodec]]) -> None:
        self._codecs = MappingProxyType(dict(codecs))

    @property
    def policy_types(self) -> tuple[str, ...]:
        return tuple(self._codecs)

    def create(self, metadata: PolicyMetadata) -> _ManifestPolicyCodec:
        try:
            codec_type = self._codecs[metadata.policy_type]
        except KeyError as exc:
            raise ValueError(
                f"policy type {metadata.policy_type!r} has no unified codec; available types: {list(self._codecs)}"
            ) from exc
        return codec_type(metadata)


POLICY_CODEC_REGISTRY = PolicyCodecRegistry(
    {
        "act": ACTPolicyCodec,
        "pi05": PI05PolicyCodec,
        "smolvla": SmolVLAPolicyCodec,
    }
)


def create_policy_codec(metadata: PolicyMetadata) -> _ManifestPolicyCodec:
    return POLICY_CODEC_REGISTRY.create(metadata)
