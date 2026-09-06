"""LeRobot policy execution through the shared model-session lifecycle."""

from __future__ import annotations

import gc
import importlib
import inspect
import math
from collections.abc import Mapping, Sequence
from contextlib import nullcontext, suppress
from typing import Any

from inference_manifest import TorchRuntimeProfile
from inference_service.auto_horizon_runtime import build_action_attention_collector
from inference_service.backends.errors import BackendInferenceError, BackendLoadError
from inference_service.backends.types import BackendAdmissionEvidence, BackendCapabilities, RuntimeContext
from inference_service.lerobot_assets import (
    TOKENIZER_REFERENCE_KEYS,
    VLM_REFERENCE_KEYS,
    resolve_local_semantic_reference,
)
from inference_service.model_sessions.base import ModelSession
from inference_service.unified_runtime import ExecutionContext, LoadRollback, ModelRequest
from robot_config.inference_runtime_options import (
    AUTO_HORIZON_RUNTIME_OPTION_DEFAULTS,
    TORCH_RUNTIME_MODEL_DTYPE_DEFAULT,
)

_ACTION_METHODS = frozenset({"predict_action_chunk", "select_action"})
_NOISE_KEYS = ("_noise", "action.noise", "noise")
_MODEL_DTYPES = {"native": None, "fp16": "half", "bf16": "bfloat16", "fp32": "float"}
# Same option set as the profile-compatibility identity in robot_config.
_AUTO_HORIZON_OPTIONS = frozenset(AUTO_HORIZON_RUNTIME_OPTION_DEFAULTS)


class LeRobotTorchModelSession(ModelSession):
    """Own one native LeRobot policy and its device-aware processors."""

    def __init__(
        self,
        device_name: str,
        *,
        priority_scheduling: bool = False,
    ) -> None:
        super().__init__(
            "model-session:lerobot-torch",
            BackendCapabilities(
                thread_safe=False,
                max_in_flight_per_instance=1,
                supports_multiple_instances=True,
                hardware_resource_id=f"torch:{device_name}" if priority_scheduling else None,
                resource_domain=f"torch:{device_name}",
                max_in_flight_per_resource_domain=1,
                admission_evidence=BackendAdmissionEvidence(
                    sdk_initialization=True,
                    multi_instance_execution=True,
                    failure_isolation=True,
                    independent_close=True,
                ),
            ),
        )
        self._configured_device_name = device_name
        self._torch: Any | None = None
        self._device: Any | None = None
        self._policy: Any | None = None
        self._policy_config: Any | None = None
        self._preprocessor: Any | None = None
        self._postprocessor: Any | None = None
        self._model_dtype = "native"
        self._last_metadata: dict[str, object] = {}
        self._attention_collector = None

    @property
    def policy(self) -> Any | None:
        return self._policy

    @property
    def policy_config(self) -> Any | None:
        return self._policy_config

    @property
    def runtime_version(self) -> str:
        return self._runtime_version(self._torch)

    def preprocess(self, inputs: Mapping[str, object]) -> Mapping[str, object]:
        if self._preprocessor is None:
            raise BackendInferenceError("LeRobot preprocessor is not loaded", code="runtime_not_loaded")
        return self._preprocessor(dict(inputs))

    def postprocess(self, action: object) -> object:
        if self._postprocessor is None:
            raise BackendInferenceError("LeRobot postprocessor is not loaded", code="runtime_not_loaded")
        return self._postprocessor(action)

    def execution_metadata(self, request_id: str) -> Mapping[str, object]:
        if self._last_metadata.get("request_id") != request_id:
            return {}
        return dict(self._last_metadata)

    def _load(self, context: RuntimeContext, rollback: LoadRollback) -> None:
        profile = context.backend_profile
        if (
            context.backend != "torch"
            or context.interface != "policy"
            or context.operation != "predict"
            or not isinstance(profile, TorchRuntimeProfile)
        ):
            raise BackendLoadError(
                "LeRobotTorchModelSession requires policy/*/predict with a typed Torch profile",
                code="invalid_deployment",
            )
        if profile.device != self._configured_device_name:
            raise BackendLoadError(
                "Torch deployment device does not match the session", code="deployment_context_mismatch"
            )
        model_dtype = self.validate_runtime_options(context.runtime_options)
        torch_module = self._import_required("torch", "PyTorch")
        self._validate_device(torch_module, profile.device)
        if profile.device == "npu":
            self._import_required("torch_npu", "torch_npu")
            self._validate_npu(torch_module)
        try:
            device = torch_module.device(profile.device)
        except Exception as exc:
            raise BackendLoadError(
                f"cannot construct Torch device {profile.device!r}: {exc}", code="invalid_device"
            ) from exc

        config_type = self._import_attribute(
            "lerobot.configs.policies", "PreTrainedConfig", "LeRobot policy configuration"
        )
        factory = self._import_required("lerobot.policies.factory", "LeRobot policy factory")
        get_policy_class = self._require_attribute(factory, "get_policy_class", "LeRobot policy factory")
        make_processors = self._require_attribute(factory, "make_pre_post_processors", "LeRobot processor factory")
        bundle_path = str(context.validated_manifest.bundle_root)
        policy_config = config_type.from_pretrained(bundle_path, local_files_only=True)
        try:
            policy_config.device = profile.device
        except (AttributeError, TypeError) as exc:
            raise BackendLoadError(
                "LeRobot policy config does not permit runtime device placement", code="incompatible_policy_config"
            ) from exc
        # Avoid network access during policy construction. The strict checkpoint load
        # restores the bundled backbone parameters from the local model artifacts.
        try:
            if getattr(policy_config, "pretrained_backbone_weights", None) is not None:
                policy_config.pretrained_backbone_weights = None
        except (AttributeError, TypeError) as exc:
            raise BackendLoadError(
                "LeRobot policy config does not permit the backbone weights override",
                code="incompatible_policy_config",
            ) from exc
        try:
            local_vlm_path = resolve_local_semantic_reference(
                context.validated_manifest.bundle_root, "config.json", VLM_REFERENCE_KEYS
            )
            tokenizer_path = resolve_local_semantic_reference(
                context.validated_manifest.bundle_root, "policy_preprocessor.json", TOKENIZER_REFERENCE_KEYS
            )
        except ValueError as exc:
            raise BackendLoadError(str(exc), code="invalid_policy_assets") from exc
        if local_vlm_path is not None:
            policy_config.vlm_model_name = local_vlm_path
        policy_class = get_policy_class(context.model_type)
        policy = policy_class.from_pretrained(
            bundle_path,
            config=policy_config,
            local_files_only=True,
            strict=True,
        )
        if context.runtime_options.get("auto_horizon_enabled", False) and context.model_type != "pi05":
            raise BackendLoadError(
                "AutoHorizon currently supports native Torch PI0.5 policies only",
                code="unsupported_auto_horizon_policy",
            )

        self._torch = torch_module
        self._device = device
        self._policy = policy
        self._policy_config = policy_config
        rollback.defer(self._release)
        moved = policy.to(device)
        if moved is not None:
            self._policy = policy = moved
        self._cast_model(policy, model_dtype)
        evaluated = policy.eval()
        if evaluated is not None:
            self._policy = policy = evaluated
        try:
            self._attention_collector = build_action_attention_collector(policy, context.runtime_options)
        except (TypeError, ValueError) as exc:
            raise BackendLoadError(f"unable to enable AutoHorizon: {exc}", code="auto_horizon_setup_failed") from exc
        preprocessor_overrides: dict[str, dict[str, object]] = {"device_processor": {"device": str(device)}}
        if tokenizer_path is not None:
            preprocessor_overrides["tokenizer_processor"] = {"tokenizer_name": tokenizer_path}
        self._preprocessor, self._postprocessor = make_processors(
            policy_cfg=policy_config,
            pretrained_path=bundle_path,
            preprocessor_overrides=preprocessor_overrides,
            postprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        self._model_dtype = model_dtype
        resettable = callable(getattr(policy, "reset", None))
        self._update_loaded_capabilities(
            resettable=resettable,
            stateful=resettable,
            supports_attention=self.observes_attention(policy),
        )

    def _execute(self, request: ModelRequest, context: ExecutionContext) -> Mapping[str, object]:
        context.check("backend")
        policy = self._policy
        torch_module = self._torch
        if policy is None or torch_module is None:
            raise BackendInferenceError("LeRobot Torch model session is not loaded", code="runtime_not_loaded")
        input_descriptors = {d.semantic: d for d in self._require_context().validated_manifest.manifest.model.inputs}
        batch = {}
        for key, value in request.inputs.items():
            tensor = self._move_input(value)
            descriptor = input_descriptors.get(key)
            if (
                descriptor is not None
                and hasattr(tensor, "dim")
                and hasattr(tensor, "shape")
                and tensor.dim() == len(descriptor.shape)
            ):
                tensor = tensor.unsqueeze(0)
            batch[key] = tensor
        action_method = request.metadata.get("action_method", "predict_action_chunk")
        if not isinstance(action_method, str) or action_method not in _ACTION_METHODS:
            raise BackendInferenceError(
                f"unsupported native action method {action_method!r}", code="invalid_action_method"
            )
        if self._attention_collector is not None and action_method != "predict_action_chunk":
            raise BackendInferenceError(
                "AutoHorizon requires predict_action_chunk",
                code="unsupported_auto_horizon_action_method",
            )
        callback = getattr(policy, action_method, None)
        if not callable(callback):
            raise BackendInferenceError(
                f"loaded policy does not implement {action_method}", code="unsupported_action_method"
            )
        noise = self._pop_noise(batch)
        kwargs = {}
        if noise is not None and self._accepts_keyword(callback, "noise"):
            kwargs["noise"] = self._place_noise(noise)
        inference_mode = getattr(torch_module, "inference_mode", None)
        if self._attention_collector is not None:
            self._attention_collector.begin()
        with inference_mode() if callable(inference_mode) else nullcontext():
            action = callback(batch, **kwargs)
        action = self._remove_batch_dimension(action)
        attention_metadata = self._attention_collector.estimate() if self._attention_collector is not None else {}
        self._last_metadata = {
            "request_id": context.request_id,
            "policy_type": self._require_context().model_type,
            "device": self._configured_device_name,
            "action_method": action_method,
            "external_noise": noise is not None and "noise" in kwargs,
            "model_dtype": self._model_dtype,
            **attention_metadata,
        }
        return {"action": action}

    def _validate_values(self, values, descriptors, direction):
        del values, descriptors, direction

    def _reset(self) -> None:
        reset = getattr(self._policy, "reset", None)
        if not callable(reset):
            super()._reset()
        reset()
        for processor in (self._preprocessor, self._postprocessor):
            processor_reset = getattr(processor, "reset", None)
            if callable(processor_reset):
                processor_reset()

    def _close(self) -> None:
        self._release()

    def _release(self) -> None:
        torch_module = self._torch
        device_name = self._configured_device_name
        if self._attention_collector is not None:
            self._attention_collector.close()
        self._policy = None
        self._policy_config = None
        self._preprocessor = None
        self._postprocessor = None
        self._device = None
        self._torch = None
        self._attention_collector = None
        self._last_metadata = {}
        gc.collect()
        cache_owner = getattr(torch_module, device_name, None) if torch_module is not None else None
        empty_cache = getattr(cache_owner, "empty_cache", None)
        if callable(empty_cache):
            with suppress(Exception):
                empty_cache()

    def _move_input(self, value: object) -> object:
        if value is None or isinstance(value, str | bytes | Mapping):
            return value
        if (
            isinstance(value, Sequence)
            and not isinstance(value, str | bytes)
            and (not value or isinstance(value[0], str | bytes | Mapping))
        ):
            return value
        is_tensor = getattr(self._torch, "is_tensor", None)
        if callable(is_tensor) and is_tensor(value):
            tensor = value
        else:
            try:
                tensor = self._torch.as_tensor(value)
            except (AttributeError, TypeError, ValueError):
                return value
        move = getattr(tensor, "to", None)
        return move(self._device) if callable(move) else tensor

    def _place_noise(self, noise: object) -> object:
        projection = getattr(getattr(self._policy, "model", None), "action_in_proj", None)
        parameter = getattr(projection, "weight", None)
        parameters = getattr(self._policy, "parameters", None)
        if parameter is None and callable(parameters):
            with suppress(StopIteration, TypeError):
                parameter = next(iter(parameters()))
        move = getattr(noise, "to", None)
        if not callable(move):
            return noise
        if parameter is not None:
            return move(device=getattr(parameter, "device", self._device), dtype=getattr(parameter, "dtype", None))
        return move(self._device)

    @staticmethod
    def validate_runtime_options(options: Mapping[str, object]) -> str:
        unknown = sorted(set(options) - ({"model_dtype"} | _AUTO_HORIZON_OPTIONS))
        if unknown:
            raise BackendLoadError(f"unknown Torch runtime options: {unknown}", code="invalid_runtime_options")
        model_dtype = options.get("model_dtype", TORCH_RUNTIME_MODEL_DTYPE_DEFAULT)
        if not isinstance(model_dtype, str) or model_dtype not in _MODEL_DTYPES:
            raise BackendLoadError(
                f"unsupported Torch model_dtype {model_dtype!r}; expected one of {sorted(_MODEL_DTYPES)}",
                code="invalid_runtime_options",
            )
        enabled = options.get("auto_horizon_enabled", AUTO_HORIZON_RUNTIME_OPTION_DEFAULTS["auto_horizon_enabled"])
        if type(enabled) is not bool:
            raise BackendLoadError("auto_horizon_enabled must be a boolean", code="invalid_runtime_options")
        for name in ("auto_horizon_hold_threshold", "auto_horizon_entropy_quantile"):
            value = options.get(name, AUTO_HORIZON_RUNTIME_OPTION_DEFAULTS[name])
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
                raise BackendLoadError(f"{name} must be a finite number", code="invalid_runtime_options")
            if not 0.0 <= float(value) <= 1.0:
                raise BackendLoadError(f"{name} must be between 0 and 1", code="invalid_runtime_options")
        for name in ("auto_horizon_run_length", "auto_horizon_sampling_step"):
            value = options.get(name, AUTO_HORIZON_RUNTIME_OPTION_DEFAULTS[name])
            if type(value) is not int or value < 1:
                raise BackendLoadError(f"{name} must be positive", code="invalid_runtime_options")
        return model_dtype

    @staticmethod
    def observes_attention(policy: object) -> bool:
        for owner in (policy, getattr(policy, "model", None)):
            if owner is None:
                continue
            declared = getattr(owner, "supports_attention", None)
            if isinstance(declared, bool):
                return declared
            if any(hasattr(owner, name) for name in ("last_attn_weights", "get_attention_maps", "get_attentions")):
                return True
            if any(
                hasattr(layer, "multihead_attn") for layer in getattr(getattr(owner, "decoder", None), "layers", ())
            ):
                return True
        return False

    @staticmethod
    def _pop_noise(batch: dict[str, object]) -> object | None:
        noise = None
        for key in _NOISE_KEYS:
            if key in batch:
                value = batch.pop(key)
                if noise is None:
                    noise = value
        return noise

    @staticmethod
    def _accepts_keyword(callback: Any, keyword: str) -> bool:
        try:
            parameters = inspect.signature(callback).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(p.kind is inspect.Parameter.VAR_KEYWORD or p.name == keyword for p in parameters)

    @staticmethod
    def _remove_batch_dimension(action: object) -> object:
        shape = getattr(action, "shape", ())
        squeeze = getattr(action, "squeeze", None)
        return squeeze(0) if len(shape) >= 2 and shape[0] == 1 and callable(squeeze) else action

    @staticmethod
    def _cast_model(policy: object, model_dtype: str) -> None:
        method_name = _MODEL_DTYPES[model_dtype]
        if method_name is None:
            return
        cast = getattr(getattr(policy, "model", None), method_name, None)
        if not callable(cast):
            raise BackendLoadError(
                f"loaded policy model does not support {model_dtype!r}", code="unsupported_model_dtype"
            )
        converted = cast()
        if converted is not None:
            policy.model = converted

    @staticmethod
    def _validate_device(torch_module: Any, device: str) -> None:
        if device in {"cpu", "npu"}:
            return
        if device == "cuda":
            owner = getattr(torch_module, "cuda", None)
        elif device == "mps":
            owner = getattr(getattr(torch_module, "backends", None), "mps", None)
        else:
            owner = None
        available = getattr(owner, "is_available", None)
        if not callable(available) or not available():
            raise BackendLoadError(
                f"Torch deployment device {device!r} is not available on this host", code="device_unavailable"
            )

    @staticmethod
    def _validate_npu(torch_module: Any) -> None:
        available = getattr(getattr(torch_module, "npu", None), "is_available", None)
        if not callable(available) or not available():
            raise BackendLoadError(
                "Torch deployment device 'npu' is not available after importing torch_npu", code="device_unavailable"
            )

    @staticmethod
    def _import_required(module_name: str, description: str) -> Any:
        try:
            return importlib.import_module(module_name)
        except (ImportError, OSError) as exc:
            raise BackendLoadError(
                f"{description} dependency {module_name!r} is unavailable: {exc}", code="missing_dependency"
            ) from exc

    @classmethod
    def _import_attribute(cls, module_name: str, attribute: str, description: str) -> Any:
        return cls._require_attribute(cls._import_required(module_name, description), attribute, description)

    @staticmethod
    def _require_attribute(module: Any, attribute: str, description: str) -> Any:
        value = getattr(module, attribute, None)
        if not callable(value):
            raise BackendLoadError(
                f"{description} does not expose callable {attribute!r}", code="incompatible_dependency"
            )
        return value


class LeRobotSessionPreprocessor:
    def __init__(self, session: LeRobotTorchModelSession) -> None:
        self._session = session

    def __call__(self, inputs: Mapping[str, object]) -> Mapping[str, object]:
        return self._session.preprocess(inputs)


class LeRobotSessionPostprocessor:
    def __init__(self, session: LeRobotTorchModelSession) -> None:
        self._session = session

    def __call__(self, action: object) -> object:
        return self._session.postprocess(action)
