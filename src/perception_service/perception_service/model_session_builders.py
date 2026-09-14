"""Perception-owned builders registered with the shared model-session registry."""

from __future__ import annotations

import importlib
import json

from inference_manifest import TorchRuntimeProfile
from inference_service.backends import RuntimeContext
from inference_service.model_sessions import AscendOmModelSession, TorchModelSession
from inference_service.unified_runtime import RuntimeDependencyError, RuntimeProviders, SessionBuilderRegistry

from .graspgen_session import GraspGenAscendSession
from .sam2_automatic_ascend_session import SAM2AutomaticAscendSession
from .siglip2_ascend_session import SigLIP2AscendSession


def _load_torch_module(model_type: str, context: RuntimeContext):
    identity_path = context.validated_manifest.bundle_root / "assets" / "adapter.json"
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        module_name, function_name = identity["torch_module_loader"].split(":", 1)
        loader = getattr(importlib.import_module(module_name), function_name)
    except (AttributeError, ImportError, KeyError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"{model_type} Torch bundle has no loadable model integration in {identity_path}: {exc}"
        ) from exc
    module = loader(context)
    if not callable(module):
        raise TypeError(f"{model_type} Torch module loader did not return a callable")
    return module


def _ascend_device_id(model_type: str, adapter, context: RuntimeContext, options, allowed: set[str]) -> int:
    if context.backend != "ascend" or context.target_runtime != "acl" or not adapter.compiled_abi_finalized:
        raise RuntimeError(f"{model_type} compiled adapter ABI is not finalized or is not an ACL deployment")
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(f"unknown {model_type} runtime options: {unknown}")
    profile_device = context.device_id
    option_device = options.get("device_id", profile_device if profile_device is not None else 0)
    if profile_device is not None and option_device != profile_device:
        raise ValueError("runtime option device_id does not match the typed Ascend profile")
    return int(option_device)


def build_perception_session(
    context: RuntimeContext,
    *,
    adapter,
    providers: RuntimeProviders | None = None,
):
    model_type = context.model_type
    options = context.runtime_options
    if context.backend == "ascend":
        device_id = _ascend_device_id(model_type, adapter, context, options, {"device_id"})
        runtime_manager = getattr(providers, "acl_runtime_provider", None) if providers is not None else None
        if model_type == "siglip2":
            return SigLIP2AscendSession(device_id=device_id, runtime_manager=runtime_manager)
        if model_type == "sam2" and adapter.identity.operation == "automatic":
            return SAM2AutomaticAscendSession(device_id=device_id, runtime_manager=runtime_manager)
        return AscendOmModelSession(device_id=device_id, runtime_manager=runtime_manager)
    if context.backend == "torch":
        if options:
            raise ValueError(
                f"Torch model plugins do not select backend/device or accept runtime options: {sorted(options)}"
            )
        profile = context.backend_profile
        if not isinstance(profile, TorchRuntimeProfile) or profile.device not in {"cpu", "cuda"}:
            raise RuntimeError("generic Torch perception sessions support only typed cpu/cuda profiles")
        return TorchModelSession(
            lambda runtime_context: _load_torch_module(model_type, runtime_context),
        )
    raise RuntimeError(f"{model_type} deployment backend {context.backend!r} is unsupported")


def build_graspgen_session(
    context: RuntimeContext,
    *,
    adapter,
    providers: RuntimeProviders | None = None,
):
    """Build a Torch CUDA or host-orchestrated Ascend GraspGen session."""

    model_type = context.model_type
    options = context.runtime_options
    if context.backend == "torch":
        if options:
            raise ValueError(f"GraspGen Torch deployment does not accept runtime options: {sorted(options)}")
        profile = context.backend_profile
        if not isinstance(profile, TorchRuntimeProfile) or profile.device != "cuda":
            raise RuntimeError("GraspGen Torch requires a typed cuda profile")
        return TorchModelSession(
            lambda runtime_context: _load_torch_module(model_type, runtime_context),
        )
    if context.backend != "ascend":
        raise RuntimeError(f"{model_type} requires a Torch CUDA or ACL deployment")
    device_id = _ascend_device_id(
        model_type,
        adapter,
        context,
        options,
        {"device_id", "random_seed"},
    )
    return GraspGenAscendSession(
        device_id=device_id,
        config=adapter.config,
        runtime_manager=(getattr(providers, "acl_runtime_provider", None) if providers is not None else None),
    )


def register_perception_session_builders(registry: SessionBuilderRegistry | None = None) -> None:
    if registry is None:
        raise RuntimeDependencyError(
            "register_perception_session_builders requires an explicit session registry",
            code="session_builder_registry_required",
        )
    for model_type, operation in (
        ("ram_plus", "recognize_tags"),
        ("sam2", "automatic"),
        ("sam2", "prompt"),
        ("siglip2", "encode"),
        ("grounding_dino", "detect"),
        ("yolox_person", "detect"),
        ("pear_parameter_network", "predict_parameters"),
    ):
        for backend in ("torch", "ascend"):
            key = ("tensor_model", model_type, operation, backend)
            if registry.get(*key) is None:
                registry.register(
                    "tensor_model",
                    model_type,
                    operation,
                    backend,
                    build_perception_session,
                )

    for backend in ("torch", "ascend"):
        key = ("tensor_model", "graspgen", "generate_grasps", backend)
        if registry.get(*key) is None:
            registry.register(
                "tensor_model",
                "graspgen",
                "generate_grasps",
                backend,
                build_graspgen_session,
            )


__all__ = ["build_graspgen_session", "build_perception_session", "register_perception_session_builders"]
