"""Voice ASR model-session builders."""

from __future__ import annotations

from inference_service.model_sessions import (
    StatefulAscendOmModelSession,
    build_ascend_model_session,
    build_onnx_model_session,
)
from inference_service.unified_runtime import RuntimeDependencyError, SessionBuilderKey

SPEECH_DIRECTION_IDENTITY = ("tensor_model", "speech_direction", "enhance_and_vad")
SPEECH_DIRECTION_ROLE_IDENTITIES = (
    ("fullsubnet", "enhance"),
    ("silero_vad", "vad"),
)
SPEECH_DIRECTION_BACKENDS = {
    "ascend": build_ascend_model_session,
    "onnx": build_onnx_model_session,
}

_STATE_ABI = {
    "silero_vad": (("host.silero.state_in", "host.silero.state_out"),),
    "fullsubnet_fb": (
        ("host.fullsubnet.fb_hidden_in", "host.fullsubnet.fb_hidden_out"),
        ("host.fullsubnet.fb_cell_in", "host.fullsubnet.fb_cell_out"),
    ),
    "fullsubnet_sb": (
        ("host.fullsubnet.sb_hidden_in", "host.fullsubnet.sb_hidden_out"),
        ("host.fullsubnet.sb_cell_in", "host.fullsubnet.sb_cell_out"),
    ),
}


def build_speech_direction_session(context, *, providers=None):
    return StatefulAscendOmModelSession(
        device_id=context.device_id or 0,
        runtime_manager=getattr(providers, "acl_runtime_provider", None),
        state_abi=_STATE_ABI,
    )


def register_speech_direction_session_builder(registry=None) -> None:
    if registry is None:
        raise RuntimeDependencyError(
            "register_speech_direction_session_builder requires an explicit session registry",
            code="session_builder_registry_required",
        )
    for model_type, operation in SPEECH_DIRECTION_ROLE_IDENTITIES:
        key = SessionBuilderKey("tensor_model", model_type, operation, "ascend")
        if registry.get(key) is None:
            registry.register(key, build_speech_direction_session)


__all__ = [
    "SPEECH_DIRECTION_BACKENDS",
    "SPEECH_DIRECTION_IDENTITY",
    "SPEECH_DIRECTION_ROLE_IDENTITIES",
    "register_speech_direction_session_builder",
]
