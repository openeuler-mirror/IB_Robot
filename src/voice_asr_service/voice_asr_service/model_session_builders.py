"""Voice ASR model-session builders."""

from __future__ import annotations

from inference_service.model_sessions import build_ascend_model_session, build_onnx_model_session
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


def register_speech_direction_session_builder(registry=None) -> None:
    if registry is None:
        raise RuntimeDependencyError(
            "register_speech_direction_session_builder requires an explicit session registry",
            code="session_builder_registry_required",
        )
    for model_type, operation in SPEECH_DIRECTION_ROLE_IDENTITIES:
        for backend, builder in SPEECH_DIRECTION_BACKENDS.items():
            key = SessionBuilderKey("tensor_model", model_type, operation, backend)
            if registry.get(key) is None:
                registry.register(key, builder)


__all__ = [
    "SPEECH_DIRECTION_BACKENDS",
    "SPEECH_DIRECTION_IDENTITY",
    "SPEECH_DIRECTION_ROLE_IDENTITIES",
    "register_speech_direction_session_builder",
]
