"""Manifest-backed Silero VAD runtime for Voice ASR."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from inference_manifest import load_inference_manifest
from inference_service.backends import RuntimeContext
from inference_service.runtime_composition import build_runtime_dependencies
from inference_service.unified_runtime import (
    ExecutionContext,
    ModelRequest,
    ModelRuntimeHandle,
    OwnedComponent,
    ResultAdapter,
    RuntimeAssembly,
)

from .model_session_builders import register_speech_direction_session_builder

_AUDIO_SEMANTIC = "host.silero.audio"
_SAMPLE_RATE_SEMANTIC = "host.silero.sample_rate"
_PROBABILITY_SEMANTIC = "host.silero.prob"
_CONTEXT_SIZE = 64


class ManifestVadRuntime:
    """Own one loaded Silero session and its input context."""

    def __init__(self, bundle_path: str | Path, deployment: str = "torch_cpu") -> None:
        self._bundle_path = Path(bundle_path)
        self._deployment = deployment
        self._session = None
        self._context = None
        self._handle = None
        self._stream = None
        self._providers = None
        self._audio_context = np.zeros(_CONTEXT_SIZE, dtype=np.float32)
        self._context_size = _CONTEXT_SIZE
        self._request_id = 0
        self._sample_rate_hz = 16000
        self._frame_size = 512
        self._chunk_size = 576

    def initialize(self) -> None:
        if self._handle is not None:
            raise RuntimeError("manifest-backed Silero VAD runtime is already initialized")
        validated = load_inference_manifest(self._bundle_path, self._deployment)
        identity = validated.top_level_identity
        if (identity.interface, identity.model_type, identity.operation) != ("tensor_model", "silero_vad", "vad"):
            raise ValueError("selected deployment is not tensor_model/silero_vad/vad")
        if validated.deployment.backend != "onnx":
            raise ValueError("Voice ASR Silero VAD requires an ONNX Runtime deployment")
        contract = validated.deployment.audio_contract
        if contract is None or contract.sample_rate_hz is None or contract.frame_size is None:
            raise ValueError("Silero deployment must declare sample_rate_hz and frame_size")
        if contract.chunk_size is None or contract.chunk_size <= contract.frame_size:
            raise ValueError("Silero deployment must declare chunk_size larger than frame_size")
        if contract.channels != 1 or contract.sample_dtype != "float32":
            raise ValueError("Voice ASR Silero VAD requires mono float32 audio")
        if contract.execution_mode not in {"streaming", "both"}:
            raise ValueError("Voice ASR Silero VAD requires a streaming deployment")
        self._sample_rate_hz = contract.sample_rate_hz
        self._frame_size = contract.frame_size
        self._chunk_size = contract.chunk_size
        self._context_size = self._chunk_size - self._frame_size
        self._audio_context = np.zeros(self._context_size, dtype=np.float32)
        self._context = RuntimeContext(validated)
        dependencies = build_runtime_dependencies(
            lambda session_registry, _assembler_registry: register_speech_direction_session_builder(session_registry)
        )
        self._providers = dependencies.providers
        try:
            self._session = dependencies.registry_set.session_builder_registry.create(
                self._context,
                backend_registry=dependencies.registry_set.backend_registry,
                providers=dependencies.providers,
            )
            adapter = _VadStreamingAdapter(self._session)
            assembly = RuntimeAssembly(
                runtime_executor=_VadStreamExecutor(),
                streaming_runtime=adapter,
                session=self._session,
                owned_components=(OwnedComponent(self._session, "silero-vad-session", load_context=self._context),),
                request_adapter=ResultAdapter(required_outputs=(_PROBABILITY_SEMANTIC,)),
                stateful=True,
                resettable=True,
                state_scope="stream",
                state_bank_mode="runtime_exclusive",
                max_open_streams=1,
                load_context=self._context,
                identity=identity,
                execution_contract=validated.deployment.execution_contract,
                deployment_fingerprint=validated.deployment_fingerprint,
                runtime_profile_fingerprint=validated.runtime_profile_fingerprint,
                artifact_integrity=validated.integrity_status,
            )
            self._handle = ModelRuntimeHandle(assembly)
            self._handle.load()
            self._stream = self._handle.open_stream(ExecutionContext("voice-asr-vad-open"))
            self._audio_context.fill(0)
        except Exception:
            self.close()
            raise

    @property
    def sample_rate_hz(self) -> int:
        return self._sample_rate_hz

    @property
    def frame_size(self) -> int:
        return self._frame_size

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    def infer(self, audio_frame: np.ndarray) -> float:
        if self._session is None or self._context is None:
            raise RuntimeError("manifest-backed Silero VAD runtime is not initialized")
        frame = np.asarray(audio_frame, dtype=np.float32).reshape(-1)
        if frame.size != self._frame_size:
            raise ValueError(f"Silero VAD expects {self._frame_size} samples, got {frame.size}")
        model_audio = np.concatenate((self._audio_context, frame)).reshape(1, -1)
        self._request_id += 1
        execution = ExecutionContext(f"voice-asr-vad-{self._request_id}")
        request = ModelRequest(
            {
                _AUDIO_SEMANTIC: model_audio,
                _SAMPLE_RATE_SEMANTIC: np.asarray(self._sample_rate_hz, dtype=np.int64),
            },
            {"deployment": self._deployment},
        )
        result = self._handle.step(self._stream, request, execution)
        self._audio_context = frame[-self._context_size :].copy()
        return float(np.asarray(result.outputs[_PROBABILITY_SEMANTIC]).reshape(-1)[0])

    def reset(self) -> None:
        if self._handle is not None and self._stream is not None:
            self._handle.reset_stream(self._stream, ExecutionContext(f"voice-asr-vad-reset-{self._request_id}"))
        self._audio_context.fill(0)

    def close(self) -> None:
        error = None
        try:
            if self._handle is not None:
                self._handle.close()
            elif self._session is not None:
                self._session.close()
        except Exception as exc:  # pragma: no cover - cleanup error is re-raised below
            error = exc
        finally:
            self._handle = None
            self._stream = None
            self._session = None
            self._context = None
            if self._providers is not None:
                self._providers.close()
                self._providers = None
        if error is not None:
            raise error


class _VadStreamExecutor:
    def execute(self, request, context):
        raise RuntimeError("Silero VAD is stream-scoped and must use step()")


class _VadStreamingAdapter:
    def __init__(self, session) -> None:
        self._session = session

    def open_stream(self, _context):
        return None

    def step(self, _stream, request, context):
        return self._session.execute_role("silero_vad", request.inputs, request, context)

    def reset_stream(self, _stream, _context):
        self._session.reset()

    def close_stream(self, _stream, _context):
        return None
