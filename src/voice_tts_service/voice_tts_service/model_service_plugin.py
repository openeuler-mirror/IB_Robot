"""Typed ZipVoice plugin for the shared model-service host."""

from __future__ import annotations

from array import array

from inference_service.backends import BackendError, RuntimeContext
from inference_service.model_service_plugin import ModelServiceError, ModelServicePlugin, PluginRuntimeStatus
from inference_service.model_sessions import ModelSession
from inference_service.runtime_composition import require_runtime_dependencies
from inference_service.unified_runtime import (
    ExecutionContext,
    ExecutionContract,
    ExecutionFailure,
    ModelRequest,
    ModelRuntimeHandle,
    RegistrySet,
    RuntimeAssembly,
    RuntimeProviders,
)

from .defaults import VOICE_TTS_DEFAULTS
from .errors import TTSError
from .service_core import TTSLimits, TTSServiceCore


class ZipVoiceSynthesizePlugin(ModelServicePlugin):
    """Expose ZipVoice through the family-neutral typed model-service host."""

    service_type = "ibrobot_msgs/srv/SynthesizeSpeech"
    interface = "tensor_model"
    model_type = "zipvoice"
    operation = "synthesize"

    def __init__(
        self,
        _host,
        validated,
        options,
        *,
        registry_set: RegistrySet | None = None,
        providers: RuntimeProviders | None = None,
    ) -> None:
        registry_set, providers = require_runtime_dependencies(
            registry_set,
            providers,
            owner=type(self).__name__,
        )
        model = validated.manifest.model
        if (
            model.interface != self.interface
            or model.model_type != self.model_type
            or model.operation != self.operation
        ):
            raise ValueError(f"ZipVoice plugin requires {self.interface}/{self.model_type}/{self.operation}")

        allowed = {
            "device_id",
            "prompt_profile",
            "segment_max_chars",
            "segment_pause_ms",
            "max_request_chars",
            "max_prompt_audio_bytes",
            "max_prompt_duration_sec",
            "max_segments",
            "max_response_audio_bytes",
        }
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise ValueError(f"unknown ZipVoice runtime options: {unknown}")

        # Service limits belong to TTSServiceCore, not to the vendor model
        # session. Keep the full options for the core but narrow the context
        # passed to Ascend/Torch session builders to their own contract.
        session_options = {name: options[name] for name in ("device_id", "prompt_profile") if name in options}
        runtime_profile = getattr(validated, "runtime_profile", None)
        if runtime_profile is None:
            role_profiles = getattr(validated, "role_runtime_profiles", {})
            runtime_profile = next(iter(role_profiles.values()), None)
        context = RuntimeContext(
            validated_manifest=validated,
            runtime_options=session_options,
            runtime_profile=runtime_profile,
        )
        self._session: ModelSession = registry_set.session_builder_registry.create(
            context,
            backend_registry=registry_set.backend_registry,
            providers=providers,
        )
        self._runtime_handle = ModelRuntimeHandle(
            RuntimeAssembly(
                runtime_executor=self._session,
                session=self._session,
                execution_contract=ExecutionContract(
                    execution_structure="iterative",
                    orchestration_visibility="session",
                    cancellation_granularity="checkpoint",
                ),
                stateful=bool(self._session.capabilities.stateful),
                resettable=bool(self._session.capabilities.resettable),
                runtime_id="voice-tts-zipvoice",
                load_context=context,
                providers=providers,
            )
        )
        self.pipeline = self._runtime_handle
        self._pipeline = self._runtime_handle
        self._core = TTSServiceCore(
            self._infer,
            TTSLimits(
                **{
                    name: options.get(name, VOICE_TTS_DEFAULTS[name])
                    for name in TTSLimits.__dataclass_fields__
                    if name != "sample_rate"
                }
            ),
        )
        self._closed = False
        try:
            self._runtime_handle.load(context)
        except Exception:
            self._runtime_handle.close()
            self._closed = True
            raise

    def _infer(self, request: ModelRequest):
        request_id = request.metadata.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("ZipVoice ModelRequest metadata must contain request_id")
        return self._runtime_handle.execute(
            request,
            ExecutionContext(request_id),
        )

    def handle(self, request, response) -> str:
        try:
            output = self._core.synthesize(
                request.text,
                bytes(request.prompt_audio),
                request.prompt_audio_format,
                request.prompt_text,
            )
        except Exception as exc:
            code = getattr(exc, "code", "")
            if isinstance(exc, TTSError):
                code = exc.code
            elif isinstance(exc, BackendError) and code == "unsupported_prompt":
                code = "UNSUPPORTED_PROMPT"
            elif isinstance(exc, BackendError | ExecutionFailure):
                code = "INFERENCE_FAILED"
            else:
                code = "INTERNAL_ERROR"
            raise ModelServiceError(
                str(exc),
                error_code=code,
                audio_segments=[],
                total_duration_sec=0.0,
                total_inference_time_ms=0.0,
            ) from exc

        from ibrobot_msgs.msg import SynthesizedAudio

        response.audio_segments = [
            SynthesizedAudio(
                index=segment.index,
                text=segment.text,
                audio_data=array("B", segment.wav_data),
                audio_format="wav_pcm_s16le",
                sample_rate=segment.sample_rate,
                channels=1,
                duration_sec=segment.duration_sec,
                inference_time_ms=segment.inference_time_ms,
                pause_after_ms=segment.pause_after_ms,
            )
            for segment in output.segments
        ]
        response.total_duration_sec = output.total_duration_sec
        response.total_inference_time_ms = output.total_inference_time_ms
        response.error_code = ""
        return f"synthesized {len(output.segments)} audio segment(s)"

    def runtime_status(self) -> PluginRuntimeStatus:
        health = self._runtime_handle.diagnostics().health
        state = health.state.value if hasattr(health.state, "value") else str(health.state)
        runtime_version = self._session.runtime_version
        ready = health.ready and bool(runtime_version)
        return PluginRuntimeStatus(
            state=state,
            ready=ready,
            failure_reason=""
            if ready
            else (
                "loaded model runtime did not expose a version"
                if health.ready
                else (health.message or health.reason_code or f"model runtime is {state}")
            ),
            metadata={"runtime_version": runtime_version} if runtime_version else {},
        )

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._runtime_handle.close()


__all__ = ["ZipVoiceSynthesizePlugin"]
