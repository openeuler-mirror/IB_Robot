"""Voice ASR package exports."""

from importlib import import_module

_EXPORTS = {
    "AudioCaptureModule": (".audio_capture_module", "AudioCaptureModule"),
    "AudioConfig": (".audio_capture_module", "AudioConfig"),
    "CaptureState": (".audio_capture_module", "CaptureState"),
    "RingBuffer": (".audio_capture_module", "RingBuffer"),
    "FileInputModule": (".file_input_module", "FileInputModule"),
    "FileResult": (".file_input_module", "FileResult"),
    "FileState": (".file_input_module", "FileState"),
    "FileError": (".file_input_module", "FileError"),
    "ASRInferenceModule": (".asr_inference_module", "ASRInferenceModule"),
    "ASRResult": (".asr_inference_module", "ASRResult"),
    "ASRState": (".asr_inference_module", "ASRState"),
    "VADModule": (".vad_module", "VADModule"),
    "VADConfig": (".vad_module", "VADConfig"),
    "VADState": (".vad_module", "VADState"),
    "VADResult": (".vad_module", "VADResult"),
    "ManifestVadRuntime": (".vad_runtime", "ManifestVadRuntime"),
    "StateMachine": (".state_machine", "StateMachine"),
    "NodeState": (".state_machine", "NodeState"),
    "ActiveMode": (".state_machine", "ActiveMode"),
}


def __getattr__(name):
    """Load package exports lazily so light-weight submodules stay importable."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "AudioCaptureModule",
    "AudioConfig",
    "CaptureState",
    "RingBuffer",
    "FileInputModule",
    "FileResult",
    "FileState",
    "FileError",
    "ASRInferenceModule",
    "ASRResult",
    "ASRState",
    "VADModule",
    "VADConfig",
    "VADState",
    "VADResult",
    "ManifestVadRuntime",
    "StateMachine",
    "NodeState",
    "ActiveMode",
]
