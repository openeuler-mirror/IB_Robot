"""Shared default values for Voice ASR configuration."""

VOICE_ASR_DEFAULTS = {
    "enabled": False,
    "active_mode": "continuous",
    "language": "zh",
    "bundle_path": "models/voice_asr/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23",
    "deployment": "torch_cpu",
    "max_recording_duration": 10.0,
    "vad_sensitivity": 0.6,
    "vad_bundle_path": "models/silero-vad",
    "vad_deployment": "torch_cpu",
    "realtime_pre_roll_seconds": 0.5,
    "publish_partial": True,
    "output_topic": "/voice_command",
    "sample_rate": 16000,
    "chunk_size": 512,
    "buffer_seconds": 5.0,
    "audio_input_channel": 1,
    "exit_on_init_failure": True,
}
