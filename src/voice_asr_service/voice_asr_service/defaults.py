"""Shared default values for Voice ASR configuration."""

VOICE_ASR_DEFAULTS = {
    "enabled": False,
    "auto_download_model": True,
    "active_mode": "continuous",
    "language": "zh",
    "model_path": "",
    "tokens_path": "",
    "provider": "cpu",
    "model_type": "auto",
    "max_recording_duration": 10.0,
    "vad_sensitivity": 0.6,
    "realtime_pre_roll_seconds": 0.5,
    "publish_partial": True,
    "output_topic": "/voice_command",
    "sample_rate": 16000,
    "chunk_size": 512,
    "buffer_seconds": 5.0,
    "device_index": -1,
    "device_name": "",
    "exit_on_init_failure": True,
}
