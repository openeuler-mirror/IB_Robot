"""Non-model I/O boundary contract tests for Voice ASR file and capture inputs.

``FileInputModule`` and ``AudioCaptureModule`` are I/O components, not model
runtimes. These tests pin their boundary contracts: decode/resample to mono
float32 at the ASR sample rate, frame accumulation at the declared chunk size,
and pre-roll retention — without any model runtime involvement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from voice_asr_service.audio_capture_module import AudioCaptureModule, AudioConfig  # noqa: E402
from voice_asr_service.file_input_module import FileError, FileInputModule  # noqa: E402


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int, channels: int = 1) -> None:
    import soundfile as sf

    data = samples if channels == 1 else np.stack([samples] * channels, axis=1)
    sf.write(str(path), data, sample_rate, subtype="PCM_16")


def test_wav_decode_produces_mono_float32_at_target_rate(tmp_path: Path):
    wav_path = tmp_path / "speech.wav"
    original = (np.sin(np.linspace(0.0, 440 * 2 * np.pi, 16000)) * 0.4).astype(np.float32)
    _write_wav(wav_path, original, sample_rate=16000)

    module = FileInputModule()
    result = module.load_file(str(wav_path))

    assert result.success is True
    assert result.error_code is None
    assert result.sample_rate == FileInputModule.TARGET_SAMPLE_RATE == 16000
    audio = result.audio_data
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert np.all(np.abs(audio) <= 1.0)
    assert len(audio) == 16000
    assert result.duration == pytest.approx(1.0)


def test_wav_resampling_converts_to_target_rate(tmp_path: Path):
    wav_path = tmp_path / "resample.wav"
    _write_wav(wav_path, np.zeros(8000, dtype=np.float32), sample_rate=8000)

    module = FileInputModule()
    result = module.load_file(str(wav_path))

    assert result.success is True
    assert result.sample_rate == 16000
    assert result.duration == pytest.approx(1.0)
    assert len(result.audio_data) == 16000


def test_stereo_wav_is_downmixed_to_first_channel(tmp_path: Path):
    import soundfile as sf

    wav_path = tmp_path / "stereo.wav"
    left = np.full(16000, 0.25, dtype=np.float32)
    right = np.full(16000, -0.75, dtype=np.float32)
    sf.write(str(wav_path), np.stack([left, right], axis=1), 16000, subtype="PCM_16")

    module = FileInputModule()
    result = module.load_file(str(wav_path))

    assert result.success is True
    audio = result.audio_data
    assert audio.ndim == 1
    assert np.all(audio > 0.0), "only the first channel must survive downmixing"


def test_missing_file_and_unsupported_format_fail_with_contract_errors(tmp_path: Path):
    module = FileInputModule()

    missing = module.load_file(str(tmp_path / "absent.wav"))
    assert missing.success is False
    assert missing.error_code is FileError.FILE_NOT_FOUND

    bad_format = tmp_path / "audio.xyz"
    bad_format.write_bytes(b"nonsense")
    unsupported = module.load_file(str(bad_format))
    assert unsupported.success is False
    assert unsupported.error_code is FileError.UNSUPPORTED_FORMAT


def test_capture_contract_emits_float32_frames_within_unit_range():
    capture = AudioCaptureModule(AudioConfig(channels=1, chunk_size=512, buffer_seconds=1.0))

    pcm = np.array([32767, -32768, 0, 16384] * 512, dtype=np.int16)
    assert capture.feed_audio(pcm.tobytes(), channels=1) is True

    frames = [capture.get_audio_chunk(timeout=0.0) for _ in range(4)]
    assert all(frame is not None for frame in frames)
    for frame in frames:
        assert frame.dtype == np.float32
        assert frame.shape == (512,)
        assert np.all(np.abs(frame) <= 1.0)
    np.testing.assert_allclose(frames[0][:4], np.array([32767, -32768, 0, 16384]) / 32768.0)


def test_capture_preroll_retains_recent_frames_for_streaming_warmup():
    capture = AudioCaptureModule(AudioConfig(channels=1, chunk_size=512, buffer_seconds=1.0))
    capture.feed_audio((np.ones(512 * 3, dtype=np.int16) * 1000).tobytes(), channels=1)

    pre_roll = capture.get_pre_roll_audio(seconds=0.05)

    assert pre_roll.dtype == np.float32
    assert len(pre_roll) == int(16000 * 0.05)
    np.testing.assert_allclose(pre_roll, np.full_like(pre_roll, 1000 / 32768.0))
