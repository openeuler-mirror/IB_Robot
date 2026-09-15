from types import SimpleNamespace

import numpy as np

from voice_asr_service.audio_capture_module import AudioCaptureModule, AudioConfig
from voice_asr_service.voice_asr_node import VoiceASRNode


def test_shared_audio_accumulates_short_multichannel_blocks_into_exact_frames():
    capture = AudioCaptureModule(AudioConfig(channels=6, input_channel=1, chunk_size=4, buffer_seconds=1.0))
    first = np.array([[0, 100, 200, 300, 400, 500], [0, 200, 0, 0, 0, 0]], dtype=np.int16)
    second = np.array([[0, 300, 0, 0, 0, 0], [0, 400, 0, 0, 0, 0], [0, 500, 0, 0, 0, 0]], dtype=np.int16)

    assert capture.feed_audio(first.tobytes(), channels=6) is False
    assert capture.get_audio_chunk(timeout=0.0) is None
    assert capture.feed_audio(second.tobytes(), channels=6) is True

    chunk = capture.get_audio_chunk(timeout=0.0)
    np.testing.assert_allclose(chunk, np.array([100, 200, 300, 400], dtype=np.float32) / 32768.0)
    assert capture.get_audio_chunk(timeout=0.0) is None
    np.testing.assert_allclose(capture._pending_shared_audio, np.array([500], dtype=np.float32) / 32768.0)


def test_dual_channel_capture_emits_paired_vad_asr_chunks():
    capture = AudioCaptureModule(
        AudioConfig(
            channels=6,
            input_channel=1,
            vad_input_channel=0,
            chunk_size=4,
            buffer_seconds=1.0,
        )
    )
    block = np.array(
        [
            [10, 100, 0, 0, 0, 0],
            [20, 200, 0, 0, 0, 0],
            [30, 300, 0, 0, 0, 0],
            [40, 400, 0, 0, 0, 0],
            [50, 500, 0, 0, 0, 0],
        ],
        dtype=np.int16,
    )

    assert capture.feed_audio(block.tobytes(), channels=6) is True

    chunk = capture.get_audio_chunk(timeout=0.0)
    assert chunk is not None and chunk.ndim == 2 and chunk.shape == (4, 2)
    np.testing.assert_allclose(chunk[:, 0], np.array([10, 20, 30, 40], dtype=np.float32) / 32768.0)
    np.testing.assert_allclose(chunk[:, 1], np.array([100, 200, 300, 400], dtype=np.float32) / 32768.0)
    assert capture.get_audio_chunk(timeout=0.0) is None
    np.testing.assert_allclose(capture._pending_shared_audio, np.array([500], dtype=np.float32) / 32768.0)
    np.testing.assert_allclose(capture._pending_vad_audio, np.array([50], dtype=np.float32) / 32768.0)

    # pre-roll（ASR 通道）只含 ASR 列
    capture.feed_audio(block.tobytes(), channels=6)
    pre_roll = capture.get_pre_roll_audio(seconds=1.0)
    assert pre_roll.ndim == 1


def test_voice_asr_callback_consumes_audio_data_stamped_payload():
    class FakeCapture:
        def __init__(self):
            self.calls = []

        def feed_audio(self, data, channels):
            self.calls.append((data, channels))
            return True

    node = VoiceASRNode.__new__(VoiceASRNode)
    node._audio_capture = FakeCapture()
    node._audio_channels = 6
    node._last_audio_chunk_time = 0.0
    message = SimpleNamespace(audio=SimpleNamespace(data=[1, 2, 3]))

    node._on_audio_message(message)

    assert node._audio_capture.calls == [(b"\x01\x02\x03", 6)]
    assert node._last_audio_chunk_time > 0.0
