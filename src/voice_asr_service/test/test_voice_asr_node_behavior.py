"""ROS behavior regression tests for VoiceASRNode service and topic handling.

These tests use the ``__new__`` construction pattern (see
``test_shared_audio_capture.py``) so service handlers, topic publishing, and
reset semantics can be exercised without spinning up rclpy. Model modules are
faked at the instance level; the node logic under test is the mapping between
runtime outcomes and the existing ROS responses/topics.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ibrobot_msgs.srv import RecognizeFile, SetHotwords  # noqa: E402
from voice_asr_service.asr_inference_module import ASRResult  # noqa: E402
from voice_asr_service.state_machine import ActiveMode, NodeState, StateMachine  # noqa: E402
from voice_asr_service.vad_module import VADState  # noqa: E402
from voice_asr_service.voice_asr_node import VoiceASRNode  # noqa: E402

_LOGGER = logging.getLogger("voice-asr-node-test")


class FakeASR:
    def __init__(self, *, streaming: bool = True, ready: bool = True):
        self.streaming = streaming
        self.ready = ready
        self.hotwords = None
        self.reset_calls = 0
        self.started = False
        self.waveforms: list[np.ndarray] = []
        self.partial = ASRResult(text="你", is_final=False)
        self.final = ASRResult(text="你好世界", is_final=True, confidence=0.9)
        self.file_results = [
            ASRResult(text="第一段", is_final=True, start_time=0.0, duration=1.5),
            ASRResult(text="第二段", is_final=True, start_time=2.0, duration=1.0),
        ]
        self.raise_on_hotwords: Exception | None = None

    @property
    def is_ready(self) -> bool:
        return self.ready

    def is_streaming(self) -> bool:
        return self.streaming

    def start_streaming(self) -> bool:
        self.started = True
        return True

    def accept_waveform(self, audio_data):
        self.waveforms.append(np.asarray(audio_data))
        return self.partial

    def end_streaming(self) -> ASRResult:
        self.started = False
        return self.final

    def recognize_file(self, audio_data, enable_vad, vad_module):
        return list(self.file_results)

    def set_hotwords(self, hotwords):
        if self.raise_on_hotwords is not None:
            raise self.raise_on_hotwords
        self.hotwords = dict(hotwords)

    def reset(self):
        self.reset_calls += 1


class FakeVAD:
    def __init__(self, state: VADState):
        self.state = state
        self.reset_calls = 0

    def process(self, audio_frame):
        from voice_asr_service.vad_module import VADResult

        return VADResult(is_speech=self.state is not VADState.SILENCE, state=self.state, confidence=0.9, energy=0.01)

    def reset(self):
        self.reset_calls += 1


class FakeVADRuntime:
    def __init__(self, sample_rate_hz: int = 16000, frame_size: int = 512):
        self._sample_rate_hz = sample_rate_hz
        self._frame_size = frame_size
        self.closed = False

    @property
    def sample_rate_hz(self) -> int:
        return self._sample_rate_hz

    @property
    def frame_size(self) -> int:
        return self._frame_size

    def infer(self, audio_frame):
        return 0.9

    def reset(self):
        return None

    def close(self):
        self.closed = True


class FakeCapture:
    def __init__(self):
        self.chunks: list[np.ndarray] = []
        self.started = False
        self.cleared = 0

    def get_audio_chunk(self, timeout=0.1):
        if self.chunks:
            return self.chunks.pop(0)
        return None

    def get_pre_roll_audio(self, seconds=0.3):
        return np.zeros(0, dtype=np.float32)

    def start_capture(self) -> bool:
        self.started = True
        return True

    def initialize(self) -> bool:
        return True

    def stop_capture(self):
        self.started = False

    def clear_buffer(self):
        self.cleared += 1


class FakeFileInput:
    def __init__(self, success: bool = True):
        self.success = success

    def load_file(self, file_path):
        if not self.success:
            from voice_asr_service.file_input_module import FileResult

            return FileResult(success=False, error_message=f"cannot read {file_path}")
        return SimpleNamespace(success=True, audio_data=np.zeros(16000, dtype=np.float32), sample_rate=16000)


def make_node(
    *,
    asr=None,
    vad_state=VADState.SILENCE,
    node_state=NodeState.LISTENING,
    mode=ActiveMode.CONTINUOUS,
    publish_partial=True,
    vad_runtime=None,
):
    node = VoiceASRNode.__new__(VoiceASRNode)
    node._asr = asr or FakeASR()
    node._pipeline_init_error = None
    node._asr_init_error = None
    node._state_machine = StateMachine()
    node._state_machine.set_mode(mode)
    node._state_machine._state = node_state
    node.get_logger = lambda: _LOGGER
    node._vad = FakeVAD(vad_state)
    node._vad_runtime = vad_runtime or FakeVADRuntime()
    node._audio_capture = FakeCapture()
    node._file_input = FakeFileInput()
    node._publish_partial = publish_partial
    node._last_partial_text = ""
    node._recording_start_time = None
    node._sample_rate = 16000
    node._chunk_size = 512
    node._vad_sensitivity = 0.6
    node._vad_bundle_path = "models/silero-vad"
    node._vad_deployment = "torch_cpu"
    node._realtime_pre_roll_seconds = 0.0
    node._last_audio_chunk_time = 0.0
    commands: list[tuple[str, float]] = []
    partials: list[str] = []

    def record_command(text, confidence=1.0):
        commands.append((text, confidence))

    def record_partial(text):
        partials.append(text)

    node._publish_command = record_command
    node._publish_partial_result = record_partial
    node.commands = commands
    node.partials = partials
    return node


def test_recognize_file_maps_results_timestamps_and_durations():
    node = make_node()
    created = []

    def fake_create_vad():
        runtime = FakeVADRuntime()
        created.append(runtime)
        return SimpleNamespace(close=lambda: setattr(runtime, "closed", True))

    node._create_vad_module = fake_create_vad

    request = RecognizeFile.Request()
    request.file_path = "/tmp/demo.wav"
    request.enable_vad = True
    response = RecognizeFile.Response()

    node._on_recognize_file(request, response)

    assert response.success is True
    assert response.error_message == ""
    assert list(response.results) == ["第一段", "第二段"]
    assert list(response.timestamps) == [0.0, 2.0]
    assert list(response.durations) == [1.5, 1.0]
    assert node.commands == [("第一段", 1.0), ("第二段", 1.0)]
    assert len(created) == 1
    assert created[0].closed is True, "per-request VAD runtime must be closed"


def test_recognize_file_fails_closed_when_pipeline_not_ready():
    node = make_node()
    node._pipeline_init_error = "bundle missing"

    request = RecognizeFile.Request()
    request.file_path = "/tmp/demo.wav"
    request.enable_vad = False
    response = RecognizeFile.Response()

    node._on_recognize_file(request, response)

    assert response.success is False
    assert "bundle missing" in response.error_message
    assert list(response.results) == []
    assert node.commands == []


def test_recognize_file_reports_file_load_failure():
    node = make_node()
    node._file_input = FakeFileInput(success=False)

    request = RecognizeFile.Request()
    request.file_path = "/tmp/missing.wav"
    request.enable_vad = False
    response = RecognizeFile.Response()

    node._on_recognize_file(request, response)

    assert response.success is False
    assert "cannot read" in response.error_message


def test_recognize_file_maps_recognition_exception_to_error_response():
    node = make_node()
    node._asr.file_results = None

    def raise_recognize(audio_data, enable_vad, vad_module):
        raise RuntimeError("decoder exploded")

    node._asr.recognize_file = raise_recognize

    request = RecognizeFile.Request()
    request.file_path = "/tmp/demo.wav"
    request.enable_vad = False
    response = RecognizeFile.Response()

    node._on_recognize_file(request, response)

    assert response.success is False
    assert "decoder exploded" in response.error_message


def test_set_hotwords_delegates_to_asr_and_reports_success():
    node = make_node()
    request = SetHotwords.Request()
    request.hotwords = ["前进", "停止"]
    request.boost_scores = [2.0, 1.0]
    response = SetHotwords.Response()

    node._on_set_hotwords(request, response)

    assert response.success is True
    assert node._asr.hotwords == {"前进": 2.0, "停止": 1.0}


def test_set_hotwords_maps_exception_to_error_response():
    node = make_node()
    node._asr.raise_on_hotwords = RuntimeError("hotwords unsupported")
    request = SetHotwords.Request()
    request.hotwords = ["前进"]
    response = SetHotwords.Response()

    node._on_set_hotwords(request, response)

    assert response.success is False
    assert "hotwords unsupported" in response.error_message


def test_start_recognition_rejected_for_offline_model():
    node = make_node(asr=FakeASR(streaming=False), node_state=NodeState.IDLE)
    response = SimpleNamespace(success=True, error_message="")

    node._on_start_recognition(SimpleNamespace(), response)

    assert response.success is False
    assert "streaming" in response.error_message
    assert not node._audio_capture.started
    assert node._state_machine.state is NodeState.IDLE


def test_start_recognition_transitions_listening_for_streaming_model():
    node = make_node(asr=FakeASR(streaming=True), node_state=NodeState.IDLE)
    response = SimpleNamespace(success=True, error_message="")

    node._on_start_recognition(SimpleNamespace(), response)

    assert response.success is True
    assert node._audio_capture.started is True
    assert node._state_machine.state is NodeState.LISTENING


def test_stop_recognition_publishes_final_result_and_returns_to_idle():
    node = make_node(node_state=NodeState.RECOGNIZING)
    node._asr.started = True

    node._on_stop_recognition(SimpleNamespace(), SimpleNamespace())

    assert node.commands == [("你好世界", 0.9)]
    assert node._state_machine.state is NodeState.IDLE
    assert node._asr.started is False
    assert node._asr.reset_calls == 1
    assert node._vad.reset_calls == 1


def test_process_audio_publishes_partial_then_final_on_silence():
    node = make_node(vad_state=VADState.SPEAKING, node_state=NodeState.LISTENING)
    node._audio_capture.chunks.append(np.zeros(512, dtype=np.float32))

    node._process_audio()

    assert node._state_machine.state is NodeState.RECOGNIZING
    assert node._asr.started is True
    assert node.partials == ["你"]
    assert node.commands == []

    node._vad.state = VADState.SILENCE
    node._audio_capture.chunks.append(np.zeros(512, dtype=np.float32))
    node._process_audio()

    assert node.commands == [("你好世界", 0.9)]
    assert node._state_machine.state is NodeState.LISTENING
    assert node._asr.reset_calls == 1


def test_process_audio_stops_realtime_capture_for_offline_model_on_speech():
    node = make_node(asr=FakeASR(streaming=False), vad_state=VADState.SPEAKING, node_state=NodeState.LISTENING)
    node._audio_capture.chunks.append(np.zeros(512, dtype=np.float32))

    node._process_audio()

    assert node._state_machine.state is NodeState.IDLE
    assert not node._audio_capture.started
    assert node._asr.started is False
    assert node.partials == []


def test_reset_recognition_runtime_resets_vad_and_asr_and_clears_audio():
    node = make_node()
    node._last_partial_text = "旧文本"
    node._recording_start_time = 123.456

    node._reset_recognition_runtime(clear_audio=True)

    assert node._recording_start_time is None
    assert node._last_partial_text == ""
    assert node._asr.reset_calls == 1
    assert node._vad.reset_calls == 1
    assert node._audio_capture.cleared == 1


def test_vad_audio_contract_mismatch_fails_closed():
    node = make_node(vad_runtime=FakeVADRuntime(sample_rate_hz=8000))
    with pytest.raises(ValueError, match="sample_rate=8000"):
        node._validate_vad_audio_contract(node._vad_runtime)

    node = make_node(vad_runtime=FakeVADRuntime(frame_size=256))
    with pytest.raises(ValueError, match="frame_size=256"):
        node._validate_vad_audio_contract(node._vad_runtime)
