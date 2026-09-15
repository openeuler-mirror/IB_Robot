"""Tests for the streaming speech-direction compute schedule."""

from __future__ import annotations

import numpy as np

from voice_asr_service.speech_direction.pipeline import DoaState, VadState
from voice_asr_service.speech_direction.pipeline_streaming import (
    StreamingPipelineParams,
    StreamingSpeechDirectionPipeline,
)


class _FakeFullSubNet:
    def __init__(self) -> None:
        self.calls = 0

    def process_4ch(self, value):
        self.calls += 1
        return np.asarray(value, dtype=np.float32)

    def reset(self) -> None:
        pass


class _FakeSilero:
    def __init__(self) -> None:
        self.calls = 0

    def inference(self, value):
        self.calls += 1
        return 1.0

    def reset_state(self) -> None:
        pass


class _FakeSrp:
    def __init__(self) -> None:
        self.angles = np.asarray([90.0, 120.0], dtype=np.float32)
        self.stft_calls = 0
        self.score_calls = 0

    def stft_4ch(self, value):
        self.stft_calls += 1
        return value

    def compute_all_scores(self, value):
        self.score_calls += 1
        return np.asarray([[0.0, 1.0]], dtype=np.float32)


class _SpeechSilero:
    def __init__(self) -> None:
        self.calls = 0

    def inference(self, value):
        self.calls += 1
        return 1.0

    def reset_state(self) -> None:
        pass


class _EnergySilero:
    def __init__(self) -> None:
        self.calls = 0

    def inference(self, value):
        self.calls += 1
        return 1.0 if np.sqrt(np.mean(np.asarray(value, dtype=np.float32) ** 2)) > 0.1 else 0.0

    def reset_state(self) -> None:
        pass


def test_voice_begin_emits_after_two_direction_scores() -> None:
    fullnet = _FakeFullSubNet()
    srp = _FakeSrp()
    params = StreamingPipelineParams(
        processing_samples=256,
        model_batch_samples=512,
        srp_frame_samples=1024,
        srp_hop_samples=512,
        srp_update_interval_hops=1,
        early_direction_min_scores=2,
        min_segment_samples=512,
        min_accum_samples=512,
        max_accum_samples=16000,
        segment_max_rms_threshold=0.0,
    )
    pipeline = StreamingSpeechDirectionPipeline(
        fullnet,
        _EnergySilero(),
        srp,
        params,
        VadState(),
        DoaState(),
        vad_threshold=0.5,
        rms_threshold=0.0,
    )

    block = np.ones((6, 256), dtype=np.float32)
    outputs = [pipeline.process_block(block) for _ in range(8)]

    assert any(item.seg_output == 120 and item.seg_seq == 1 for item in outputs)
    assert pipeline.get_segment_history()[0][3] == "voice_begin"


def test_voice_begin_does_not_prevent_next_segment_output() -> None:
    fullnet = _FakeFullSubNet()
    srp = _FakeSrp()
    params = StreamingPipelineParams(
        processing_samples=256,
        model_batch_samples=512,
        srp_frame_samples=1024,
        srp_hop_samples=512,
        srp_update_interval_hops=1,
        early_direction_min_scores=2,
        segment_end_gap_samples=1536,
        min_segment_samples=512,
        min_accum_samples=512,
        max_accum_samples=16000,
        segment_max_rms_threshold=0.0,
    )
    pipeline = StreamingSpeechDirectionPipeline(
        fullnet,
        _EnergySilero(),
        srp,
        params,
        VadState(),
        DoaState(),
        vad_threshold=0.5,
        rms_threshold=0.0,
    )

    active = np.ones((6, 256), dtype=np.float32)
    silent = np.zeros((6, 256), dtype=np.float32)
    outputs = []
    outputs.extend(pipeline.process_block(active) for _ in range(8))
    outputs.extend(pipeline.process_block(silent) for _ in range(12))
    outputs.extend(pipeline.process_block(active) for _ in range(8))

    history = pipeline.get_segment_history()
    assert [item[3] for item in history] == ["voice_begin", "seg_end", "voice_begin"]
    assert [item[0] for item in history] == [1, 2, 3]


def test_repeated_voice_begin_and_segment_end_outputs_keep_incrementing_sequence() -> None:
    fullnet = _FakeFullSubNet()
    srp = _FakeSrp()
    params = StreamingPipelineParams(
        processing_samples=256,
        model_batch_samples=512,
        srp_frame_samples=1024,
        srp_hop_samples=512,
        srp_update_interval_hops=1,
        early_direction_min_scores=2,
        segment_end_gap_samples=1536,
        min_segment_samples=512,
        min_accum_samples=512,
        max_accum_samples=16000,
        segment_max_rms_threshold=0.0,
    )
    pipeline = StreamingSpeechDirectionPipeline(
        fullnet,
        _EnergySilero(),
        srp,
        params,
        VadState(),
        DoaState(),
        vad_threshold=0.5,
        rms_threshold=0.0,
    )

    active = np.ones((6, 256), dtype=np.float32)
    silent = np.zeros((6, 256), dtype=np.float32)
    for _ in range(8):
        pipeline.process_block(active)
    for _ in range(12):
        pipeline.process_block(silent)
    for _ in range(8):
        pipeline.process_block(active)

    history = pipeline.get_segment_history()
    assert [item[0] for item in history] == [1, 2, 3]
    assert [item[3] for item in history] == ["voice_begin", "seg_end", "voice_begin"]


def test_voice_begin_preserves_scores_for_segment_end_output() -> None:
    fullnet = _FakeFullSubNet()
    srp = _FakeSrp()
    params = StreamingPipelineParams(
        processing_samples=256,
        model_batch_samples=512,
        srp_frame_samples=1024,
        srp_hop_samples=512,
        srp_update_interval_hops=1,
        early_direction_min_scores=2,
        segment_end_gap_samples=512,
        min_segment_samples=512,
        min_accum_samples=512,
        max_accum_samples=16000,
        segment_max_rms_threshold=0.0,
    )
    pipeline = StreamingSpeechDirectionPipeline(
        fullnet,
        _SpeechSilero(),
        srp,
        params,
        VadState(),
        DoaState(),
        vad_threshold=0.5,
        rms_threshold=0.0,
    )

    active = np.ones((6, 256), dtype=np.float32)
    silent = np.zeros((6, 256), dtype=np.float32)
    outputs = [pipeline.process_block(active) for _ in range(8)]
    outputs.extend(pipeline.process_block(silent) for _ in range(18))

    history = pipeline.get_segment_history()
    assert [item[3] for item in history] == ["voice_begin"]
    assert all(item.seg_seq == 1 for item in outputs if item.seg_output is not None)


def test_raw_activity_gate_skips_silence_and_resets_before_next_utterance() -> None:
    fullnet = _FakeFullSubNet()
    srp = _FakeSrp()
    params = StreamingPipelineParams(
        processing_samples=256,
        model_batch_samples=512,
        srp_frame_samples=1024,
        srp_hop_samples=512,
        srp_update_interval_hops=1,
        raw_activity_gate_enabled=True,
        raw_activity_start_rms=0.01,
        raw_activity_stop_rms=0.01,
        raw_activity_tail_samples=512,
        min_segment_samples=512,
        min_accum_samples=512,
        segment_max_rms_threshold=0.0,
    )
    pipeline = StreamingSpeechDirectionPipeline(
        fullnet,
        _EnergySilero(),
        srp,
        params,
        VadState(),
        DoaState(),
        vad_threshold=0.5,
        rms_threshold=0.0,
    )

    silent = np.zeros((6, 256), dtype=np.float32)
    active = np.ones((6, 256), dtype=np.float32)
    for _ in range(4):
        pipeline.process_block(silent)
    assert fullnet.calls == 0

    pipeline.process_block(active)
    pipeline.process_block(active)
    assert fullnet.calls == 1

    for _ in range(6):
        pipeline.process_block(silent)
    assert fullnet.calls == 4

    for _ in range(2):
        pipeline.process_block(silent)
    assert fullnet.calls == 4


def test_64ms_srp_interval_keeps_enhancement_and_vad_continuous() -> None:
    fullnet = _FakeFullSubNet()
    silero = _FakeSilero()
    srp = _FakeSrp()
    params = StreamingPipelineParams(
        processing_samples=256,
        model_batch_samples=512,
        srp_frame_samples=4096,
        srp_hop_samples=512,
        srp_update_interval_hops=2,
        min_segment_samples=512,
        min_accum_samples=512,
        max_accum_samples=16000,
        segment_max_rms_threshold=0.0,
    )
    pipeline = StreamingSpeechDirectionPipeline(
        fullnet,
        silero,
        srp,
        params,
        VadState(),
        DoaState(),
        vad_threshold=0.5,
        rms_threshold=0.0,
    )

    block = np.ones((6, 256), dtype=np.float32)
    for _ in range(24):
        pipeline.process_block(block)

    assert fullnet.calls == 12
    assert silero.calls == 12
    assert srp.stft_calls == 3
    assert srp.score_calls == 3
    assert pipeline._srp_history.shape == (4096, 4)
