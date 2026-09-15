"""cumulative stateful FullSubNet + 时间制门控 + 4096/512 SRP 生产编排。"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .diagnostics import DiagnosticsPacket, FrameMetrics
from .pipeline import DoaState, HopResult, VadState
from .temporal_gate import TemporalSpeechGate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamingPipelineParams:
    """低延迟流式链路参数，时间门限统一保存为sample或毫秒。"""

    sample_rate: int = 16000
    processing_samples: int = 256
    model_batch_samples: int = 512
    srp_update_interval_hops: int = 2
    early_direction_min_scores: int = 2
    raw_activity_gate_enabled: bool = True
    raw_activity_start_rms: float = 0.026
    raw_activity_stop_rms: float = 0.020
    raw_activity_tail_samples: int = 8000
    input_channels: tuple[int, int, int, int] = (1, 2, 3, 4)
    srp_frame_samples: int = 4096
    srp_hop_samples: int = 512
    candidate_window_samples: int = 1024
    segment_end_gap_samples: int = 2400
    min_segment_samples: int = 1600
    min_accum_samples: int = 1536  # 3×512=96ms；真实短语音回归仍可稳定定位
    max_accum_samples: int = 16000
    segment_max_rms_threshold: float = 0.005


class StreamingSpeechDirectionPipeline:
    """512样本调度、内部256频谱推进的流式定位链路。"""

    def __init__(
        self,
        fullnet,
        silero_engine,
        srp,
        params: StreamingPipelineParams,
        vad_state: VadState,
        doa_state: DoaState,
        *,
        vad_threshold: float = 0.65,
        rms_threshold: float = 0.002,
        diagnostics=None,
        initialize_backend: bool = True,
    ):
        self.fullnet = fullnet
        self.srp = srp
        self.params = params
        self.vad_state = vad_state
        self.doa_state = doa_state
        self.diagnostics = diagnostics
        self.sr = params.sample_rate
        self.hop_size = params.processing_samples
        self.model_batch_samples = params.model_batch_samples
        if self.model_batch_samples != 2 * self.hop_size:
            raise ValueError("stateful T=2要求model_batch_samples恰好为2个processing hop")
        self.frame_size = params.srp_frame_samples
        self.input_channels = list(params.input_channels)
        self.gate = TemporalSpeechGate(
            silero_engine,
            vad_threshold=vad_threshold,
            rms_threshold=rms_threshold,
            candidate_window_samples=params.candidate_window_samples,
            exit_gap_samples=params.segment_end_gap_samples,
            initialize_backend=initialize_backend,
        )
        self._lock = threading.Lock()
        self._closed = False  # 关闭一旦开始即禁新推理（process/reset 见此即拒绝）
        self._cleanup_complete = False  # 清理已尝试完毕（含失败项），重入据此返回
        self._output_seq = 0
        self._segment_seq = 0
        self._history: list[tuple[int, int, float, str]] = []
        self._samples_processed = 0
        self._diagnostics_error_reported = False
        # 单次 model_batch 处理耗时样本(有界,避免长跑内存增长)。
        # 仅在真正执行 fullnet+vad+srp 的 tick(偶数 hop,累积满 512 样本)记录,
        # 早退 tick 不计入,故可直接用做"单次块处理时延"基线。reset 清空。
        self._block_latency_ms: deque[float] = deque(maxlen=8192)
        self._reset_temporal_context(reset_backend=initialize_backend)

    def _reset_temporal_context(self, *, reset_backend: bool = True) -> None:
        if reset_backend:
            self.fullnet.reset()
            self.gate.reset()
        self._srp_history = np.zeros((0, 4), np.float32)
        self._state = "IDLE"
        self._scores: list[np.ndarray] = []
        self._rms: list[float] = []
        self._segment_start = 0
        self._last_gray_end = 0
        self._candidate_score: np.ndarray | None = None
        self._candidate_rms = 0.0
        self._candidate_start = 0
        self._input_pending = np.zeros((0, 4), np.float32)
        self._pending_raw_start: int | None = None
        self._model_hop_count = 0
        self._early_direction_emitted = False
        self._raw_active = not self.params.raw_activity_gate_enabled
        self._raw_quiet_samples = 0
        self._backend_reset_requested = False

    def process_block(self, data: np.ndarray, *, capture_start_sample: int | None = None) -> HopResult:
        """处理一个[6,256] tick；两个tick合并后调用一次T=2 stateful 模型。"""
        if self._closed:
            raise RuntimeError("流式 speech_direction pipeline 已关闭")
        started = time.perf_counter()
        value = np.asarray(data, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != self.hop_size:
            raise ValueError(f"流式pipeline输入必须为[channels,{self.hop_size}]，得到{value.shape}")
        raw_start = self._samples_processed if capture_start_sample is None else int(capture_start_sample)
        mic4 = np.ascontiguousarray(value[np.asarray(self.input_channels), :].T)
        self._samples_processed = raw_start + self.hop_size
        if self._pending_raw_start is None:
            self._pending_raw_start = raw_start
        self._input_pending = np.concatenate([self._input_pending, mic4], axis=0)
        if self._input_pending.shape[0] < self.model_batch_samples:
            return HopResult(
                raw_start_sample=raw_start,
                enh_start_sample=None,
                session_sample=raw_start,
                hop_t=raw_start / self.sr,
            )
        model_input = self._input_pending[: self.model_batch_samples]
        model_raw_start = self._pending_raw_start
        self._input_pending = self._input_pending[self.model_batch_samples :]
        self._pending_raw_start = model_raw_start + self.model_batch_samples if self._input_pending.shape[0] else None

        raw_rms = float(np.sqrt(np.mean(model_input.astype(np.float64) ** 2)) + 1e-12)
        if self.params.raw_activity_gate_enabled and not self._raw_active:
            if raw_rms < self.params.raw_activity_start_rms:
                self.vad_state.update(
                    0.0,
                    False,
                    raw_rms,
                    False,
                    (model_raw_start + self.model_batch_samples) / self.sr,
                )
                return HopResult(
                    raw_start_sample=raw_start,
                    enh_start_sample=None,
                    session_sample=model_raw_start,
                    hop_t=model_raw_start / self.sr,
                )
            self._srp_history = np.zeros((0, 4), np.float32)
            self._state = "IDLE"
            self._scores = []
            self._rms = []
            self._candidate_score = None
            self._early_direction_emitted = False
            self._model_hop_count = 0
            self._raw_active = True
            self._raw_quiet_samples = 0
        elif self.params.raw_activity_gate_enabled:
            if raw_rms < self.params.raw_activity_stop_rms:
                self._raw_quiet_samples += self.model_batch_samples
            else:
                self._raw_quiet_samples = 0
        fullnet_start = time.perf_counter()
        enhanced_full = self.fullnet.process_4ch(model_input)
        fullnet_ms = (time.perf_counter() - fullnet_start) * 1000.0
        if enhanced_full is None:
            return HopResult(
                raw_start_sample=model_raw_start,
                enh_start_sample=None,
                session_sample=model_raw_start,
                hop_t=model_raw_start / self.sr,
            )
        self._model_hop_count += 1
        # T=2输出512样本，对应前一个完整512输入块；下游VAD和SRP也按512推进。
        enhanced = enhanced_full
        output_start = model_raw_start - self.model_batch_samples
        output_end = output_start + self.model_batch_samples
        enh_ch1 = enhanced[:, 0].copy()
        gate_start = time.perf_counter()
        decision = self.gate.process_frame(enh_ch1, frame_start_sample=output_start)
        vad_ms = (time.perf_counter() - gate_start) * 1000.0

        # 无论门控是否通过都维护增强4ch历史，避免后续4096窗口跨静音出现缺口。
        self._srp_history = np.concatenate([self._srp_history, enhanced], axis=0)
        self._srp_history = self._srp_history[-self.params.srp_frame_samples :]
        score = None
        frame_doa = None
        srp_ms = 0.0
        should_update_srp = (
            self._srp_history.shape[0] == self.params.srp_frame_samples
            and decision.is_gray
            and self._model_hop_count % self.params.srp_update_interval_hops == 0
        )
        if should_update_srp:
            srp_start = time.perf_counter()
            spectrum = self.srp.stft_4ch(self._srp_history)
            score = self.srp.compute_all_scores(spectrum)[0]
            frame_doa = int(self.srp.angles[int(np.argmax(score))])
            srp_ms = (time.perf_counter() - srp_start) * 1000.0

        previous_state = self._state
        self._state = decision.gate_state
        segment_output = self._update_segment(
            previous_state=previous_state,
            decision=decision,
            score=score,
        )
        if (
            self.params.raw_activity_gate_enabled
            and self._raw_active
            and self._raw_quiet_samples >= self.params.raw_activity_tail_samples
            and self._state == "IDLE"
        ):
            # The current block has completed segment finalization.  Reset
            # only after that boundary so no active utterance is truncated.
            self._srp_history = np.zeros((0, 4), np.float32)
            self._model_hop_count = 0
            self._raw_active = False
            self._raw_quiet_samples = 0
            self._backend_reset_requested = True
            self.vad_state.update(
                0.0,
                False,
                raw_rms,
                False,
                output_end / self.sr,
            )
        self.vad_state.update(
            decision.vad_prob,
            decision.is_speech,
            decision.rms,
            decision.is_gray,
            output_end / self.sr,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        other_ms = max(elapsed_ms - fullnet_ms - vad_ms - srp_ms, 0.0)
        # 记录本块处理耗时,供运行时观测与离线回归断言时延预算(32ms/block)。
        # 这是生产路径真实耗时,非测试桩;早退 tick 不进此处。
        self._block_latency_ms.append(elapsed_ms)
        metrics = FrameMetrics(
            session_sample=output_start,
            sample_count=self.model_batch_samples,
            segment_seq=self._segment_seq,
            vad_probability=decision.vad_prob,
            is_speech=decision.is_speech,
            is_gray=decision.is_gray,
            rms=decision.rms,
            frame_doa_degree=frame_doa,
            segment_doa_degree=segment_output,
            inference_elapsed_ms=elapsed_ms,
            fullsubnet_elapsed_ms=fullnet_ms,
            silero_vad_elapsed_ms=vad_ms,
            srp_elapsed_ms=srp_ms,
            other_elapsed_ms=other_ms,
            processing_tick_samples=self.hop_size,
            model_batch_samples=self.model_batch_samples,
            vad_frame_samples=512,
            srp_hop_samples=self.params.srp_hop_samples,
            stage_executed=True,
        )
        self._enqueue_diagnostics(
            raw_start_sample=raw_start,
            enh_start_sample=output_start,
            enhanced=enhanced,
            metrics=metrics,
        )
        logger.debug(
            "[StreamingPipeline] sample=%d state=%s vad=%.3f rms=%.5f "
            "doa=%s fullnet=%.2fms vad=%.2fms srp=%.2fms total=%.2fms",
            output_start,
            decision.gate_state,
            decision.vad_prob,
            decision.rms,
            frame_doa,
            fullnet_ms,
            vad_ms,
            srp_ms,
            elapsed_ms,
        )
        return HopResult(
            enh4=enhanced,
            enh_ch1=enh_ch1,
            is_gray_hop=decision.is_gray,
            scores_frame=score,
            frame_doa=frame_doa,
            seg_output=segment_output,
            seg_seq=self._segment_seq,
            raw_start_sample=raw_start,
            enh_start_sample=output_start,
            session_sample=output_start,
            hop_t=output_start / self.sr,
        )

    def _enqueue_diagnostics(
        self, *, raw_start_sample: int, enh_start_sample: int, enhanced: np.ndarray, metrics: FrameMetrics
    ) -> None:
        """维测写盘完全旁路实时主路径，入队异常只停用维测。"""
        if self.diagnostics is None or self._diagnostics_error_reported:
            return
        try:
            self.diagnostics.enqueue(
                DiagnosticsPacket(
                    raw_start_sample=raw_start_sample,
                    raw6ch=None,
                    enh_start_sample=enh_start_sample,
                    enh4ch=enhanced.copy(),
                    metrics=metrics,
                )
            )
        except Exception:
            self._diagnostics_error_reported = True
            logger.exception("流式 pipeline 维测入队异常，已停用维测旁路")

    def _append_score(self, score: np.ndarray | None, rms: float) -> None:
        if score is not None:
            self._scores.append(np.asarray(score, dtype=np.float32))
            self._rms.append(float(rms))

    def _update_segment(self, *, previous_state: str, decision, score) -> int | None:
        """把 CANDIDATE/ACTIVE 门控映射到段级RMS加权累计。"""
        if decision.gate_state == "CANDIDATE":
            self._candidate_score = None if score is None else np.asarray(score).copy()
            self._candidate_rms = decision.rms
            self._candidate_start = decision.frame_start_sample
            return None

        if previous_state == "CANDIDATE" and decision.gate_state == "ACTIVE":
            self._segment_seq += 1
            self._segment_start = self._candidate_start
            self._last_gray_end = decision.frame_end_sample
            self._scores = []
            self._rms = []
            self._append_score(self._candidate_score, self._candidate_rms)
            self._append_score(score, decision.rms)
            self._candidate_score = None
            self._early_direction_emitted = False
            return None

        if decision.gate_state == "ACTIVE" and decision.is_gray:
            self._last_gray_end = decision.frame_end_sample
            self._append_score(score, decision.rms)
            if (
                not self._early_direction_emitted
                and len(self._scores) >= self.params.early_direction_min_scores
                and self._rms
                and max(self._rms) >= self.params.segment_max_rms_threshold
            ):
                early_output = self._emit_segment("voice_begin", decision.frame_end_sample)
                self._early_direction_emitted = early_output is not None
                return early_output
            if self._last_gray_end - self._segment_start >= self.params.max_accum_samples:
                return self._emit_segment("mid_long_seg", decision.frame_end_sample)
            return None

        if previous_state == "ACTIVE" and decision.gate_state == "IDLE":
            segment_samples = self._last_gray_end - self._segment_start
            score_coverage_samples = self.params.srp_hop_samples * self.params.srp_update_interval_hops
            accumulated_samples = len(self._scores) * score_coverage_samples
            result = None
            if (
                segment_samples >= self.params.min_segment_samples
                and accumulated_samples >= self.params.min_accum_samples
                and self._rms
                and max(self._rms) >= self.params.segment_max_rms_threshold
            ):
                result = self._emit_segment("seg_end", decision.frame_end_sample)
            self._early_direction_emitted = False
            return result
        return None

    def _emit_segment(self, output_type: str, decision_sample: int) -> int | None:
        if not self._scores:
            return None
        scores = np.stack(self._scores)
        rms = np.maximum(np.asarray(self._rms, np.float32), 1e-8)
        weights = rms / (rms.max() + 1e-8)
        accumulated = (scores * weights[:, None]).sum(0) / (weights.sum() + 1e-8)
        angle = int(self.srp.angles[int(np.argmax(accumulated))])
        self._output_seq += 1
        hop_t = decision_sample / self.sr
        self._history.append((self._output_seq, angle, hop_t, output_type))
        self.doa_state.update(
            angle,
            time.time(),
            self._output_seq,
            meta={"type": output_type, "seg_seq": self._segment_seq},
        )
        if output_type == "mid_long_seg":
            self._scores = []
            self._rms = []
            self._segment_start = decision_sample
        return angle

    def reset_for_gap(self, *, next_capture_sample: int, dropped_samples: int) -> None:
        if dropped_samples < 0:
            raise ValueError("dropped_samples不能为负数")
        if self._closed:
            raise RuntimeError("流式 speech_direction pipeline 已关闭")
        self._reset_temporal_context()
        self._samples_processed = int(next_capture_sample)
        self._model_hop_count = 0

    def reset_for_activity(self, *, next_capture_sample: int) -> None:
        """Reset host and recurrent state at a completed quiet boundary.

        The caller invokes this only after the enclosing runtime step has
        returned, so resetting the stateful Session is no longer re-entrant.
        Unlike ``reset()``, the absolute capture timeline and output history
        remain continuous for diagnostics and consumer de-duplication.
        """

        if next_capture_sample < 0:
            raise ValueError("next_capture_sample不能为负数")
        if self._closed:
            raise RuntimeError("流式 speech_direction pipeline 已关闭")
        self._reset_temporal_context()
        self._samples_processed = int(next_capture_sample)
        self._model_hop_count = 0

    def consume_backend_reset_request(self) -> bool:
        """Return and clear a deferred recurrent-state reset request.

        Stateful backend resets must happen between runtime ``step`` calls;
        invoking a Session reset from inside ``process_block`` would mutate
        the same exclusive state bank while the enclosing step is active.
        """

        requested = self._backend_reset_requested
        self._backend_reset_requested = False
        return requested

    def reset(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._reset_temporal_context()
            self._samples_processed = 0
            self._model_hop_count = 0
            self._output_seq = 0
            self._segment_seq = 0
            self._history = []
            self._block_latency_ms.clear()
            self._backend_reset_requested = False

    def close(self, *, close_backends: bool = True) -> None:
        """best-effort terminal 关闭：尽力释放 enhancer 与 Silero，一个失败仍继续关闭另一个，末尾汇总异常。

        关闭一旦开始即禁新推理；重入只在清理已尝试完毕后才直接返回，
        因此失败后再次调用并不会"重新执行整个关闭"——属 best-effort terminal，非可重试。
        """
        with self._lock:
            if self._cleanup_complete:
                # 清理已尝试完毕（含失败项），重入直接返回，避免重复释放已成功资源。
                return
            self._closed = True  # 禁新推理；释放动作放在锁外，避免持锁阻塞推理线程的退出
        errors = []
        try:
            for component in (self.gate, self.fullnet):
                try:
                    if component is self.gate:
                        component.close(close_silero=close_backends)
                    elif close_backends:
                        component.close(close_executor=close_backends)
                    else:
                        close_host = getattr(component, "close_host", None)
                        if callable(close_host):
                            close_host()
                        else:
                            component.close(close_executor=False)
                except Exception as exc:
                    errors.append(str(exc) or type(exc).__name__)
        finally:
            # 整个遍历已尝试完毕（含部分失败项），重入据此返回。
            self._srp_history = np.zeros((0, 4), np.float32)
            self._scores = []
            self._rms = []
            self._input_pending = np.zeros((0, 4), np.float32)
            self._history = []
            self._block_latency_ms.clear()
            self._cleanup_complete = True
        if errors:
            raise RuntimeError("; ".join(errors))

    def get_segment_history(self) -> list[tuple[int, int, float, str]]:
        return list(self._history)

    def get_block_latency_ms(self) -> list[float]:
        """返回单次 model_batch 处理耗时快照(ms),reset 后清空。

        仅含真正执行 fullnet+vad+srp 的块(每 512 样本一次),早退 tick 不计入。
        供离线回归在真实生产路径(worker 线程→process_block)下断言时延预算。
        """
        return list(self._block_latency_ms)


__all__ = ["StreamingPipelineParams", "StreamingSpeechDirectionPipeline"]
