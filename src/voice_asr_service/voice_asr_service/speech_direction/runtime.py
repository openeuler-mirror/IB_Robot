"""Shared ROS audio buffering and worker lifecycle management.

设计要点:
    1. MultiChannelRingBuffer(多消费者):ROS 回调写入 6ch,worker 独立 reader。
    2. SpeechDirectionRuntime:管理 worker 线程、共享状态和优雅停止。
    3. feed_audio:audio_common 与离线测试共用同一数据入口。
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from inference_service.unified_runtime import (
    ExecutionContext,
    ExecutionContract,
    LifecycleState,
    ModelRequest,
    ModelRuntimeHandle,
    OwnedComponent,
    RuntimeAssembly,
)

from .streaming_runtime import SpeechDirectionStreamingRuntime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RingWriteResult:
    """一次 Ring 写入对应的完整采集区间和解码结果。"""

    start_sample: int
    end_sample: int
    samples: np.ndarray  # (channels, frames)，包含本次完整输入


@dataclass(frozen=True)
class RingReadResult:
    """一次 Ring 读取结果，采样位置使用原始采集绝对时轴。"""

    start_sample: int
    end_sample: int
    samples: np.ndarray  # (channels, frames)
    dropped_before: int


class MultiChannelRingBuffer:
    """多消费者环形缓冲区(采用绝对计数法)。

    线程安全地缓存采集到的多通道音频,供多个消费者各自独立按帧读取。
    写入接口接收 audio_common 的 interleaved int16 PCM bytes，
    内部解码为 float32 并存成 (channels, capacity) 的环形数组。

    关键设计(绝对计数法):write_pos / reader_pos 用**单调递增的绝对计数**,
    只在访问 _buf 时对 capacity 取模。这样:
      - available = write_pos - reader_pos(无需取模,天然正确)
      - 缓冲满时丢弃最旧,把落后 reader 推到 write_pos - capacity

    读策略:
        - "block"  :不足时阻塞等待,直到凑够 n 帧(实时处理线程)
        - "latest" :不足时返回当前全部可读帧(预取/刷新场景)
    """

    def __init__(self, capacity_frames: int, channels: int = 6, dtype=np.float32):
        """
        Args:
            capacity_frames: 缓冲容量(帧数,1 帧 = channels 个采样点)
            channels: 通道数
            dtype: 存储精度(float32 归一化到 [-1, 1])
        """
        self.channels = channels
        self.capacity = int(capacity_frames)
        self.dtype = dtype
        # 环形存储:(channels, capacity)
        self._buf = np.zeros((channels, self.capacity), dtype=dtype)
        self._write_pos = 0  # 已写入的绝对帧数(单调递增,访问 _buf 时 % capacity)
        self._reader_pos: dict[int, int] = {}
        self._reader_overwritten_frames: dict[int, int] = {}
        self._reader_pending_dropped: dict[int, int] = {}
        self._overwrite_events = 0
        self._overwritten_frames = 0
        self._next_reader_id = 0
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def write(self, data) -> RingWriteResult:
        """写入音频并返回完整输入对应的绝对采集区间。

        data: interleaved int16 bytes,或 (n_frames, channels)/(channels, n_frames) 数组。
        单次写入超过容量时，Ring 只保留尾部，但绝对采集时轴仍按完整输入推进。
        """
        with self._cond:
            decoded = self._decode(data)  # (n_frames, channels) float32
            input_n = decoded.shape[0]
            input_start = self._write_pos
            input_end = input_start + input_n
            result = RingWriteResult(
                start_sample=input_start,
                end_sample=input_end,
                samples=decoded.T.copy(),
            )
            if input_n == 0:
                return result

            # 绝对采集位置必须按原始输入长度推进；Ring 只保留尾部不能改变会话时轴。
            frames = decoded[-self.capacity :]
            stored_n = frames.shape[0]
            stored_start = input_end - stored_n
            start = stored_start % self.capacity
            end = start + stored_n
            if end <= self.capacity:
                # 不回绕:一次写入(frames.T 形状 (channels, stored_n))
                self._buf[:, start:end] = frames.T
            else:
                # 回绕:分两段写
                first = self.capacity - start
                self._buf[:, start:] = frames[:first].T
                second = stored_n - first
                self._buf[:, :second] = frames[first:].T

            self._write_pos = input_end

            # 容量溢出时显式累计每个 reader 被覆盖的帧数，禁止静默丢失。
            low = max(0, self._write_pos - self.capacity)
            for rid in list(self._reader_pos):
                old_pos = self._reader_pos[rid]
                if old_pos < low:
                    dropped = low - old_pos
                    self._reader_pos[rid] = low
                    self._reader_overwritten_frames[rid] += dropped
                    self._reader_pending_dropped[rid] += dropped
                    self._overwrite_events += 1
                    self._overwritten_frames += dropped

            self._cond.notify_all()
            return result

    def _decode(self, data) -> np.ndarray:
        """bytes/int16/float → (n_frames, channels) float32,范围 [-1, 1]。"""
        if isinstance(data, bytes | bytearray):
            arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            n_total = arr.shape[0]
            if n_total % self.channels != 0:
                # 不完整帧:丢弃尾部不足一帧的样本,避免 reshape 报错
                arr = arr[: (n_total // self.channels) * self.channels]
            return arr.reshape(-1, self.channels)  # (n_frames, channels)
        # 已是 numpy 数组(或可转换为数组)
        # int16 整数数组需要除 32768 归一化到 [-1, 1](与 bytes 路径一致);
        # float 数组假定已在 [-1, 1] 区间,不再重复归一化,避免改变幅度。
        arr_raw = np.asarray(data)
        if arr_raw.dtype == np.int16:
            arr = arr_raw.astype(np.float32) / 32768.0
        else:
            arr = arr_raw.astype(self.dtype)
        if arr.ndim == 1:
            # 单通道数据,扩展为 (n, 1)
            return arr.reshape(-1, 1)
        if arr.ndim == 2:
            # (channels, n_frames) → 转置为 (n_frames, channels)
            if arr.shape[0] == self.channels and arr.shape[1] != self.channels:
                return arr.T
            return arr
        raise ValueError(f"不支持的数据维度: {arr.shape}")

    def register(self, start_latest: bool = True) -> int:
        """注册一个 reader,返回 reader_id。

        Args:
            start_latest: True=从最新写入位置开始读(丢弃历史);
                       False=从最早可读位置开始读(尽量不丢)
        """
        with self._cond:
            rid = self._next_reader_id
            self._next_reader_id += 1
            if start_latest:
                self._reader_pos[rid] = self._write_pos
            else:
                self._reader_pos[rid] = max(0, self._write_pos - self.capacity)
            self._reader_overwritten_frames[rid] = 0
            self._reader_pending_dropped[rid] = 0
            return rid

    def read_with_position(
        self, reader_id: int, n_frames: int, read_mode: str = "block", timeout: float = 1.0
    ) -> RingReadResult | None:
        """读取数据并返回原始采集绝对位置；旧 read API 只返回 samples。"""
        deadline = time.monotonic() + timeout if read_mode == "block" else 0.0
        with self._cond:
            while True:
                rpos = self._reader_pos.get(reader_id, 0)
                avail = self._write_pos - rpos
                if avail >= n_frames:
                    return self._read_n_with_position(reader_id, n_frames)
                if read_mode == "latest":
                    return self._read_n_with_position(reader_id, avail) if avail > 0 else None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(timeout=remaining)

    def read(self, reader_id: int, n_frames: int, read_mode: str = "block", timeout: float = 1.0):
        """读取 n_frames 帧，保留旧 API 返回 ndarray 的兼容行为。"""
        result = self.read_with_position(reader_id, n_frames, read_mode, timeout)
        return None if result is None else result.samples

    def _read_n_with_position(self, reader_id: int, n: int) -> RingReadResult | None:
        """从 reader_pos 读取并返回区间；调用方必须持有 Condition 锁。"""
        if n <= 0:
            return None
        n = min(n, self.capacity)
        rpos = self._reader_pos.get(reader_id, 0)
        out = np.zeros((self.channels, n), dtype=self.dtype)
        start = rpos % self.capacity
        end = start + n
        if end <= self.capacity:
            out[:] = self._buf[:, start:end]
        else:
            first = self.capacity - start
            out[:, :first] = self._buf[:, start:]
            second = n - first
            out[:, first:] = self._buf[:, :second]
        self._reader_pos[reader_id] = rpos + n
        dropped = self._reader_pending_dropped.get(reader_id, 0)
        self._reader_pending_dropped[reader_id] = 0
        return RingReadResult(rpos, rpos + n, out, dropped)

    def reader_stats(self, reader_id: int) -> dict[str, int]:
        with self._lock:
            pos = self._reader_pos.get(reader_id, self._write_pos)
            return {
                "position": pos,
                "dropped_frames_total": self._reader_overwritten_frames.get(reader_id, 0),
                "available_frames": max(0, self._write_pos - pos),
            }

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "total_written_frames": self._write_pos,
                "capacity_frames": self.capacity,
                "overwrite_events": self._overwrite_events,
                "overwritten_frames": self._overwritten_frames,
            }

    def _read_n(self, reader_id: int, n: int) -> np.ndarray | None:
        """兼容旧内部调用，返回 (channels, n)。"""
        result = self._read_n_with_position(reader_id, n)
        return None if result is None else result.samples

    def total_written(self) -> int:
        """累计写入帧数(供诊断)。"""
        with self._lock:
            return self._write_pos


class SpeechDirectionRuntime:
    """speech_direction 运行时:ROS 音频 + worker 线程生命周期 + 共享状态。

    node.py 持有本类,通过 get_speech_direction() 轮询段级方向。
    """

    def __init__(
        self,
        config,
        pipeline,
        *,
        offline: bool = False,
        enable_capture: bool = True,
        on_fatal_error: Callable[[str], None] | None = None,
        model_runtime_handle: ModelRuntimeHandle | None = None,
    ):
        """
        Args:
            config: SpeechDirectionConfig
            pipeline: SpeechDirectionPipeline 实例
            offline: 是否为离线回归输入；实时输入始终来自 audio_common
            on_fatal_error: 不可恢复错误回调；每个 runtime 最多调用一次
            model_runtime_handle: optional unified-runtime owner.  When it is
                omitted, a compatibility handle is created around the pipeline.
        """
        self.config = config
        self.pipeline = pipeline
        self.enable_capture = bool(enable_capture)
        if model_runtime_handle is None:
            streaming_runtime = SpeechDirectionStreamingRuntime(pipeline, close_backends=True)
            model_runtime_handle = ModelRuntimeHandle(
                RuntimeAssembly(
                    runtime_executor=streaming_runtime,
                    streaming_runtime=streaming_runtime,
                    owned_components=(OwnedComponent(streaming_runtime, "speech_direction_streaming_runtime"),),
                    stateful=True,
                    resettable=True,
                    state_scope="stream",
                    state_bank_mode="runtime_exclusive",
                    max_open_streams=1,
                    execution_contract=ExecutionContract(
                        state_scope="stream",
                        execution_structure="direct",
                        cancellation_granularity="checkpoint",
                        state_bank_mode="runtime_exclusive",
                        max_open_streams=1,
                    ),
                    identity=("tensor_model", "speech_direction", "enhance_and_vad"),
                    runtime_id="speech-direction-compat",
                )
            )
        self._model_runtime_handle = model_runtime_handle
        self._stream_handle = None
        self.vad_state = pipeline.vad_state
        self.doa_state = pipeline.doa_state
        self.max_age_ms = config.speech_direction_max_age_ms
        self._on_fatal_error = on_fatal_error
        self._fatal_error_reported = False
        self._fatal_error_lock = threading.Lock()

        # 在线突发缓冲为 10 秒；离线 CUDA 回归允许 60 秒，避免测试调度抖动
        # 被误判成算法丢帧。两者仍使用同一个 RingBuffer/worker 主流程。
        self.ring_buffer = MultiChannelRingBuffer(
            capacity_frames=int(config.audio.sample_rate) * (60 if offline else 10),
            channels=int(config.audio.channels),
        )
        self._diagnostics = getattr(pipeline, "diagnostics", None)
        self._pipeline_gap_count = 0
        self._pipeline_gap_frames = 0

        self._running = False
        self._threads: list[threading.Thread] = []
        self._reader_id: int | None = None
        self._pipeline_closed = False
        self._handle_closed = False

    def _record_raw_chunk(self, result: RingWriteResult) -> None:
        """把完整 raw 块非阻塞投递到 diagnostics。"""
        if self._diagnostics is not None:
            self._diagnostics.enqueue_raw(
                start_sample=result.start_sample,
                samples=result.samples.T.copy(),
            )

    def _write_audio(self, data) -> RingWriteResult:
        """统一 ROS/WAV/测试输入的绝对采样位置和 raw 记录路径。"""
        result = self.ring_buffer.write(data)
        self._record_raw_chunk(result)
        return result

    def _report_fatal_error(self, reason: str) -> None:
        """线程安全地向节点报告首个不可恢复错误。"""
        with self._fatal_error_lock:
            if self._fatal_error_reported:
                return
            self._fatal_error_reported = True
        if self._on_fatal_error is not None:
            self._on_fatal_error(reason)

    def start(self) -> None:
        """启动 worker 线程。"""
        if self._running:
            return
        self._ensure_stream_runtime()
        self._running = True
        logger.info("SpeechDirection runtime 启动中...")

        if self.enable_capture:
            logger.info(
                "等待 audio_common PCM: channels=%d, sample_rate=%d",
                self.config.audio.channels,
                self.config.audio.sample_rate,
            )

        # 核心 worker 线程；启动失败时保留原始启动错误。
        try:
            self._start_thread(self._worker_loop, "SpeechDirectionWorker")
        except Exception as e:
            reason = f"SpeechDirectionWorker 启动失败: {e}"
            logger.error(reason, exc_info=True)
            self._running = False
            self._threads = []
            self._report_fatal_error(reason)

    def _ensure_stream_runtime(self) -> None:
        """Load the handle and open the one runtime-exclusive stream."""

        if self._handle_closed:
            raise RuntimeError("Speech Direction runtime handle is closed")
        state = self._model_runtime_handle.state
        if state is LifecycleState.CREATED:
            self._model_runtime_handle.load(ExecutionContext("speech-direction-load"))
        elif state is not LifecycleState.READY:
            raise RuntimeError(f"Speech Direction runtime handle is not ready: {state.value}")
        if self._stream_handle is None:
            self._stream_handle = self._model_runtime_handle.open_stream(ExecutionContext("speech-direction-open"))

    @property
    def runtime_handle(self) -> ModelRuntimeHandle:
        """Expose the lifecycle owner for diagnostics and controlled reset."""

        return self._model_runtime_handle

    @property
    def stream_handle(self):
        """Return the active unified stream identity, if the runtime is started."""

        return self._stream_handle

    def reset(self) -> None:
        """Reset the active stream without changing its stable stream ID."""

        if self._stream_handle is None:
            self.pipeline.reset()
            return
        self._model_runtime_handle.reset_stream(
            self._stream_handle,
            ExecutionContext("speech-direction-reset"),
        )

    def _start_thread(self, target: Callable, name: str) -> None:
        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    def _worker_loop(self):
        """阻塞主循环:每 hop 读 6ch → pipeline 处理。"""
        self._reader_id = self.ring_buffer.register(start_latest=True)
        hop_size = self.pipeline.hop_size
        logger.info(
            "SpeechDirectionWorker 启动: frame=%d hop=%d sr=%d",
            self.pipeline.frame_size,
            hop_size,
            self.pipeline.sr,
        )
        expected_sample: int | None = None
        try:
            while self._running:
                item = self.ring_buffer.read_with_position(self._reader_id, hop_size, read_mode="block", timeout=1.0)
                if item is None or item.samples.shape[1] == 0:
                    continue
                if expected_sample is not None and item.start_sample != expected_sample:
                    gap_frames = max(0, item.start_sample - expected_sample)
                    self._pipeline_gap_count += 1
                    self._pipeline_gap_frames += gap_frames
                    # Ring 覆盖后不能把不相邻音频继续拼进增强上下文和
                    # STFT overlap.  The next request carries the absolute
                    # capture position, so the adapter can restart its host
                    # state without exposing a second stream identity.
                    if self._stream_handle is None:
                        raise RuntimeError("Speech Direction stream is not open")
                    self._model_runtime_handle.reset_stream(
                        self._stream_handle,
                        ExecutionContext(f"speech-direction-gap-{item.start_sample}"),
                    )
                    logger.debug(
                        "RingBuffer 覆盖导致 pipeline 跳过 %d 帧(%.3fs)，从 sample=%d 冷启动",
                        gap_frames,
                        gap_frames / self.pipeline.sr,
                        item.start_sample,
                    )
                try:
                    if self._stream_handle is None:
                        raise RuntimeError("Speech Direction stream is not open")
                    self._model_runtime_handle.step(
                        self._stream_handle,
                        ModelRequest(
                            {
                                "audio": item.samples,
                                "capture_start_sample": item.start_sample,
                            },
                            {"request_id": f"speech-direction-{item.start_sample}"},
                        ),
                        ExecutionContext(f"speech-direction-step-{item.start_sample}"),
                    )
                    expected_sample = item.end_sample
                except Exception as e:
                    reason = f"pipeline 处理异常: {e}"
                    logger.error(reason, exc_info=True)
                    self._running = False
                    self._report_fatal_error(reason)
        except Exception as e:
            reason = f"SpeechDirectionWorker 异常退出: {e}"
            logger.error(reason, exc_info=True)
            self._report_fatal_error(reason)
        finally:
            # 只有 stop() 主动清除 running 才属于正常退出。
            if self._running:
                self._running = False
                self._report_fatal_error("SpeechDirectionWorker 非预期退出")
            logger.info("SpeechDirectionWorker 停止")

    def stop(self) -> None:
        """幂等停止线程并释放资源，即使单项清理失败也完成其余收尾。"""
        if not self._running and not self._threads and self._pipeline_closed and self._handle_closed:
            return
        logger.info("正在停止 SpeechDirection runtime...")
        self._running = False
        cleanup_error = None
        current_thread = threading.current_thread()
        remaining_threads: list[threading.Thread] = []
        worker_alive = False
        for thread in self._threads:
            if thread is current_thread:
                # fatal 回调可能从 worker 间接触发销毁，禁止线程 join 自身；
                # 当前线程本身即未退出的 worker，计入 worker_alive 并保留句柄，
                # 否则下方会误判"无存活 worker"而 close 与自身并发使用的 pipeline 资源。
                remaining_threads.append(thread)
                worker_alive = True
                continue
            try:
                thread.join(timeout=2.0)
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
                logger.warning("等待 SpeechDirection worker 退出失败", exc_info=True)
            if thread.is_alive():
                # 仍存活的 worker 必须保留句柄：否则二次 stop() 看不到它，
                # 会误判已全部退出而 close pipeline，与该 worker 的 NPU 推理竞态。
                remaining_threads.append(thread)
                worker_alive = True
        # 只移除已确认退出的线程；仍存活 worker（含当前线程本身）保留在 self._threads，
        # 二次 stop() 才能再次 join 它，直到全部退出后才允许 close。
        self._threads = remaining_threads
        if self._diagnostics is not None:
            self._diagnostics.update_capture_stats(self.stats())
        if worker_alive:
            # 仍有 worker（含当前线程本身）可能在使用 pipeline，绝不能 close：
            # close 会销毁 ACL buffer/dataset，与 worker 的 execute_bank 竞态导致
            # use-after-free。保留资源供后续 stop() 重试，直到 worker 真正退出。
            if cleanup_error is None:
                cleanup_error = RuntimeError("SpeechDirection worker 未全部退出，pipeline 资源保留供后续 stop() 重试")
            logger.warning("SpeechDirection worker 仍未退出，跳过 pipeline.close 避免竞态")
        else:
            stream_handle = self._stream_handle
            if stream_handle is not None:
                try:
                    self._model_runtime_handle.close_stream(
                        stream_handle,
                        ExecutionContext("speech-direction-close-stream"),
                    )
                except Exception as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
                    logger.warning("关闭 speech_direction stream 失败", exc_info=True)
                finally:
                    self._stream_handle = None
            try:
                self._model_runtime_handle.close()
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
                logger.warning("关闭 speech_direction runtime handle 失败", exc_info=True)
            finally:
                # Handle close is terminal even when one owned component
                # reports a cleanup error; subsequent stop() calls are
                # idempotent and cannot close Sessions a second time.
                self._handle_closed = True
                self._pipeline_closed = True
        logger.info("SpeechDirection runtime 已停止")
        if cleanup_error is not None:
            raise cleanup_error

    def feed_audio(self, data) -> None:
        """离线模式外部灌入 6ch 音频(无采集设备时使用)。

        data: audio_common 的 6ch interleaved int16 PCM bytes，
              也接受离线测试使用的 (n, 6) int16/float 数组。
        """
        if not self._running:
            raise RuntimeError("runtime 未启动,请先调用 start()")
        self._write_audio(data)

    def stats(self) -> dict[str, int]:
        """汇总采集、Ring 覆盖和 pipeline gap 统计。"""
        ring = self.ring_buffer.stats()
        frames_captured = ring["total_written_frames"]
        return {
            "frames_captured": frames_captured,
            "ring_capacity_frames": ring["capacity_frames"],
            "ring_overwrite_events": ring["overwrite_events"],
            "ring_overwritten_frames": ring["overwritten_frames"],
            "pipeline_gap_count": self._pipeline_gap_count,
            "pipeline_gap_frames": self._pipeline_gap_frames,
        }

    def get_speech_direction(self, max_age_ms: int | None = None):
        """当前讲话方向:有段级角度且未过期 → 返回 angle,否则 None。

        语义(段级 DOA 透传):
          - 段级 DOA 输出 = 已过完整灰区验证 = 已验证可信方向,直接透传,不再二次过滤
          - angle 非 None 且 age_ms < max_age → 返回 angle
          - 其他(冷启动/方向过期)→ angle=None

        返回字段:
          - wall_clock_ts: 来源墙钟时间戳(doa_state._wall_clock_ts, time.time 域),
            用于 age_ms 计算;与 ROS 时钟(get_clock().now)不同域,见 node.py 注释
          - age_ms: now - ts,方向真实年龄
          - type: 方向类型(mid_long_seg/seg_end),供 node 按类型构造 stamp
        """
        if max_age_ms is None:
            max_age_ms = self.max_age_ms

        full = self.doa_state.get_full()
        angle = full.get("angle")
        ts = full.get("wall_clock_ts", 0.0)
        seq_id = full.get("seq_id", 0)
        meta = full.get("meta") or {}
        # 方向类型(mid_long_seg/seg_end),供 node 按类型构造 stamp:
        # 长语音中间方向用发布时刻(age≈0),段末方向用来源真实 age
        direction_type = meta.get("type")
        now = time.time()
        age_ms = max(0.0, (now - ts) * 1000.0)

        out_angle = None
        if angle is not None and age_ms < max_age_ms:
            out_angle = float(angle)

        _, is_speech = self.vad_state.get()
        return {
            "wall_clock_ts": ts,
            "angle": out_angle,
            "is_speech": bool(is_speech),
            "age_ms": age_ms,
            "seq_id": int(seq_id),
            "type": direction_type,
        }


__all__ = [
    "MultiChannelRingBuffer",
    "SpeechDirectionRuntime",
    "SpeechDirectionStreamingRuntime",
]
