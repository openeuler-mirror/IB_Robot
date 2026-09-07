"""人声门控:Silero VAD + RMS 灰区段判定。

设计要点(灰区切段逻辑与算法基线一致):
    1. 每个 hop(2048 样本)的增强 ch1 分 4 个 512 子帧,逐子帧算 Silero VAD 概率与 RMS。
    2. 灰区判据:逐子帧 VAD>vad_threshold(0.65) 且 RMS>rms_threshold(0.002)。
    3. hop 灰区 = 4 子帧 ≥1 灰区(决定是否算 SRP,省算力)。
    4. Top-2 VAD 概率:挡单帧瞬态脉冲(占 1 个 32ms 子帧高概率,Top-2 压回低值),
       保 64ms 短口令(占 ≥2 子帧,Top-2 正常拉高)。

SileroVadEngine 内联在此(speech_direction 独有的推理实例,不与 ASR 共用——
ASR 有自己的 vad_module.py,LSTM state 不共享因输入不同)。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Silero VAD 标准帧长:512 samples = 32ms @ 16kHz
SILERO_FRAME_SIZE = 512
SILERO_SAMPLE_RATE = 16000
# Silero LSTM state shape:[2, batch, 128]（官方各版本 ONNX 接口一致）
_SILERO_VAD_V5_STATE_SHAPE = (2, 1, 128)
# context 样本数(@16kHz=64)
SILERO_CONTEXT_SIZE_16K = 64


class SileroVadEngine:
    """Silero VAD ONNX 推理器,隐藏状态跨帧传递。

    纯推理类,不含端点状态机(端点状态机由 pipeline 的段级触发实现)。
    线程安全性:每个实例的 state/context 由持有者保证单线程访问(或自行加锁)。
    """

    def __init__(
        self,
        model_path: str,
        sample_rate: int = SILERO_SAMPLE_RATE,
        backend: str = "ascend",
        inference_runner=None,
    ):
        """
        Args:
            model_path: 模型路径(.om 走 NPU,.onnx 走 CPU)
            sample_rate: 输入采样率(默认 16000)
            backend: 推理后端 "ascend"(Ascend NPU,默认) 或
                     "onnx"(CPU,onnxruntime,Ubuntu 回归基线)
            inference_runner: 可选,注入已构造好的推理器(需实现 infer/reset/close,
                         如 manifest/RuntimeContext 驱动的 SpeechDirectionRoleRunner)。
                         提供时复用其已加载的模型会话,不再另建推理器,
                         避免同一模型被重复加载。
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Silero VAD 模型不存在: {model_path}")

        self.model_path = model_path
        self.sample_rate = sample_rate
        self.frame_size = SILERO_FRAME_SIZE
        self.backend = backend
        self._inference_runner = None
        self._closed = False
        self._backend_closed = False

        if self.backend == "onnx" and inference_runner is not None:
            # Reuse the manifest-backed ONNX session supplied by the node.
            self._inference_runner = inference_runner
            self._sess = None
            self._input_names = []
            self._output_names = []
            self._audio_in = self._state_in = self._sr_in = None
            self._out_name = self._state_out = None
            self._state = self._zero_state()
            self._context_size = 64 if sample_rate == 16000 else 32
            self._context = np.zeros(self._context_size, dtype=np.float32)
            logger.info("Silero VAD loaded through manifest runner: sr=%d, backend=%s", sample_rate, backend)
            return

        if self.backend == "ascend":
            if inference_runner is not None:
                # 复用调用方已构造好的推理器(如 manifest 驱动的 SpeechDirectionRoleRunner)，
                # 仅借用本类的 context 拼接逻辑，不重复加载模型。
                self._inference_runner = inference_runner
                self._sess = None
                self._input_names = []
                self._output_names = []
                self._audio_in = self._state_in = self._sr_in = None
                self._out_name = self._state_out = None
                self._state = self._zero_state()
                self._context_size = 64 if sample_rate == 16000 else 32
                self._context = np.zeros(self._context_size, dtype=np.float32)
                logger.info("Silero VAD 已加载(复用外部 Ascend ACL 推理器): sr=%d, backend=%s", sample_rate, backend)
                return
            # 310P 生产路径直接使用 Ascend ACL，Silero 的 LSTM state 保留在 Device。
            from .silero_acl import SileroVadAclRunner

            self._inference_runner = SileroVadAclRunner(model_path)
            self._sess = None
            self._input_names = []
            self._output_names = []
            self._audio_in = self._state_in = self._sr_in = None
            self._out_name = self._state_out = None
        elif backend == "onnx":
            # CPU 基线:onnxruntime,用于 Ubuntu CUDA/CPU 回归对照
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ImportError("未安装 onnxruntime。请 `pip install onnxruntime` 后再使用 Silero VAD。") from exc
            self._sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self._input_names = [i.name for i in self._sess.get_inputs()]
            self._output_names = [o.name for o in self._sess.get_outputs()]
            self._audio_in = "input" if "input" in self._input_names else self._input_names[0]
            self._state_in = "state" if "state" in self._input_names else None
            self._sr_in = "sr" if "sr" in self._input_names else None
            self._out_name = "output" if "output" in self._output_names else self._output_names[0]
            self._state_out = (
                "stateN"
                if "stateN" in self._output_names
                else next((n for n in self._output_names if n != self._out_name), None)
            )
        else:
            raise ValueError(f"不支持的 Silero VAD backend: {backend}(仅支持 ascend/onnx)")

        # 隐藏状态 [2, 1, 128],跨帧传递
        self._state = self._zero_state()
        # context:帧前拼上一帧末尾 context_size 样本,初始化为零
        self._context_size = 64 if sample_rate == 16000 else 32
        self._context = np.zeros(self._context_size, dtype=np.float32)

        logger.info("Silero VAD 已加载: %s (sr=%d, backend=%s)", model_path, sample_rate, backend)

    def _zero_state(self) -> np.ndarray:
        """零状态:[2, 1, 128] float32"""
        return np.zeros(_SILERO_VAD_V5_STATE_SHAPE, dtype=np.float32)

    def inference(self, audio_chunk: np.ndarray) -> float:
        """对一帧音频做人声概率推理,更新隐藏状态。

        Args:
            audio_chunk: (512,) 或 (N,) float32,范围 [-1, 1]
                         不足 frame_size 时尾部补零,超出时截断

        Returns:
            人声概率 0.0 ~ 1.0
        """
        if audio_chunk is None:
            return 0.0

        chunk = np.asarray(audio_chunk, dtype=np.float32)
        # 对齐到 frame_size:不足补零,超出截断
        if len(chunk) < self.frame_size:
            chunk = np.pad(chunk, (0, self.frame_size - len(chunk)))
        elif len(chunk) > self.frame_size:
            chunk = chunk[: self.frame_size]

        # 关键:帧前拼上一帧末尾 context_size 样本,组成 [context | chunk]
        x = np.concatenate([self._context, chunk]).astype(np.float32)
        audio_in = x.reshape(1, -1)

        if self._inference_runner is not None:
            # Ascend ACL 走 ACL runner；仅 onnx 走 ONNX Runtime。
            prob = self._inference_runner.infer(audio_in)
            out_map = {}
        else:
            feed = {self._audio_in: audio_in}
            if self._sr_in is not None:
                feed[self._sr_in] = np.array(self.sample_rate, dtype=np.int64)
            if self._state_in is not None:
                feed[self._state_in] = self._state
            outputs = self._sess.run(self._output_names, feed)
            out_map = dict(zip(self._output_names, outputs, strict=False))

            # 概率输出 shape 通常 [1, 1] 或 [1]
            prob = float(np.asarray(out_map[self._out_name]).reshape(-1)[0])
            # ONNX 兼容后端由 Host 保存状态；OM 状态由 Ascend ACL Device 双 bank 保存。
            if self._state_out is not None and self._state_out in out_map:
                self._state = np.asarray(out_map[self._state_out])
        # 更新 context:取本帧末尾 context_size 样本供下一帧拼接
        self._context = chunk[-self._context_size :].astype(np.float32).copy()

        return max(0.0, min(1.0, prob))

    def reset_state(self):
        """重置隐藏状态为零(段间切换 / 冷启动 / 长静音后复位)。"""
        if self._closed:
            return
        self._state = self._zero_state()
        self._context = np.zeros(self._context_size, dtype=np.float32)
        if self._inference_runner is not None:
            self._inference_runner.reset()

    def close(self, *, close_runner: bool = True) -> None:
        """释放 Ascend ACL Silero 资源；ONNX Runtime 会话由其运行库管理。"""
        if self._backend_closed:
            return
        self._closed = True
        if close_runner and self._inference_runner is not None:
            self._inference_runner.close()
        if close_runner:
            self._backend_closed = True

    def close_host(self) -> None:
        """Invalidate host context while leaving an injected runner owned elsewhere."""

        if self._backend_closed:
            return
        self._closed = True


@dataclass
class GateResult:
    """单 hop 的人声门控判定结果。

    Attributes:
        vad_prob: hop 的 Top-2 VAD 概率(挡单帧瞬态脉冲)
        is_speech: 是否人声(vad_prob > vad_threshold)
        rms: hop 粒度 RMS(段级加权用,FRAME=4096 窗的 enh_ch1)
        is_gray_hop: hop 是否灰区(4 子帧 ≥1 灰区,决定是否算 SRP)
        sub_probs: 4 个子帧的 VAD 概率
        sub_rms: 4 个子帧的 RMS
        sub_gray: 4 个子帧的灰区标记
    """

    vad_prob: float
    is_speech: bool
    rms: float
    is_gray_hop: bool
    sub_probs: list[float]
    sub_rms: list[float]
    sub_gray: list[bool]


class SpeechGate:
    """人声门控:Silero VAD + RMS 灰区判定。

    内部持有一个 SileroVadEngine 实例(LSTM state 跨 hop 传递)。
    """

    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        vad_threshold: float = 0.65,
        rms_threshold: float = 0.002,
        backend: str = "ascend",
        silero_engine=None,
    ):
        """
        Args:
            model_path: 模型路径(.om 走 NPU,.onnx 走 CPU)
            sample_rate: 采样率(16000)
            vad_threshold: 灰区 VAD 门限(增强 ch1 的 Silero 概率)
            rms_threshold: 灰区 RMS 门限(增强 ch1 RMS,真语音段约 0.02 量级)
            backend: 推理后端 "ascend"(NPU,默认) 或 "onnx"(CPU 基线)
        """
        self.silero = silero_engine or SileroVadEngine(
            model_path=model_path,
            sample_rate=sample_rate,
            backend=backend,
        )
        self.vad_threshold = vad_threshold
        self.rms_threshold = rms_threshold
        self.silero_frame = SILERO_FRAME_SIZE

    def process_hop(self, enh_ch1: np.ndarray) -> GateResult:
        """对一个 hop 的增强 ch1 做人声门控判定。

        Args:
            enh_ch1: (hop_size,) float32,增强后 ch1 单声道

        Returns:
            GateResult
        """
        n_sub = len(enh_ch1) // self.silero_frame  # 4(hop=2048 / 512)
        probs: list[float] = []
        sub_rms: list[float] = []
        sub_gray: list[bool] = []

        for k in range(n_sub):
            sf = enh_ch1[k * self.silero_frame : (k + 1) * self.silero_frame]
            prob = self.silero.inference(sf)
            probs.append(prob)
            # 512 点窗 RMS(灰区切段逻辑与算法基线一致)
            r = float(np.sqrt(np.mean(sf**2)) + 1e-12)
            sub_rms.append(r)
            sub_gray.append((prob > self.vad_threshold) and (r > self.rms_threshold))

        # Top-2 VAD 概率:挡单帧瞬态脉冲
        if len(probs) >= 2:
            vad_prob = float(np.partition(probs, -2)[-2])
        else:
            vad_prob = float(probs[0]) if probs else 0.0

        # hop 灰区(4 子帧 ≥1 灰区)
        is_gray_hop = sum(sub_gray) >= 1

        # hop 粒度 RMS(段级加权用)
        rms = float(np.sqrt(np.mean(enh_ch1**2)) + 1e-12)
        is_speech = vad_prob > self.vad_threshold

        return GateResult(
            vad_prob=vad_prob,
            is_speech=is_speech,
            rms=rms,
            is_gray_hop=is_gray_hop,
            sub_probs=probs,
            sub_rms=sub_rms,
            sub_gray=sub_gray,
        )

    def reset(self):
        """重置 Silero 状态(段间切换 / 冷启动 / 长静音后复位)。"""
        self.silero.reset_state()

    def close(self, *, close_silero: bool = True) -> None:
        """释放 Silero 后端资源。"""
        if close_silero:
            self.silero.close()
        else:
            close_host = getattr(self.silero, "close_host", None)
            if callable(close_host):
                close_host()


__all__ = ["SileroVadEngine", "SpeechGate", "GateResult", "SILERO_FRAME_SIZE", "SILERO_CONTEXT_SIZE_16K"]
