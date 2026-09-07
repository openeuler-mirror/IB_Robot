#!/usr/bin/env python3
"""
ASRInferenceModule - ASR推理模块

职责边界：管理 sherpa-onnx 识别器的生命周期，执行解码

支持流式识别(OnlineRecognizer)和非流式识别(OfflineRecognizer)
支持热词增强
支持自动检测模型类型
"""

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from inference_manifest import load_inference_manifest
from inference_service.unified_runtime import (
    ExecutionContext,
    ExecutionContract,
    ModelRequest,
    ModelRuntimeHandle,
    RuntimeAssembly,
)

_MAX_FINAL_DECODE_STEPS = 200


class _SherpaStream:
    def __init__(self, recognizer: Any, sample_rate: int) -> None:
        self.recognizer = recognizer
        self.sample_rate = sample_rate
        self.stream = recognizer.create_stream()


class _SherpaStreamingRuntime:
    """Adapt one sherpa recognizer to the unified stream lifecycle."""

    def __init__(self, recognizer: Any, sample_rate: int) -> None:
        self._recognizer = recognizer
        self._sample_rate = sample_rate

    def load(self, _context=None) -> None:
        return None

    def open_stream(self, _context: ExecutionContext) -> _SherpaStream:
        return _SherpaStream(self._recognizer, self._sample_rate)

    def step(self, stream: _SherpaStream, request: ModelRequest, context: ExecutionContext) -> dict[str, object]:
        context.check("sherpa.accept_waveform")
        audio = request.inputs.get("audio")
        if not isinstance(audio, np.ndarray):
            raise TypeError("ASR stream request requires a numpy audio input")
        stream.stream.accept_waveform(stream.sample_rate, audio)
        if request.metadata.get("final", False):
            # Final decode must drain the recognizer completely: silent
            # truncation would drop undecoded audio while reporting success.
            stream.stream.input_finished()
            while self._recognizer.is_ready(stream.stream):
                self._recognizer.decode_stream(stream.stream)
        elif self._recognizer.is_ready(stream.stream):
            # Live streaming: decode one step per admitted chunk when the
            # recognizer has work queued; the queue drains across chunks.
            self._recognizer.decode_stream(stream.stream)
        result = self._recognizer.get_result(stream.stream)
        text = result if isinstance(result, str) else getattr(result, "text", "")
        return {"text": text if isinstance(text, str) else "", "is_final": bool(request.metadata.get("final", False))}

    def reset_stream(self, stream: _SherpaStream, _context: ExecutionContext) -> None:
        stream.stream = self._recognizer.create_stream()

    def close_stream(self, stream: _SherpaStream, _context: ExecutionContext) -> None:
        # Release the native sherpa stream so its buffers do not stay
        # referenced by the runtime's closed-stream records.
        stream.stream = None

    def close(self) -> None:
        return None


class ASRState(Enum):
    IDLE = "idle"
    READY = "ready"
    RECOGNIZING = "recognizing"
    ERROR = "error"


class ModelType(Enum):
    STREAMING = "streaming"
    OFFLINE = "offline"
    AUTO = "auto"


@dataclass
class ASRResult:
    text: str
    is_final: bool
    confidence: float = 1.0
    tokens: list[str] | None = None
    timestamps: list[float] | None = None
    start_time: float | None = None
    duration: float | None = None


class ASRInferenceModule:
    """
    ASR推理模块

    管理 sherpa-onnx 识别器的生命周期，执行解码
    支持流式识别(OnlineRecognizer)和非流式识别(OfflineRecognizer)
    支持热词增强
    支持自动检测模型类型
    """

    def __init__(self):
        self.state = ASRState.IDLE
        self._recognizer: Any | None = None
        self._stream: Any | None = None
        self._model_type: ModelType = ModelType.OFFLINE
        self._hotwords: dict[str, float] = {}
        self._sample_rate: int = 16000

        self._active_stream: Any | None = None
        self._pending_stream: Any | None = None
        self._last_stream_text = ""

        self._bundle_path: str | None = None
        self._deployment: str | None = None
        self._language: str = "zh"
        self._last_error: str | None = None
        self._runtime_handle: ModelRuntimeHandle | None = None
        self._runtime_context: ExecutionContext | None = None

        self._lock = threading.Lock()

    def initialize(
        self,
        bundle_path: str,
        deployment: str,
        language: str = "zh",
    ) -> bool:
        """
        初始化 ASR 模型

        Args:
            bundle_path: schema-v3 ASR bundle directory
            deployment: named deployment in the bundle manifest

        Returns:
            bool: 是否初始化成功
        """
        try:
            import importlib.util

            if importlib.util.find_spec("sherpa_onnx") is None:
                raise ImportError("sherpa_onnx not installed. Install with: pip install sherpa-onnx")

            self._recognizer = None
            self._active_stream = None
            self._pending_stream = None
            self._last_error = None
            validated = load_inference_manifest(bundle_path, deployment)
            identity = validated.identity
            if (identity.interface, identity.model_type, identity.operation) != (
                "tensor_model",
                "sherpa_onnx",
                "recognize",
            ):
                raise ValueError("ASR bundle must use tensor_model/sherpa_onnx/recognize")
            self._bundle_path = str(validated.bundle_root)
            self._deployment = deployment
            self._language = language

            roles = validated.resolved_artifacts
            role_paths = {str(role): str(path) for role, path in roles.items()}
            provider = "cuda" if validated.deployment.device == "cuda" else "cpu"
            tokens_path = role_paths.get("tokens")
            if tokens_path is None:
                raise ValueError("ASR deployment must declare a tokens artifact")
            if {"encoder", "decoder", "joiner"}.issubset(role_paths):
                self._model_type = ModelType.STREAMING
                recognizer = self._create_streaming_recognizer(
                    role_paths["encoder"], role_paths["decoder"], role_paths["joiner"], tokens_path, provider
                )
            elif {"encoder", "decoder"}.issubset(role_paths):
                self._model_type = ModelType.STREAMING
                recognizer = self._create_streaming_paraformer_recognizer(
                    role_paths["encoder"], role_paths["decoder"], tokens_path, provider
                )
            elif "model" in role_paths:
                self._model_type = ModelType.OFFLINE
                recognizer = self._create_offline_recognizer(role_paths["model"], tokens_path, provider)
            else:
                raise ValueError("ASR deployment must declare streaming or offline artifact roles")

            if recognizer is None:
                raise RuntimeError("Recognizer factory returned None for the selected ASR deployment")

            # 流式和离线模型都必须从 recognizer 配置读取真实采样率，
            # 不能用默认值代替，否则节点层无法验证完整音频链路契约。
            feat_config = getattr(getattr(recognizer, "config", None), "feat_config", None)
            sample_rate = getattr(feat_config, "sampling_rate", None)
            if sample_rate is None:
                raise RuntimeError(
                    "Unable to determine ASR model sampling rate; "
                    "the Voice ASR audio pipeline contract cannot be verified"
                )

            self._recognizer = recognizer
            self._sample_rate = int(sample_rate)
            if self._model_type == ModelType.STREAMING:
                streaming_runtime = _SherpaStreamingRuntime(self._recognizer, self._sample_rate)
                self._runtime_handle = ModelRuntimeHandle(
                    RuntimeAssembly(
                        runtime_executor=streaming_runtime,
                        streaming_runtime=streaming_runtime,
                        session=streaming_runtime,
                        execution_contract=ExecutionContract(
                            state_scope="stream",
                            state_bank_mode="per_stream",
                            max_open_streams=1,
                        ),
                        stateful=True,
                        resettable=True,
                        state_scope="stream",
                        state_bank_mode="per_stream",
                        max_open_streams=1,
                    )
                )
                self._runtime_context = ExecutionContext.create("voice-asr")
                self._runtime_handle.load()
            self.state = ASRState.READY
            return True

        except ImportError:
            self._last_error = "sherpa_onnx not installed. Install with: pip install sherpa-onnx"
            self.state = ASRState.ERROR
            raise RuntimeError(self._last_error) from None
        except Exception as e:
            self._recognizer = None
            self._active_stream = None
            self._pending_stream = None
            self._last_error = str(e)
            self.state = ASRState.ERROR
            raise RuntimeError(f"Failed to initialize ASR: {e}") from e

    def _create_streaming_recognizer(
        self, encoder: str, decoder: str, joiner: str, tokens_path: str, provider: str
    ) -> Any:
        """
        创建流式识别器 (OnlineRecognizer)
        使用 sherpa-onnx 工厂方法
        """
        import sherpa_onnx

        return sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens_path,
            num_threads=4,
            provider=provider,
        )

    def _create_streaming_paraformer_recognizer(
        self, encoder: str, decoder: str, tokens_path: str, provider: str
    ) -> Any:
        import sherpa_onnx

        return sherpa_onnx.OnlineRecognizer.from_paraformer(
            encoder=encoder,
            decoder=decoder,
            tokens=tokens_path,
            num_threads=4,
            provider=provider,
        )

    def _create_offline_recognizer(self, model_path: str, tokens_path: str, provider: str) -> Any:
        """
        创建非流式识别器 (OfflineRecognizer)
        使用 sherpa-onnx 工厂方法
        """
        import sherpa_onnx

        return sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=model_path,
            tokens=tokens_path,
            num_threads=4,
            provider=provider,
            decoding_method="greedy_search",
        )

    def create_stream(self) -> Any:
        """创建新的识别流"""
        if self._runtime_handle is None or self._runtime_context is None:
            raise RuntimeError("streaming ASR runtime not initialized")
        return self._runtime_handle.open_stream(self._runtime_context)

    def _require_recognizer(self, operation: str) -> None:
        """Ensure the underlying recognizer exists before using it."""
        if self._recognizer is not None:
            return

        detail = f": {self._last_error}" if self._last_error else ""
        raise RuntimeError(f"Cannot run {operation} because the ASR recognizer is not initialized{detail}")

    def is_streaming(self) -> bool:
        """检查是否为流式模型"""
        return self._model_type == ModelType.STREAMING

    def _extract_streaming_text(self, stream: Any) -> str:
        """Extract text from sherpa-onnx get_result() across return-shape variants.

        Some sherpa-onnx builds return an object with a ``text`` attribute,
        while others return a plain string. Ignore unknown return objects
        instead of converting them into unhelpful ``<object at ...>`` text.
        """
        result = self._recognizer.get_result(stream)
        if isinstance(result, str):
            return result
        text = getattr(result, "text", None)
        return text if isinstance(text, str) else ""

    def _extract_offline_text(self, stream: Any) -> str:
        """Extract text from sherpa-onnx OfflineRecognizer result in a version-tolerant way.

        sherpa-onnx OfflineRecognizer stream.result may be an object with a
        ``.text`` attribute or a plain string depending on the build. This
        helper normalises the output to always return str.
        """
        result = stream.result
        if isinstance(result, str):
            return result
        text = getattr(result, "text", None)
        return text if isinstance(text, str) else ""

    def _decode_online_stream_until_idle(self, stream: Any, max_steps: int | None = None) -> bool:
        steps = 0
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)
            steps += 1
            if max_steps is not None and steps >= max_steps:
                return not self._recognizer.is_ready(stream)
        return True

    def start_streaming(self) -> bool:
        """开始流式识别"""
        with self._lock:
            if self.state != ASRState.READY:
                return False

            if self._model_type != ModelType.STREAMING:
                raise RuntimeError(
                    "start_streaming() is only for streaming models. "
                    "Current model is offline. Use recognize_file() instead."
                )

            self._active_stream = self.create_stream()
            self._last_stream_text = ""
            self.state = ASRState.RECOGNIZING
            return True

    def accept_waveform(self, audio_data: np.ndarray) -> ASRResult | None:
        """
        输入音频数据并获取识别结果

        Args:
            audio_data: 音频数据 (float32, 16kHz)

        Returns:
            ASRResult: 识别结果（如果有）
        """
        with self._lock:
            if self._active_stream is None:
                return None

            result = self._runtime_handle.step(
                self._active_stream,
                ModelRequest(inputs={"audio": audio_data}),
                self._runtime_context,
            )
            text = str(result.outputs.get("text", ""))
            self._last_stream_text = text
            if text:
                return ASRResult(text=text, is_final=False, confidence=1.0)

            return None

    def get_partial_result(self) -> ASRResult:
        """获取当前中间结果"""
        with self._lock:
            if self._active_stream is None:
                return ASRResult(text="", is_final=False)

            text = self._last_stream_text
            return ASRResult(text=text, is_final=False, confidence=1.0)

    def get_final_result(self) -> ASRResult:
        """获取最终结果并结束识别"""
        with self._lock:
            if self._active_stream is None:
                return ASRResult(text="", is_final=True)

            result = self._runtime_handle.step(
                self._active_stream,
                ModelRequest(inputs={"audio": np.zeros(0, dtype=np.float32)}, metadata={"final": True}),
                self._runtime_context,
            )
            text = str(result.outputs.get("text", ""))

            result = ASRResult(text=text, is_final=True, confidence=1.0)

            self._runtime_handle.close_stream(self._active_stream, self._runtime_context)
            self._active_stream = None
            self._last_stream_text = ""
            self.state = ASRState.READY

            return result

    def end_streaming(self) -> ASRResult:
        """结束流式识别并返回最终结果"""
        return self.get_final_result()

    def recognize_file(
        self, audio_data: np.ndarray, enable_vad: bool = True, vad_module: Any | None = None
    ) -> list[ASRResult]:
        """
        识别音频文件

        支持流式和非流式模型:
        - 流式模型: 使用 OnlineRecognizer 的流式接口
        - 非流式模型: 使用 OfflineRecognizer 的批量接口

        Args:
            audio_data: 完整音频数据
            enable_vad: 是否启用 VAD 分段
            vad_module: VAD 模块实例

        Returns:
            List[ASRResult]: 识别结果列表
        """
        self._require_recognizer("file recognition")

        results = []

        if self._model_type == ModelType.STREAMING:
            results = self._recognize_streaming(audio_data, enable_vad, vad_module)
        else:
            results = self._recognize_offline(audio_data, enable_vad, vad_module)

        return results

    def _create_file_result(self, text: str, start_sample: int, end_sample: int) -> ASRResult:
        """构造带显式时间边界的文件识别结果"""
        start_time = start_sample / self._sample_rate
        duration = max(end_sample - start_sample, 0) / self._sample_rate
        return ASRResult(
            text=text,
            is_final=True,
            confidence=1.0,
            timestamps=[start_time],
            start_time=start_time,
            duration=duration,
        )

    def _recognize_streaming(self, audio_data: np.ndarray, enable_vad: bool, vad_module: Any | None) -> list[ASRResult]:
        """使用流式模型识别"""
        self._require_recognizer("streaming file recognition")

        results = []

        if not enable_vad or vad_module is None:
            segments = [(0, len(audio_data), audio_data)]
        else:
            segments = vad_module.segment_audio(audio_data)
        for start_sample, end_sample, audio_segment in segments:
            # File recognition shares the runtime handle protocol with live
            # streaming: one admitted stream per segment, final decode, and
            # guaranteed release so the single open-stream slot is free again.
            stream = self.create_stream()
            try:
                result = self._runtime_handle.step(
                    stream,
                    ModelRequest(
                        inputs={"audio": np.ascontiguousarray(audio_segment, dtype=np.float32)},
                        metadata={"final": True},
                    ),
                    self._runtime_context,
                )
                text = str(result.outputs.get("text", ""))
            finally:
                self._runtime_handle.close_stream(stream, self._runtime_context)
            if text:
                results.append(
                    self._create_file_result(
                        text=text,
                        start_sample=start_sample,
                        end_sample=end_sample,
                    )
                )

        return results

    def _recognize_offline(self, audio_data: np.ndarray, enable_vad: bool, vad_module: Any | None) -> list[ASRResult]:
        """使用非流式模型识别"""
        self._require_recognizer("offline file recognition")

        results = []

        if not enable_vad or vad_module is None:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(self._sample_rate, audio_data)
            self._recognizer.decode_stream(stream)
            text = self._extract_offline_text(stream)
            if text:
                results.append(
                    self._create_file_result(
                        text=text,
                        start_sample=0,
                        end_sample=len(audio_data),
                    )
                )
        else:
            segments = vad_module.segment_audio(audio_data)
            for segment in segments:
                start_sample, end_sample, audio_segment = segment
                stream = self._recognizer.create_stream()
                stream.accept_waveform(self._sample_rate, audio_segment)
                self._recognizer.decode_stream(stream)
                text = self._extract_offline_text(stream)
                if text:
                    results.append(
                        self._create_file_result(
                            text=text,
                            start_sample=start_sample,
                            end_sample=end_sample,
                        )
                    )

        return results

    def set_hotwords(self, hotwords: dict[str, float]):
        """
        设置热词

        Args:
            hotwords: 热词字典 {word: boost_score}
        """
        self._hotwords = hotwords

        if self._recognizer and hasattr(self._recognizer, "set_hotwords"):
            self._recognizer.set_hotwords(hotwords)

    def add_hotword(self, word: str, boost: float = 1.5):
        """添加单个热词"""
        self._hotwords[word] = boost
        self.set_hotwords(self._hotwords)

    def remove_hotword(self, word: str):
        """移除热词"""
        if word in self._hotwords:
            del self._hotwords[word]
            self.set_hotwords(self._hotwords)

    def clear_hotwords(self):
        """清除所有热词"""
        self._hotwords.clear()
        if self._recognizer and hasattr(self._recognizer, "set_hotwords"):
            self._recognizer.set_hotwords({})

    def reset(self):
        """重置识别状态"""
        with self._lock:
            if self._active_stream is not None and self._runtime_handle is not None:
                # Facade reset ends the current recognition and returns to
                # READY: close the stream so the single open-stream slot is
                # released for the next start_streaming()/recognize_file().
                self._runtime_handle.close_stream(self._active_stream, self._runtime_context)
            self._active_stream = None
            self._pending_stream = None
            self._last_stream_text = ""
            if self._recognizer is not None:
                self.state = ASRState.READY
            elif self._last_error:
                self.state = ASRState.ERROR
            else:
                self.state = ASRState.IDLE

    def cleanup(self):
        """清理资源"""
        with self._lock:
            if self._active_stream is not None and self._runtime_handle is not None:
                self._runtime_handle.close_stream(self._active_stream, self._runtime_context)
            self._active_stream = None
            self._pending_stream = None
            if self._runtime_handle is not None:
                self._runtime_handle.close()
                self._runtime_handle = None
            self._recognizer = None
            self.state = ASRState.IDLE

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_ready(self) -> bool:
        return self.state in [ASRState.READY, ASRState.RECOGNIZING]
