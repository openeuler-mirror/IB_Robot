#!/usr/bin/env python3
"""
VADModule - 语音活动检测模块

职责边界：检测语音起点和终点，过滤无效音频

两级检测策略：
- 第一级：silero-vad 检测人声活动（粗筛）
- 第二级：基于能量的自适应阈值（精确定位端点）
"""

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

_SILERO_VAD_V4_STATE_SHAPE = (2, 1, 64)
_SILERO_VAD_V5_STATE_SHAPE = (2, 1, 128)
_MIN_ENERGY_GATE = 1e-4


class VADState(Enum):
    SILENCE = "silence"
    STARTING = "starting"
    SPEAKING = "speaking"
    ENDING = "ending"


@dataclass
class VADConfig:
    sample_rate: int = 16000
    frame_size: int = 512
    pre_roll_ms: float = 300.0
    post_roll_ms: float = 500.0
    speech_threshold: float = 0.5
    silence_threshold: float = 0.3
    min_speech_duration: float = 0.3
    min_silence_duration: float = 0.5
    energy_threshold: float = 0.01
    adaptive_threshold: bool = True


@dataclass
class VADResult:
    is_speech: bool
    state: VADState
    confidence: float = 0.0
    energy: float = 0.0


class VADModule:
    """
    语音活动检测模块

    检测语音起点和终点，过滤无效音频
    支持两级检测策略和自适应阈值
    """

    def __init__(self, config: VADConfig | None = None):
        self.config = config or VADConfig()
        self.state = VADState.SILENCE
        self._logger = None
        self._sensitivity = 0.5

        self._model: Any | None = None
        self._model_loaded = False
        self._model_backend: str = "none"
        self._onnx_session: Any | None = None
        self._onnx_state_h: np.ndarray | None = None
        self._onnx_state_c: np.ndarray | None = None
        self._onnx_state: np.ndarray | None = None

        self._noise_floor: float = 0.0
        self._noise_samples: int = 0
        self._calibration_frames: int = int(2.0 * self.config.sample_rate / self.config.frame_size)

        self._speech_start_sample: int = 0
        self._speech_frames: int = 0
        self._silence_frames: int = 0
        self._total_samples: int = 0

        self._pre_roll_buffer: list[np.ndarray] = []
        self._pre_roll_samples: int = 0
        self._max_pre_roll_samples: int = int(self.config.sample_rate * self.config.pre_roll_ms / 1000)

        self._post_roll_frames: int = 0
        self._max_post_roll_frames: int = int(
            self.config.sample_rate * self.config.post_roll_ms / 1000 / self.config.frame_size
        )

    def set_logger(self, logger) -> None:
        """Attach an optional logger for runtime warnings."""
        self._logger = logger

    def _warn(self, message: str) -> None:
        """Emit warnings through ROS logging when available."""
        if self._logger is not None:
            warn = getattr(self._logger, "warning", None) or getattr(self._logger, "warn", None)
            if callable(warn):
                warn(message)
                return
        print(message)

    def initialize(self, model_path: str | None = None) -> bool:
        """初始化 VAD 模型"""
        try:
            import torch

            # 优先加载本地 JIT 模型，避免首次运行时联网下载
            if model_path is None:
                model_path = self._get_default_local_model_path()

            if model_path and os.path.exists(model_path):
                try:
                    self._model = torch.jit.load(model_path, map_location="cpu")
                    self._model.eval()
                    self._model_loaded = True
                    self._model_backend = "torch_jit"
                    return True
                except Exception as e:
                    self._warn(f"Warning: Failed to load local VAD model from {model_path}: {e}")

            # 优先使用本地 ONNX（CPU provider），避免 torch.hub 触发外部 CUDA 依赖
            onnx_model_path = self._get_default_local_onnx_model_path()
            if onnx_model_path and os.path.exists(onnx_model_path):
                try:
                    import onnxruntime as ort

                    session_options = ort.SessionOptions()
                    session_options.intra_op_num_threads = 1
                    session_options.inter_op_num_threads = 1
                    session = ort.InferenceSession(
                        onnx_model_path,
                        sess_options=session_options,
                        providers=["CPUExecutionProvider"],
                    )
                    input_names = {inp.name for inp in session.get_inputs()}
                    if {"x", "h", "c"}.issubset(input_names):
                        self._onnx_session = session
                        # silero_vad.onnx (v4) uses separate hidden/cell LSTM state
                        # tensors shaped as (layers=2, batch=1, hidden=64).
                        self._onnx_state_h = np.zeros(_SILERO_VAD_V4_STATE_SHAPE, dtype=np.float32)
                        self._onnx_state_c = np.zeros(_SILERO_VAD_V4_STATE_SHAPE, dtype=np.float32)
                        self._model_loaded = True
                        self._model_backend = "onnx_v4"
                        return True
                    if {"input", "state", "sr"}.issubset(input_names):
                        self._onnx_session = session
                        # silero_vad_v5.onnx exposes a single recurrent state tensor
                        # shaped as (layers=2, batch=1, hidden=128).
                        self._onnx_state = np.zeros(_SILERO_VAD_V5_STATE_SHAPE, dtype=np.float32)
                        self._model_loaded = True
                        self._model_backend = "onnx_v5"
                        return True
                except Exception as e:
                    self._warn(f"Warning: Failed to load local ONNX VAD model from {onnx_model_path}: {e}")

            # 本地不存在时回退到 torch.hub
            self._model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                source="github",
                force_reload=False,
                onnx=False,
                trust_repo=True,
            )
            self._model.eval()
            self._model_loaded = True
            self._model_backend = "torch_hub"
            return True

        except Exception as e:
            self._warn(f"Warning: Failed to load silero-vad: {e}")
            self._warn("Falling back to energy-based VAD")
            self._model_loaded = False
            self._model_backend = "energy"
            return True

    def _get_default_local_model_path(self) -> str | None:
        """获取默认本地 silero-vad 模型路径"""
        candidate = Path(__file__).resolve().parents[3] / "models" / "voice_asr" / "silero-vad" / "silero_vad.jit"
        resolved = candidate.resolve()
        return str(resolved) if resolved.exists() else None

    def _get_default_local_onnx_model_path(self) -> str | None:
        """获取默认本地 silero-vad ONNX 模型路径。"""
        candidates = (
            Path(__file__).resolve().parents[3] / "models" / "voice_asr" / "silero-vad" / "silero_vad.onnx",
            Path(__file__).resolve().parents[3] / "models" / "voice_asr" / "silero-vad" / "silero_vad_v5.onnx",
        )
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.exists():
                return str(resolved)
        return None

    def process(self, audio_frame: np.ndarray) -> VADResult:
        """
        处理音频帧，返回 VAD 结果

        Args:
            audio_frame: 音频帧 (float32)

        Returns:
            VADResult: VAD 检测结果
        """
        if len(audio_frame) == 0:
            return VADResult(is_speech=False, state=self.state)

        energy = self._compute_energy(audio_frame)
        confidence = self._get_speech_probability(audio_frame)

        self._update_noise_floor(energy)

        if self.config.adaptive_threshold:
            threshold = self._get_adaptive_threshold()
        else:
            threshold = self.config.speech_threshold

        result = self._update_state(confidence, energy, threshold)

        self._update_pre_roll(audio_frame)
        self._total_samples += len(audio_frame)

        return result

    def _compute_energy(self, audio_frame: np.ndarray) -> float:
        """计算音频帧能量"""
        return float(np.sqrt(np.mean(audio_frame**2)))

    def _get_speech_probability(self, audio_frame: np.ndarray) -> float:
        """获取语音概率"""
        if self._model_backend.startswith("onnx") and self._onnx_session is not None:
            return self._get_onnx_speech_probability(audio_frame)
        if self._model_loaded and self._model is not None:
            return self._get_torch_speech_probability(audio_frame)
        return self._get_energy_fallback_probability(audio_frame)

    def _get_onnx_speech_probability(self, audio_frame: np.ndarray) -> float:
        """ONNX 后端语音概率检测"""
        try:
            if len(audio_frame) < 512:
                audio_frame = np.pad(audio_frame, (0, 512 - len(audio_frame)))
            if len(audio_frame) > 512:
                audio_frame = audio_frame[:512]

            frame = np.asarray(audio_frame, dtype=np.float32).reshape(1, -1)

            if self._model_backend == "onnx_v4":
                if self._onnx_state_h is None or self._onnx_state_c is None:
                    self._onnx_state_h = np.zeros(_SILERO_VAD_V4_STATE_SHAPE, dtype=np.float32)
                    self._onnx_state_c = np.zeros(_SILERO_VAD_V4_STATE_SHAPE, dtype=np.float32)
                prob, new_h, new_c = self._onnx_session.run(
                    None,
                    {"x": frame, "h": self._onnx_state_h, "c": self._onnx_state_c},
                )
                self._onnx_state_h = new_h
                self._onnx_state_c = new_c
                return float(prob[0][0])

            if self._model_backend == "onnx_v5":
                if self._onnx_state is None:
                    self._onnx_state = np.zeros(_SILERO_VAD_V5_STATE_SHAPE, dtype=np.float32)
                prob, new_state = self._onnx_session.run(
                    None,
                    {
                        "input": frame,
                        "state": self._onnx_state,
                        "sr": np.array(self.config.sample_rate, dtype=np.int64),
                    },
                )
                self._onnx_state = new_state
                return float(prob[0][0])
        except Exception:
            pass
        return self._get_energy_fallback_probability(audio_frame)

    def _get_torch_speech_probability(self, audio_frame: np.ndarray) -> float:
        """Torch 后端语音概率检测"""
        try:
            import torch

            if len(audio_frame) < 512:
                audio_frame = np.pad(audio_frame, (0, 512 - len(audio_frame)))

            audio_tensor = torch.from_numpy(audio_frame).unsqueeze(0)

            with torch.no_grad():
                confidence = self._model(audio_tensor, self.config.sample_rate).item()

            return confidence

        except Exception:
            return self._get_energy_fallback_probability(audio_frame)

    def _get_energy_fallback_probability(self, audio_frame: np.ndarray) -> float:
        """基于能量的回退语音概率检测"""
        energy = self._compute_energy(audio_frame)
        return min(1.0, energy / max(self._noise_floor * 2, 0.01))

    def _update_noise_floor(self, energy: float):
        """更新噪声底估计"""
        if self._noise_samples < self._calibration_frames and self.state == VADState.SILENCE:
            alpha = 0.9 if self._noise_samples > 0 else 0.0
            self._noise_floor = alpha * self._noise_floor + (1 - alpha) * energy
            self._noise_samples += 1

    def _get_adaptive_threshold(self) -> float:
        """获取自适应阈值"""
        base_threshold = self.config.speech_threshold

        if self._noise_floor > 0:
            snr_factor = min(2.0, max(0.5, self._noise_floor / 0.01))
            return base_threshold * snr_factor

        return base_threshold

    def _get_energy_gate(self) -> float:
        static_gate = self.config.energy_threshold * max(0.5, 1.0 - self._sensitivity * 0.5)
        if self._noise_floor <= 0:
            return static_gate

        noise_gate = max(_MIN_ENERGY_GATE, self._noise_floor * max(1.0, 2.5 - self._sensitivity))
        return min(static_gate, noise_gate)

    def _update_state(self, confidence: float, energy: float, threshold: float) -> VADResult:
        """更新 VAD 状态"""
        energy_gate = self._get_energy_gate()
        is_speech = confidence > threshold and energy >= energy_gate

        if self.state == VADState.SILENCE:
            if is_speech:
                self.state = VADState.STARTING
                self._speech_start_sample = self._total_samples
                self._speech_frames = 1
                self._silence_frames = 0

        elif self.state == VADState.STARTING:
            if is_speech:
                self._speech_frames += 1
                speech_duration = self._speech_frames * self.config.frame_size / self.config.sample_rate

                if speech_duration >= self.config.min_speech_duration:
                    self.state = VADState.SPEAKING
            else:
                self.state = VADState.SILENCE
                self._speech_frames = 0

        elif self.state == VADState.SPEAKING:
            if is_speech:
                self._speech_frames += 1
                self._silence_frames = 0
            else:
                self._silence_frames += 1
                silence_duration = self._silence_frames * self.config.frame_size / self.config.sample_rate

                if silence_duration >= self.config.min_silence_duration:
                    self.state = VADState.ENDING
                    self._post_roll_frames = 0

        elif self.state == VADState.ENDING:
            self._post_roll_frames += 1

            if is_speech:
                self.state = VADState.SPEAKING
                self._silence_frames = 0
            elif self._post_roll_frames >= self._max_post_roll_frames:
                self.state = VADState.SILENCE
                self._speech_frames = 0
                self._silence_frames = 0

        return VADResult(is_speech=is_speech, state=self.state, confidence=confidence, energy=energy)

    def _update_pre_roll(self, audio_frame: np.ndarray):
        """更新预录缓冲区"""
        self._pre_roll_buffer.append(audio_frame.copy())
        self._pre_roll_samples += len(audio_frame)

        while self._pre_roll_samples > self._max_pre_roll_samples and self._pre_roll_buffer:
            removed = self._pre_roll_buffer.pop(0)
            self._pre_roll_samples -= len(removed)

    def get_pre_roll_audio(self) -> np.ndarray:
        """获取预录音频"""
        if not self._pre_roll_buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self._pre_roll_buffer)

    def is_speech_started(self) -> bool:
        """检测是否语音开始"""
        return self.state in [VADState.STARTING, VADState.SPEAKING, VADState.ENDING]

    def is_speech_ended(self) -> bool:
        """检测是否语音结束"""
        return self.state == VADState.SILENCE and self._speech_frames == 0

    def get_speech_start_time(self) -> float:
        """获取语音开始时间"""
        return self._speech_start_sample / self.config.sample_rate

    def segment_audio(
        self, audio_data: np.ndarray, min_segment_duration: float = 0.5
    ) -> list[tuple[int, int, np.ndarray]]:
        """
        对音频进行分段

        Args:
            audio_data: 完整音频数据
            min_segment_duration: 最小分段时长

        Returns:
            List of (start_sample, end_sample, audio_segment)
        """
        self.reset()
        segments = []
        current_start = None
        current_audio = []

        frame_size = self.config.frame_size
        n_frames = len(audio_data) // frame_size

        for i in range(n_frames):
            start = i * frame_size
            end = start + frame_size
            frame = audio_data[start:end]

            result = self.process(frame)

            if result.state in (VADState.STARTING, VADState.SPEAKING, VADState.ENDING):
                if current_start is None:
                    current_start = start
                    pre_roll = self.get_pre_roll_audio()
                    if len(pre_roll) > 0:
                        current_audio.append(pre_roll)
                current_audio.append(frame)

            elif result.state == VADState.SILENCE and current_start is not None:
                if current_audio:
                    segment_audio = np.concatenate(current_audio)
                    duration = len(segment_audio) / self.config.sample_rate

                    if duration >= min_segment_duration:
                        segments.append((current_start, current_start + len(segment_audio), segment_audio))

                current_start = None
                current_audio = []
                self.reset()

        if current_start is not None and current_audio:
            segment_audio = np.concatenate(current_audio)
            duration = len(segment_audio) / self.config.sample_rate

            if duration >= min_segment_duration:
                segments.append((current_start, current_start + len(segment_audio), segment_audio))

        return segments

    def reset(self):
        """重置 VAD 状态"""
        self.state = VADState.SILENCE
        self._speech_start_sample = 0
        self._speech_frames = 0
        self._silence_frames = 0
        self._total_samples = 0
        self._post_roll_frames = 0
        self._pre_roll_buffer = []
        self._pre_roll_samples = 0
        self._noise_floor = 0.0
        self._noise_samples = 0
        if self._model_backend == "onnx_v4":
            self._onnx_state_h = np.zeros(_SILERO_VAD_V4_STATE_SHAPE, dtype=np.float32)
            self._onnx_state_c = np.zeros(_SILERO_VAD_V4_STATE_SHAPE, dtype=np.float32)
        elif self._model_backend == "onnx_v5":
            self._onnx_state = np.zeros(_SILERO_VAD_V5_STATE_SHAPE, dtype=np.float32)

    def set_sensitivity(self, sensitivity: float):
        """
        设置 VAD 灵敏度

        Args:
            sensitivity: 灵敏度 (0.0-1.0)，越高越灵敏
        """
        sensitivity = max(0.0, min(1.0, sensitivity))
        self._sensitivity = sensitivity
        self.config.speech_threshold = 0.7 - sensitivity * 0.4
        self.config.silence_threshold = self.config.speech_threshold - 0.1
