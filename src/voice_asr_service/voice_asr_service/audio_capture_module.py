#!/usr/bin/env python3
"""
AudioCaptureModule - 音频采集模块

职责边界：专注于原始音频数据的获取，不做任何业务处理

支持多种后端：pyaudio/sounddevice，通过统一接口屏蔽差异
"""

import queue
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class CaptureState(Enum):
    UNINITIALIZED = "uninitialized"
    OPENING = "opening"
    CAPTURING = "capturing"
    PAUSED = "paused"
    CLOSED = "closed"
    ERROR = "error"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 512
    buffer_seconds: float = 5.0


class RingBuffer:
    """环形缓冲区，用于存储预录音频"""

    def __init__(self, max_samples: int):
        self.max_samples = max_samples
        self.buffer = np.zeros(max_samples, dtype=np.float32)
        self.write_pos = 0
        self.size = 0
        self.lock = threading.Lock()

    def write(self, data: np.ndarray):
        with self.lock:
            n = len(data)
            if n >= self.max_samples:
                self.buffer[:] = data[-self.max_samples :]
                self.write_pos = self.max_samples
                self.size = self.max_samples
            else:
                end_pos = (self.write_pos + n) % self.max_samples
                if end_pos < self.write_pos:
                    split = self.max_samples - self.write_pos
                    self.buffer[self.write_pos :] = data[:split]
                    self.buffer[:end_pos] = data[split:]
                else:
                    self.buffer[self.write_pos : end_pos] = data
                self.write_pos = end_pos
                self.size = min(self.size + n, self.max_samples)

    def read_all(self) -> np.ndarray:
        with self.lock:
            if self.size == 0:
                return np.array([], dtype=np.float32)
            if self.size < self.max_samples:
                return self.buffer[: self.size].copy()
            # Buffer is full; chronological data starts at write_pos
            return np.concatenate([self.buffer[self.write_pos :], self.buffer[: self.write_pos]])

    def read_last(self, n_samples: int) -> np.ndarray:
        with self.lock:
            if n_samples >= self.size:
                return self.read_all()
            start_pos = (self.write_pos - n_samples) % self.max_samples
            if start_pos < self.write_pos:
                return self.buffer[start_pos : self.write_pos].copy()
            else:
                return np.concatenate([self.buffer[start_pos:], self.buffer[: self.write_pos]])

    def clear(self):
        with self.lock:
            self.buffer.fill(0)
            self.write_pos = 0
            self.size = 0


class AudioCaptureModule:
    """
    音频采集模块

    支持多种后端（pyaudio/sounddevice），通过统一接口屏蔽差异
    维护环形缓冲区用于预录音频
    异步采集，通过线程安全队列与主线程交互
    """

    def __init__(self, config: AudioConfig | None = None):
        self.config = config or AudioConfig()
        self.state = CaptureState.UNINITIALIZED

        self._audio_queue: queue.Queue = queue.Queue(maxsize=100)
        self._ring_buffer = RingBuffer(int(self.config.sample_rate * self.config.buffer_seconds))

        self._capture_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

        self._backend: Any | None = None
        self._backend_type: str | None = None
        self._stream: Any | None = None
        self._stream_lock = threading.Lock()

        self._retry_count = 0
        self._max_retries = 5
        self._retry_delay = 1.0

        self._on_error_callback: Callable[[str], None] | None = None
        self._device_index: int | None = None

    def set_error_callback(self, callback: Callable[[str], None]):
        self._on_error_callback = callback

    def set_device(self, device_index: int | None = None):
        self._device_index = device_index

    @staticmethod
    def list_audio_devices() -> list[dict[str, Any]]:
        """枚举可用音频输入设备"""
        devices = []
        try:
            import sounddevice as sd

            for idx, info in enumerate(sd.query_devices()):
                if info.get("max_input_channels", 0) > 0:
                    devices.append(
                        {
                            "index": idx,
                            "name": info.get("name", "Unknown"),
                            "channels": info.get("max_input_channels", 0),
                            "sample_rate": int(info.get("default_samplerate", 0)),
                            "backend": "sounddevice",
                        }
                    )
        except Exception:
            pass

        if not devices:
            try:
                import pyaudio

                pa = pyaudio.PyAudio()
                for i in range(pa.get_device_count()):
                    info = pa.get_device_info_by_index(i)
                    if info.get("maxInputChannels", 0) > 0:
                        devices.append(
                            {
                                "index": i,
                                "name": info.get("name", "Unknown"),
                                "channels": info.get("maxInputChannels", 0),
                                "sample_rate": int(info.get("defaultSampleRate", 0)),
                                "backend": "pyaudio",
                            }
                        )
                pa.terminate()
            except Exception:
                pass

        return devices

    def set_device_by_name(self, name_substring: str) -> bool:
        """通过名称子串匹配选择输入设备"""
        devices = self.list_audio_devices()
        for dev in devices:
            if name_substring.lower() in dev["name"].lower():
                self._device_index = dev["index"]
                return True
        return False

    def initialize(self) -> bool:
        if (
            self.state in [CaptureState.CLOSED, CaptureState.PAUSED, CaptureState.CAPTURING]
            and self._backend is not None
            and self._backend_type is not None
        ):
            return True

        self.state = CaptureState.OPENING

        try:
            import sounddevice as sd

            self._backend = sd
            self._backend_type = "sounddevice"
        except ImportError:
            try:
                import pyaudio

                self._backend = pyaudio.PyAudio()
                self._backend_type = "pyaudio"
            except ImportError:
                self._handle_error("No audio backend available (sounddevice or pyaudio required)")
                return False

        self.state = CaptureState.CLOSED
        self._retry_count = 0
        return True

    def start_capture(self) -> bool:
        if self.state == CaptureState.CAPTURING:
            return True

        if self.state == CaptureState.PAUSED:
            self._pause_event.clear()
            self.state = CaptureState.CAPTURING
            return True

        if (
            self.state != CaptureState.CLOSED or self._backend is None or self._backend_type is None
        ) and not self.initialize():
            return False

        self._stop_event.clear()
        self._pause_event.clear()

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        for _ in range(50):
            if self.state == CaptureState.CAPTURING:
                return True
            if self.state == CaptureState.ERROR:
                return False
            time.sleep(0.1)

        return self.state == CaptureState.CAPTURING

    def stop_capture(self):
        self._stop_event.set()
        self._close_stream()
        if self._capture_thread and self._capture_thread.is_alive():
            with suppress(KeyboardInterrupt):
                self._capture_thread.join(timeout=0.5)
        if self._capture_thread and not self._capture_thread.is_alive():
            self._capture_thread = None

        self.state = CaptureState.CLOSED

    def pause(self):
        if self.state == CaptureState.CAPTURING:
            self._pause_event.set()
            self.state = CaptureState.PAUSED

    def resume(self):
        if self.state == CaptureState.PAUSED:
            self._pause_event.clear()
            self.state = CaptureState.CAPTURING

    def get_audio_chunk(self, timeout: float = 0.1) -> np.ndarray | None:
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_pre_roll_audio(self, seconds: float = 0.3) -> np.ndarray:
        n_samples = int(self.config.sample_rate * seconds)
        return self._ring_buffer.read_last(n_samples)

    def clear_buffer(self):
        self._ring_buffer.clear()
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def _capture_loop(self):
        while not self._stop_event.is_set():
            try:
                if self._pause_event.is_set():
                    time.sleep(0.01)
                    continue

                if self._backend_type == "sounddevice":
                    self._capture_sounddevice()
                else:
                    self._capture_pyaudio()

            except Exception as e:
                if self._stop_event.is_set():
                    return
                self._handle_error(f"Capture error: {e}")
                if self._retry_count < self._max_retries:
                    self._retry_count += 1
                    time.sleep(self._retry_delay * (2**self._retry_count))
                    self.state = CaptureState.OPENING
                else:
                    self._handle_error(f"Max retries ({self._max_retries}) exceeded")
                    return

    def _capture_sounddevice(self):
        import sounddevice as sd

        def callback(indata, frames, time_info, status):
            if status:
                pass

            if self._pause_event.is_set():
                return

            audio_data = indata[:, 0].astype(np.float32)
            self._ring_buffer.write(audio_data)

            with suppress(queue.Full):
                self._audio_queue.put_nowait(audio_data)

        try:
            with sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=np.float32,
                blocksize=self.config.chunk_size,
                device=self._device_index,
                callback=callback,
            ) as stream:
                with self._stream_lock:
                    self._stream = stream
                self.state = CaptureState.CAPTURING
                self._retry_count = 0

                while not self._stop_event.is_set():
                    if self._pause_event.is_set():
                        self.state = CaptureState.PAUSED
                    else:
                        self.state = CaptureState.CAPTURING
                    time.sleep(0.01)

        except Exception as e:
            raise e
        finally:
            with self._stream_lock:
                self._stream = None

    def _capture_pyaudio(self):
        stream = self._backend.open(
            format=8,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=self.config.chunk_size,
            input_device_index=self._device_index,
        )
        with self._stream_lock:
            self._stream = stream

        self.state = CaptureState.CAPTURING
        self._retry_count = 0

        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                self.state = CaptureState.PAUSED
                time.sleep(0.01)
                continue

            try:
                data = stream.read(self.config.chunk_size, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                self._ring_buffer.write(audio_data)

                with suppress(queue.Full):
                    self._audio_queue.put_nowait(audio_data)

            except Exception as e:
                raise e
        with self._stream_lock:
            if self._stream is stream:
                self._stream = None

    def _close_stream(self):
        with self._stream_lock:
            stream = self._stream
            self._stream = None

        if stream is None:
            return

        if self._backend_type == "pyaudio":
            with suppress(Exception):
                stream.stop_stream()
                stream.close()

    def _handle_error(self, message: str):
        self.state = CaptureState.ERROR
        if self._on_error_callback:
            self._on_error_callback(message)

    def cleanup(self):
        self.stop_capture()
        if self._backend_type == "pyaudio" and self._backend:
            with suppress(Exception):
                self._backend.terminate()
        self._backend = None
