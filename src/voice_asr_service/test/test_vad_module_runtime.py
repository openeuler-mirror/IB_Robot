from __future__ import annotations

import numpy as np
import pytest

from voice_asr_service.vad_module import VADConfig, VADModule


class _FakeVadRuntime:
    def __init__(self, probability: float = 0.9) -> None:
        self.probability = probability
        self.infer_calls = 0
        self.reset_calls = 0
        self.close_calls = 0

    def infer(self, audio_frame: np.ndarray) -> float:
        self.infer_calls += 1
        assert audio_frame.shape == (512,)
        return self.probability

    def reset(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def test_vad_requires_manifest_runtime_and_forwards_state_controls() -> None:
    module = VADModule(VADConfig(sample_rate=16000, frame_size=512))
    with pytest.raises(RuntimeError, match="manifest-backed"):
        module.initialize()

    runtime = _FakeVadRuntime()
    module.set_runtime(runtime)
    assert module.initialize()
    result = module.process(np.zeros(512, dtype=np.float32))
    assert result.confidence == pytest.approx(0.9)
    assert runtime.infer_calls == 1
    module.reset()
    module.close()
    assert runtime.reset_calls == 1
    assert runtime.close_calls == 1


def test_vad_rejects_non_contract_frame_without_fallback() -> None:
    module = VADModule(VADConfig(sample_rate=16000, frame_size=512))
    module.set_runtime(_FakeVadRuntime())
    module.initialize()
    with pytest.raises(ValueError, match="512 samples"):
        module.process(np.zeros(256, dtype=np.float32))
