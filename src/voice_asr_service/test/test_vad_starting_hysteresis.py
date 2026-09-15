"""VAD STARTING 状态迟滞回归测试。

噪声环境下 Silero 置信度会在自适应阈值附近抖动；STARTING 状态必须允许
短暂（≤3 帧）回落，否则单帧丢失会重置整个语音起点检测，导致 ASR 只拿到
碎片音频后输出空结果。
"""

from voice_asr_service.vad_module import VADConfig, VADModule, VADState


def _gate_ok(vad: VADModule) -> bool:
    energy = 0.05
    gate = vad._get_energy_gate()
    return energy >= gate


def test_starting_survives_single_frame_miss():
    vad = VADModule(VADConfig())
    threshold = 0.5

    # 首帧命中 → STARTING
    assert vad._update_state(0.9, 0.05, threshold).state is VADState.STARTING
    # 连续命中累积
    for _ in range(4):
        vad._update_state(0.9, 0.05, threshold)
    # 单帧丢失（置信度抖动）不应重置到 SILENCE
    result = vad._update_state(0.2, 0.05, threshold)
    assert result.state is VADState.STARTING
    # 恢复命中后继续累积并进入 SPEAKING
    for _ in range(6):
        result = vad._update_state(0.9, 0.05, threshold)
    assert result.state is VADState.SPEAKING


def test_starting_resets_after_sustained_misses():
    vad = VADModule(VADConfig())
    threshold = 0.5

    vad._update_state(0.9, 0.05, threshold)
    # 连续 3 帧丢失才回 SILENCE
    vad._update_state(0.2, 0.05, threshold)
    vad._update_state(0.2, 0.05, threshold)
    assert vad._update_state(0.2, 0.05, threshold).state is VADState.SILENCE


def test_reset_clears_starting_hysteresis():
    vad = VADModule(VADConfig())
    threshold = 0.5

    vad._update_state(0.9, 0.05, threshold)
    vad._update_state(0.2, 0.05, threshold)
    vad.reset()
    assert vad._starting_miss_frames == 0
    assert vad.state is VADState.SILENCE
