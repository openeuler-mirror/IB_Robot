"""离线 WAV 回归测试:6 个真实录音 → pipeline 段级 DOA → 比对 GT,误差 ≤15°。

回归基线:
  - 6 个 6ch/16kHz 真实录音,文件名含 GT 角度
  - 通过 runtime.feed_audio 灌入(离线模式,无采集设备)
  - 段级 DOA 输出与 GT 比对,圆周误差 ≤15°(5° 步长量化内)

GT:
  sound_16s_180.flac        → 180° 单段
  sound_10s_90.flac         → 90°  单段
  sound_18s_0_0.flac        → 0°, 0° 两段
  sound_10s_90_0.flac       → 90°, 0° 两段
  sound_9s_0_90.flac        → 0°, 90° 两段(第3段碎段无GT,不统计)
  sound_19s_90_90_90_90.flac → 90°×4 四段

音频文件以 FLAC 无损压缩随仓库提供(见 audio/ 目录,合计约 6MB),
soundfile 原生支持 FLAC 读取,DOA 结果与原始 PCM 逐位一致。
环境变量 SPEECH_DIRECTION_AUDIO_DIR 可覆盖默认音频目录(供外部夹具调试)。

基线证据见 audio/BASELINE.md(6 文件 12 段,每段误差均 ≤ 15°)。

运行:
  # 默认读 audio/ 目录;模型缺失或音频缺失时整体 skip
  python -m pytest test_offline_regression.py -v -s
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from voice_asr_service.speech_direction.config import SpeechDirectionConfig  # noqa: E402
from voice_asr_service.speech_direction.doa.srp_phat import StftSrpPhat  # noqa: E402
from voice_asr_service.speech_direction.enhancement.factory import (  # noqa: E402
    build_stateful_fullsubnet,
)
from voice_asr_service.speech_direction.pipeline import (  # noqa: E402
    DoaState,
    VadState,
)
from voice_asr_service.speech_direction.pipeline_streaming import (  # noqa: E402
    StreamingPipelineParams,
    StreamingSpeechDirectionPipeline,
)
from voice_asr_service.speech_direction.runtime import (  # noqa: E402
    SpeechDirectionRuntime,
)
from voice_asr_service.speech_direction.speech_gate import SpeechGate  # noqa: E402

# ============================ 测试音频与 GT ============================

# 文件 → GT 角度序列(按时间顺序,None=碎段不统计)
SOUND_FILES = [
    ("sound_16s_180.flac", [180]),
    ("sound_10s_90.flac", [90]),
    ("sound_18s_0_0.flac", [0, 0]),
    ("sound_10s_90_0.flac", [90, 0]),
    ("sound_9s_0_90.flac", [0, 90, None]),
    ("sound_19s_90_90_90_90.flac", [90, 90, 90, 90]),
]

# 误差门限:max err 15°(5° 步长量化内)
ERR_THRESHOLD = 15.0

# 音频目录:默认为同目录 audio/(FLAC 夹具随仓库提供);
# 环境变量 SPEECH_DIRECTION_AUDIO_DIR 可覆盖,供外部夹具调试。
_audio_dir_env = os.environ.get("SPEECH_DIRECTION_AUDIO_DIR", "")
AUDIO_DIR = Path(_audio_dir_env) if _audio_dir_env else Path(__file__).with_name("audio")


def _circular_diff(a: float, b: float) -> float:
    """圆周绝对误差(度),结果 [0,180]。"""
    return float(abs(np.mod(a - b + 180, 360) - 180))


def _audio_available() -> bool:
    """检查测试音频是否可用(环境变量已指定且文件齐全)。"""
    if AUDIO_DIR is None:
        return False
    if not AUDIO_DIR.exists():
        return False
    return all((AUDIO_DIR / fname).exists() for fname, _ in SOUND_FILES)


# 真实录音类在资产不可用时整体 skip；资产无关契约仍应固定执行。
_REAL_AUDIO_REQUIRED = pytest.mark.skipif(
    not _audio_available(),
    reason="测试音频不可用:默认目录 audio/ 缺文件,或 SPEECH_DIRECTION_AUDIO_DIR 指向的目录缺文件",
)


def _build_pipeline():
    """构造与 310P 相同的 cumulative stateful + temporal gate + 新 SRP 主流程。"""
    cfg = SpeechDirectionConfig()
    # Ubuntu 仅替换模型执行后端，算法编排和 256/512/4096/512 时序不变。
    cfg.fullnet.backend = "stateful_torch_cuda"
    cfg.fullnet.device = "cuda"
    cfg.fullnet.ckpt = str(_SRC.parents[1] / "models/fullsubnet/assets/cum_fullsubnet_best_model_218epochs.tar")
    cfg.fullnet.stateful_manifest_path = str(
        _SRC.parents[1] / "models/fullsubnet/assets/cum_fullsubnet_best_model_218epochs.manifest.json"
    )
    cfg.vad.backend = "onnx"
    cfg.vad.model_path = str(_SRC.parents[1] / "models/silero-vad/assets/silero_vad.onnx")
    if not __import__("torch").cuda.is_available():
        pytest.skip("Ubuntu 主流程回归要求 CUDA，不允许静默回退 CPU")

    from voice_asr_service.speech_direction.model_downloader import require_configured_models

    try:
        require_configured_models(
            cfg.vad.model_path,
            cfg.fullnet.ckpt,
            silero_backend=cfg.vad.backend,
            fullsubnet_backend=cfg.fullnet.backend,
            fullsubnet_stateful_manifest_path=cfg.fullnet.stateful_manifest_path,
        )
    except FileNotFoundError as exc:
        pytest.skip(f"cumulative stateful 模型未就绪，跳过回归测试: {exc}")

    fullnet = build_stateful_fullsubnet(
        backend=cfg.fullnet.backend,
        checkpoint_path=cfg.fullnet.ckpt,
        manifest_path=cfg.fullnet.stateful_manifest_path,
        device=cfg.fullnet.device,
    )
    speech_gate = SpeechGate(
        model_path=cfg.vad.model_path,
        sample_rate=cfg.vad.sample_rate,
        vad_threshold=cfg.gray_region.vad_threshold,
        rms_threshold=cfg.gray_region.rms_threshold,
        backend=cfg.vad.backend,
    )
    angles = np.arange(0, 360, cfg.doa.angle_step_degree, dtype=np.float32)
    srp = StftSrpPhat(
        frame_size=cfg.doa.frame_size,
        hop_size=cfg.doa.hop_size,
        angles_deg=angles,
        sample_rate=cfg.doa.sample_rate,
        mic_positions=np.array(cfg.doa.mic_positions, dtype=np.float64),
        sound_speed=cfg.doa.sound_speed,
        freq_lo=cfg.doa.freq_band_hz[0],
        freq_hi=cfg.doa.freq_band_hz[1],
        diag_freq_hi=cfg.doa.diag_pair_freq_max_hz,
    )
    params = StreamingPipelineParams(
        sample_rate=cfg.pipeline.sample_rate,
        processing_samples=cfg.pipeline.processing_hop_samples,
        model_batch_samples=cfg.pipeline.model_batch_samples,
        srp_update_interval_hops=cfg.pipeline.srp_update_interval_hops,
        input_channels=tuple(cfg.pipeline.input_channels),
        srp_frame_samples=cfg.doa.frame_size,
        srp_hop_samples=cfg.doa.hop_size,
        candidate_window_samples=round(0.064 * cfg.pipeline.sample_rate),
        segment_end_gap_samples=round(cfg.gray_region.seg_end_gap_s * cfg.pipeline.sample_rate),
        min_segment_samples=round(cfg.gray_region.min_seg_dur_s * cfg.pipeline.sample_rate),
        min_accum_samples=cfg.gray_region.min_accum_frames * cfg.doa.hop_size,
        max_accum_samples=round(cfg.gray_region.max_accum_dur_s * cfg.pipeline.sample_rate),
        segment_max_rms_threshold=cfg.gray_region.seg_max_rms_threshold,
    )
    pipeline = StreamingSpeechDirectionPipeline(
        fullnet,
        speech_gate.silero,
        srp,
        params,
        VadState(),
        DoaState(),
        vad_threshold=cfg.gray_region.vad_threshold,
        rms_threshold=cfg.gray_region.rms_threshold,
    )
    runtime = SpeechDirectionRuntime(cfg, pipeline, offline=True)
    return runtime, pipeline


def _feed_wav(runtime: SpeechDirectionRuntime, wav_path: Path, chunk: int = 256) -> None:
    """把 6ch WAV 按统一 processing tick 灌入 runtime。

    每次写一个 256-sample tick 的 6ch int16 PCM，并按 16ms 真实节奏喂入，
    让 worker 不堆积且不会因测试端停顿形成错误的采样缺口。
    """
    audio, sr = sf.read(str(wav_path), always_2d=True, dtype="float32")
    if sr != 16000:
        raise ValueError(f"采样率 {sr} != 16000")
    if audio.shape[1] < 6:
        raise ValueError(f"通道数 {audio.shape[1]} < 6")
    int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    n = int16.shape[0]
    for s in range(0, n, chunk):
        block = int16[s : s + chunk]
        if block.shape[0] < chunk:
            block = np.pad(block, ((0, chunk - block.shape[0]), (0, 0)))
        runtime.feed_audio(block.tobytes())
        time.sleep(chunk / sr)


@_REAL_AUDIO_REQUIRED
class TestOfflineRegression:
    """离线 WAV 回归测试:6 个真实录音,段级 DOA 误差 ≤15°。"""

    @pytest.fixture(scope="class")
    def runtime(self):
        """构造算法链(所有测试共享,模型只加载一次)。"""
        rt, pipeline = _build_pipeline()
        rt.start()
        yield rt
        rt.stop()

    @pytest.mark.parametrize(
        "fname,gt_list",
        SOUND_FILES,
        ids=[f[0].replace(".flac", "") for f in SOUND_FILES],
    )
    def test_segment_doa_within_15deg(self, runtime, fname, gt_list):
        """单个音频:段级 DOA 输出与 GT 圆周误差 ≤15°。"""
        wav_path = AUDIO_DIR / fname
        if not wav_path.exists():
            pytest.skip(f"音频不存在: {wav_path}")

        # 重置 pipeline 状态(每个文件独立测,丢弃上个文件残留)
        runtime.reset()
        runtime.doa_state._angle = None
        runtime.doa_state._wall_clock_ts = 0.0
        runtime.doa_state._seq_id = 0
        # 重新注册 reader 从最新位置读
        runtime._reader_id = runtime.ring_buffer.register(start_latest=True)
        time.sleep(0.3)  # 给 worker 时间消化残留
        runtime._reader_id = runtime.ring_buffer.register(start_latest=True)

        # 灌入音频
        _feed_wav(runtime, wav_path)
        # 等 worker 处理完尾部(seg_end_gap=0.15s,多等一会)
        time.sleep(0.8)

        # 收集所有段级输出(中间方向 + 段末方向,按时间顺序)
        history = runtime.pipeline.get_segment_history()

        # 期望段数 = GT 中非 None 的个数
        expected_segs = [g for g in gt_list if g is not None]

        # 段数不足(漏检)直接失败
        if len(history) < len(expected_segs):
            pytest.fail(f"{fname}: 段数不足,期望 {len(expected_segs)} 实际 {len(history)}(漏检)")

        # 按时间顺序逐段与 GT 比对
        # history[i] 对应 gt_list[i],GT 为 None 的段(碎段)跳过不统计
        errors = []
        for i, (_output_seq, angle, _hop_t, seg_type) in enumerate(history):
            gt = gt_list[i] if i < len(gt_list) else None
            if gt is None:
                # GT 无此段(碎段或超出预期),跳过不统计
                continue
            err = _circular_diff(float(angle), float(gt))
            errors.append((fname, i, angle, gt, err, seg_type))

        # 所有段误差 ≤15°
        failed = [e for e in errors if e[4] > ERR_THRESHOLD]
        if failed:
            detail = "; ".join(f"段{e[1]}({e[5]}): DOA={e[2]}° GT={e[3]}° err={e[4]:.0f}°" for e in failed)
            pytest.fail(f"{fname}: {len(failed)}/{len(errors)} 段误差 >{ERR_THRESHOLD}° — {detail}")

        # 打印通过信息(调试用,-s 时可见)
        for fname_, seg_i, angle, gt, err, seg_type in errors:
            print(f"    {fname_} 段{seg_i}({seg_type}): DOA={angle}° GT={gt}° err={err:.0f}° [PASS]")

        # 时延回归:在真实生产路径(worker 线程→ringbuf→process_block)下,
        # 断言单次 model_batch(512 样本 = 32ms 预算)处理耗时满足实时性。
        # reset 在用例开头已清空 _block_latency_ms,此处仅本文件样本。
        latency_ms = runtime.pipeline.get_block_latency_ms()
        _assert_block_latency(fname, latency_ms)


def _assert_block_latency(fname: str, latency_ms: list[float]) -> None:
    """断言单次块处理时延满足 32ms 实时预算,打印分布供排查。

    生产链路每 512 样本(32ms)调度一次 fullnet+vad+srp,处理耗时须 < 32ms
    才能跟上实时回放;p99 与 max 均须达标,p99 反映稳态、max 反映最坏抖动。
    样本过少(极短音频)时不断言,只打印,避免统计无意义。
    """
    # block 预算:512 样本 / 16000 Hz = 32ms
    BLOCK_BUDGET_MS = 32.0
    MIN_SAMPLES = 10  # 样本过少时统计无意义,只打印不断言
    if not latency_ms:
        pytest.fail(f"{fname}: 未收集到块处理时延样本(process_block 未执行 fullnet)")
    arr = np.asarray(latency_ms, dtype=np.float64)
    p50 = float(np.percentile(arr, 50))
    p99 = float(np.percentile(arr, 99))
    mx = float(arr.max())
    print(f"    {fname} 时延: n={len(arr)} p50={p50:.2f}ms p99={p99:.2f}ms max={mx:.2f}ms (预算 {BLOCK_BUDGET_MS}ms)")
    if len(arr) < MIN_SAMPLES:
        return
    if p99 > BLOCK_BUDGET_MS:
        pytest.fail(f"{fname}: 块处理 p99={p99:.2f}ms > {BLOCK_BUDGET_MS}ms 预算,无法实时回放")
    if mx > BLOCK_BUDGET_MS:
        pytest.fail(f"{fname}: 块处理 max={mx:.2f}ms > {BLOCK_BUDGET_MS}ms 预算,最坏抖动超标")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
