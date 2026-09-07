"""speech_direction 配置。

speech_direction 链路必需参数。默认值与算法基线对齐,保证算法精度可复现。

模型路径默认用绝对路径(基于本文件位置推算工作区根),避免相对路径在
不同 cwd(pytest / ros2 launch / 命令行)下找不到模型文件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# 工作区根目录(voice_asr_service/speech_direction/config.py 向上 4 级到 IB_Robot/)
_WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
_MODELS_ROOT = _WORKSPACE_ROOT / "models"


def _model_path(rel: str) -> str:
    """把相对模型路径解析为绝对路径(基于工作区 models/ 根)。"""
    return str(_MODELS_ROOT / rel)


@dataclass
class AudioConfig:
    """音频采集参数(硬件固定值,不可更改)。"""

    sample_rate: int = 16000  # 硬件固定
    channels: int = 6  # 6_channels_firmware
    chunk_size: int = 160  # 10ms @ 16kHz
    sample_format: str = "int16"
    # 0-based 设备通道索引:[1,2,3,4] 跳过 ch0(DSP)与 ch5(playback)
    channel_indices: list[int] = field(default_factory=lambda: [1, 2, 3, 4])


@dataclass
class PipelineConfig:
    """统一流式 pipeline 的显式时序参数。"""

    sample_rate: int = 16000
    # runtime 每 256 samples 调度一次；两个 tick 聚合为一次 T=2 模型调用。
    processing_hop_samples: int = 256
    model_batch_samples: int = 512
    srp_update_interval_hops: int = 2
    frame_size: int = 4096  # 兼容旧调用方，语义等同 srp_frame_samples
    hop_size: int = 512  # 兼容旧调用方，语义等同 srp_hop_samples，禁止作为 processing tick
    fft_size: int = 4096
    window: str = "hann"
    input_channels: list[int] = field(default_factory=lambda: [1, 2, 3, 4])


@dataclass
class FullSubNetConfig:
    """FullSubNet 4ch 增强参数。"""

    # 独立 fullsubnet bundle（models/fullsubnet）是唯一资产来源；
    # cumulative stateful checkpoint 与 310P OM 同权重同 bundle。
    ckpt: str = field(default_factory=lambda: _model_path("fullsubnet/assets/cum_fullsubnet_best_model_218epochs.tar"))
    # cumulative stateful T=2 拆分 OM；每次处理512 samples并显式续传FB/SB h/c。
    stateful_fb_om_path: str = field(
        default_factory=lambda: _model_path(
            "fullsubnet/artifacts/ascend/fullsubnet/fullsubnet_cum_stateful_fb_b4_t2_fp16.om"
        )
    )
    stateful_sb_om_path: str = field(
        default_factory=lambda: _model_path(
            "fullsubnet/artifacts/ascend/fullsubnet/fullsubnet_cum_stateful_sb_b4_t2_fp16.om"
        )
    )
    stateful_manifest_path: str = field(
        default_factory=lambda: _model_path("fullsubnet/assets/cum_fullsubnet_best_model_218epochs.manifest.json")
    )
    inference_bundle: str = field(default_factory=lambda: _model_path("fullsubnet"))
    device_id: int = 0
    device: str = "cuda"  # Ubuntu stateful Torch 固定 CUDA；禁止静默回退 CPU
    # ACL is the backend identity; statefulness is selected by the streaming execution path.
    backend: str = "ascend"
    deployment: str = "ascend_310p"
    num_freqs: int = 257
    n_fft: int = 512  # FullSubNet 内部 STFT(固定)
    hop: int = 256
    win: int = 512


@dataclass
class VadConfig:
    """Silero VAD 参数。"""

    # Silero VAD OM 在 310P 使用 Ascend ACL；ONNX 用于 Ubuntu 的相同门控流程。
    # 默认指向独立 bundle 的 310P 部署工件；实际路径由 _apply_bundle_artifacts 推导。
    model_path: str = field(
        default_factory=lambda: _model_path("silero-vad/artifacts/ascend/ascend_310p/silero_vad_v6_310p_mixed16.om")
    )
    sample_rate: int = 16000
    frame_size: int = 512  # Silero 子帧 = 32ms @ 16kHz
    input_source: str = "enh_mic1_mono"  # 增强 ch1 单麦
    # 推理后端:ascend(Ascend NPU,默认) 或 onnx(Ubuntu)
    backend: str = "ascend"
    deployment: str = "ascend_310p"
    # 独立 Silero VAD bundle(models/silero-vad):多业务共享的同源部署,唯一解析来源。
    inference_bundle: str = field(default_factory=lambda: _model_path("silero-vad"))


@dataclass
class GrayRegionConfig:
    """灰区切段参数(灰区切段逻辑与算法基线一致)。"""

    vad_threshold: float = 0.65  # 灰区 VAD 门限(增强 ch1 的 Silero 概率)
    rms_threshold: float = 0.002  # 灰区 RMS 门限(增强 ch1 RMS,真语音段约 0.02 量级)
    seg_end_gap_s: float = 0.15  # 非灰区持续多久判段结束(等价算法基线 merge_gap)
    min_seg_dur_s: float = 0.10  # 最短段时长(短于此丢弃)
    min_accum_frames: int = 3  # 3×512=96ms；保留已验证真实音频中的短语音段
    max_accum_dur_s: float = 1.0  # 长段中间输出阈值(累积到此先出一次 DOA 再清缓冲)
    seg_max_rms_threshold: float = 0.005  # 段内子帧 max RMS 门限(<此值丢弃,真语音段 max>0.025)


@dataclass
class DoaConfig:
    """保相位 STFT-SRP-PHAT 参数。"""

    sample_rate: int = 16000
    input_channels: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    frame_size: int = 4096
    hop_size: int = 512
    fft_size: int = 4096
    freq_band_hz: tuple[float, float] = (300.0, 3400.0)  # 边相邻对频段
    diag_pair_freq_max_hz: float = 2600.0  # 对角对上限(防空间混叠栅瓣)
    angle_step_degree: int = 5  # 5° 步长 → 72 角度
    sound_speed: float = 343.0
    # 麦克风物理坐标 (4, 2) 米,顺序对应 ch1~4(配置驱动,传入 srp_phat)
    mic_positions: list[list[float]] = field(
        default_factory=lambda: [
            [0.02285, 0.02285],  # Mic1/ch1 右上
            [-0.02285, 0.02285],  # Mic2/ch2 左上
            [-0.02285, -0.02285],  # Mic3/ch3 左下
            [0.02285, -0.02285],  # Mic4/ch4 右下
        ]
    )
    max_age_ms: int = 1300  # 1s段累计+状态机与look-ahead余量


@dataclass
class DiagnosticsConfig:
    """高通量离线维测参数。"""

    high_throughput_enabled: bool = False  # 默认关闭，避免部署即写盘到磁盘满
    rollover_seconds: int = 300
    save_raw6ch: bool = True
    save_enh4ch: bool = True
    save_frame_metrics: bool = True
    save_gray_events: bool = True
    queue_size: int = 128
    drop_when_full: bool = True
    # FullSubNet 分阶段 timing 会增加热路径计时与低频日志，生产默认关闭。
    fullsubnet_timing_enabled: bool = False


@dataclass
class SpeechDirectionConfig:
    """speech_direction 完整配置。"""

    audio: AudioConfig = field(default_factory=AudioConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    fullnet: FullSubNetConfig = field(default_factory=FullSubNetConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    gray_region: GrayRegionConfig = field(default_factory=GrayRegionConfig)
    doa: DoaConfig = field(default_factory=DoaConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)

    # 运行时控制
    audio_topic: str = "/audio/capture_stamped"
    mount_yaw_deg: float = 0.0  # 阵列安装偏角(度)
    speech_direction_max_age_ms: int = 1300


__all__ = [
    "AudioConfig",
    "PipelineConfig",
    "FullSubNetConfig",
    "VadConfig",
    "GrayRegionConfig",
    "DoaConfig",
    "DiagnosticsConfig",
    "SpeechDirectionConfig",
]
