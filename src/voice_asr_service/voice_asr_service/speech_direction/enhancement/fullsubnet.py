"""FullSubNet 4 通道独立增强。

算法逻辑与回归基线逐字对齐(经回归验证:5 文件 8 段 max err 15°)。

设计要点:
    1. 4ch 各自增强(实测必要):FullSubNet 对噪声频点压幅、对语音频点保幅,
       拉开信噪比,让段级能量加权累积能精准选中好帧。增强后真语音段 RMS 仍在 ~0.02 量级
       (decompress_cirm 对齐上游逆变换,不引入额外增益)。
    2. FullSubNet 内部 STFT(512 点 / hop 256 / Hann)→ 幅度谱 → 模型输出 cRM
       (2 通道:实部/虚部压缩域)→ decompress_cirm 还原 → 增强谱 Ŝ=X⊙cRM → istft 回时域。
    3. 输出长度严格等于输入(istft length 对齐)。
    4. DOA 用增强 4ch 的复数谱(重新 STFT 4096/Hann),互谱相位来自增强波形的通道间相对关系。
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# FullSubNet 内部 STFT 固定参数(与算法基线一致)
N_FFT = 512
HOP = 256
WIN = 512
NUM_FREQS = 257


def decompress_cirm(mask: torch.Tensor, K: float = 10.0, limit: float = 9.9) -> torch.Tensor:
    """还原压缩域 cRM 到真实复数比值 S/X。

    对齐 audio_zen/acoustics/mask.py 的 decompress_cIRM 逆变换:
        m = -K * log((K - y) / (K + y))
    压缩常数 C=0.1 仅用于正向 compress_cIRM,逆变换不出现除以 C。
    """
    y = torch.clamp(mask, -limit, limit)
    return -K * torch.log((K - y) / (K + y) + 1e-8)


class FullSubNetEnhancer:
    """FullSubNet 4 通道独立增强器。"""

    def __init__(
        self,
        ckpt: str,
        device: str = "cuda",
        n_fft: int = N_FFT,
        hop: int = HOP,
        win: int = WIN,
        num_freqs: int = NUM_FREQS,
    ):
        """
        Args:
            ckpt: models/fullsubnet/assets/cum_fullsubnet_best_model_218epochs.tar 路径
            device: cuda / cpu / auto(auto=优先 CUDA,无 CUDA 提示风险退 CPU)
            n_fft/hop/win: FullSubNet 内部 STFT 参数(固定 512/256/512)
            num_freqs: FullSubNet 频点数(257 = 512//2+1)
        """
        # 解析设备策略:auto → 优先 CUDA,无 CUDA 退 CPU 并提示风险
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("FullSubNet device=auto → 使用 CUDA(GPU)")
            else:
                device = "cpu"
                logger.warning(
                    "FullSubNet device=auto → 无 CUDA,退回 CPU。"
                    "CPU 上 4ch 增强可能超过 hop=128ms 预算导致丢帧/方向延迟,"
                    "建议在有 CUDA 的环境运行。"
                )
        self.device = device
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.num_freqs = num_freqs

        self._model = self._load_model(ckpt)
        # 预创建 Hann 窗(避免每帧重建)
        self._window = torch.hann_window(self.win, device=device)
        logger.info("FullSubNet 已加载: ckpt=%s device=%s", ckpt, device)

    def _load_model(self, ckpt: str):
        """加载 FullSubNet 模型(参数与算法基线一致)。"""
        from fullsubnet.model import Model

        m = Model(
            num_freqs=self.num_freqs,
            look_ahead=2,
            sequence_model="LSTM",
            fb_num_neighbors=0,
            sb_num_neighbors=15,
            fb_output_activate_function="ReLU",
            sb_output_activate_function=False,
            fb_model_hidden_size=512,
            sb_model_hidden_size=384,
            norm_type="offline_laplace_norm",
            num_groups_in_drop_band=2,
            weight_init=False,
        )
        ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        m.load_state_dict(ck["model"])
        m.eval()
        m.to(self.device)
        return m

    def enhance_one(self, wav_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """对单通道跑一次 FullSubNet,返回 (增强波形, cRM 复数 mask)。

        Args:
            wav_np: (T,) float32 [-1,1]
        Returns:
            enhanced: (T,) float32
            crm: (F=257, T_frames) complex64,含该通道自己的相位+幅度校正
        """
        wav_t = torch.from_numpy(wav_np).float().to(self.device)
        complex_stft = torch.stft(
            wav_t.unsqueeze(0),
            self.n_fft,
            self.hop,
            self.win,
            window=self._window,
            return_complex=True,
        )  # (1, F, T)
        mag = torch.abs(complex_stft)
        inp = mag.unsqueeze(1)  # (1, 1, F, T)
        with torch.no_grad():
            out = self._model(inp)  # (1, 2, F, T)
        crm = torch.complex(decompress_cirm(out[:, 0]), decompress_cirm(out[:, 1]))  # (1, F, T)
        enhanced_spec = complex_stft * crm  # X ⊙ cRM = Ŝ
        enhanced = (
            torch.istft(
                enhanced_spec,
                self.n_fft,
                self.hop,
                self.win,
                window=self._window,
                length=wav_np.shape[0],
            )
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        crm_np = crm.squeeze(0).cpu().numpy().astype(np.complex64)
        return enhanced, crm_np

    def enhance_4ch(self, audio4: np.ndarray) -> tuple[np.ndarray, list]:
        """4ch 各自 FullSubNet 增强,返回 (enh4, crms4)。

        Args:
            audio4: (T, 4) float32
        Returns:
            enh4: (T, 4) 增强后时域
            crms4: list of 4 个 (F=257, T_frm) 复数 cRM
        """
        enh4 = np.zeros_like(audio4, dtype=np.float32)
        crms4 = []
        for c in range(4):
            enh, crm = self.enhance_one(audio4[:, c].astype(np.float32))
            enh4[:, c] = enh
            crms4.append(crm)
        return enh4, crms4


__all__ = ["FullSubNetEnhancer", "decompress_cirm", "N_FFT", "HOP", "WIN", "NUM_FREQS"]
