#!/usr/bin/env python3
"""speech_direction 模型校验。

对齐 perception_service/grounded_sam2_wrapper.py 的风格:节点启动时校验模型,
缺失则抛 FileNotFoundError 并提示运行下载脚本,节点直接退出(不降级保持运行)。

生产路径使用 require_configured_models():从 speech_direction.yaml 读取实际模型路径
校验,与平台 profile 一致。下方的旧常量/resolve_assets()/require_models() 仅供 CLI
自检与兼容入口,不再代表生产资产清单。

生产资产(均不入库,放 models/ 下,通过 python3 scripts/verify_speech_direction_assets.py 校验):
   1. Silero VAD OM/ONNX — models/silero-vad 独立 bundle
   2. FullSubNet cumulative ckpt/OM — models/fullsubnet 独立 bundle

FullSubNet 源码(model.py + audio_zen)已打包为 ibrobot-fullsubnet wheel,
通过 scripts/setup.sh 安装到 venv;不再需要 git clone 上游源码仓到 models/ 下。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# 工作区根目录(voice_asr_service/speech_direction/model_downloader.py 向上 5 级到 IB_Robot/)
_WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_MODELS_ROOT = _WORKSPACE_ROOT / "models"

# Silero VAD(与 voice_asr_service 共用目录);310P 用 OM,Ubuntu 用 ONNX,具体文件名由 yaml 指定。
SILERO_VAD_REL = "silero-vad/assets/silero_vad.onnx"

# FullSubNet cumulative ckpt(218epochs,两平台共用同一权重)
FULLSUBNET_CKPT_REL = "fullsubnet/assets/cum_fullsubnet_best_model_218epochs.tar"


@dataclass(frozen=True)
class ModelAsset:
    """单个模型资产描述。"""

    name: str  # 资产名(展示用)
    path: Path  # 绝对路径
    required_for: str  # 用途说明

    def exists(self) -> bool:
        return self.path.is_file()


@dataclass(frozen=True)
class ModelStatus:
    """模型校验结果。"""

    silero_vad: ModelAsset
    fullsubnet_ckpt: ModelAsset

    def missing(self) -> list[ModelAsset]:
        """返回缺失的资产列表。"""
        return [a for a in (self.silero_vad, self.fullsubnet_ckpt) if not a.exists()]

    def all_present(self) -> bool:
        return not self.missing()


def resolve_assets(models_root: Path | None = None) -> ModelStatus:
    """解析所有模型资产路径。

    Args:
        models_root: 模型根目录(默认 <workspace>/models)

    Returns:
        ModelStatus: 各资产的存在性可通过 .exists() 检查
    """
    root = models_root or _DEFAULT_MODELS_ROOT
    return ModelStatus(
        silero_vad=ModelAsset(
            name="Silero VAD",
            path=root / SILERO_VAD_REL,
            required_for="人声门控(Silero VAD 推理)",
        ),
        fullsubnet_ckpt=ModelAsset(
            name="FullSubNet ckpt",
            path=root / FULLSUBNET_CKPT_REL,
            required_for="语音增强(模型权重,~67MB)",
        ),
    )


def _raise_missing(status: ModelStatus, location_summary: str) -> None:
    """统一报告缺失资产，确保默认布局与显式路径使用相同错误语义。"""
    missing = status.missing()
    if not missing:
        return

    detail = "\n".join(f"{asset.name} not found: {asset.path}" for asset in missing)
    raise FileNotFoundError(
        f"speech_direction 模型缺失(共 {len(missing)} 项):\n{detail}\n"
        "Run: python3 scripts/verify_speech_direction_assets.py（仅校验；310P 资产需从 NAS 手动获取）\n"
        f"模型位置: {location_summary}"
    )


def require_configured_models(
    silero_vad_path: str | Path,
    fullsubnet_ckpt_path: str | Path,
    *,
    silero_backend: str = "onnx",
    fullsubnet_backend: str = "torch",
    fullsubnet_stateful_fb_om_path: str | Path = "",
    fullsubnet_stateful_sb_om_path: str | Path = "",
    fullsubnet_stateful_manifest_path: str | Path = "",
) -> None:
    """校验 ROS 参数指定的模型资产路径。

    ascend 后端(310P1 部署形态):
      - silero 校验 silero_vad_path 指向的 .om 文件
      - fullsubnet 校验 FB/SB/manifest 三件套
    torch/onnx 后端(回归基线):
      - 校验 silero .onnx、fullsubnet ckpt;Model 类由 ibrobot-fullsubnet wheel 提供
    """
    assets: list[ModelAsset] = []

    if silero_backend not in {"ascend", "onnx"}:
        raise ValueError(f"不支持的 Silero VAD backend: {silero_backend}")
    if fullsubnet_backend not in {"ascend", "stateful_torch_cuda", "stateful_torch_cpu", "torch"}:
        raise ValueError(f"不支持的 FullSubNet backend: {fullsubnet_backend}")

    # Silero VAD: Ascend ACL 校验 .om，onnx 校验 .onnx。
    silero_path = Path(silero_vad_path)
    if silero_backend == "ascend":
        assets.append(
            ModelAsset(
                name="Silero VAD om",
                path=silero_path,
                required_for="人声门控(Silero VAD om 推理,NPU)",
            )
        )
    else:
        assets.append(
            ModelAsset(
                name="Silero VAD ONNX",
                path=silero_path,
                required_for="人声门控(Silero VAD onnx 推理,CPU 基线)",
            )
        )

    # FullSubNet: Ascend ACL 校验 FB/SB/manifest；stateful Torch 校验 ckpt/manifest。
    if fullsubnet_backend == "ascend":
        for name, path, purpose in (
            ("FullSubNet cumulative FB Ascend artifact", fullsubnet_stateful_fb_om_path, "FB stateful推理"),
            ("FullSubNet cumulative SB Ascend artifact", fullsubnet_stateful_sb_om_path, "SB stateful推理"),
            ("FullSubNet cumulative manifest", fullsubnet_stateful_manifest_path, "checkpoint/norm契约校验"),
        ):
            if not path:
                raise ValueError(f"fullsubnet Ascend ACL 后端未提供{name}")
            assets.append(ModelAsset(name=name, path=Path(path), required_for=purpose))
    else:
        # stateful Torch 与 legacy Torch 校验 ckpt;Model 类来自 wheel 包。
        assets.append(
            ModelAsset(
                name="FullSubNet ckpt",
                path=Path(fullsubnet_ckpt_path),
                required_for="语音增强(模型权重)",
            )
        )
        if fullsubnet_backend in {"stateful_torch_cuda", "stateful_torch_cpu"}:
            if not fullsubnet_stateful_manifest_path:
                raise ValueError("stateful Torch 后端未提供 cumulative manifest")
            assets.append(
                ModelAsset(
                    name="FullSubNet cumulative manifest",
                    path=Path(fullsubnet_stateful_manifest_path),
                    required_for="checkpoint/norm契约校验",
                )
            )

    missing = [a for a in assets if not a.exists()]
    if missing:
        detail = "\n".join(f"{a.name} not found: {a.path}" for a in missing)
        raise FileNotFoundError(
            f"speech_direction 模型缺失(共 {len(missing)} 项):\n{detail}\n"
            "Run: python3 scripts/verify_speech_direction_assets.py（仅校验；310P 资产需从 NAS 手动获取）\n"
            "模型位置: 来自 speech_direction ROS 参数"
        )


def require_models(models_root: Path | None = None) -> None:
    """校验默认下载布局,保留 CLI 与既有调用兼容性。

    与 perception_service/grounded_sam2_wrapper.py 一致:模型缺失时抛
    FileNotFoundError 并提示运行下载脚本,节点直接退出,不降级保持运行。

    Args:
        models_root: 模型根目录(默认 <workspace>/models)

    Raises:
        FileNotFoundError: 任一模型缺失,消息含缺失项路径与下载脚本提示
    """
    root = models_root or _DEFAULT_MODELS_ROOT
    _raise_missing(resolve_assets(models_root), str(root))


def main() -> int:
    """命令行入口:检查模型状态并打印指引。"""
    import argparse

    parser = argparse.ArgumentParser(description="Check speech_direction model assets.")
    parser.add_argument(
        "--models-root",
        type=Path,
        default=_DEFAULT_MODELS_ROOT,
        help=f"Models root directory (default: {_DEFAULT_MODELS_ROOT})",
    )
    args = parser.parse_args()

    status = resolve_assets(args.models_root)
    print(f"Models root: {args.models_root}")
    print()
    for a in (status.silero_vad, status.fullsubnet_ckpt):
        mark = "OK" if a.exists() else "MISSING"
        print(f"  [{mark}] {a.name}")
        print(f"       path: {a.path}")
        print(f"       use: {a.required_for}")
        print()

    if status.all_present():
        print("All models present.")
        return 0

    print("Missing models detected. Run:")
    print("  python3 scripts/verify_speech_direction_assets.py（仅校验；310P 资产需从 NAS 手动获取）")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
