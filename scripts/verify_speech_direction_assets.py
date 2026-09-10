#!/usr/bin/env python3
"""校验 speech_direction 独立 bundle 的模型资产（纯校验，不下载）。

走两条校验：
1. 标准 inference_manifest.json —— 用 load_inference_manifest_metadata 校验
   bundle 结构、deployment bindings/execution 与 semantic_identity，不校验
   文件存在。
2. manifest deployment artifacts —— 逐资产校验文件存在性和 SHA-256，确保
   下载或从 NAS 手动获取的资产没有损坏或被篡改。

资产不在本仓库管理；缺失的资产会打印提示并跳过（不终止），已存在的资产
校验不通过则报错。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# 脚本从源码树运行时，把两个包源码根加入路径：inference_manifest（标准加载入口）
# 与 voice_asr_service（契约 SSOT STATEFUL_FULLSUBNET_CONTRACT）。
_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
_INFERENCE_MANIFEST_ROOT = _WORKSPACE_ROOT / "src" / "inference_manifest"
for _root in (_INFERENCE_MANIFEST_ROOT,):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from inference_manifest import load_inference_manifest_metadata  # noqa: E402

# 独立 bundle 名单。每个 bundle 支持哪些 deployment 以其 inference_manifest.json
# 的 deployments 字典为唯一事实来源（SSOT）：新 deployment 自动纳入校验，
# 无需在本脚本维护平行清单。
_BUNDLES: tuple[str, ...] = ("silero-vad", "fullsubnet")

_NAS_HINT = (
    "优先统一入口: python3 scripts/download_models.py --models <bundle>（HF openEuler org 已发布完整 bundle）；"
    "离线制作: 310P OM 从 NAS 获取、310B OM 由 models/_work/{fullsubnet,silero-vad} 导出流程生成、"
    "Ubuntu 依赖用 scripts/download_speech_direction_models.sh"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, expected: str) -> None:
    """校验通过静默返回，不通过抛异常（用于已存在的资产）。"""
    if not path.is_file():
        raise FileNotFoundError(f"模型文件不存在: {path}")
    actual_sha = _sha256(path)
    if actual_sha != expected:
        raise ValueError(f"{path}: sha256={actual_sha},期望={expected}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-root",
        type=Path,
        default=_WORKSPACE_ROOT / "models",
        help="模型根目录（含 silero-vad/ 与 fullsubnet/ 独立 bundle）",
    )
    parser.add_argument("--bundle", choices=list(_BUNDLES), default=None, help="只校验该 bundle")
    parser.add_argument("--deployment", default=None, help="只校验该 deployment 的资产（默认校验全部 deployment）")
    args = parser.parse_args()

    models_root = args.models_root.resolve()
    missing = False
    for bundle_name in _BUNDLES:
        if args.bundle and bundle_name != args.bundle:
            continue
        bundle_dir = models_root / bundle_name
        manifest_path = bundle_dir / "inference_manifest.json"
        if not manifest_path.is_file():
            print(f"[missing] bundle {bundle_name}: {manifest_path}")
            print(f"           {_NAS_HINT}")
            missing = True
            continue
        # deployment 清单以 manifest 为唯一事实来源，顺序保持 manifest 写入顺序。
        deployment_names = tuple(json.loads(manifest_path.read_text(encoding="utf-8")).get("deployments", {}))
        if not deployment_names:
            print(f"[error] bundle {bundle_name}: manifest declares no deployments")
            missing = True
            continue
        if args.deployment and args.deployment not in deployment_names:
            print(
                f"[error] bundle {bundle_name}: requested deployment {args.deployment!r} "
                f"not in manifest deployments {list(deployment_names)}"
            )
            missing = True
            continue
        loaded = []
        for dep_name in deployment_names:
            if args.deployment and dep_name != args.deployment:
                continue
            vm = load_inference_manifest_metadata(bundle_dir, dep_name)
            loaded.append((dep_name, vm))
            print(f"[manifest] {bundle_name}/{dep_name} 结构校验 OK (fingerprint={vm.fingerprint[:16]}...)")

        assets: dict[Path, str] = {}
        for dep_name, vm in loaded:
            for role, artifact in vm.deployment.artifacts.items():
                if artifact.sha256 is None:
                    raise ValueError(f"{bundle_name}/{dep_name}/{role} is missing artifact sha256")
                assets[bundle_dir / artifact.path] = artifact.sha256
        for target, expected_sha in sorted(assets.items()):
            desc = str(target.relative_to(bundle_dir))
            if not target.is_file():
                print(f"[missing] {bundle_name}/{desc}: {target}")
                print(f"           {_NAS_HINT}")
                missing = True
                continue
            _verify(target, expected_sha)
            print(f"[ok] {bundle_name}/{desc} SHA-256 OK")

    if missing:
        print("\n存在缺失资产或 deployment 选择未命中，请补齐后重新校验。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
