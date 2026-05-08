#!/usr/bin/env python3
"""Download and resolve default ASR model bundles for voice_asr_service."""

from __future__ import annotations

import argparse
import os
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_REALTIME_MODES = {"continuous", "wake_word"}


@dataclass(frozen=True)
class ModelBundle:
    profile: str
    directory: str
    url: str
    required_patterns: tuple[str, ...]
    model_path: str = ""
    tokens_path: str = "tokens.txt"

    def bundle_dir(self, model_root: Path) -> Path:
        return model_root / self.directory

    def resolved_model_path(self, model_root: Path) -> str:
        bundle_dir = self.bundle_dir(model_root)
        return str(bundle_dir / self.model_path) if self.model_path else str(bundle_dir)

    def resolved_tokens_path(self, model_root: Path) -> str:
        return str(self.bundle_dir(model_root) / self.tokens_path)

    def is_available(self, model_root: Path) -> bool:
        bundle_dir = self.bundle_dir(model_root)
        if not bundle_dir.exists():
            return False
        return all(any(bundle_dir.glob(pattern)) for pattern in self.required_patterns)


STREAMING_ZH_BUNDLE = ModelBundle(
    profile="streaming_zh",
    directory="sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23",
    url=(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
        "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23.tar.bz2"
    ),
    required_patterns=("tokens.txt", "encoder*.onnx", "decoder*.onnx", "joiner*.onnx"),
)

OFFLINE_ZH_BUNDLE = ModelBundle(
    profile="offline_zh",
    directory="sherpa-onnx-paraformer-zh-int8-2025-10-07",
    url=(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
        "sherpa-onnx-paraformer-zh-int8-2025-10-07.tar.bz2"
    ),
    required_patterns=("tokens.txt", "model*.onnx"),
    model_path="model.int8.onnx",
)

MODEL_BUNDLES = {
    STREAMING_ZH_BUNDLE.profile: STREAMING_ZH_BUNDLE,
    OFFLINE_ZH_BUNDLE.profile: OFFLINE_ZH_BUNDLE,
}


@dataclass(frozen=True)
class ResolvedModelAssets:
    model_path: str
    tokens_path: str
    downloaded: bool
    profile: str | None = None


def default_model_root() -> Path:
    return Path(__file__).resolve().parents[3] / "models" / "voice_asr"


def infer_model_bundle_from_path_hint(model_path: str) -> ModelBundle | None:
    """Infer a bundle purely from the configured path or directory name."""
    hint = model_path.lower()
    if "paraformer" in hint and "stream" not in hint:
        return OFFLINE_ZH_BUNDLE
    if any(keyword in hint for keyword in ("stream", "zipformer", "transducer", "encoder", "joiner")):
        return STREAMING_ZH_BUNDLE
    return None


def _log(logger, level: str, message: str) -> None:
    if logger is None:
        print(message)
        return
    log_fn = getattr(logger, level, None)
    if callable(log_fn):
        log_fn(message)
    else:
        print(message)


def _safe_extract_tar(archive_path: Path, target_dir: Path) -> None:
    target_dir = target_dir.resolve()
    with tarfile.open(archive_path, "r:bz2") as tar:
        for member in tar.getmembers():
            member_path = (target_dir / member.name).resolve()
            if os.path.commonpath([target_dir, member_path]) != str(target_dir):
                raise RuntimeError(f"Unsafe archive member path detected: {member.name}")
        extract_kwargs = {}
        if hasattr(tarfile, "data_filter"):
            extract_kwargs["filter"] = "data"
        # Python < 3.12 has no tarfile data filter. The commonpath guard above
        # remains the effective path-traversal protection on those runtimes.
        tar.extractall(target_dir, **extract_kwargs)


def infer_model_bundle(
    model_path: str,
    model_type: str = "auto",
    active_mode: str = "manual",
    language: str = "zh",
) -> ModelBundle:
    hinted_bundle = infer_model_bundle_from_path_hint(model_path)
    if hinted_bundle is not None:
        return hinted_bundle
    if model_type == "offline":
        return OFFLINE_ZH_BUNDLE
    if model_type == "streaming" or active_mode in _REALTIME_MODES:
        return STREAMING_ZH_BUNDLE
    if language.startswith("zh"):
        return OFFLINE_ZH_BUNDLE
    return STREAMING_ZH_BUNDLE


def download_model_bundle(
    bundle: ModelBundle,
    model_root: Path | None = None,
    logger=None,
) -> Path:
    model_root = model_root or default_model_root()
    model_root.mkdir(parents=True, exist_ok=True)

    if bundle.is_available(model_root):
        _log(logger, "info", f"ASR model bundle already present: {bundle.directory}")
        return bundle.bundle_dir(model_root)

    archive_path = model_root / f"{bundle.directory}.tar.bz2"
    _log(logger, "info", f"Downloading ASR model bundle '{bundle.profile}' from {bundle.url}")

    try:
        with urllib.request.urlopen(bundle.url, timeout=60) as response, archive_path.open("wb") as archive:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                archive.write(chunk)

        archive_size = archive_path.stat().st_size
        if archive_size < 1024:
            raise RuntimeError(
                f"Downloaded ASR bundle '{bundle.profile}' is too small "
                f"({archive_size} bytes); the download was likely interrupted."
            )

        _log(logger, "info", f"Extracting ASR model bundle to {model_root}")
        _safe_extract_tar(archive_path, model_root)
    finally:
        if archive_path.exists():
            archive_path.unlink()

    if not bundle.is_available(model_root):
        raise RuntimeError(
            f"Downloaded ASR bundle '{bundle.profile}' is incomplete under {bundle.bundle_dir(model_root)}"
        )

    _log(logger, "info", f"ASR model bundle ready: {bundle.bundle_dir(model_root)}")
    return bundle.bundle_dir(model_root)


def resolve_model_assets(
    model_path: str,
    tokens_path: str = "",
    model_type: str = "auto",
    active_mode: str = "manual",
    language: str = "zh",
    auto_download_model: bool = True,
    model_root: Path | None = None,
    logger=None,
) -> ResolvedModelAssets:
    if not model_path:
        if not auto_download_model:
            return ResolvedModelAssets(model_path="", tokens_path=tokens_path, downloaded=False, profile=None)

        bundle = infer_model_bundle(
            model_path="",
            model_type=model_type,
            active_mode=active_mode,
            language=language,
        )
        download_model_bundle(bundle=bundle, model_root=model_root, logger=logger)
        _log(
            logger,
            "info",
            f"ASR model_path is empty; using default bundle '{bundle.directory}'.",
        )
        return ResolvedModelAssets(
            model_path=bundle.resolved_model_path(model_root or default_model_root()),
            tokens_path=bundle.resolved_tokens_path(model_root or default_model_root()),
            downloaded=True,
            profile=bundle.profile,
        )

    requested_model_path = Path(model_path)
    requested_tokens_path = Path(tokens_path) if tokens_path else None
    if requested_model_path.exists():
        resolved_tokens = (
            str(requested_tokens_path)
            if requested_tokens_path and requested_tokens_path.exists()
            else str(requested_model_path.parent / "tokens.txt")
            if requested_model_path.is_file()
            else str(requested_model_path / "tokens.txt")
        )
        return ResolvedModelAssets(
            model_path=str(requested_model_path),
            tokens_path=resolved_tokens if Path(resolved_tokens).exists() else tokens_path,
            downloaded=False,
            profile=None,
        )

    if not auto_download_model:
        return ResolvedModelAssets(model_path=model_path, tokens_path=tokens_path, downloaded=False, profile=None)

    bundle = infer_model_bundle(
        model_path=model_path,
        model_type=model_type,
        active_mode=active_mode,
        language=language,
    )
    download_model_bundle(bundle=bundle, model_root=model_root, logger=logger)
    _log(
        logger,
        "info",
        f"Configured ASR model path '{model_path}' is missing; using downloaded bundle '{bundle.directory}' instead.",
    )
    return ResolvedModelAssets(
        model_path=bundle.resolved_model_path(model_root or default_model_root()),
        tokens_path=bundle.resolved_tokens_path(model_root or default_model_root()),
        downloaded=True,
        profile=bundle.profile,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download default voice ASR model bundles.")
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=sorted(MODEL_BUNDLES.keys()),
        default=[],
        help="Model bundle profiles to download.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all default model bundles.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    profiles = list(MODEL_BUNDLES.keys()) if args.all else args.profiles
    if not profiles:
        return 0

    model_root = default_model_root()
    for profile in profiles:
        download_model_bundle(MODEL_BUNDLES[profile], model_root=model_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
