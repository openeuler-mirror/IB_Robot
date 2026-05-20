"""Shared helpers for Ascend OM PolicyWrapper adapters (ACT, PI05, ...).

Keeps a single source of truth for config loading, OM file-path resolution,
device-name normalization, and tensor coercion.  Adapter modules
(``policy_wrapper.py`` and ``pi05/policy_wrapper.py``) compose these helpers
to express their backend-specific layouts without copy/pasting plumbing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Device name
# ---------------------------------------------------------------------------


def normalize_device_name(device: str) -> str:
    return str(device).lower().strip().replace("-", "_")


# ---------------------------------------------------------------------------
# config.json loading
# ---------------------------------------------------------------------------


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def policy_config_path(path: str) -> Path | None:
    candidate = Path(path).expanduser()
    if candidate.is_file() and candidate.name == "config.json":
        return candidate
    if candidate.is_dir():
        cfg = candidate / "config.json"
        if cfg.is_file():
            return cfg
    return None


def load_policy_config(path: str) -> dict[str, Any]:
    cfg_path = policy_config_path(path)
    return read_json(cfg_path) if cfg_path is not None else {}


def policy_type_from_path(path: str) -> str:
    """Best-effort lookup of the ``type`` field in a policy ``config.json``."""
    cfg_path = policy_config_path(path)
    if cfg_path is None:
        return ""
    try:
        return str(read_json(cfg_path).get("type", "")).lower()
    except Exception:
        return ""


def chunk_size_from_config(config: dict[str, Any]) -> int:
    for key in ("chunk_size", "n_action_steps", "action_chunk_size"):
        value = config.get(key)
        if value is not None:
            return int(value)
    return 1


# ---------------------------------------------------------------------------
# OM file-path resolution (env > config.json > directory conventions)
# ---------------------------------------------------------------------------


def candidate_paths(
    path: str,
    config: dict[str, Any],
    *,
    env_names: tuple[str, ...],
    config_key: str,
    basename_candidates: tuple[str, ...] = (),
    extra_dir_globs: tuple[str, ...] = (),
) -> list[Path]:
    """Collect candidate paths in priority order.

    Args:
        path: The policy directory or file path provided by the caller.
        config: Parsed ``config.json`` of the policy.
        env_names: Environment variables consulted first.
        config_key: Field in ``config.json`` consulted next.
        basename_candidates: File basenames searched under ``<path>`` and
            ``<path>/model``.
        extra_dir_globs: Glob patterns evaluated under ``<path>`` and
            ``<path>/model`` (e.g. ``("*.om",)``).
    """
    raw_path = Path(path).expanduser()
    candidates: list[Path] = []

    for env_name in env_names:
        env_val = os.environ.get(env_name, "").strip()
        if env_val:
            candidates.append(Path(env_val).expanduser())

    cfg_val = str(config.get(config_key) or "").strip()
    if cfg_val:
        cfg_path = Path(cfg_val).expanduser()
        candidates.append(cfg_path)
        if not cfg_path.is_absolute() and raw_path.is_dir():
            candidates.append(raw_path / cfg_path)

    if raw_path.is_dir():
        for basename in basename_candidates:
            candidates.append(raw_path / basename)
            candidates.append(raw_path / "model" / basename)
        for pattern in extra_dir_globs:
            candidates.extend(sorted(raw_path.glob(pattern)))
            model_dir = raw_path / "model"
            if model_dir.is_dir():
                candidates.extend(sorted(model_dir.glob(pattern)))

    return candidates


def resolve_first_existing(
    candidates: list[Path],
    description: str,
    *,
    predicate=None,
) -> Path:
    """Return the first candidate whose file exists (and matches ``predicate``)."""
    checked: list[str] = []
    for candidate in candidates:
        candidate = candidate.expanduser()
        checked.append(str(candidate))
        if not candidate.is_file():
            continue
        if predicate is not None and not predicate(candidate):
            continue
        return candidate.resolve()
    raise FileNotFoundError(f"{description} not found. Checked: " + ", ".join(checked))


# ---------------------------------------------------------------------------
# Output tensor coercion
# ---------------------------------------------------------------------------


def as_action_tensor(output: Any, device: torch.device) -> Tensor:
    """Coerce a wrapper output into a ``(chunk_size, action_dim)`` tensor on
    the requested device.

    Accepts ``Tensor``, ``np.ndarray`` or anything ``torch.as_tensor`` can
    handle.  Drops a singleton leading batch dimension when present so the
    result matches LeRobot's ``predict_action_chunk`` convention.
    """
    tensor = output if isinstance(output, Tensor) else torch.as_tensor(output)
    if tensor.ndim >= 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor.to(device)
