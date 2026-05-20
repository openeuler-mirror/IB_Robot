"""Shared helper for loading lerobot ``PreTrainedConfig`` from on-disk policies.

The stock ``PreTrainedConfig.from_pretrained`` uses draccus to strictly decode
``config.json`` against the policy dataclass.  IB-Robot extends some policies'
``config.json`` with hardware-backend hints (Ascend OM / RKNN paths) that are
**not** part of the upstream dataclass and therefore make draccus raise
``DecodingError``.

This helper materializes a sanitized copy of ``config.json`` in a tempdir with
the IB-Robot-only keys removed, then defers to upstream ``from_pretrained``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

# Keys that IB-Robot writes into ``config.json`` but are not part of the
# upstream lerobot policy dataclasses.  Keep this list conservative; anything
# upstream may eventually adopt should be removed from here.
_IBROBOT_ONLY_KEYS: frozenset[str] = frozenset(
    {
        # Ascend OM backend hints
        "is_ascend_om_enabled",
        "om_model_path",
        "om_vlm_model_path",
        "om_action_expert_model_path",
        # RKNN backend hints
        "is_rknn_enabled",
        "rknn_model_path",
    }
)


def load_pretrained_policy_config(policy_path: str) -> Any:
    """Load a ``PreTrainedConfig`` instance, tolerating IB-Robot custom keys.

    Falls back to plain ``PreTrainedConfig.from_pretrained`` when ``policy_path``
    is not a local directory (e.g. an HF hub repo id) or when ``config.json``
    contains no IB-Robot-only keys.
    """
    from lerobot.configs.policies import PreTrainedConfig

    src_dir = Path(policy_path)
    src_cfg = src_dir / "config.json"
    if not src_cfg.is_file():
        # Not a local dir layout — let upstream handle hub download / errors.
        return PreTrainedConfig.from_pretrained(policy_path)

    with open(src_cfg) as f:
        raw = json.load(f)

    stripped = {k: v for k, v in raw.items() if k not in _IBROBOT_ONLY_KEYS}
    if stripped.keys() == raw.keys():
        return PreTrainedConfig.from_pretrained(policy_path)

    tmpdir = Path(tempfile.mkdtemp(prefix="ibrobot_policy_cfg_"))
    with open(tmpdir / "config.json", "w") as f:
        json.dump(stripped, f)
    return PreTrainedConfig.from_pretrained(str(tmpdir))
