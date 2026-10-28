from __future__ import annotations

import json
from pathlib import Path


class PreTrainedConfig:
    def __init__(self, policy_type: str, device: str) -> None:
        self.type = policy_type
        self.device = device

    @classmethod
    def from_pretrained(cls, path: str, **_kwargs):
        config = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
        return cls(config["type"], config["device"])
