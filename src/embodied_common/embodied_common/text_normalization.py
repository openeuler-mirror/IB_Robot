"""Shared text normalization helpers for trigger matching."""

from __future__ import annotations

import unicodedata


def normalize_trigger_text(text: str) -> str:
    """Normalize a trigger phrase for exact fixed-phrase matching."""

    chars = [char.casefold() for char in text if not char.isspace()]
    while chars and unicodedata.category(chars[0]).startswith("P"):
        chars.pop(0)
    while chars and unicodedata.category(chars[-1]).startswith("P"):
        chars.pop()
    return "".join(chars)
