"""Synthetic image generators for hardware_mock.

The mock only needs visually-stable, deterministic frames so downstream nodes
can prove the data path works. We deliberately keep this dependency-light
(numpy only) and never spin up a camera device.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

ImageGenerator = Callable[[], np.ndarray]


@dataclass(frozen=True)
class ImageSourceSpec:
    """Resolved spec for one camera's synthetic frame stream."""

    width: int
    height: int
    kind: str  # 'checkerboard' | 'solid' | 'gradient'
    tile: int = 40
    color_rgb: tuple = (128, 128, 128)
    encoding: str = "bgr8"


def _parse_hex_color(text: str) -> tuple:
    """Parse #RRGGBB / RRGGBB into an (R, G, B) uint8 tuple."""
    s = text.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"Invalid hex color '{text}', expected #RRGGBB")
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
    except ValueError as exc:
        raise ValueError(f"Invalid hex color '{text}'") from exc
    return (r, g, b)


def resolve_spec(
    camera_name: str,
    width: int,
    height: int,
    overrides: dict[str, dict] | None,
    *,
    encoding: str = "bgr8",
) -> ImageSourceSpec:
    """Resolve final image-source spec from optional YAML overrides.

    Overrides come from ``robot.hardware_mock.image_sources.<name>`` and accept:
        kind:  checkerboard (default) | solid | gradient
        tile:  int (checkerboard only)
        color: '#RRGGBB' (solid only)
    """
    cfg = (overrides or {}).get(camera_name, {}) or {}
    kind = str(cfg.get("kind", "checkerboard")).lower()
    if kind not in ("checkerboard", "solid", "gradient"):
        raise ValueError(
            f"hardware_mock.image_sources.{camera_name}.kind='{kind}' invalid; "
            "use 'checkerboard', 'solid', or 'gradient'."
        )
    tile = int(cfg.get("tile", 40))
    if tile <= 0:
        raise ValueError(f"hardware_mock.image_sources.{camera_name}.tile must be > 0")
    normalized_encoding = encoding.lower()
    if normalized_encoding not in {"bgr8", "rgb8"}:
        raise ValueError(f"hardware_mock image encoding must be 'bgr8' or 'rgb8', got {encoding!r}")
    color = _parse_hex_color(cfg["color"]) if "color" in cfg else (128, 128, 128)
    return ImageSourceSpec(
        width=width,
        height=height,
        encoding=normalized_encoding,
        kind=kind,
        tile=tile,
        color_rgb=color,
    )


def make_generator(spec: ImageSourceSpec) -> ImageGenerator:
    """Build a zero-arg generator returning an HxWx3 uint8 color frame.

    The image is computed once and reused; mocking does not require novelty
    and avoids needless CPU at 30/60 Hz.
    """
    if spec.kind == "solid":
        frame = np.zeros((spec.height, spec.width, 3), dtype=np.uint8)
        r, g, b = spec.color_rgb
        first, third = (r, b) if spec.encoding == "rgb8" else (b, r)
        frame[..., 0] = first
        frame[..., 1] = g
        frame[..., 2] = third
    elif spec.kind == "gradient":
        x = np.linspace(0, 255, spec.width, dtype=np.uint8)
        y = np.linspace(0, 255, spec.height, dtype=np.uint8)
        gx, gy = np.meshgrid(x, y)
        frame = np.zeros((spec.height, spec.width, 3), dtype=np.uint8)
        blue = gx
        red = (gx // 2 + gy // 2).astype(np.uint8)
        frame[..., 0] = red if spec.encoding == "rgb8" else blue
        frame[..., 1] = gy  # G
        frame[..., 2] = blue if spec.encoding == "rgb8" else red
    else:  # checkerboard
        tile = spec.tile
        yy, xx = np.indices((spec.height, spec.width))
        mask = ((xx // tile) + (yy // tile)) % 2 == 0
        frame = np.zeros((spec.height, spec.width, 3), dtype=np.uint8)
        frame[mask] = (220, 220, 220)
        frame[~mask] = (32, 32, 32)

    def _gen() -> np.ndarray:
        # Return a view; downstream cv_bridge copies into the message.
        return frame

    return _gen
