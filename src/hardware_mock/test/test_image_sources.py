"""Tests for hardware_mock.image_sources."""

import numpy as np
import pytest

from hardware_mock.image_sources import make_generator, resolve_spec


def test_default_is_checkerboard():
    spec = resolve_spec("cam", 64, 48, overrides=None)
    assert spec.kind == "checkerboard"
    frame = make_generator(spec)()
    assert frame.shape == (48, 64, 3)
    assert frame.dtype == np.uint8
    # Checkerboard must have at least two distinct intensities.
    assert len(np.unique(frame)) >= 2


def test_solid_color_override():
    spec = resolve_spec("cam", 4, 4, overrides={"cam": {"kind": "solid", "color": "#10203040"[:7]}})
    # '#102030' → (R=0x10, G=0x20, B=0x30); we encode bgr8 so first channel = B=0x30.
    frame = make_generator(spec)()
    assert frame[0, 0, 0] == 0x30
    assert frame[0, 0, 1] == 0x20
    assert frame[0, 0, 2] == 0x10


def test_gradient_shape():
    spec = resolve_spec("cam", 8, 4, overrides={"cam": {"kind": "gradient"}})
    frame = make_generator(spec)()
    assert frame.shape == (4, 8, 3)


def test_invalid_kind_rejected():
    with pytest.raises(ValueError):
        resolve_spec("cam", 8, 8, overrides={"cam": {"kind": "noise"}})


def test_invalid_color_rejected():
    with pytest.raises(ValueError):
        resolve_spec("cam", 8, 8, overrides={"cam": {"kind": "solid", "color": "not-a-color"}})
