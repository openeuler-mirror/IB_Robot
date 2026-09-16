import math
import struct
from pathlib import Path

import numpy as np
import pytest
from so101_suite.gripper_geometry import (
    _quaternion_matrix,
    _TablePlane,
    gripper_geometry_metrics_batch,
    gripper_mesh_min_z,
    gripper_mesh_vertices,
    tabletop_clearance,
)

_SO101_MESHES = Path(__file__).parents[2] / "so101_description" / "meshes" / "lerobot" / "so101"


def _synthetic_gripper_directory(directory: Path) -> Path:
    """Write a minimal binary-STL gripper pair (fixed + moving jaw) into ``directory``."""

    def write_stl(name: str, triangles: list[tuple[tuple[float, float, float], ...]]) -> None:
        payload = bytearray(b"\0" * 80)
        payload += struct.pack("<I", len(triangles))
        for a, b, c in triangles:
            payload += struct.pack("<3f", 0.0, 0.0, 1.0)
            for vertex in (a, b, c):
                payload += struct.pack("<3f", *vertex)
            payload += struct.pack("<H", 0)
        (directory / name).write_bytes(payload)

    write_stl(
        "wrist_roll_follower_so101_v1.stl",
        [
            ((0.0, 0.0, 0.0), (0.02, 0.0, 0.0), (0.0, 0.02, 0.0)),
            ((0.02, 0.0, 0.0), (0.02, 0.02, 0.0), (0.0, 0.02, 0.0)),
        ],
    )
    write_stl(
        "moving_jaw_so101_v1.stl",
        [
            ((0.0, 0.0, 0.005), (0.01, 0.0, 0.005), (0.0, 0.01, 0.005)),
            ((0.01, 0.0, 0.005), (0.01, 0.01, 0.005), (0.0, 0.01, 0.005)),
        ],
    )
    return directory


def test_quaternion_matrix_normalizes_non_unit_quaternion():
    unit = _quaternion_matrix((0.0, 0.0, 0.0, 1.0))
    doubled = _quaternion_matrix((0.0, 0.0, 0.0, 2.0))

    assert np.allclose(unit, doubled)
    assert np.allclose(unit, np.eye(3))

    with pytest.raises(ValueError):
        _quaternion_matrix((0.0, 0.0, 0.0, 0.0))


def test_mesh_vertices_with_synthetic_stl(tmp_path):
    mesh_directory = _synthetic_gripper_directory(tmp_path)

    fixed, moving = gripper_mesh_vertices(
        mesh_directory,
        (0.10, -0.16, 0.18),
        (0.0, 0.0, 0.0, 1.0),
        0.035,
    )

    assert len(fixed) > 0 and len(moving) > 0
    # The hardcoded SO101 visual chain only adds sub-millimeter z offsets to
    # the fixed part; the moving jaw hangs below through gripper_to_jaw.
    assert abs(float(np.min(fixed[:, 2])) - 0.18) < 0.005
    assert -0.05 < float(np.min(moving[:, 2])) < 0.18
    min_z = gripper_mesh_min_z(mesh_directory, (0.10, -0.16, 0.18), (0.0, 0.0, 0.0, 1.0), 0.035)
    assert math.isclose(
        min_z,
        min(float(np.min(fixed[:, 2])), float(np.min(moving[:, 2]))),
        abs_tol=1e-9,
    )


def test_tabletop_clearance_with_synthetic_stl(tmp_path):
    mesh_directory = _synthetic_gripper_directory(tmp_path)
    plane = _TablePlane(normal=(0.0, 0.0, 1.0), offset=-0.10)

    clearance = tabletop_clearance(
        mesh_directory,
        (0.10, -0.16, 0.22),
        (0.10, -0.16, 0.18),
        (0.0, 0.0, 0.0, 1.0),
        0.035,
        plane,
        sweep_steps=1,
    )

    # The approach point sits above the grasp point along the plane normal,
    # so the binding clearance is the grasp-pose mesh height plus the offset.
    expected = gripper_mesh_min_z(mesh_directory, (0.10, -0.16, 0.18), (0.0, 0.0, 0.0, 1.0), 0.035) - 0.10
    assert math.isclose(clearance, expected, abs_tol=1e-9)


def test_batch_gripper_geometry_matches_scalar_checks(tmp_path):
    mesh_directory = _synthetic_gripper_directory(tmp_path)
    plane = _TablePlane(normal=(0.0, 0.0, 1.0), offset=0.02)
    candidates = [
        ((0.10, -0.16, 0.18), (0.10, -0.16, 0.08), (0.0, 0.0, 0.0, 1.0), 0.035),
        ((0.14, -0.12, 0.16), (0.14, -0.12, 0.06), (0.0, 0.0, 0.0, 1.0), 0.020),
    ]

    batch = gripper_geometry_metrics_batch(mesh_directory, candidates, plane)

    for candidate, (batch_min_z, batch_clearance) in zip(candidates, batch, strict=True):
        approach, grasp, quaternion, width = candidate
        scalar_min_z = gripper_mesh_min_z(mesh_directory, grasp, quaternion, width)
        scalar_clearance = tabletop_clearance(
            mesh_directory,
            approach,
            grasp,
            quaternion,
            width,
            plane,
            sweep_steps=5,
        )
        assert math.isclose(batch_min_z, scalar_min_z, abs_tol=1e-10)
        assert batch_clearance is not None
        assert math.isclose(batch_clearance, scalar_clearance, abs_tol=1e-10)


@pytest.mark.skipif(not _SO101_MESHES.is_dir(), reason="SO-101 meshes are not available in this workspace")
def test_batch_gripper_geometry_with_real_so101_meshes():
    plane = _TablePlane(normal=(0.0, 0.0, 1.0), offset=0.02)
    candidates = [
        ((0.10, -0.16, 0.18), (0.10, -0.16, 0.08), (0.0, 0.0, 0.0, 1.0), 0.035),
        ((0.14, -0.12, 0.16), (0.14, -0.12, 0.06), (0.0, 0.0, 0.0, 1.0), 0.020),
    ]

    batch = gripper_geometry_metrics_batch(_SO101_MESHES, candidates, plane)

    for candidate, (batch_min_z, batch_clearance) in zip(candidates, batch, strict=True):
        approach, grasp, quaternion, width = candidate
        scalar_min_z = gripper_mesh_min_z(_SO101_MESHES, grasp, quaternion, width)
        scalar_clearance = tabletop_clearance(
            _SO101_MESHES,
            approach,
            grasp,
            quaternion,
            width,
            plane,
            sweep_steps=5,
        )
        assert math.isclose(batch_min_z, scalar_min_z, abs_tol=1e-10)
        assert batch_clearance is not None
        assert math.isclose(batch_clearance, scalar_clearance, abs_tol=1e-10)
