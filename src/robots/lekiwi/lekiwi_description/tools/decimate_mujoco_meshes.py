"""Generate the MuJoCo visual meshes for lekiwi_description (dev tool, NOT installed).

Why this exists
---------------
MuJoCo 3.12.0's STL decoder rejects meshes with more than 200000 faces
(``stl_decoder: number of faces should be between 1 and 200000``), and the
shipped chassis/wheel meshes exceed that limit:

  meshes/lekiwi_base.stl   440888 faces
  meshes/omni_wheel.stl    543010 faces

The MuJoCo robot template (``urdf/../mujoco/lekiwi_base.xml.template``) needs
visual meshes under the limit, so this script derives reduced copies into
``meshes/mujoco/`` (visual only — ``contype=0 conaffinity=0``; collision stays
on the analytic hulls in the template). The derived files are committed;
regeneration is only needed when the source meshes change.

How
---
Per mesh, two strategies, tried in order:

1. Quadric decimation (``fast_simplification``) straight to the target face
   count. Works for the wheel; stalls well above the target on the chassis
   (a 593-component CAD assembly with genus-9 shells that the edge-collapse
   queue refuses to collapse further).
2. Fallback: voxelise at ``mc_pitch`` (seals sub-pitch holes and produces one
   smooth watertight shell), marching-cubes surface, transform back to world
   coordinates, then quadric decimation to the target — marches over the
   topology that stalls strategy 1.

Both strategies are deterministic for fixed library versions, so the same
inputs and the dependency versions below reproduce the committed outputs
byte-for-byte.

Dependencies (dev-time only; none of them is a package dependency)
------------------------------------------------------------------
  trimesh             4.5.3   (mesh I/O, voxelisation, decimation wrapper)
  fast_simplification 0.2.0   (quadric edge-collapse decimation)
  scikit-image        0.25.2  (marching cubes, fallback strategy only)
  mujoco              3.12.0  (verification compile; must match the simulator's
                              libmujoco — mujoco_vendor 0.1.1)

Usage
-----
  python3 tools/decimate_mujoco_meshes.py [--check-only]

Run from the ``lekiwi_description`` package root. ``--check-only`` verifies
the committed outputs against the limits (face count, MuJoCo compile) without
rewriting them.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import trimesh

PKG_ROOT = Path(__file__).resolve().parent.parent
SOURCE_MESHES = PKG_ROOT / "meshes"
OUTPUT_DIR = PKG_ROOT / "meshes" / "mujoco"

#: MuJoCo's STL decoder face limit (stl_decoder, libmujoco 3.12.0).
MUJOCO_STL_FACE_LIMIT = 200000

#: (source file name, target face count, marching-cubes fallback pitch or None).
#: The pitch is chosen so marching cubes yields ~7x the target faces — enough
#: surface detail for the second-stage quadric pass to pick good collapses.
MESHES = (
    ("lekiwi_base.stl", 25000, 0.0025),
    ("omni_wheel.stl", 25000, None),
)


def stl_face_count(path: Path) -> int:
    """The declared triangle count of a binary STL."""
    with open(path, "rb") as handle:
        return struct.unpack("<I", handle.read(84)[80:84])[0]


def decimate(source: Path, target_faces: int, mc_pitch: float | None) -> trimesh.Trimesh:
    """One mesh through the two-strategy pipeline (see module docstring)."""
    mesh = trimesh.load(source)
    simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
    if len(simplified.faces) <= target_faces:
        return simplified
    if mc_pitch is None:
        raise RuntimeError(
            f"{source.name}: quadric decimation stalled at {len(simplified.faces)} faces "
            f"(target {target_faces}) and no marching-cubes fallback pitch is configured"
        )
    voxels = mesh.voxelized(pitch=mc_pitch)
    surface = voxels.marching_cubes
    surface.apply_transform(voxels.transform)
    simplified = surface.simplify_quadric_decimation(face_count=target_faces)
    if len(simplified.faces) > target_faces:
        raise RuntimeError(
            f"{source.name}: fallback pipeline produced {len(simplified.faces)} faces (target {target_faces})"
        )
    return simplified


def verify(source: Path, output: Path, tolerance: float = 0.004) -> None:
    """The output loads in MuJoCo, is under the face limit and keeps the silhouette.

    ``tolerance`` bounds how far the decimated bounding box may drift from the
    source's (the marching-cubes stage pads by up to half a pitch; the value
    covers the configured pitches with margin).
    """
    import mujoco

    if stl_face_count(output) > MUJOCO_STL_FACE_LIMIT:
        raise RuntimeError(f"{output.name}: {stl_face_count(output)} faces exceeds the MuJoCo STL limit")
    xml = (
        f'<mujoco><asset><mesh name="m" file="{output}"/></asset>'
        '<worldbody><body><geom name="g" type="mesh" mesh="m"/></body></worldbody></mujoco>'
    )
    mujoco.MjSpec.from_string(xml).compile()
    original = trimesh.load(source)
    derived = trimesh.load(output)
    drift = np.abs(np.array(derived.bounds) - np.array(original.bounds)).max()
    if drift > tolerance:
        raise RuntimeError(f"{output.name}: bounding box drifted {drift:.4f} m (> {tolerance} m)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check-only", action="store_true", help="verify the committed outputs only; write nothing")
    args = parser.parse_args()

    failures = []
    for name, target_faces, mc_pitch in MESHES:
        source = SOURCE_MESHES / name
        output = OUTPUT_DIR / name
        if not source.is_file():
            failures.append(f"missing source mesh {source}")
            continue
        if args.check_only:
            if not output.is_file():
                failures.append(f"missing derived mesh {output}")
                continue
            try:
                verify(source, output)
                print(f"OK    {name}: {stl_face_count(output)} faces, {output.stat().st_size} bytes")
            except RuntimeError as exc:
                failures.append(str(exc))
            continue
        derived = decimate(source, target_faces, mc_pitch)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        derived.export(output)
        try:
            verify(source, output)
        except RuntimeError as exc:
            failures.append(str(exc))
        print(f"{name}: {stl_face_count(source)} -> {stl_face_count(output)} faces, {output.stat().st_size} bytes")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
