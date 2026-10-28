# MuJoCo visual meshes

Decimated visual meshes for the MuJoCo platform
(`mujoco/lekiwi_base.xml.template`, lekiwi-mujoco-runtime-platform D2b).

## Why they exist

MuJoCo 3.12.0's STL decoder loads at most 200000 faces per STL file
(`stl_decoder: number of faces should be between 1 and 200000`). The shipped
source meshes exceed that limit:

| source (`..`)     | faces  |
| ------------------ | -----: |
| `lekiwi_base.stl` | 440888 |
| `omni_wheel.stl`  | 543010 |

The runtime therefore cannot reference the source meshes in the robot MJCF.
These copies are decimated to 25000 faces each (about 1.2 MB per file), which
the decoder accepts and which keeps the chassis and wheel silhouettes
recognizable in the MuJoCo viewer.

They are **visual only**: the MJCF template renders them with
`contype="0" conaffinity="0"` (group 1). Collision stays on the analytic
hulls declared in the template (box chassis, sphere wheels), exactly as
design D2 specifies. MuJoCo sensor rays ignore `alpha=0` geoms but do hit
visible ones, so these meshes are part of the lidar's self-hit surface
(measured in the platform preflight, design D5 self-hits).

## Provenance and license

Derived from `../lekiwi_base.stl` and `../omni_wheel.stl` of this same
package by `../../tools/decimate_mujoco_meshes.py` (development tool; not
installed and not a package dependency). Same license as the source meshes:
Apache-2.0, see the package `package.xml`.

Regenerate after changing a source mesh:

```bash
python3 tools/decimate_mujoco_meshes.py            # regenerate + verify
python3 tools/decimate_mujoco_meshes.py --check-only   # verify committed files
```

The script header documents the exact dependency versions the committed files
were produced with (trimesh 4.5.3, fast_simplification 0.2.0,
scikit-image 0.25.2, mujoco 3.12.0); with those versions the pipeline is
deterministic.
