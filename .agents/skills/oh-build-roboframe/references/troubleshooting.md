# Typical Recovery Cases

## When to Read

Read this file when one of the following scenarios occurs during the OpenHarmony host-side build of RoboFrame:

- `libs/lerobot` submodule metadata becomes broken in `/tmp` or the working copy, surfacing `fatal: not a git repository ... .git/modules/libs/lerobot`.
- The staged runtime tree still imports training-only dependencies (`datasets`, `pyarrow`, `av`).
- A stale build cache causes `COLCON_CURRENT_PREFIX` mismatches or references to a previously installed prefix.
- A package is reported as missing at runtime after deploy (`embodied_common`, `so101_description`, `so101_motion`).

## Case 1: `libs/lerobot` submodule metadata is broken in `/tmp`

Symptom:

```text
fatal: not a git repository ... .git/modules/libs/lerobot
```

Correct action:

- fix `build_roboframe_oh.sh` so runtime staging can fall back to upstream clone
- re-run the official build
- do not manually inject `lerobot/src`

## Case 2: runtime tree still imports training-only dependencies

Symptoms include:

- `ModuleNotFoundError: datasets`
- `ModuleNotFoundError: pyarrow`
- `ImportError ... av`

Likely cause:

- The dependencies archive is incomplete or the staged upstream LeRobot tree
  is inconsistent

Correct action:

- rebuild with the official script
- verify staged `install/lerobot/src`
- redeploy the rebuilt `install/` tree

## Case 3: stale build cache after install prefix change

Symptom:

```text
The build time path "/data/ibrobot/install/controller_manager_msgs" doesn't exist.
Either source a script for a different shell or set the environment variable
"COLCON_CURRENT_PREFIX" explicitly.
```

Likely cause:

- A previous build used a different `--custom-prefix` (e.g. `/data/ibrobot/install`),
  and the stale `colcon_command_prefix_build.sh` in `build/` still references the
  old path. Subsequent builds inherit the stale prefix and fail.

Correct action:

- remove `build/` and `install/` directories (they are Docker-owned, use the
  alpine container pattern from the Canonical Build Command section)
- re-run the official build with the default package list
- **do NOT** exclude the failing package from `--packages` — the package itself
  is fine, only the cache is stale

## Case 4: missing package at runtime after deploy

Symptom:

```text
ModuleNotFoundError: No module named 'embodied_common'
package 'so101_description' not found, searching: [...]
```

Likely cause:

- The build used a manually reduced `--packages` list that omitted a transitive
  dependency (`embodied_common`, `voice_asr_service`) or an SO-101 suite package
  (`so101_description`, `so101_motion`, `so101_robot`).

Correct action:

- rebuild with the default `PACKAGES` list (do not pass `--packages`)
- the default list in `build_roboframe_oh.sh` is authoritative and includes
  all required packages
