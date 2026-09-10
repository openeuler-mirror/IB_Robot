---
name: ibrobot-worktree-env
description: "Handles environment setup inside a git worktree of IB_Robot. Use when user asks to 'worktree 环境', 'git worktree', '复用主仓库 venv', 'worktree venv', 'worktree source .shrc_local', 'worktree PYTHONPATH', '混合环境', '测错分支', 'worktree setup', or when running commands inside a worktree and imports/paths resolve to the wrong repo. Triggers whenever environment inheritance across main repo and worktree is required."
---

# IB_Robot Worktree Environment Skill

This skill handles the **mixed-environment trap** that appears when working inside a `git worktree` of IB_Robot. It is distinct from `ibrobot-env`, which only covers environment setup inside a single repo.

## When to Use This Skill

- User created a worktree via `git worktree add ...` and wants to run / build / test inside it.
- User reports "imports come from the main repo", "tests run against the wrong branch", "PYTHONPATH points to main repo", or similar cross-repo pollution.
- User asks whether to `source /path/to/main/IB_Robot/.shrc_local` from a worktree (the answer is **no**, see below).

## Core Problem

IB_Robot's `.shrc_local` uses **two different path resolution bases**, which is what makes the worktree environment tricky:

- `venv/bin/activate` → resolved relative to the **caller's PWD** (the command is `source venv/bin/activate` with no path prefix, so it depends on where you run `source .shrc_local`)
- `WORKSPACE`, `PYTHONPATH`, `install/setup.sh` → resolved relative to `.shrc_local`'s **file location** (via `BASH_SOURCE` / `%x`), so they always point to the directory containing the sourced `.shrc_local`

If you source the **main repo's** `.shrc_local` from a worktree, you get a **mixed environment**: the venv activation depends on your PWD, but `WORKSPACE` / `PYTHONPATH` / `install/setup.sh` all point to the main repo. This silently makes you test the wrong branch.

## ❌ Wrong Approach — Do NOT Source Main Repo's .shrc_local

```bash
cd /path/to/worktree
source /path/to/main/IB_Robot/.shrc_local   # ❌ WRONG
```

Why this is wrong:

- `.shrc_local` activates `venv/bin/activate` **relative to the current directory** — so it may activate the worktree's own venv (if present) or fail.
- `WORKSPACE` is computed from `.shrc_local`'s file location — so it points to the **main repo**, not the worktree.
- `PYTHONPATH` points to the **main repo's** `libs/lerobot/src` and venv site-packages path.
- The loaded `install/setup.sh` is the **main repo's** build artifacts.

Result: "worktree's Python venv + main repo's source code and ROS build artifacts". You will test the wrong branch without any visible error.

## ✅ Correct Approach — Initialize in a Clean Child Shell

Do not repair an already-loaded main-repo/worktree environment by sourcing another
`.shrc_local`. Activation scripts prepend paths; they do not reliably remove the old
`PYTHONPATH`, ROS overlay, or CMake prefix. If `VIRTUAL_ENV`, `PYTHONPATH`,
`AMENT_PREFIX_PATH`, `CMAKE_PREFIX_PATH`, or `COLCON_PREFIX_PATH` is already set,
refuse to initialize the shared-venv flow in that shell and open this clean child shell:

```bash
env -i \
  HOME="$HOME" \
  USER="${USER:-$(id -un)}" \
  TERM="${TERM:-xterm}" \
  PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  bash --noprofile --norc
```

At the new prompt, initialize the worktree before activating the shared venv:

```bash
MAIN_REPO=/path/to/main/IB_Robot
WORKTREE=/path/to/worktree
cd "$WORKTREE"

# A local venv would take over when .shrc_local is sourced. Fail closed.
if [[ -e venv ]]; then
  echo "Refusing shared-venv setup: $WORKTREE/venv already exists" >&2
  exit 1
fi

# This is the only setup.sh mode allowed in the shared-venv flow. It initializes
# missing submodules and verifies/applies the managed LeRobot patch stack, but
# skips dependency installation, venv setup, and environment verification. Use
# the shared venv's Python for patch metadata helpers without activating it.
test -x "$MAIN_REPO/venv/bin/python3"
VENV_PYTHON="$MAIN_REPO/venv/bin/python3" ./scripts/setup.sh --only-patch --yes
test -e libs/lerobot/.git
test -f libs/lerobot/src/lerobot/__init__.py
git -C libs/lerobot rev-parse --is-inside-work-tree

source "$MAIN_REPO/venv/bin/activate"
# ros2 CLI entry points use a hardcoded `#!/usr/bin/python3` shebang, so they
# run under the SYSTEM python regardless of venv activation. The main repo's
# .shrc_local exposes its venv site-packages to that system python via
# PYTHONPATH; the worktree's .shrc_local instead points at the worktree's own
# (nonexistent) venv path, which drops the shared venv's packages. Prepend the
# shared venv site-packages explicitly, or `ros2 launch` fails with e.g.
# `ImportError: cannot import name 'model_validator' from 'pydantic'
# (/usr/lib/python3/dist-packages/pydantic/__init__.py)` and similar
# venv-only-symbol errors before the launch file even loads.
_PY_VER="$(python3 -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
export PYTHONPATH="$MAIN_REPO/venv/lib/${_PY_VER}/site-packages${PYTHONPATH:+:$PYTHONPATH}"
unset _PY_VER
source "$WORKTREE/.shrc_local"
```

Do not set `IBR_LEROBOT_FORCE_REBUILD=1` for routine initialization. That escape
hatch may discard local `libs/lerobot` changes. A successful `--only-patch` run is
the authoritative submodule and patch-stack precondition for this workflow.

Why this works:

1. `env -i` prevents a previous checkout's Python and ROS paths from surviving as fallbacks.
2. `--only-patch` makes a fresh worktree's `libs/lerobot` importable without modifying the shared venv.
3. The first `source` reuses the main repo's installed dependencies.
4. The worktree's `.shrc_local` resolves `WORKSPACE`, `libs/lerobot/src`, and `install/setup.sh` from the current worktree.

## Verification Snippet

Run this in the initialized child shell. It fails if LeRobot resolves outside the
current worktree or if `sys.path`/environment prefixes contain another registered
Git worktree (the shared venv directory itself is the only allowed main-repo path):

```bash
python3 - "$MAIN_REPO" "$WORKTREE" <<'PY'
import os
import subprocess
import sys
from pathlib import Path

import lerobot

main_repo = Path(sys.argv[1]).resolve()
worktree = Path(sys.argv[2]).resolve()
shared_venv = (main_repo / "venv").resolve()


def is_within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_path(value):
    path = Path(value or ".").expanduser()
    return (path if path.is_absolute() else worktree / path).resolve()


problems = []
workspace_value = os.environ.get("WORKSPACE")
if not workspace_value or resolve_path(workspace_value) != worktree:
    problems.append(f"WORKSPACE={workspace_value!r}")
if resolve_path(os.environ.get("VIRTUAL_ENV", "")) != shared_venv:
    problems.append(f"VIRTUAL_ENV={os.environ.get('VIRTUAL_ENV')!r}")
if Path(sys.prefix).resolve() != shared_venv:
    problems.append(f"sys.prefix={sys.prefix}")

lerobot_file = Path(lerobot.__file__).resolve()
if not is_within(lerobot_file, worktree / "libs/lerobot/src"):
    problems.append(f"lerobot={lerobot_file}")

worktree_lines = subprocess.check_output(
    ["git", "worktree", "list", "--porcelain"], cwd=worktree, text=True
).splitlines()
other_worktrees = [
    Path(line.removeprefix("worktree ")).resolve()
    for line in worktree_lines
    if line.startswith("worktree ")
    and Path(line.removeprefix("worktree ")).resolve() != worktree
]

path_entries = [("sys.path", value) for value in sys.path]
for name in (
    "PYTHONPATH",
    "AMENT_PREFIX_PATH",
    "CMAKE_PREFIX_PATH",
    "COLCON_PREFIX_PATH",
    "ROS_PACKAGE_PATH",
):
    path_entries.extend((name, value) for value in os.environ.get(name, "").split(os.pathsep) if value)

for source, value in path_entries:
    path = resolve_path(value)
    for other in other_worktrees:
        if is_within(path, other) and not is_within(path, shared_venv):
            problems.append(f"{source} contains other checkout path: {path}")
            break

# ros2 CLI entry points use a hardcoded system-python shebang; they only see
# venv-provided packages when PYTHONPATH exposes the active venv site-packages.
# In the shared-venv flow the worktree's .shrc_local cannot do this (it points
# at the worktree's own nonexistent venv path), so the caller must prepend it.
venv_site = shared_venv / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
pythonpath_entries = [resolve_path(v) for v in os.environ.get("PYTHONPATH", "").split(os.pathsep) if v]
if resolve_path(os.environ.get("VIRTUAL_ENV", "")) == shared_venv and venv_site not in pythonpath_entries:
    problems.append(
        "PYTHONPATH does not expose the shared venv site-packages, so ros2 CLI "
        f"(system python shebang) cannot import venv packages: add {venv_site}"
    )

if problems:
    raise SystemExit("Unsafe mixed worktree environment:\n  - " + "\n  - ".join(problems))

print(f"VIRTUAL_ENV: {shared_venv}")
print(f"WORKSPACE: {worktree}")
print(f"Python: {sys.executable}")
print(f"LeRobot: {lerobot_file}")
PY
```

This intentionally rejects an editable `.pth` that exposes the main repo's source,
even when the worktree path appears first. If verification fails, exit the child
shell and create a dedicated worktree venv with `./scripts/setup.sh`; do not edit the
shared venv's `.pth` files or continue with a fallback-prone environment.

## Known Limitations of the Shared-venv Workaround

This workaround is a **manual stopgap**, not a full solution. Be aware of:

- **Requirements drift across branches**: if branches have incompatible `requirements/*.txt`, the shared venv may be missing packages or have wrong versions. Symptom: `ModuleNotFoundError` or version-mismatch errors in one worktree but not another.
- **Concurrent setup contamination**: a full `./scripts/setup.sh` run mutates or creates a venv. In the shared-venv flow, only `./scripts/setup.sh --only-patch` is allowed, and it must run before shared-venv activation.
- **LeRobot patch / extras mismatch**: source code comes from the current worktree, but LeRobot's installed dependencies (e.g. `pip install -e '.[extras]'` extras) still come from the shared venv. If patch levels or extras differ across branches, runtime errors may appear.
- **Editable source leakage**: an absolute editable-install `.pth` in the shared venv may expose the main repo's source. The verification snippet rejects this; use a dedicated worktree venv instead of relying on import order.
- **Local venv takeover**: once a worktree has its own `venv/bin/activate`, its `.shrc_local` switches to the local venv. The shared-venv flow therefore refuses to start when `venv/` already exists.
- **Absolute paths in pre-commit / entry scripts**: `.git/hooks/pre-commit` and similar entry scripts may contain absolute paths captured at venv-creation time. Usually fine as long as the main repo is not moved on disk; breaks if the main repo is relocated.
- **Empty install/ in fresh worktrees**: a fresh shared-venv worktree has no build
  artifacts. `build.sh --packages-up-to <pkg>` covers only that package's
  dependency closure — NOT the full launch graph. A `ros2 launch robot_config
  robot.launch.py` stack additionally needs e.g. `hardware_mock` (mock platform)
  and `so101_hardware` (URDF `$(find ...)` references); build them explicitly
  (`--packages-up-to hardware_mock --packages-up-to so101_hardware`) or expect
  `package '<name>' not found` at launch time.

## Decision Flowchart

```
Inside a worktree and need to run Python / ROS 2 / build?
│
├── Does the worktree have its own venv/?
│   ├── YES → use the local venv; do not layer the main venv over it.
│   └── NO  → follow the shared-venv flow:
│             1. open an env-cleared child shell
│             2. run ./scripts/setup.sh --only-patch --yes
│             3. activate the main repo venv
│             4. source the worktree's .shrc_local
│             5. run the fail-closed verification snippet
│                └── failure → use a dedicated worktree venv
│
└── Need to run ./scripts/setup.sh or ./scripts/build.sh?
    ├── setup.sh --only-patch → allowed before shared-venv activation
    ├── full setup.sh → only for a dedicated worktree venv
    └── build.sh → safe to run from the worktree; produces worktree-local
                   install/ artifacts (ROS build outputs stay in the worktree)
```

## Relationship to ibrobot-env

| Aspect | ibrobot-env | ibrobot-worktree-env (this skill) |
|---|---|---|
| Scope | Single repo, source `.shrc_local` | Worktree sharing main repo's venv |
| Key question | "How do I load env vars?" | "How do I avoid mixed main/worktree env?" |
| Typical mistake | Forgetting to prefix `source .shrc_local &&` | Sourcing main repo's `.shrc_local` from a worktree |
| Verification | Commands simply run | Python check of venv/workspace/import origins and all checkout-bearing path prefixes |

If the user is **not** inside a worktree, defer to `ibrobot-env`. Only invoke this skill when worktree-specific path/venv drift is the concern.

## Issue Tracker Reference

The underlying fragility of this manual flow is tracked in **Issue #98**. The issue proposes formalizing the workaround with compatibility checks and concurrency guards. Until that issue is resolved, shared-venv reuse is supported only when the fail-closed checks above pass; otherwise use a dedicated worktree venv.
