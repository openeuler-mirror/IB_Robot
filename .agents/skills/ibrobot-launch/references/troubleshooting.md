# Troubleshooting

## Nodes Cannot Discover Each Other

Confirm every process uses identical `ROS_DOMAIN_ID` and `RMW_IMPLEMENTATION`. For distributed
operation, also confirm `ROS_LOCALHOST_ONLY=0` and network reachability.

## `ModuleNotFoundError: lerobot`

- Ubuntu/openEuler: source `.shrc_local` in the same shell.
- OpenHarmony: source `/data/roboframe/scripts/robooh_1.0.1.env`; if the staged LeRobot tree is
  absent, rebuild and redeploy the official release rather than copying source manually.

## Package or Launch File Not Found

- Ubuntu/openEuler: build first, then source `install/setup.zsh` in the launch shell.
- OpenHarmony: verify `/data/roboframe/install` and the package with `ros2 pkg list`. Rebuild with
  the default `oh-build-roboframe` package set if it is missing.

## OpenHarmony Native Process Crashes or RKNN Import Fails

Confirm the board env was loaded, the artifact is aarch64/musl compatible, the deployment exists in
`inference_manifest.json`, and the required NPU runtime is installed. For RKNN, verify:

```sh
python3 -c "from rknnlite.api import RKNNLite; print('RKNNLite OK')"
```

## `/joint_states` Publishes Joints in an Unexpected Order

This is expected Humble behavior, not a misconfiguration. The `joints` list in
`so101_controllers.yaml` only selects which state interfaces the broadcaster
claims; the Humble `joint_state_broadcaster` builds the message by iterating an
`std::unordered_map`, so the published order (e.g. `2,4,1,3,5,6`) is a
hash-bucket order and differs from any declared order. The order is stable per
process but unspecified, and no parameter can pin it (fixed only in newer
ros2_controllers releases, not backported to Humble).

Implications and rules:

- Never read a JointState positionally; always resolve by `msg.name`
  (the shipped consumers — tensormsg, conversions, servo, MoveIt — all do).
- The ordered joint list for datasets, commands and limits is the public
  description's `joint.state` `joint_names` (via `/runtime/get_status`).
- To eyeball values in a fixed order, sort by name:

```sh
ros2 topic echo /joint_states --once | sed -n '/^header:/,$p' | python3 -c \
  "import sys, yaml; m = next(y for y in yaml.safe_load_all(sys.stdin) if y); [print(f'{n:>3} {p:+.4f}') for n, p in sorted(zip(m['name'], m['position']), key=lambda t: int(t[0]))]"
```

## Controller or Hardware Device Missing

Check the selected control mode, YAML paths, calibration, permissions, `/dev/ttyACM*`, camera
devices, and required OpenHarmony kernel configuration before changing launch logic.

## ROS Package Version Mismatch Between Ubuntu and openEuler

### Symptoms (only reproduces on one distro)

- `spawner: error: unrecognized arguments: <names...> --activate-as-group` — older
  `ros-humble-controller-manager` (e.g. openEuler Embedded 2.25.2) only accepts a single
  `controller_name` positional arg; multi-controller names and `--activate-as-group` /
  `--switch-timeout` are rejected. (Precedent: #67 / #105 / #187.)
- ros2 binaries segfault or emit `undefined symbol` on one distro only.
- A CLI flag, service interface, or spawner behaviour exists on Ubuntu but is missing or
  differs on openEuler.

### Hard rule: compare versions BEFORE editing code

This is a hard rule in AGENTS.md ("跨发行版 ROS 包版本一致性"). Do **not** dumb down launch
or control code to the lowest common denominator. First prove the version mismatch, then push
for an environment upgrade on openEuler. Code-level downgrade is a last resort and must be
justified in the PR/issue as to why an environment upgrade is impossible.

### How to compare versions

Full sweep (recommended — catch every divergent package, not just the suspect one):

```bash
# Ubuntu (deb)
dpkg-query -W -f='${Package} ${Version}\n' 'ros-humble-*' | sort

# openEuler (rpm)
rpm -qa --qf '%{NAME} %{VERSION}-%{RELEASE}\n' 'ros-humble-*' | sort
```

Targeted query for a single suspect package:

```bash
# Ubuntu
dpkg -s ros-humble-controller-manager | grep -i version
# openEuler
rpm -q ros-humble-controller-manager
```

### Upgrade procedure on openEuler

```bash
sudo dnf remove 'ros-humble-*'
./scripts/setup.sh        # re-runs scripts/install_ros.sh
source .shrc_local        # reload ROS + venv + workspace overlays
```

### Re-verify after upgrade (mandatory)

Re-run the same version query on openEuler and compare against Ubuntu again. Versions must
match, or openEuler must be strictly newer. Do not assume the upgrade succeeded without
re-checking — the openEuler ROS repo may still ship an older build.

### If still mismatched after upgrade

Do **not** fall back to code-level workarounds. Have the user file an issue (or PR) with the
openEuler ROS package maintainer requesting a version sync, and attach the following evidence
so the maintainer can act without back-and-forth:

- Mismatched package name(s) with both version strings — paste the raw `dpkg` / `rpm` output.
- The exact runtime error log. For the spawner case, include the full
  `spawner: error: unrecognized arguments: ...` line and the surrounding
  `[ERROR] [spawner-N]: process has died [pid ... exit code 2, cmd '...']` block.
- Ubuntu upstream version and changelog link — from packages.ubuntu.com or the ROS index.
- Impact on IB_Robot: which launch file, which node, which controller fails, and the
  reproduction command.

Precedent trail: #67 (original bug report with the spawner error log), #105 (design
discussion questioning the code-level workaround), #187 (the code-level downgrade itself —
per-controller spawning instead of `--activate-as-group`; this is the pattern to avoid
repeating for future mismatches).
