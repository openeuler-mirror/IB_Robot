---
name: ibrobot-bq3588hm-oh
description: "Captures the verified OpenHarmony runtime facts for the Bearkey BQ3588HM board. Use when user wants to bootstrap ROS on-device, source ros2ohos.env, check Python, inspect /data/install or /data/out, or debug the read-only-root workaround on the board."
---

# IB-Robot BQ3588HM OpenHarmony Board Skill

This skill captures the current verified runtime state of the Bearkey BQ3588HM OpenHarmony board.

Use `ibrobot-hdc` for connectivity and file transfer. Use this skill for board-local runtime facts, ROS bootstrap, Python availability, and known platform quirks.

## Current Verified Board State

- Board: Bearkey BQ3588HM
- OpenHarmony access path: user-provided HDC target, preferably TCP `<board-ip>:8710`, with USB
  HDC also supported as a direct transport
- HDC binary available in `$PATH` as `hdc`
- Official runtime archives currently on the board:
  - `/data/ohos-humble-build-aarch64-20260115100449.tar.gz`
  - `/data/ohos-18-sysdeps-aarch64-20260115.tar.gz`
- Extracted ROS runtime roots:
  - `/data/install`
  - `/data/out`
- Deployed IB_Robot runtime root:
  - `/data/ibrobot/install`
- Deployed RKNN policy root:
  - `/data/ibrobot/models/502000/pretrained_model`
- ROS environment file:
  - `/data/ros2ohos.env`
- Local board patch already applied:
  - `/data/sysdeps.env` now inserts `mount -o remount,rw /` before the SSH setup `mkdir`
- Verified third-party torch runtime root:
  - `/data/local/skh-run/usr`
- Verified preinstalled RKNN Lite Python package root:
  - `/sys_prod/robot/out/lib/python3.12/site-packages`
- Verified RKNN runtime library:
  - `/usr/lib/librknnrt.so`

## RKNN Runtime Preparation Checklist

Before using this skill for live board interaction, first confirm transport with the user:

1. preferred: TCP target `<board-ip>:8710`
2. alternative: USB HDC connected and used directly

Do not assume a fixed board IP. If the user wants TCP but it has not been enabled yet, first use
USB HDC to run `hdc tmode port 8710`, then reconnect via `hdc tconn <board-ip>:8710`.

Before trying to launch IB_Robot with `device:=rknn`, confirm all of the following are present:

1. official OpenHarmony ROS runtime unpacked under `/data/install` and `/data/out`
2. `/data/ros2ohos.env` available and sourceable
3. `rknnlite` import path available under `/sys_prod/robot/out/lib/python3.12/site-packages`
4. `librknnrt.so` copied to `/usr/lib/librknnrt.so`
5. `skh-run` torch runtime available under `/data/local/skh-run/usr`
6. deployed IB_Robot install under `/data/ibrobot/install`
7. deployed RKNN policy under `/data/ibrobot/models/502000/pretrained_model`

If any item above is missing, the board is **not** ready for the RKNN cloud inference path yet.

## Core Runtime Bootstrap

The official documentation is directionally correct: stay in `/data` and source `./ros2ohos.env`.
On this board, the local `/data/sysdeps.env` has already been patched so that it remounts `/` read-write before the SSH setup branch creates directories on the root filesystem.

### Preferred Verified Pattern

```sh
cd /data
. ./ros2ohos.env
ros2 topic list
python3 --version
python3 -m pip --version
mount -o remount,ro /
```

Keep all commands in the same shell session. Remount `/` back to read-only after the commands that depend on the sourced environment have completed.

### Fallback Pattern

If remounting `/` is undesired in a specific session, the SSH setup branch can still be bypassed by temporarily moving the sysdeps `sshd_config` file:

```sh
cd /data
mv out/etc/sshd_config out/etc/sshd_config.disabled
. ./ros2ohos.env
ros2 topic list
python3 --version
python3 -m pip --version
mv out/etc/sshd_config.disabled out/etc/sshd_config
```

## Why Remounting Matters

The originally installed `/data/sysdeps.env` did **not** contain a `mount -o remount,rw /` line before its SSH setup branch. The board-local file has since been patched so that the relevant part is now:

```sh
if [ -f "${OHOS_ROS2_SYSDEPS}/etc/sshd_config" ]; then
    mount -o remount,rw /
    mkdir -p /var/empty /var/run /root/.ssh /libexec
    ...
fi
```

On this board, the root filesystem starts as read-only for those paths, so the `mkdir` operations would abort the environment setup before ROS and Python are fully prepared unless `/` is remounted read-write first.

The author-suggested remount path is valid in practice: `mount -o remount,rw /` works on this board, and after patching `sysdeps.env` the plain `source ./ros2ohos.env` path succeeds without the `sshd_config` workaround.

Temporarily moving `/data/out/etc/sshd_config` out of the way remains a fallback that avoids modifying the official scripts when remounting `/` is not desired.

## Python Environment on the Board

The board does **not** provide a ready-to-use global `python`, `python3`, or `pip` before the ROS/sysdeps environment is loaded.

After sourcing the environment successfully:

- `python3` resolves to `/data/out/bin/python3`
- Real interpreter: `/data/out/bin/python3.12`
- Verified version: `Python 3.12.12`
- `pip` is available via:

```sh
python3 -m pip --version
```

Verified output:

```text
pip 25.1.1 from /data/out/lib/python3.12/site-packages/pip (python 3.12)
```

### Important Limitation: `ros2ohos.env` Alone Is Not Enough for RKNN + torch

For plain ROS/Python checks, sourcing `/data/ros2ohos.env` is enough.

For IB_Robot's RKNN inference path, that environment is **not sufficient** on its own:

- `rknnlite` lives under `/sys_prod/robot/out/lib/python3.12/site-packages`
- `torch` lives under `/data/local/skh-run/usr/lib/python3.12/site-packages`
- `torch` must see the `skh-run` NumPy 1.26.4 before the system NumPy 2.4.0, or `torch.from_numpy` will fail

If the user wants to actually launch `device:=rknn` inference on the board, prefer the dedicated
`bq3588-oh-rknn` skill.

## Preparing Missing RKNN Pieces on the Board

### 1. Fix `rknnlite` extension suffixes if needed

The preinstalled `rknnlite` package may ship `.cpython-312-aarch64-linux-gnu.so` files while the
board expects `linux-ohos` suffixes. The verified repair pattern is:

```sh
for f in /sys_prod/robot/out/lib/python3.12/site-packages/rknnlite/api/*.cpython-312-aarch64-linux-gnu.so; do
    cp "$f" "${f%-gnu.so}-ohos.so"
done
for f in /sys_prod/robot/out/lib/python3.12/site-packages/rknnlite/api/npu_config/*.cpython-312-aarch64-linux-gnu.so; do
    cp "$f" "${f%-gnu.so}-ohos.so"
done
for f in /sys_prod/robot/out/lib/python3.12/site-packages/rknnlite/utils/*.cpython-312-aarch64-linux-gnu.so; do
    cp "$f" "${f%-gnu.so}-ohos.so"
done
```

### 2. Ensure `/usr/lib/librknnrt.so` exists

If `rknnlite` cannot find the runtime library, repair it with:

```sh
mount -o rw,remount /
mkdir -p /usr/lib
cp /vendor/lib64/librknnrt.so /usr/lib/
```

### 3. Ensure `skh-run` exists

The verified torch runtime root is:

```text
/data/local/skh-run/usr
```

If it is missing, the board still lacks the Python runtime required by `RKNNPolicyWrapper` and
`pure_inference_node`. In that case, deploy the `thirdparty_pytorch` runtime payload before
attempting RKNN launch.

### 4. Ensure IB_Robot install and policy are deployed

The verified runtime expects:

```text
/data/ibrobot/install
/data/ibrobot/models/502000/pretrained_model
```

The install tarball is produced from the host-side OpenHarmony build workspace, and the policy
directory must include at least:

- `config.json`
- `act_ros2_rknn.onnx`
- `model.rknn`

## Verified ROS Runtime Signals

After sourcing with the workaround, `ros2 topic list` succeeds on the board and currently shows at least:

- `/joint_states`
- `/parameter_events`
- `/robot_status/ee_pose`
- `/rosout`
- `/tf_static`

## Verified RKNN Runtime Signals

The following were re-verified on this board:

- `rknnlite` imports successfully once `/sys_prod/robot/out/lib/python3.12/site-packages` is visible
- direct RKNN inference against `/data/ibrobot/models/502000/pretrained_model/model.rknn` succeeds
- verified direct inference output shape: `(1, 100, 6)`
- verified direct inference latency: about `0.383s`
- `RKNNPolicyWrapper` from `inference_service` runs successfully when the `skh-run` torch runtime is layered in
- `pure_inference_node` can start and wait for `/preprocessed/batch`
- `ros2 launch inference_service cloud_inference.launch.py policy_path:=/data/ibrobot/models/502000/pretrained_model device:=rknn` can start successfully

## Scope Boundary

Use this skill when the user wants to:

- bootstrap ROS on the BQ3588HM board
- source `ros2ohos.env`
- check on-device Python or pip
- inspect `/data/install`, `/data/out`, or `/data/ibrobot/install`
- understand the read-only-root workaround
- confirm what is already installed on this board
- confirm where `rknnlite`, `torch`, `skh-run`, or the deployed RKNN policy live

Do **not** use this skill for:

- HDC transport and reconnect logic (`ibrobot-hdc`)
- local workspace builds (`ibrobot-build`)
- local Ubuntu ROS environment setup (`ibrobot-env`)
- the exact RKNN launch overlay for `cloud_inference.launch.py` (`bq3588-oh-rknn`)

## Practical Consequence for IB_Robot

The board now has:

- official OpenHarmony ROS runtime unpacked
- working on-device `ros2`
- working on-device Python 3.12 from sysdeps
- working deployed IB_Robot install at `/data/ibrobot/install`
- working deployed RKNN policy under `/data/ibrobot/models/502000/pretrained_model`
- usable RKNN Lite runtime plus `skh-run` torch runtime

The next major step is usually **not** rebuilding the board image again.
It is either:

1. cross-building and deploying updated IB_Robot packages, or
2. launching the RKNN cloud inference path with the correct `skh-run` + `rknnlite` environment overlay.
