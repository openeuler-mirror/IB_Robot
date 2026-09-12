# Known Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| `sudo: a terminal is required to read the password` | Docker exec has no tty; `use_pty` in sudoers blocks `sudo -v` | Phase 1 **must** set `NOPASSWD:ALL`; `setup.sh` code uses `sudo -n true` first |
| `sh: 1: rosdep: not found` | Was caused by rosdepc calling `os.system('rosdep ...')` | Replaced rosdepc with direct `pip install rosdep` |
| `error loading sources list: Permission denied` | `write_rosdep_sources_list` wrote file as 600 root | Now does `chmod 644` after writing |
| `rosdep update` times out | `ROSDISTRO_INDEX_URL` not passed to platform script | Platform scripts now pass `env ROSDISTRO_INDEX_URL=...` |
| pip downloads from pypi.org at ~10 KB/s | No pip mirror configured in container | `ensure_workspace_venv` writes `${VENV_PATH}/pip.conf` |
| rosdep installs from `packages.ros.org` at ~10 KB/s | The ROS desktop-full image keeps its deb822 `ros2.sources` file | Phase 2 switches both `.sources` and `.list` ROS entries to TUNA before `apt-get update` |
| TUNA ROS source index returns 404 | TUNA serves binary ROS packages but not the `deb-src` index | Phase 2 changes deb822 `Types` to `deb` and removes traditional `deb-src` entries |
| `Permission denied: ~/.cache/pre-commit` | Bind mount caused Docker to create the home cache parent as root | Mount pip at `/var/cache/ibrobot-pip`, not below `~/.cache` |
| `pip cache has been disabled` | Container UID/GID does not match the host cache owner, often because `1000:1000` was hard-coded | Recreate `testuser` from host `id -u`/`id -g` and require the user-level write probe before setup |
| `The build time path ... doesn't exist` | Copied or iterative source retained host-specific venv/build paths | Remove `venv build install log`, then rerun setup from the selected source |
| `git: command not found` mid-setup | `install_ros.sh` apt install may remove git | Phase 1 already installed git; re-run `apt-get install -y git git-lfs` if needed |
| lerobot patch stack fetch fails | Submodule base commit not in local checkout | Rebase branch onto `upstream/master` before copying |
| `GraspGen pointnet2_ops requires a CUDA toolkit` | Host has no CUDA toolkit; `setup.sh` will warn and skip grasp install | Phase 0 detects nvcc; if absent, setup.sh skips grasp gracefully (like SAM2_BUILD_CUDA=0 pattern) — verification continues without grasp smoke test |
| `nvcc: command not found` during `install_graspgen_pip` | CUDA toolkit mount failed or `CUDA_HOME` env var not inherited by `docker exec` | Verify Phase 2.5 `nvcc --version` probe passed; ensure `docker run -e CUDA_HOME=...` is set so `docker exec` inherits it |
| `CUDA_HOME` set but `nvcc` not executable in container | Bind mount used a symlink path that Docker did not resolve | Use `readlink -f` to resolve the actual directory before mounting; mount the real directory (e.g. `/usr/local/cuda-12.4`), not the `/usr/local/cuda` symlink |
| Bind mount of `/usr` overwrites container `/usr` | Host `nvcc` is at `/usr/bin/nvcc` (apt-installed), `dirname(dirname())` yields `/usr` | Phase 0 rejects system directories (`/`, `/usr`, `/usr/local`); install CUDA toolkit under `/usr/local/cuda-<version>` instead |
| `fatal: detected dubious ownership in repository at '/home/testuser/IB_Robot'` in later git probes (e.g. the Phase 6 `git rev-parse` provenance line) | `docker cp` preserves host UIDs; the container user has no `safe.directory` entry for the copied workspace | Phase 2.3b registers `/home/testuser/IB_Robot` as a git `safe.directory` for `testuser` right after `useradd` (observed 2026-09: report step produced the error instead of HEAD/tree) |
