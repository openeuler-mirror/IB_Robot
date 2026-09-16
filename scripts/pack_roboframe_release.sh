#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_INSTALL=""
DEPS_ARCHIVE=""
SKH_TAR=""
PATCHES_DIR=""
PYSITE_EXTRAS_DIR=""
OUTPUT=""
VERSION="1.0.0-$(date +%Y%m%d)"
STAGE_DIR=""

log_info()  { printf "[INFO]  %s\n" "$*" >&2; }
log_warn()  { printf "[WARN]  %s\n" "$*" >&2; }
log_error() { printf "[ERROR] %s\n" "$*" >&2; }

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Pack a RoboFrame release tarball for RoboPi (RoboOH 1.0.1).

Options:
  --build-install <dir>   Colcon install tree from build_roboframe_oh.sh
  --deps-archive <file>   Published roboframe-deps archive containing pysite/
  --skh-run <tarball>     Legacy skh-run archive and optional syslib source
  --patches-dir <dir>     Directory of patched .so files (e.g. libcontroller_manager.so)
  --pysite-extras <dir>   OpenHarmony-compatible Python archives and wheels
  --output <path>         Output .tar.gz path (default: roboframe-robopi-<ver>.tar.gz)
  --version <ver>         Release version string (default: 1.0.0-YYYYMMDD)
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-install) BUILD_INSTALL="$2"; shift 2 ;;
        --deps-archive)  DEPS_ARCHIVE="$2"; shift 2 ;;
        --skh-run)       SKH_TAR="$2"; shift 2 ;;
        --patches-dir)   PATCHES_DIR="$2"; shift 2 ;;
        --pysite-extras) PYSITE_EXTRAS_DIR="$2"; shift 2 ;;
        --output)        OUTPUT="$2"; shift 2 ;;
        --version)       VERSION="$2"; shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        *) log_error "Unknown option: $1"; usage; exit 1 ;;
    esac
done

[[ -z "${BUILD_INSTALL}" ]] && { log_error "--build-install required"; usage; exit 1; }
[[ -d "${BUILD_INSTALL}" ]] || { log_error "Not a directory: ${BUILD_INSTALL}"; exit 1; }
if [[ -n "${DEPS_ARCHIVE}" ]]; then
    [[ -f "${DEPS_ARCHIVE}" ]] || { log_error "Dependencies archive not found: ${DEPS_ARCHIVE}"; exit 1; }
else
    [[ -z "${SKH_TAR}" ]] && SKH_TAR="${REPO_ROOT}/../thirdparty_pytorch/test/skh-run.tar.gz"
    [[ -f "${SKH_TAR}" ]] || { log_error "skh-run.tar.gz not found: ${SKH_TAR}"; exit 1; }
fi
[[ -z "${OUTPUT}" ]] && OUTPUT="${PWD}/roboframe-robopi-${VERSION}.tar.gz"

STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "${STAGE_DIR}"' EXIT

PKG_ROOT="${STAGE_DIR}/roboframe-ohos"
mkdir -p "${PKG_ROOT}"/{install,pysite,syslib,scripts}

log_info "Stage dir: ${STAGE_DIR}"

# [1] Copy ROS install packages
log_info "Copying install tree..."
for pkg_dir in "${BUILD_INSTALL}"/*/; do
    pkg="$(basename "${pkg_dir}")"
    [[ "${pkg}" = "lerobot" ]] && continue
    cp -a "${pkg_dir}" "${PKG_ROOT}/install/${pkg}"
done

if [[ -d "${BUILD_INSTALL}/lerobot" ]]; then
    cp -a "${BUILD_INSTALL}/lerobot" "${PKG_ROOT}/install/lerobot"
fi

# Copy colcon top-level setup files
for f in setup.sh setup.bash setup.zsh setup.ps1 local_setup.sh local_setup.bash local_setup.zsh local_setup.ps1 _local_setup_util_sh.py _local_setup_util_ps1.py COLCON_IGNORE; do
    [[ -f "${BUILD_INSTALL}/${f}" ]] && cp -a "${BUILD_INSTALL}/${f}" "${PKG_ROOT}/install/${f}"
done

# [2] Extract the base pysite and dependency syslib
if [[ -n "${DEPS_ARCHIVE}" ]]; then
    log_info "Extracting pysite and syslib from $(basename "${DEPS_ARCHIVE}")..."
    tar xzf "${DEPS_ARCHIVE}" -C "${STAGE_DIR}" ./pysite
    rm -rf "${PKG_ROOT}/pysite"
    mv "${STAGE_DIR}/pysite" "${PKG_ROOT}/pysite"
    if tar tzf "${DEPS_ARCHIVE}" ./syslib >/dev/null 2>&1; then
        tar xzf "${DEPS_ARCHIVE}" -C "${STAGE_DIR}" ./syslib
        cp -a "${STAGE_DIR}/syslib/." "${PKG_ROOT}/syslib/"
        rm -rf "${STAGE_DIR}/syslib"
    fi
else
    log_info "Extracting pysite from skh-run..."
    tar xzf "${SKH_TAR}" -C "${STAGE_DIR}" \
        "skh-run/usr/lib/python3.12/site-packages" 2>/dev/null || true
    if [[ -d "${STAGE_DIR}/skh-run/usr/lib/python3.12/site-packages" ]]; then
        rm -rf "${PKG_ROOT}/pysite"
        mv "${STAGE_DIR}/skh-run/usr/lib/python3.12/site-packages" "${PKG_ROOT}/pysite"
    fi
fi

if [[ ! -d "${PKG_ROOT}/pysite/torch" ]]; then
    log_error "Failed to extract pysite (torch not found)"
    exit 1
fi

# [2.5] Overlay extra OpenHarmony-compatible Python packages
log_info "Extracting pysite extras..."
EXTRAS_DIR="${PYSITE_EXTRAS_DIR:-${REPO_ROOT}/third_party/pysite-extras}"
if [[ -d "${EXTRAS_DIR}" ]]; then
    for tar_file in "${EXTRAS_DIR}"/*.tar.gz; do
        [[ -f "${tar_file}" ]] || continue
        tar xzf "${tar_file}" -C "${PKG_ROOT}/pysite" 2>/dev/null || true
        log_info "  Extracted $(basename "${tar_file}")"
    done
    for whl_file in "${EXTRAS_DIR}"/*.whl; do
        [[ -f "${whl_file}" ]] || continue
        python3 -m zipfile -e "${whl_file}" "${PKG_ROOT}/pysite"
        log_info "  Extracted $(basename "${whl_file}") (+ .so renamed)"
    done
    while IFS= read -r so_file; do
        mv "${so_file}" "${so_file%linux-musl.so}linux-ohos.so"
    done < <(find "${PKG_ROOT}/pysite" -type f -name '*.cpython-312-aarch64-linux-musl.so')
    while IFS= read -r so_file; do
        mv "${so_file}" "${so_file%linux-gnu.so}linux-ohos.so"
    done < <(find "${PKG_ROOT}/pysite" -type f -name '*.cpython-312-aarch64-linux-gnu.so')
else
    log_warn "No extras directory at ${EXTRAS_DIR}"
fi

declare -a REQUIRED_PYSITE_ENTRIES=(
    "huggingface_hub"
    "jsonschema"
    "rpds"
    "rknnlite"
    "ruamel"
)
for entry in "${REQUIRED_PYSITE_ENTRIES[@]}"; do
    if [[ ! -e "${PKG_ROOT}/pysite/${entry}" ]]; then
        log_error "Required Python package missing from release: ${entry}"
        log_error "Add its OpenHarmony-compatible archive or wheel to ${EXTRAS_DIR}"
        exit 1
    fi
done

# [2.6] Copy pymoveit2 from source (pure Python, not installed by colcon)
PYMOVEIT2_SRC="${REPO_ROOT}/src/pymoveit2/pymoveit2"
if [[ -d "${PYMOVEIT2_SRC}" ]]; then
    log_info "Copying pymoveit2 from source..."
    cp -a "${PYMOVEIT2_SRC}" "${PKG_ROOT}/pysite/pymoveit2"
else
    log_warn "pymoveit2 source not found at ${PYMOVEIT2_SRC}, motion_server will fail"
fi

# [3] Extract optional syslib from legacy skh-run.tar.gz
[[ -n "${SKH_TAR}" ]] && log_info "Extracting syslib..."
declare -a SYSLIB_NAMES=(
    "skh-run/usr/lib/libc++.so.1"
    "skh-run/usr/lib/libc++.so.1.0"
    "skh-run/usr/lib/libc++abi.so.1"
    "skh-run/usr/lib/libc++abi.so.1.0"
    "skh-run/usr/lib/libunwind.so.1"
    "skh-run/usr/lib/libunwind.so.1.0"
    "skh-run/usr/lib/libomp.so"
    "skh-run/usr/lib/libiomp5.so"
    "skh-run/usr/lib/libjpeg.so.62"
    "skh-run/usr/lib/libjpeg.so.62.4.0"
    "skh-run/usr/lib/libintl.so.8"
    "skh-run/usr/lib/libintl.so.8.4.0"
)
for name in "${SYSLIB_NAMES[@]}"; do
    if [[ -n "${SKH_TAR}" ]] && tar xzf "${SKH_TAR}" -C "${STAGE_DIR}" "${name}" 2>/dev/null; then
        mv "${STAGE_DIR}/${name}" "${PKG_ROOT}/syslib/"
    fi
done

# [3.5] Copy patched .so files (e.g. libcontroller_manager.so with defer_lock fix)
# Use default patches directory if not specified
if [ -z "${PATCHES_DIR}" ]; then
    PATCHES_DIR="${REPO_ROOT}/third_party/patches/oh"
fi
if [ -n "${PATCHES_DIR}" ] && [ -d "${PATCHES_DIR}" ]; then
    log_info "Copying patched .so files from ${PATCHES_DIR}..."
    mkdir -p "${PKG_ROOT}/install/patches/lib"
    cp -a "${PATCHES_DIR}"/*.so* "${PKG_ROOT}/install/patches/lib/" 2>/dev/null || true
    log_info "  $(ls -1 "${PKG_ROOT}/install/patches/lib/"*.so* 2>/dev/null | wc -l) patched .so files copied"
else
    log_info "WARNING: patches directory not found (${PATCHES_DIR:-unset}), ros2_control crash fix will be missing"
fi

# [4] Copy env script
log_info "Copying robooh_1.0.1.env..."
cp "${REPO_ROOT}/scripts/robooh_1.0.1.env" "${PKG_ROOT}/scripts/"

log_info "Copying setup_sshd.sh..."
cp "${REPO_ROOT}/scripts/setup_sshd.sh" "${PKG_ROOT}/scripts/"

log_info "Copying Houmo environment script..."
mkdir -p "${PKG_ROOT}/scripts/setup"
cp "${REPO_ROOT}/scripts/setup/houmo_hmm_env.sh" "${PKG_ROOT}/scripts/setup/"

# [5] Re-generate wrapper scripts for new layout
log_info "Rewriting wrapper scripts..."

declare -A WRAPPERS=(
    ["lib/inference_service/pipeline_policy_node"]="inference_service.pipeline_policy_node"
    ["lib/inference_service/pure_inference_node"]="inference_service.pure_inference_node"
    ["lib/dataset_tools/policy_eval"]="dataset_tools.policy_eval"
)

for rel_path in "${!WRAPPERS[@]}"; do
    module="${WRAPPERS[$rel_path]}"
    for script_path in "${PKG_ROOT}/install"/inference_service/"${rel_path}" \
                       "${PKG_ROOT}/install"/dataset_tools/"${rel_path}"; do
        [[ -f "${script_path}" ]] || continue

        cat > "${script_path}" <<'WRAP_EOF'
#!/system/bin/sh
export LD_PRELOAD=/sys_prod/robot/out/lib/libpython3.12.so.1.0

RF_LIB=/data/roboframe/syslib:\
/data/roboframe/pysite/rpds_py.libs:\
/sys_prod/robot/out/lib:/sys_prod/robot/install/lib:\
/data/roboframe/install/ibrobot_msgs/lib:\
/data/roboframe/install/tensormsg/lib:\
/data/roboframe/install/embodied_common/lib:\
/data/roboframe/install/robot_config/lib:\
/data/roboframe/install/inference_service/lib:\
/data/roboframe/install/hardware_mock/lib:\
/data/roboframe/install/action_dispatch/lib:\
/data/roboframe/install/feetech_sdk/lib:\
/data/roboframe/install/so101_sdk/lib:\
/data/roboframe/install/so101_hardware/lib:\
/data/roboframe/install/so101_description/lib:\
/data/roboframe/install/so101_motion/lib:\
/data/roboframe/install/so101_suite/lib:\
/data/roboframe/install/so101_robot/lib:\
/data/roboframe/install/robot_runtime/lib:\
/data/roboframe/install/task_dispatch/lib:\
/data/roboframe/install/dataset_tools/lib
export LD_LIBRARY_PATH="${RF_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PY_PATH=/data/roboframe/pysite:\
/data/roboframe/install/lerobot/src:\
/data/roboframe/install/ibrobot_msgs/lib/python3.12/site-packages:\
/data/roboframe/install/tensormsg/lib/python3.12/site-packages:\
/data/roboframe/install/embodied_common/lib/python3.12/site-packages:\
/data/roboframe/install/inference_manifest/lib/python3.12/site-packages:\
/data/roboframe/install/ibrobot_tracing/lib/python3.12/site-packages:\
/data/roboframe/install/robot_config/lib/python3.12/site-packages:\
/data/roboframe/install/inference_service/lib/python3.12/site-packages:\
/data/roboframe/install/hardware_mock/lib/python3.12/site-packages:\
/data/roboframe/install/action_dispatch/lib/python3.12/site-packages:\
/data/roboframe/install/feetech_sdk/lib/python3.12/site-packages:\
/data/roboframe/install/so101_sdk/lib/python3.12/site-packages:\
/data/roboframe/install/so101_hardware/lib/python3.12/site-packages:\
/data/roboframe/install/so101_description/lib/python3.12/site-packages:\
/data/roboframe/install/so101_motion/lib/python3.12/site-packages:\
/data/roboframe/install/so101_suite/lib/python3.12/site-packages:\
/data/roboframe/install/so101_robot/lib/python3.12/site-packages:\
/data/roboframe/install/robot_runtime/lib/python3.12/site-packages:\
/data/roboframe/install/task_dispatch/lib/python3.12/site-packages:\
/data/roboframe/install/dataset_tools/lib/python3.12/site-packages:\
/sys_prod/robot/out/lib/python3.12/site-packages:\
/sys_prod/robot/install/lib/python3.12/site-packages
export PYTHONPATH="${PY_PATH}${PYTHONPATH:+:$PYTHONPATH}"

export AMENT_PREFIX_PATH="/sys_prod/robot/install:/data/roboframe/install:/data/roboframe/install/embodied_common:/data/roboframe/install/inference_manifest${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"
export ROS_DISTRO=humble
export ROS_VERSION=2
export ROS_PYTHON_VERSION=3

WRAP_EOF
        printf 'exec /sys_prod/robot/out/bin/python3 -m %s "$@"\n' "${module}" >> "${script_path}"
        chmod +x "${script_path}"
    done
done

# [5.5] Fix shebang for all Python entry scripts (launch subprocess needs absolute python path)
log_info "Fixing Python entry script shebangs..."
find "${PKG_ROOT}/install" -type f -executable 2>/dev/null | while read -r script; do
    if head -1 "$script" 2>/dev/null | grep -q "^#!/usr/bin/env python"; then
        head -1 "$script" | grep -qE '^#!/usr/bin/env python3?$' && \
            sed -i '1s|^#!/usr/bin/env python3\?$|#!/sys_prod/robot/out/bin/python3|' "$script" && \
            log_info "  fixed: $(echo "$script" | sed "s|${PKG_ROOT}/||")"
    fi
done

# [6] Generate install.sh
log_info "Generating install.sh..."
cat > "${PKG_ROOT}/install.sh" <<'INSTALL_EOF'
#!/system/bin/sh
set -e

DEST=/data/roboframe
SYS_LIB=/sys_prod/robot/out/lib
ROS_LIB=/sys_prod/robot/install/lib
DIR="$(cd "$(dirname "$0")" && pwd)"

printf '=== RoboFrame for RoboOH 1.0.1 installer ===\n'

printf '[1/5] Deploying RoboFrame packages...\n'
mkdir -p "${DEST}/install"
cp -a "${DIR}/install/"* "${DEST}/install/"

printf '[2/5] Deploying Python packages (pysite)...\n'
rm -rf "${DEST}/pysite"
cp -a "${DIR}/pysite" "${DEST}/pysite"

printf '[3/5] Deploying environment script...\n'
mkdir -p "${DEST}/scripts"
cp "${DIR}/scripts/robooh_1.0.1.env" "${DEST}/scripts/"
cp "${DIR}/scripts/setup_sshd.sh" "${DEST}/scripts/"
mkdir -p "${DEST}/scripts/setup"
cp "${DIR}/scripts/setup/houmo_hmm_env.sh" "${DEST}/scripts/setup/"
chmod +x "${DEST}/scripts/setup_sshd.sh"

printf '[4/5] Installing system native libs + patches...\n'
# Remount read-only partitions as writable
mount -o remount,rw / 2>/dev/null || true
mount -o remount,rw /sys_prod 2>/dev/null || true

# Keep dependency libraries under /data so the release remains self-contained.
rm -rf "${DEST}/syslib"
mkdir -p "${DEST}/syslib"
cp -a "${DIR}/syslib/." "${DEST}/syslib/"

# Create /lib/libintl.so.8 symlink (musl LD_PRELOAD doesn't use LD_LIBRARY_PATH)
if [ ! -e /lib/libintl.so.8 ]; then
    ln -sf "${SYS_LIB}/libintl.so.8" /lib/libintl.so.8
fi

# Apply patched .so overrides (e.g. libcontroller_manager.so with defer_lock fix)
PATCHES_DIR="${DEST}/install/patches/lib"
if [ -d "${PATCHES_DIR}" ]; then
    for so in "${PATCHES_DIR}"/*.so*; do
        [ -f "$so" ] || continue
        so_name=$(basename "$so")
        if [ -f "${ROS_LIB}/${so_name}" ]; then
            cp "${ROS_LIB}/${so_name}" "${ROS_LIB}/${so_name}.bak"
            cp "$so" "${ROS_LIB}/${so_name}"
            chmod 755 "${ROS_LIB}/${so_name}"
            printf '  patched: %s\n' "$so_name"
        fi
    done
fi

# Remount back to read-only
mount -o remount,ro /sys_prod 2>/dev/null || true
mount -o remount,ro / 2>/dev/null || true

printf '[5/5] Creating /data/out symlink...\n'
[ -L /data/out ] || ln -sf /sys_prod/robot/out /data/out

printf '\n=== Setting up SSH... ===\n'
sh "${DEST}/scripts/setup_sshd.sh" 2>/dev/null || true

printf '\n================================================\n'
printf 'RoboFrame installed successfully!\n'
printf '================================================\n'
printf '\nNext steps:\n'
printf '  1. Push your SSH public key to the board (run on HOST):\n'
printf '     hdc -t <board-ip>:8710 shell "echo YOUR_PUBLIC_KEY >> /root/.ssh/authorized_keys"\n'
printf '     (replace YOUR_PUBLIC_KEY with content of ~/.ssh/id_rsa.pub)\n'
printf '\n  2. SSH into the board:\n'
printf '     ssh root@<board-ip>\n'
printf '\n  3. Load the environment:\n'
printf '     . /data/roboframe/scripts/robooh_1.0.1.env\n'
printf '================================================\n'
INSTALL_EOF
chmod +x "${PKG_ROOT}/install.sh"

# [7] Create tarball
log_info "Creating tarball: ${OUTPUT}"
tar czf "${OUTPUT}" -C "${STAGE_DIR}" "roboframe-ohos"

log_info "Done: ${OUTPUT}"
du -sh "${OUTPUT}" | cut -f1-2
