#!/bin/bash
# build.sh - Modern ROS 2 Humble build script with mixin support
#
# Usage:
#   ./scripts/build.sh                    # Default dev build
#   ./scripts/build.sh --mixin release    # Release build
#   ./scripts/build.sh --mixin debug test # Debug with tests
#   ./scripts/build.sh --list-mixins      # Show available mixins
#   ./scripts/build.sh --packages-select tensormsg  # Build specific package
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-$(dirname "${SCRIPT_DIR}")}"
MIXIN_DIR="${WORKSPACE}/.colcon/mixin"

# Logging Utilities
RED='\033[0;31m'
NC='\033[0m' # No Color
log_info()    { echo -e "\033[0;32m[INFO] $*${NC}"; }
log_error()   { echo -e "${RED}[ERROR] $*${NC}"; }
log_warning() { echo -e "\033[1;33m[WARNING] $*${NC}"; }

# Reuse setup.sh platform detection so build-time package skips follow the same
# openEuler/OpenHarmony fallback rules as dependency installation.
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup/detect.sh"

# ============================================================================
# Help & Mixin Listing
# ============================================================================
show_help() {
    cat << 'EOF'
ROS 2 Humble Build Script

Usage: ./scripts/build.sh [OPTIONS] [-- COLCON_ARGS]

Options:
  --mixin NAME [NAME...]   Use specified mixin(s) (can combine multiple)
  --list-mixins            List available mixins and exit
  --base                   Build the shared base packages only
  --agent                  Build the agent stack plus its base dependencies
  --so101                  Build the SO-101 robot plus its base dependencies
  --lekiwi                 Build the LeKiwi robot plus its base dependencies
  --rosclaw                Build rosclaw plus its base dependencies
  --list-groups            Resolve and list group members, then exit
  --clean                  Clean build (cmake-clean-cache)
  --this                   Build only packages in current directory
  -v, --verbose            Show detailed build output
  -h, --help               Show this help

Group flags are additive and resolved through the colcon dependency
closure, so each robot/agent flag automatically brings in the base
packages it depends on:
  ./scripts/build.sh --so101                  # SO-101 + base
  ./scripts/build.sh --base --agent --lekiwi  # LeKiwi complete content
  ./scripts/build.sh --agent                  # Agent stack + base
Group flags cannot be combined with --this or explicit --packages-*
selections passed after "--".

Common mixins:
  dev               Development (debug, no tests, symlink-install) [DEFAULT]
  debug             Debug build with full symbols
  release           Optimized release build
  rel-with-deb-info Release with debug info (RelWithDebInfo)
  test              Enable testing
  no-test           Disable testing
  lint              Enable testing + ament_lint_auto
  prod              Production (optimized, no tests, tracing disabled)

Examples:
  ./scripts/build.sh                           # Default dev build
  ./scripts/build.sh --mixin release           # Release build
  ./scripts/build.sh --mixin debug test        # Debug with tests
  ./scripts/build.sh --mixin release lint      # Release with linting
  ./scripts/build.sh --clean --mixin release   # Clean release build
  ./scripts/build.sh -- --packages-select foo  # Pass args to colcon
EOF
}

list_mixins() {
    echo "Available mixins in ${MIXIN_DIR}:"
    echo ""
    if command -v yq &> /dev/null; then
        yq -r '.[] | "  \(.name)\t\(.description // "")"' "${MIXIN_DIR}/build.mixin.yaml" | column -t -s $'\t'
    else
        # Fallback: parse with grep/sed
        grep -E "^- name:|^  description:" "${MIXIN_DIR}/build.mixin.yaml" | \
        sed 'N;s/- name: \(.*\)\n  description: "\(.*\)"/  \1\t\2/' | \
        column -t -s $'\t'
    fi
    echo ""
    echo "Combine mixins: --mixin debug test lint"
}

# ============================================================================
# Argument Parsing
# ============================================================================
MIXINS=()
CLEAN_BUILD=false
BUILD_THIS=false
VERBOSE=false
LIST_GROUPS=false
GROUP_FLAGS=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mixin)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                MIXINS+=("$1")
                shift
            done
            ;;
        --list-mixins)
            list_mixins
            exit 0
            ;;
        --base|--agent|--so101|--lekiwi|--rosclaw)
            GROUP_FLAGS+=("${1#--}")
            shift
            ;;
        --list-groups)
            LIST_GROUPS=true
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --this)
            BUILD_THIS=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Default mixin if none specified
[[ ${#MIXINS[@]} -eq 0 ]] && MIXINS=("dev")

# ============================================================================
# Virtual Environment Setup
# ============================================================================
setup_venv() {
    local venv_paths=(
        "${WORKSPACE}/venv"
        "/home/ros/colcon_venv/venv"
        "${VIRTUAL_ENV:-}"
    )
    
    for venv in "${venv_paths[@]}"; do
        if [[ -n "${venv}" && -f "${venv}/bin/activate" ]]; then
            source "${venv}/bin/activate"
            export PATH="${venv}/bin:$PATH"
            return 0
        fi
    done
    return 1
}

ensure_python_deps() {
    [[ -z "${VIRTUAL_ENV:-}" ]] && return 0
    
    local deps=("typing_extensions:typing-extensions" "serial:pyserial" "feetech_servo_sdk:feetech-servo-sdk" "sherpa_onnx:sherpa-onnx" "soundfile:soundfile" "sounddevice:sounddevice")
    for dep in "${deps[@]}"; do
        local module="${dep%%:*}"
        local package="${dep##*:}"
        if ! python3 -c "import ${module}" 2>/dev/null; then
            echo "Installing ${package} in venv..."
            python3 -m pip install --quiet "${package}"
        fi
    done
}

check_lerobot_python_compat() {
    local pyproject="${WORKSPACE}/libs/lerobot/pyproject.toml"
    [[ ! -f "${pyproject}" ]] && return 0

    local min_python
    min_python="$(grep -E '^requires-python = ">=[0-9]+\.[0-9]+"' "${pyproject}" | sed -E 's/.*">=([0-9]+\.[0-9]+)".*/\1/' | head -n1)"
    [[ -z "${min_python}" ]] && return 0

    if ! python3 - "${min_python}" <<'PY'
import re
import sys

required = sys.argv[1]
match = re.fullmatch(r"(\d+)\.(\d+)", required)
if not match:
    raise SystemExit(0)

major, minor = map(int, match.groups())
raise SystemExit(0 if sys.version_info >= (major, minor) else 1)
PY
    then
            local current_py
            current_py="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
            log_error "libs/lerobot requires Python >= ${min_python}, but the active interpreter is ${current_py}."
            log_error "The current ROS 2 Humble workspace venv is not new enough for LeRobot v0.6.0."
            exit 1
    fi
}

require_setup_environment() {
    if ! setup_venv; then
        log_error "Virtual environment not found. Please run ./scripts/setup.sh first."
        exit 1
    fi

    # Verify colcon is importable from the venv. PYTHONNOUSERSITE=1 (set by
    # setup_venv) means a colcon installed under ~/.local cannot rescue us
    # here; it must live in the venv's site-packages. Surface a precise,
    # actionable error early instead of letting `python3 -m colcon` fail with
    # a bare "No module named colcon" deep inside the build pipeline.
    if ! python3 -m colcon --help >/dev/null 2>&1; then
        log_error "colcon is not importable from the workspace venv."
        log_error "This usually means setup.sh installed colcon into ~/.local"
        log_error "(via 'pip install --user') instead of into the venv, while"
        log_error "build.sh sets PYTHONNOUSERSITE=1 to ignore ~/.local."
        log_error ""
        log_error "Fix it by reinstalling colcon into the venv:"
        log_error "  source venv/bin/activate"
        log_error "  PYTHONNOUSERSITE=1 python3 -m pip install --upgrade colcon-common-extensions colcon-mixin"
        log_error ""
        log_error "Or re-run ./scripts/setup.sh which now installs colcon into venv automatically."
        exit 1
    fi

    check_lerobot_python_compat

    if ! python3 -c "import lerobot" 2>/dev/null; then
        log_warning "lerobot is not importable in the current venv."
        log_warning "Run ./scripts/setup.sh to install or repair the Python environment before building."
    fi

    if ! python3 -c "import numpy; assert numpy.__version__.startswith('1.26.')" 2>/dev/null; then
        log_warning "NumPy is not pinned to the expected ROS-compatible 1.26.x series."
        log_warning "Run ./scripts/setup.sh to repair the Python environment before building."
    fi
}

require_setup_environment

# ============================================================================
# Package Group Selection (--base / --agent / --so101 / --lekiwi / --rosclaw)
# Groups map to src/ path prefixes; membership is resolved from the colcon
# package index so the lists follow the tree instead of rotting. The build
# uses --packages-up-to, which adds each selected package's workspace
# dependencies — that is how a robot flag "brings its base" automatically.
# ============================================================================
declare -A GROUP_PATHS=(
    [base]="ibrobot_msgs robot_config robot_runtime robot_teleop tensormsg robot_calibration model_utils hardware_mock observation_transport perception_service manipulation_service action_dispatch task_dispatch inference_service inference_manifest torch_models ibrobot_tracing voice_asr_service voice_tts_service manipulation_execution semantic_mapping object_tracker sim_models dataset_tools benchmark aero_hand_hardware attention_viz pymoveit2"
    [agent]="embodied_agent embodied_bringup embodied_common skill_library skill_catalog robot_skill_cli safety_guard workflows ibrobot_agent"
    [so101]="robots/so101 robots/feetech"
    [lekiwi]="lekiwi_hardware lekiwi_description omni_wheel_controller robot_navigation fast_calib fast_lio livox_ros_driver2"
    [rosclaw]="rosclaw"
)

resolve_group_names() {
    local flag="$1"
    local matched=0
    while IFS=$'\t' read -r pkg_name pkg_path _pkg_type; do
        for prefix in ${GROUP_PATHS[$flag]}; do
            if [[ "${pkg_path}" == "src/${prefix}" || "${pkg_path}" == "src/${prefix}/"* ]]; then
                GROUP_PKGS["${pkg_name}"]=1
                matched=$((matched + 1))
                break
            fi
        done
    done < <(python3 -m colcon list --base-paths src 2>/dev/null)
    if [[ ${matched} -eq 0 ]]; then
        log_error "Group '${flag}' matched no packages under src/ (prefixes: ${GROUP_PATHS[$flag]})."
        exit 1
    fi
}

if [[ "${LIST_GROUPS}" == "true" || ${#GROUP_FLAGS[@]} -gt 0 ]]; then
    declare -A GROUP_PKGS=()
    for flag in ${GROUP_FLAGS[@]+"${GROUP_FLAGS[@]}"}; do
        resolve_group_names "${flag}"
        log_info "Group '${flag}' resolved."
    done
    if [[ "${LIST_GROUPS}" == "true" ]]; then
        if [[ ${#GROUP_FLAGS[@]} -eq 0 ]]; then
            log_warning "--list-groups without group flags; pass one or more of --base --agent --so101 --lekiwi --rosclaw."
        fi
        echo "Selected packages (${#GROUP_PKGS[@]}):"
        for pkg_name in ${!GROUP_PKGS[@]}; do echo "  ${pkg_name}"; done | sort
        exit 0
    fi
    if [[ ${#GROUP_FLAGS[@]} -gt 0 ]]; then
        if [[ "${BUILD_THIS}" == "true" ]]; then
            log_error "Group flags cannot be combined with --this."
            exit 1
        fi
        for arg in ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; do
            if [[ "${arg}" == --packages-* || "${arg}" == --paths ]]; then
                log_error "Group flags cannot be combined with explicit package selection ('${arg}' after --)."
                exit 1
            fi
        done
        GROUP_SELECTION=()
        for pkg_name in ${!GROUP_PKGS[@]}; do GROUP_SELECTION+=("${pkg_name}"); done
        log_info "Group selection: ${#GROUP_SELECTION[@]} packages (dependency closure via --packages-up-to)."
    fi
fi



# ============================================================================
# ROS 2 Environment
# ============================================================================
if [[ ! -f /opt/ros/humble/setup.sh ]]; then
    log_error "ROS 2 Humble is not installed at /opt/ros/humble/."
    exit 1
fi
# ROS 2 setup.sh uses unbound variables internally (e.g. AMENT_TRACE_SETUP_FILES);
# temporarily disable nounset to avoid false failures.
set +u
source /opt/ros/humble/setup.sh
set -u

# Clean build: remove stale dirs BEFORE sourcing to prevent overlay chain leaks.
# Without this, install/setup.sh (which may chain to stale overlays like a
# dev_worktree) pollutes AMENT_PREFIX_PATH and colcon records it again.
if ${CLEAN_BUILD}; then
    log_info "Removing stale install/, build/, log/ to prevent overlay chain leaks..."
    rm -rf "${WORKSPACE}/install" "${WORKSPACE}/build" "${WORKSPACE}/log"
fi

if ! ${CLEAN_BUILD}; then
    if [[ -f "${WORKSPACE}/install/setup.sh" ]]; then
        set +u
        source "${WORKSPACE}/install/setup.sh"
        set -u
    fi
fi

# ============================================================================
# openEuler / RedHat FFmpeg header fix
# On these systems, FFmpeg headers live under /usr/include/ffmpeg/ instead of
# /usr/include/.  Export CPATH so that packages like usb_cam can find them.
# ============================================================================
if [[ -d /usr/include/ffmpeg ]]; then
    export CPATH="/usr/include/ffmpeg${CPATH:+:$CPATH}"
fi

# ============================================================================
# Build
# ============================================================================
cd "${WORKSPACE}"

# Build mixin arguments
MIXIN_ARGS=()
if [[ -f "${MIXIN_DIR}/build.mixin.yaml" ]]; then
    MIXIN_ARGS+=("--mixin-files" "${MIXIN_DIR}/build.mixin.yaml")
    MIXIN_ARGS+=("--mixin" "${MIXINS[@]}")
fi

# Clean build if requested
CLEAN_ARGS=()
${CLEAN_BUILD} && CLEAN_ARGS+=("--cmake-clean-cache")

# Build specific directory if --this
THIS_ARGS=()
${BUILD_THIS} && THIS_ARGS+=("--paths" "$(pwd)")

# Platform-specific package skips
PLATFORM_ARGS=()
detect_host_metadata
SETUP_PLATFORM_ID="$(detect_platform_id)"
if [[ "${SETUP_PLATFORM_ID}" == "openeuler-embedded-24.03" ]]; then
    OPENEULER_SKIP_PACKAGES=()
    if [[ "${IBR_BUILD_INCLUDE_FAST_CALIB_ON_OPENEULER:-0}" != "1" ]]; then
        OPENEULER_SKIP_PACKAGES+=("fast_calib")
    fi
    if [[ "${IBR_BUILD_INCLUDE_SIM_MODELS_ON_OPENEULER:-0}" != "1" ]]; then
        OPENEULER_SKIP_PACKAGES+=("sim_models")
    fi
    if [[ ${#OPENEULER_SKIP_PACKAGES[@]} -gt 0 ]]; then
        log_info "openEuler detected: skipping unavailable runtime packages: ${OPENEULER_SKIP_PACKAGES[*]}."
        PLATFORM_ARGS+=("--packages-ignore" "${OPENEULER_SKIP_PACKAGES[@]}")
    fi
fi

echo "════════════════════════════════════════════════════════════════════"
echo "Building with mixin(s): ${MIXINS[*]}"
echo "════════════════════════════════════════════════════════════════════"

# Select event handlers based on verbosity
EVENT_HANDLERS="status- summary-"
${VERBOSE} && EVENT_HANDLERS="console_cohesion+"

PYTHONNOUSERSITE=1 python3 -m colcon build \
    --continue-on-error \
    --parallel-workers "$(nproc)" \
    --merge-install \
    --symlink-install \
    --event-handlers ${EVENT_HANDLERS} \
    --cmake-args -Wno-dev \
    --base-paths src \
    "${MIXIN_ARGS[@]}" \
    "${CLEAN_ARGS[@]}" \
    "${THIS_ARGS[@]}" \
    "${PLATFORM_ARGS[@]}" \
    ${GROUP_SELECTION[@]+--packages-up-to "${GROUP_SELECTION[@]}"} \
    "${EXTRA_ARGS[@]}"

echo ""
echo "Build complete. Source with: source install/setup.sh"
