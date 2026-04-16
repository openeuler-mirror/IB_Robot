#!/bin/bash

detect_libc() {
    if command -v ldd >/dev/null 2>&1; then
        if ldd --version 2>&1 | grep -qi musl; then
            echo "musl"
            return 0
        fi
    fi
    echo "glibc"
}

detect_platform_id() {
    if [[ -n "${PLATFORM_OVERRIDE:-}" ]]; then
        echo "${PLATFORM_OVERRIDE}"
        return 0
    fi

    local os_id="" os_version="" os_name=""
    local libc
    libc="$(detect_libc)"

    if [[ -r /etc/os-release ]]; then
        os_id="$(awk -F= '$1=="ID"{gsub(/"/,"",$2); print tolower($2)}' /etc/os-release)"
        os_version="$(awk -F= '$1=="VERSION_ID"{gsub(/"/,"",$2); print tolower($2)}' /etc/os-release)"
        os_name="$(awk -F= '$1=="NAME"{gsub(/"/,"",$2); print tolower($2)}' /etc/os-release)"
    fi

    if [[ "${os_id}" == "ubuntu" && "${os_version}" == "22.04" ]]; then
        echo "ubuntu-22.04"
        return 0
    fi

    if [[ "${os_id}" == "openeuler" || "${os_name}" == *"openeuler"* || "$(uname -r)" == *"openeuler"* ]]; then
        echo "openeuler-embedded-24.03"
        return 0
    fi

    if [[ "${libc}" == "musl" ]] && \
        { [[ "${os_id}" == "openharmony" || "${os_id}" == "ohos" ]] || [[ "${os_name}" == *"openharmony"* ]]; }; then
        echo "openharmony-5.1.0-musl"
        return 0
    fi

    echo "unknown"
}

load_platform_impl() {
    case "${SETUP_PLATFORM_ID}" in
        ubuntu-22.04)
            if [[ -f "${SCRIPT_DIR}/setup/platforms/ubuntu-22.04.sh" ]]; then
                # shellcheck disable=SC1090
                source "${SCRIPT_DIR}/setup/platforms/ubuntu-22.04.sh"
            fi
            ;;
        openeuler-embedded-24.03)
            if [[ -f "${SCRIPT_DIR}/setup/platforms/openeuler-embedded-24.03.sh" ]]; then
                # shellcheck disable=SC1090
                source "${SCRIPT_DIR}/setup/platforms/openeuler-embedded-24.03.sh"
            fi
            ;;
        openharmony-5.1.0-musl)
            if [[ -f "${SCRIPT_DIR}/setup/platforms/openharmony-5.1.0-musl.sh" ]]; then
                # shellcheck disable=SC1090
                source "${SCRIPT_DIR}/setup/platforms/openharmony-5.1.0-musl.sh"
            fi
            ;;
        *)
            log_error "Unsupported platform '${SETUP_PLATFORM_ID}'."
            log_error "Expected one of: ubuntu-22.04, openeuler-embedded-24.03, openharmony-5.1.0-musl."
            exit 1
            ;;
    esac
}

initialize_platform() {
    SETUP_LIBC="$(detect_libc)"
    SETUP_PLATFORM_ID="$(detect_platform_id)"
    load_platform_impl
}

print_platform_summary() {
    echo ""
    echo -e "${YELLOW}--- Platform Detection ---${NC}"
    log_info "Detected platform: ${SETUP_PLATFORM_ID}"
    log_info "Detected libc: ${SETUP_LIBC}"
}
