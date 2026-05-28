#!/bin/bash
# rosdep.sh - Python virtual environment resolution and rosdep management

resolve_venv_python() {
    local venv_path="$1"
    local candidate=""

    for candidate in "${venv_path}/bin/python3" "${venv_path}/bin/python"; do
        if [[ -e "${candidate}" ]] && "${candidate}" -c "import sys" >/dev/null 2>&1; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    return 1
}

ensure_python_ssl_cert_file() {
    if [[ -n "${SSL_CERT_FILE:-}" && -f "${SSL_CERT_FILE}" ]]; then
        export PYTHONHTTPSVERIFY="${PYTHONHTTPSVERIFY:-1}"
        return 0
    fi

    local ca_bundle
    for ca_bundle in \
        /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem \
        /etc/pki/tls/cert.pem \
        /etc/ssl/certs/ca-certificates.crt \
        /etc/ssl/cert.pem
    do
        if [[ -f "${ca_bundle}" ]]; then
            export SSL_CERT_FILE="${ca_bundle}"
            export PYTHONHTTPSVERIFY=1
            log_info "Using Python SSL_CERT_FILE=${SSL_CERT_FILE}"
            return 0
        fi
    done

    log_warn "No system CA bundle found for Python SSL; rosdep HTTPS may fail."
    return 1
}

rosdep_sources_list_needs_refresh() {
    local target_file="${SETUP_ROSDEP_DEFAULT_SOURCES_FILE}"

    if [[ ! -f "${target_file}" ]]; then
        return 0
    fi

    grep -q "https://raw.githubusercontent.com/ros/rosdistro/master/rosdep/" "${target_file}"
}

write_rosdep_sources_list() {
    local target_dir
    target_dir="$(dirname "${SETUP_ROSDEP_DEFAULT_SOURCES_FILE}")"
    local target_file="${SETUP_ROSDEP_DEFAULT_SOURCES_FILE}"
    local tmpfile
    tmpfile="$(mktemp)"

    if [[ ! -d "${target_dir}" ]]; then
        run_sudo mkdir -p "${target_dir}"
    fi

    if ! "${VENV_PYTHON}" -c 'from pathlib import Path; import sys; from rosdep2.sources_list import download_default_sources_list; Path(sys.argv[2]).write_text(download_default_sources_list(sys.argv[1]), encoding="utf-8")' "${SETUP_ROSDEP_DEFAULT_SOURCES_URL}" "${tmpfile}"; then
        rm -f "${tmpfile}"
        return 1
    fi

    if ! run_sudo mv "${tmpfile}" "${target_file}"; then
        rm -f "${tmpfile}"
        return 1
    fi

    run_sudo chmod 644 "${target_file}"

    rm -f "${tmpfile}"
}

ensure_workspace_venv() {
    VENV_PATH="${WORKSPACE}/venv"

    if [[ ! -d "${VENV_PATH}" ]]; then
        log_info "Creating workspace venv at ${VENV_PATH} (early, for rosdep)..."
        run_cmd python3 -m venv --system-site-packages "${VENV_PATH}"
    fi

    VENV_PYTHON="$(resolve_venv_python "${VENV_PATH}" || true)"
    if [[ -z "${VENV_PYTHON}" ]]; then
        log_error "No working Python interpreter found under ${VENV_PATH}/bin."
        exit 1
    fi
    ROSDEP_BIN="${VENV_PATH}/bin/rosdep"

    local pip_conf="${VENV_PATH}/pip.conf"
    if [[ ! -f "${pip_conf}" ]]; then
        cat > "${pip_conf}" <<PIP_CONF
[global]
index-url = ${SETUP_PIP_INDEX_URL}
trusted-host = ${SETUP_PIP_TRUSTED_HOST}
PIP_CONF
        log_info "Configured pip mirror: ${SETUP_PIP_INDEX_URL}"
    fi
}

ensure_rosdep() {
    ensure_workspace_venv
    ensure_python_ssl_cert_file || true

    if ! "${ROSDEP_BIN}" --version &>/dev/null; then
        log_info "Installing rosdep into the workspace venv..."
        run_cmd "${VENV_PYTHON}" -m pip install --upgrade pip --quiet
        run_cmd "${VENV_PYTHON}" -m pip install --ignore-installed --upgrade rosdep --quiet
        if ! "${ROSDEP_BIN}" --version &>/dev/null; then
            log_error "rosdep install did not produce a working CLI at ${ROSDEP_BIN}."
            log_error "Re-run with VERBOSE=1 to see the full pip output."
            exit 1
        fi
    fi
    log_info "Using venv rosdep: ${ROSDEP_BIN}"

    if rosdep_sources_list_needs_refresh; then
        log_info "Configuring rosdep sources list..."
        ensure_sudo_session

        local init_output=""
        local init_exit=0
        init_output=$(write_rosdep_sources_list 2>&1) || init_exit=$?

        # Check both exit code and output for SSL/network errors
        if [[ ${init_exit} -ne 0 ]] || echo "${init_output}" | grep -qi "error\|failed\|certificate\|urlopen"; then
            if echo "${init_output}" | grep -qi "certificate\|ssl\|urlopen"; then
                log_warn "SSL certificate error detected while preparing rosdep sources:"
                echo "${init_output}"
                
                local ssl_pem=""
                ssl_pem=$("${VENV_PYTHON}" -c "import ssl; print(ssl.get_default_verify_paths().openssl_cafile)" 2>/dev/null)
                if [[ -n "${ssl_pem}" && -f "${ssl_pem}" ]]; then
                    log_info "Retrying with explicit SSL_CERT_FILE=${ssl_pem} (Python CA bundle)..."
                    if init_output=$(SSL_CERT_FILE="${ssl_pem}" PYTHONHTTPSVERIFY=1 write_rosdep_sources_list 2>&1); then
                        log_done "rosdep sources list configured using explicit SSL_CERT_FILE"
                        return 0
                    fi
                fi
                
                log_error "Failed to configure rosdep sources list due to SSL/network errors."
                log_error "Ensure your system's CA certificates are up to date."
                exit 1
            else
                log_error "Failed to configure rosdep sources list:"
                echo "${init_output}"
                exit 1
            fi
        fi
        log_done "rosdep sources list configured"
    fi
}
