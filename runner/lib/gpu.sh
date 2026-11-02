#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

# A job uses one GPU platform. Explicit selection is useful on login nodes or
# hosts with both management toolkits installed.
detect_gpu_platform() {
    case "${PRIMUS_GPU_PLATFORM:-auto}" in
        amd) echo amd; return ;;
        cuda|nvidia) echo cuda; return ;;
        auto) ;;
        *) echo "Invalid PRIMUS_GPU_PLATFORM: ${PRIMUS_GPU_PLATFORM}" >&2; return 1 ;;
    esac
    if [[ -e /dev/kfd ]]; then
        echo amd
    elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        echo cuda
    else
        echo amd
    fi
}

# Apply platform defaults before CLI options so explicit --image/--gpus win.
configure_container_gpu() {
    local -n gpu_config="$1"
    local platform
    platform=$(detect_gpu_platform) || return 1
    if [[ "$platform" == cuda ]]; then
        gpu_config[options.gpus]="${gpu_config[options.gpus]:-all}"
        if [[ -n "${PRIMUS_CUDA_IMAGE:-}" ]]; then
            gpu_config[options.image]="$PRIMUS_CUDA_IMAGE"
        elif [[ "${gpu_config[options.image]:-}" == "rocm/primus:"* || -z "${gpu_config[options.image]:-}" ]]; then
            gpu_config[options.image]="primus-cuda:latest"
        fi
    fi
}
