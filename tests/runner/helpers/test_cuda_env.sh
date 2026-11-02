#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG_INFO_RANK0() { :; }
export GPUS_PER_NODE=2
export CUDA_VISIBLE_DEVICES=4,7
export HIP_VISIBLE_DEVICES=0,1
export NVTE_USE_HIPBLASLT=1
export NCCL_SOCKET_IFNAME=eth1
source "$ROOT/runner/helpers/envs/cuda.sh"
[[ "$CUDA_VISIBLE_DEVICES" == "4,7" ]]
[[ -z "${HIP_VISIBLE_DEVICES+x}" ]]
[[ -z "${NVTE_USE_HIPBLASLT+x}" ]]
[[ "$NCCL_SOCKET_IFNAME" == "eth1" ]]
unset CUDA_VISIBLE_DEVICES
source "$ROOT/runner/helpers/envs/cuda.sh"
[[ "$CUDA_VISIBLE_DEVICES" == "0,1" ]]
export CUDA_VISIBLE_DEVICES=""
source "$ROOT/runner/helpers/envs/cuda.sh"
[[ "$CUDA_VISIBLE_DEVICES" == "" ]]
echo "CUDA environment checks passed"

source "$ROOT/runner/lib/gpu.sh"
export PRIMUS_GPU_PLATFORM=cuda
[[ "$(detect_gpu_platform)" == cuda ]]
declare -A config=([options.image]="rocm/primus:v26.8")
configure_container_gpu config
[[ "${config[options.image]}" == "primus-cuda:latest" ]]
[[ "${config[options.gpus]}" == all ]]
config[options.image]="customer/cuda:tested"
configure_container_gpu config
[[ "${config[options.image]}" == "customer/cuda:tested" ]]
export PRIMUS_CUDA_IMAGE="customer/override:tested"
configure_container_gpu config
[[ "${config[options.image]}" == "$PRIMUS_CUDA_IMAGE" ]]
unset PRIMUS_CUDA_IMAGE
export PRIMUS_GPU_PLATFORM=amd
config=([options.image]="customer/rocm:tested")
configure_container_gpu config
[[ "${config[options.image]}" == "customer/rocm:tested" ]]
[[ -z "${config[options.gpus]+x}" ]]
export PRIMUS_GPU_PLATFORM=invalid
if detect_gpu_platform >/dev/null 2>&1; then
    echo "Invalid platform accepted" >&2
    exit 1
fi
echo "Launcher platform defaults checks passed"
