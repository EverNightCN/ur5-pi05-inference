#!/usr/bin/env bash
set -euo pipefail
DEPLOY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PI05_DEPLOY_ROOT="$DEPLOY_ROOT"
export UR5_REPO="${UR5_REPO:-$DEPLOY_ROOT}"
export PYTHONPATH="$DEPLOY_ROOT/.runtime/openpi-src:$DEPLOY_ROOT:$UR5_REPO"
export LD_LIBRARY_PATH="$DEPLOY_ROOT/.runtime/libcudss-linux-aarch64-0.7.1.4_cuda12-archive/lib:/usr/local/cuda-12.6/targets/aarch64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export CPATH="/usr/local/cuda-12.6/targets/aarch64-linux/include${CPATH:+:$CPATH}"
export TRITON_PTXAS_PATH="$DEPLOY_ROOT/.venv/lib/python3.10/site-packages/triton/backends/nvidia/bin/ptxas-blackwell"
export TORCHINDUCTOR_CACHE_DIR="$DEPLOY_ROOT/.runtime/cache/inductor"
export TRITON_CACHE_DIR="$DEPLOY_ROOT/.runtime/cache/triton"
export TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1
export JAXTYPING_DISABLE=1
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"
exec "$DEPLOY_ROOT/.venv/bin/python" "$@"
