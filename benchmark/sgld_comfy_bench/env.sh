#!/usr/bin/env bash
# Source this file: sets paths and caches for every kit script from kit.env.
KIT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$KIT_ROOT/kit.env" ]; then
  echo "kit.env not found: copy kit.env.example to kit.env and edit it" >&2
  return 1 2>/dev/null || exit 1
fi
set -a
# shellcheck disable=SC1091
source "$KIT_ROOT/kit.env"
set +a
export KIT_ROOT
export H3_BENCH_WORKSPACE="$WORKSPACE"
if [ -d "$VENV" ]; then export VIRTUAL_ENV="$VENV" PATH="$VENV/bin:$PATH"; fi
# Only SGLang goes on PYTHONPATH: ComfyUI inherits this environment, so kit modules must not shadow its packages.
export PYTHONPATH="$SGLANG_SRC/python${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$WORKSPACE/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HF_TOKEN_FILE:-}" ]; then
  HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"; export HF_TOKEN
fi
export XDG_CACHE_HOME="$WORKSPACE/cache"
export UV_CACHE_DIR="$WORKSPACE/cache/uv"
export PIP_CACHE_DIR="$WORKSPACE/cache/pip"
export TORCH_HOME="$WORKSPACE/cache/torch"
export TORCH_EXTENSIONS_DIR="$WORKSPACE/cache/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$WORKSPACE/cache/torchinductor"
export TRITON_CACHE_DIR="$WORKSPACE/cache/triton"
export CUDA_CACHE_PATH="$WORKSPACE/cache/cuda"
export FLASHINFER_WORKSPACE_BASE="$WORKSPACE/cache/flashinfer"
export SGLANG_DIFFUSION_CACHE_ROOT="$WORKSPACE/cache/sglang_diffusion"
export TMPDIR="$WORKSPACE/tmp"
export PYTHONNOUSERSITE=1
mkdir -p "$TMPDIR" "$HF_HUB_CACHE" "$WORKSPACE/models" "$WORKSPACE/results"
