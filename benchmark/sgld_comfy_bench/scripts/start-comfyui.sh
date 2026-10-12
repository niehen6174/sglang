#!/usr/bin/env bash
# Usage: start-comfyui.sh <gpus> <port>   (foreground; the runner backgrounds it)
# COMFY_EXTRA_ARGS adds ComfyUI flags (e.g. --disable-smart-memory, --use-sage-attention).
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
export CUDA_VISIBLE_DEVICES="${1:?gpus}"
PORT="${2:?port}"
cd "$COMFY_DIR"
# --disable-cuda-malloc: the SGLD worker shares tensors with ComfyUI over CUDA IPC, which
# cudaMallocAsync (ComfyUI's default allocator) cannot export.
exec python main.py --listen 127.0.0.1 --port "$PORT" --disable-auto-launch --disable-cuda-malloc \
  --cache-classic --preview-method none \
  --database-url "sqlite:///$TMPDIR/comfyui-$PORT.db" ${COMFY_EXTRA_ARGS:-}
