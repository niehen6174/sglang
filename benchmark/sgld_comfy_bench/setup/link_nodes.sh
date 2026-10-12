#!/usr/bin/env bash
# Link the SGLD plugin and the H3Benchmark node into ComfyUI, and copy the test inputs.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
nodes="$COMFY_DIR/custom_nodes"
mkdir -p "$nodes" "$COMFY_DIR/input"
ln -sfn "$SGLANG_SRC/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion" "$nodes/ComfyUI_SGLDiffusion"
ln -sfn "$KIT_ROOT/custom_nodes/H3Benchmark" "$nodes/H3Benchmark"
cp -f "$KIT_ROOT"/inputs/* "$COMFY_DIR/input/"
ls -la "$nodes" | grep -E "ComfyUI_SGLDiffusion|H3Benchmark"
echo "LINK_DONE"
