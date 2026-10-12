#!/usr/bin/env bash
# Install everything the kit needs on one machine (no docker). Re-running skips finished steps.
#   1. code: SGLang bench branch, ComfyUI, vllm-omni (pinned commits below)
#   2. venv-sglang: SGLang (editable) + ComfyUI requirements + SageAttention 2/3 built for this card
#   3. venv-vllm-omni: vLLM 0.31.0 + vllm-omni (editable), kept separate from SGLang's torch
#   4. link the SGLD plugin and the H3Benchmark node into ComfyUI, copy test inputs
# Usage: setup/install.sh [--skip-vllm] [--skip-sage]
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
source ./env.sh

SGLANG_REPO=https://github.com/niehen6174/sglang.git
SGLANG_BRANCH=bench/comfyui-h3-ltx-qwen
COMFY_TAG=v0.39.0
VLLM_OMNI_COMMIT=156ddb3f4a
VLLM_VERSION=0.31.0
VLLM_INDEX=https://wheels.vllm.ai/db9527a46873454610df6dbedf79a36d6bf1a7f6
SAGE_COMMIT=d1a57a5
SKIP_VLLM=0; SKIP_SAGE=0
for a in "$@"; do case "$a" in --skip-vllm) SKIP_VLLM=1 ;; --skip-sage) SKIP_SAGE=1 ;; esac; done

case "$CARD" in
  4090) ARCH=8.9 ;; 5090) ARCH=12.0 ;; h200) ARCH=9.0 ;; b200) ARCH=10.0 ;;
  *) echo "CARD must be 4090|5090|h200|b200" >&2; exit 2 ;;
esac
command -v ffprobe >/dev/null || { echo "install ffmpeg first (ffprobe is required)" >&2; exit 2; }
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

step() { echo; echo "==== $*"; }

step "code"
[ -d "$SGLANG_SRC/.git" ] || git clone -b "$SGLANG_BRANCH" "$SGLANG_REPO" "$SGLANG_SRC"
git -C "$SGLANG_SRC" fetch -q origin "$SGLANG_BRANCH" && git -C "$SGLANG_SRC" checkout -q "origin/$SGLANG_BRANCH"
[ -d "$COMFY_DIR/.git" ] || git clone https://github.com/Comfy-Org/ComfyUI.git "$COMFY_DIR"
git -C "$COMFY_DIR" fetch -q --tags && git -C "$COMFY_DIR" checkout -q "$COMFY_TAG"
[ -d "$VLLM_OMNI_SRC/.git" ] || git clone https://github.com/vllm-project/vllm-omni.git "$VLLM_OMNI_SRC"
git -C "$VLLM_OMNI_SRC" fetch -q && git -C "$VLLM_OMNI_SRC" checkout -q "$VLLM_OMNI_COMMIT"
echo "sglang $(git -C "$SGLANG_SRC" rev-parse --short=10 HEAD)  comfyui $(git -C "$COMFY_DIR" describe --tags)  vllm-omni $(git -C "$VLLM_OMNI_SRC" rev-parse --short=10 HEAD)"

step "venv-sglang ($VENV)"
[ -x "$VENV/bin/python" ] || uv venv -q --python 3.12 "$VENV"
export VIRTUAL_ENV="$VENV" PATH="$VENV/bin:$PATH"
uv pip install -q -e "$SGLANG_SRC/python[diffusion]"
# ComfyUI's own pins (comfy-kitchen, frontend); torch stays the one SGLang installed.
grep -v -E '^(torch|torchvision|torchaudio)([<>=].*)?$' "$COMFY_DIR/requirements.txt" > "$TMPDIR/comfy-req.txt"
uv pip install -q -r "$TMPDIR/comfy-req.txt" nvidia-ml-py psutil
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0))"

if [ "$SKIP_SAGE" = 0 ]; then
  step "SageAttention 2 (and 3 on Blackwell) for sm${ARCH/./}"
  SAGE_DIR="$WORKSPACE/deps/SageAttention"
  [ -d "$SAGE_DIR/.git" ] || git clone https://github.com/thu-ml/SageAttention.git "$SAGE_DIR"
  git -C "$SAGE_DIR" checkout -q "$SAGE_COMMIT"
  if ! python -c "import sageattention" 2>/dev/null; then
    (cd "$SAGE_DIR" && TORCH_CUDA_ARCH_LIST=$ARCH EXT_PARALLEL=4 MAX_JOBS=16 uv pip install --no-build-isolation --no-deps .)
  fi
  if [ "$CARD" = 5090 ] || [ "$CARD" = b200 ]; then
    if ! python -c "import sageattn3" 2>/dev/null; then
      (cd "$SAGE_DIR/sageattention3_blackwell" && TORCH_CUDA_ARCH_LIST=$ARCH MAX_JOBS=16 uv pip install --no-build-isolation --no-deps .) \
        || echo "!! SageAttention 3 build failed; A-*-21 experiments will fail on this card (record it)"
    fi
  fi
fi

if [ "$SKIP_VLLM" = 0 ]; then
  step "venv-vllm-omni ($VENV_VLLM)"
  [ -x "$VENV_VLLM/bin/python" ] || uv venv -q --python 3.12 "$VENV_VLLM"
  VIRTUAL_ENV="$VENV_VLLM" uv pip install -q "vllm==$VLLM_VERSION" --torch-backend=auto --extra-index-url "$VLLM_INDEX"
  VIRTUAL_ENV="$VENV_VLLM" uv pip install -q -e "$VLLM_OMNI_SRC"
  "$VENV_VLLM/bin/python" -c "import vllm, vllm_omni, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
fi

step "link ComfyUI nodes and inputs"
bash setup/link_nodes.sh
echo
echo "INSTALL_DONE  next: python setup/download.py   then ./run.sh check"
