#!/bin/bash
# Second (official-path) ComfyUI on GPU 0 / port 8190; PID in logs/comfyui_b.pid
source /scratch/data/sgld_comfy/env.sh
source /scratch/data/sgld_comfy/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=${CUDA_GPU:-0}
export PYTHONPATH=/scratch/data/sgld_comfy/ComfyUI-ltx2-b
cd /scratch/data/sgld_comfy/ComfyUI-ltx2-b
nohup python main.py --listen 127.0.0.1 --port 8190 --disable-auto-launch --cache-classic --preview-method none --disable-cuda-malloc "$@" > /scratch/data/sgld_comfy/logs/comfyui_ltx2_b.log 2>&1 &
echo $! > /scratch/data/sgld_comfy/results/ltx25/logs/comfyui_b.pid
