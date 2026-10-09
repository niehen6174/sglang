#!/bin/bash
# Start the LTX ComfyUI instance on GPU 1 / port 8189 (background); PID in logs/comfyui.pid
source /scratch/data/sgld_comfy/results/ltx25/scripts/envrc.sh
cd /scratch/data/sgld_comfy/ComfyUI-ltx25
nohup python main.py --listen 127.0.0.1 --port 8189 --disable-auto-launch --cache-classic --preview-method none --disable-cuda-malloc "$@" > /scratch/data/sgld_comfy/logs/comfyui_ltx25.log 2>&1 &
echo $! > /scratch/data/sgld_comfy/results/ltx25/logs/comfyui.pid
