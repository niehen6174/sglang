#!/bin/bash
# Fresh server, INT8 t2i only. $1 = tag prefix.
set -uo pipefail
TAG=$1
R=/scratch/data/sgld_comfy/results/qwen_image21
source $R/scripts/env_qwen21.sh
LOG=/scratch/data/sgld_comfy/logs/comfy_qwen21_r8_${TAG}.log
cd /scratch/data/sgld_comfy/ComfyUI-qwen21
python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --cache-classic \
  --preview-method none --disable-cuda-malloc > $LOG 2>&1 &
PID=$!
echo $PID > /scratch/data/sgld_comfy/logs/comfy_qwen21.pid
trap "kill $PID; sleep 6" EXIT
for i in $(seq 1 60); do curl -s 127.0.0.1:8188/system_stats >/dev/null && break; sleep 2; done
cd $R
OUT=$R/r8/${TAG}.jsonl; : > $OUT
python scripts/run_workflow.py --mode integrated --out-dir $R/images/r8 --unet qwen_image_2.1_int8_convrot.safetensors \
  --task t2i --runs 3 --seed 1000 --tag ${TAG}_t2i | tee -a $OUT
grep -oE "Prompt executed in [0-9.]+ seconds|[0-9]+/25 \[[^]]*\]$" $LOG > $R/r8/${TAG}_comfylog.txt || true
