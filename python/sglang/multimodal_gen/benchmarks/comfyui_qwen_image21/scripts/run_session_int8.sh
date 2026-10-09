#!/bin/bash
# INT8 ConvRot DiT (the official template's default): t2i + 1-ref edit for one mode.
set -euo pipefail
MODE=$1
R=/scratch/data/sgld_comfy/results/qwen_image21
source $R/scripts/env_qwen21.sh
LOG=/scratch/data/sgld_comfy/logs/comfy_qwen21_session_int8_${MODE}.log
cd /scratch/data/sgld_comfy/ComfyUI-qwen21
python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --cache-classic \
  --preview-method none --disable-cuda-malloc > $LOG 2>&1 &
PID=$!
echo $PID > /scratch/data/sgld_comfy/logs/comfy_qwen21.pid
trap "kill $PID; sleep 5" EXIT
for i in $(seq 1 60); do curl -s 127.0.0.1:8188/system_stats >/dev/null && break; sleep 2; done
cd $R
RW="python scripts/run_workflow.py --mode $MODE --unet qwen_image_2.1_int8_convrot.safetensors"
OUT=$R/session_int8_${MODE}.jsonl
: > $OUT
$RW --task t2i --runs 3 --seed 1000 --tag int8_t2i_${MODE} | tee -a $OUT
$RW --task edit --images qi21_ref_portrait.png --runs 2 --seed 2000 --tag int8_edit1_${MODE} | tee -a $OUT
grep -oE "Prompt executed in [0-9.]+ seconds|[0-9]+/25 \[[^]]*\]$|prefix K/V[^[]*" $LOG > $R/session_int8_${MODE}_comfylog.txt || true
