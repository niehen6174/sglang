#!/bin/bash
# Regression smoke after porting the LTX-branch plugin fixes (t2i + edit), plus
# prompt-change runs that make ComfyUI re-encode (text encoder reload cost).
set -uo pipefail
MODE=$1
R=/scratch/data/sgld_comfy/results/qwen_image21
source $R/scripts/env_qwen21.sh
LOG=/scratch/data/sgld_comfy/logs/comfy_qwen21_smoke2_${MODE}${SMOKE_SUFFIX:-}.log
cd /scratch/data/sgld_comfy/ComfyUI-qwen21
python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --cache-classic \
  --preview-method none --disable-cuda-malloc > $LOG 2>&1 &
PID=$!
echo $PID > /scratch/data/sgld_comfy/logs/comfy_qwen21.pid
trap "kill $PID; sleep 6" EXIT
for i in $(seq 1 60); do curl -s 127.0.0.1:8188/system_stats >/dev/null && break; sleep 2; done
cd $R
OUT=$R/smoke2_${MODE}${SMOKE_SUFFIX:-}.jsonl
: > $OUT
RW="python scripts/run_workflow.py --mode $MODE --out-dir $R/images/smoke2"
$RW --task t2i --runs 3 --seed 1000 --tag smoke2_t2i_${MODE}${SMOKE_SUFFIX:-} | tee -a $OUT
for i in 1 2; do
  $RW --task t2i --runs 1 --seed 1001 --prompt "A red fox sitting in a snowy birch forest at golden hour, variant $i." --tag smoke2_t2i_newprompt${i}_${MODE}${SMOKE_SUFFIX:-} | tee -a $OUT
done
$RW --task edit --images qi21_ref_portrait.png example.png --runs 2 --seed 3000 \
  --prompt "Put the woman from image 1 into the scene of image 2, keep her sunglasses." --tag smoke2_edit2_${MODE}${SMOKE_SUFFIX:-} | tee -a $OUT
grep -oE "Prompt executed in [0-9.]+ seconds|[0-9]+/25 \[[^]]*\]$|prefix K/V[^[]*" $LOG > $R/smoke2_${MODE}${SMOKE_SUFFIX:-}_comfylog.txt || true
