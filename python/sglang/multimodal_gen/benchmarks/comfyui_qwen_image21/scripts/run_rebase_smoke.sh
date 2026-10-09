#!/bin/bash
# Smoke after rebasing onto feat/comfyui-plugin-fixes: integrated t2i / edit + turbo LoRA bf16.
set -uo pipefail
R=/scratch/data/sgld_comfy/results/qwen_image21
source $R/scripts/env_qwen21.sh
LOG=/scratch/data/sgld_comfy/logs/comfy_qwen21_rebase.log
cd /scratch/data/sgld_comfy/ComfyUI-qwen21
python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --cache-classic \
  --preview-method none --disable-cuda-malloc > $LOG 2>&1 &
PID=$!
echo $PID > /scratch/data/sgld_comfy/logs/comfy_qwen21.pid
trap "kill $PID; sleep 6" EXIT
for i in $(seq 1 60); do curl -s 127.0.0.1:8188/system_stats >/dev/null && break; sleep 2; done
cd $R
D=$R/images/rebase; OUT=$R/rebase/smoke.jsonl; mkdir -p $R/rebase; : > $OUT
RW="python scripts/run_workflow.py --mode integrated --out-dir $D"
$RW --task t2i --runs 2 --seed 1000 --tag t2i | tee -a $OUT
$RW --task edit --images qi21_ref_portrait.png --runs 1 --seed 2000 --tag edit1 | tee -a $OUT
$RW --task edit --images qi21_ref_portrait.png example.png --runs 1 --seed 3000 \
  --prompt "Put the woman from image 1 into the scene of image 2, keep her sunglasses." --tag edit2 | tee -a $OUT
L="python scripts/run_lora.py --loader sgld --out-dir $D"
$L --tag lora1_s7 --seed 7 | tee -a $OUT
$L --tag nolora_s7 --seed 7 --strength 0 | tee -a $OUT
$L --tag lora05_s7 --seed 7 --strength 0.5 | tee -a $OUT
python scripts/run_lora.py --loader native_on_sgld --tag native --seed 11 --expect-error | tee -a $OUT
grep -oE "Prompt executed in [0-9.]+ seconds|[0-9]+/(25|6) \[[^]]*\]$" $LOG > $R/rebase/smoke_comfylog.txt || true
