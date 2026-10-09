#!/bin/bash
# Fresh ComfyUI server, then the LoRA matrix for one loader (official | sgld).
set -uo pipefail
MODE=$1
UNET=${2:-qwen_image_2.1_bf16.safetensors}
SUFFIX=${3:-}
R=/scratch/data/sgld_comfy/results/qwen_image21
source $R/scripts/env_qwen21.sh
LOG=/scratch/data/sgld_comfy/logs/comfy_qwen21_lora_${MODE}${SUFFIX}.log
cd /scratch/data/sgld_comfy/ComfyUI-qwen21
python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --cache-classic \
  --preview-method none --disable-cuda-malloc > $LOG 2>&1 &
PID=$!
echo $PID > /scratch/data/sgld_comfy/logs/comfy_qwen21.pid
trap "kill $PID; sleep 6" EXIT
for i in $(seq 1 60); do curl -s 127.0.0.1:8188/system_stats >/dev/null && break; sleep 2; done
cd $R
OUT=$R/lora_session_${MODE}${SUFFIX}.jsonl
: > $OUT
L="python scripts/run_lora.py --loader $MODE --unet $UNET"
T=${MODE}${SUFFIX}
$L --tag ${T}_turbo6_lora1_s7 --seed 7 | tee -a $OUT            # cold
$L --tag ${T}_turbo6_lora1_s8 --seed 8 | tee -a $OUT            # warm
$L --tag ${T}_turbo6_lora1_s9 --seed 9 | tee -a $OUT            # warm
$L --tag ${T}_turbo6_nolora_s7 --seed 7 --strength 0 | tee -a $OUT   # LoRA removed between prompts
$L --tag ${T}_turbo6_lora05_s7 --seed 7 --strength 0.5 | tee -a $OUT # strength change
$L --tag ${T}_turbo6_lora1b_s7 --seed 7 --nickname viggle_turbo_again | tee -a $OUT  # back to 1.0
if [ "$MODE" = sgld ]; then
  python scripts/run_lora.py --loader native_on_sgld --unet $UNET --tag ${T}_native_lora_on_sgld --seed 11 --expect-error | tee -a $OUT
fi
$L --tag ${T}_base25_nolora_s7 --seed 7 --strength 0 --sampler base25 | tee -a $OUT
grep -oE "Prompt executed in [0-9.]+ seconds|[0-9]+/[0-9]+ \[[^]]*\]$|LoRA[^[]*|lora[^[]*" $LOG > $R/lora_session_${MODE}${SUFFIX}_comfylog.txt || true
