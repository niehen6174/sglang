#!/bin/bash
# R8 (generic ComfyUI path + condition stage): fresh server, same seeds as the slim session.
set -uo pipefail
R=/scratch/data/sgld_comfy/results/qwen_image21
source $R/scripts/env_qwen21.sh
LOG=/scratch/data/sgld_comfy/logs/comfy_qwen21_r8.log
cd /scratch/data/sgld_comfy/ComfyUI-qwen21
python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --cache-classic \
  --preview-method none --disable-cuda-malloc > $LOG 2>&1 &
PID=$!
echo $PID > /scratch/data/sgld_comfy/logs/comfy_qwen21.pid
trap "kill $PID; sleep 6" EXIT
for i in $(seq 1 60); do curl -s 127.0.0.1:8188/system_stats >/dev/null && break; sleep 2; done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i 0 -lms 200 > $R/r8/smi.csv &
SMI=$!
cd $R
D=$R/images/r8; OUT=$R/r8/smoke.jsonl; mkdir -p $D; : > $OUT
RW="python scripts/run_workflow.py --mode integrated --out-dir $D"
$RW --task t2i --runs 3 --seed 1000 --tag t2i_integrated | tee -a $OUT
$RW --task t2i --runs 2 --seed 4000 --cfg 4.0 --negative "blurry, low quality, watermark, text" --tag cfg4_integrated | tee -a $OUT
$RW --task edit --images qi21_ref_portrait.png --runs 2 --seed 2000 --tag edit1_integrated | tee -a $OUT
$RW --task edit --images qi21_ref_portrait.png example.png --runs 2 --seed 3000 \
  --prompt "Put the woman from image 1 into the scene of image 2, keep her sunglasses." --tag edit2_integrated | tee -a $OUT
$RW --task edit --images qi21_ref_portrait.png --runs 1 --seed 2000 --edit-target --width 1008 --height 1008 --tag edit1odd_integrated | tee -a $OUT
L="python scripts/run_lora.py --loader sgld --out-dir $D"
$L --tag sgld_turbo6_lora1_s7 --seed 7 | tee -a $OUT
$L --tag sgld_turbo6_lora1_s8 --seed 8 | tee -a $OUT
$L --tag sgld_turbo6_nolora_s7 --seed 7 --strength 0 | tee -a $OUT
$L --tag sgld_turbo6_lora05_s7 --seed 7 --strength 0.5 | tee -a $OUT
python scripts/run_lora.py --loader native_on_sgld --out-dir $D --tag native --seed 11 --expect-error | tee -a $OUT
$RW --unet qwen_image_2.1_int8_convrot.safetensors --task t2i --runs 3 --seed 1000 --tag int8_t2i_integrated | tee -a $OUT
kill $SMI
grep -oE "Prompt executed in [0-9.]+ seconds|[0-9]+/(25|6) \[[^]]*\]$" $LOG > $R/r8/smoke_comfylog.txt || true
