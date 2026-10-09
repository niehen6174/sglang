#!/bin/bash
# Slimmed-branch e2e: fresh server, t2i cold+warm, CFG 4, edit 1/2 refs, odd-size edit.
set -uo pipefail
MODE=$1
R=/scratch/data/sgld_comfy/results/qwen_image21
source $R/scripts/env_qwen21.sh
LOG=/scratch/data/sgld_comfy/logs/comfy_qwen21_slim_${MODE}.log
cd /scratch/data/sgld_comfy/ComfyUI-qwen21
python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --cache-classic \
  --preview-method none --disable-cuda-malloc > $LOG 2>&1 &
PID=$!
echo $PID > /scratch/data/sgld_comfy/logs/comfy_qwen21.pid
trap "kill $PID; sleep 6" EXIT
for i in $(seq 1 60); do curl -s 127.0.0.1:8188/system_stats >/dev/null && break; sleep 2; done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i 0 -lms 200 > $R/slim/smi_${MODE}.csv &
SMI=$!
cd $R
OUT=$R/slim/session_${MODE}.jsonl; : > $OUT
RW="python scripts/run_workflow.py --mode $MODE --out-dir $R/images/slim"
$RW --task t2i --runs 3 --seed 1000 --tag t2i_${MODE} | tee -a $OUT
$RW --task t2i --runs 2 --seed 4000 --cfg 4.0 --negative "blurry, low quality, watermark, text" --tag cfg4_${MODE} | tee -a $OUT
$RW --task edit --images qi21_ref_portrait.png --runs 2 --seed 2000 --tag edit1_${MODE} | tee -a $OUT
$RW --task edit --images qi21_ref_portrait.png example.png --runs 2 --seed 3000 \
  --prompt "Put the woman from image 1 into the scene of image 2, keep her sunglasses." --tag edit2_${MODE} | tee -a $OUT
$RW --task edit --images qi21_ref_portrait.png --runs 1 --seed 2000 --edit-target --width 1008 --height 1008 --tag edit1odd_${MODE} | tee -a $OUT
kill $SMI
grep -oE "Prompt executed in [0-9.]+ seconds|[0-9]+/25 \[[^]]*\]$" $LOG > $R/slim/session_${MODE}_comfylog.txt || true
