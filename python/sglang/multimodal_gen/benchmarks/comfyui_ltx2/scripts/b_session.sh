#!/bin/bash
# b_session.sh NAME "comfy args" REPEAT [json] : fresh official ComfyUI-b on GPU0 (port 8190) running the 2.3 template
NAME=$1; CARGS=$2; REP=$3; KW=${4:-'{}'}
D=/scratch/data/sgld_comfy/results/ltx25
kill $(cat $D/logs/comfyui_b.pid 2>/dev/null) 2>/dev/null; sleep 5
GPU=${GPU:-0}
CUDA_GPU=$GPU $D/scripts/start_comfy_b.sh $CARGS
for i in $(seq 1 60); do curl -s -m 2 127.0.0.1:8190/system_stats >/dev/null && break; sleep 3; done
source $D/scripts/envrc.sh; export CUDA_VISIBLE_DEVICES=$GPU
nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > $D/logs/vram_$NAME.csv &
SMI=$!
cd $D && python scripts/run_ltx23_tpl.py $NAME official $REP 8190 "$KW" | cut -c1-250
kill $SMI 2>/dev/null
echo "peak VRAM MiB (GPU$GPU, all processes): $(sort -n $D/logs/vram_$NAME.csv | tail -1)"
cp /scratch/data/sgld_comfy/logs/comfyui_ltx2_b.log $D/logs/comfyb_$NAME.log
grep -o "[0-9]*/[0-9]* \[[0-9:]*<[0-9:]*, *[0-9.]*[s/it]*" $D/logs/comfyb_$NAME.log | grep -E "^(8/8|3/3)" | awk 'NR%2==0' | tr '\n' ' '; echo
grep "Prompt executed" $D/logs/comfyb_$NAME.log
kill $(cat $D/logs/comfyui_b.pid) 2>/dev/null
