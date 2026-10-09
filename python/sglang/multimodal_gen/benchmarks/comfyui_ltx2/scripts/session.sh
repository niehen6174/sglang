#!/bin/bash
# session.sh TAG "comfy args" "command..." : fresh ComfyUI, run command, save log + sampler speeds
TAG=$1; CARGS=$2; shift 2
D=/scratch/data/sgld_comfy/results/ltx25
$D/scripts/stop_comfy.sh
$D/scripts/start_comfy.sh $CARGS
for i in $(seq 1 60); do curl -s -m 2 127.0.0.1:8189/system_stats >/dev/null && break; sleep 3; done
source $D/scripts/envrc.sh
nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > $D/logs/vram_$TAG.csv &
SMI=$!
cd $D && eval "$@"
kill $SMI 2>/dev/null
echo "peak VRAM MiB (GPU1, all processes): $(sort -n $D/logs/vram_$TAG.csv | tail -1)"
cp /scratch/data/sgld_comfy/logs/comfyui_ltx25.log $D/logs/comfy_$TAG.log
grep -o "[0-9]*/[0-9]* \[[0-9:]*<[0-9:]*, *[0-9.]*[s/it]*" $D/logs/comfy_$TAG.log | grep -E "^(8/8|3/3)" | awk 'NR%2==0' | tr '\n' ' '; echo
grep "Prompt executed" $D/logs/comfy_$TAG.log
$D/scripts/stop_comfy.sh
