#!/bin/bash
# mg_session.sh TAG "command..." : fresh ComfyUI-ltx25 seeing GPUs 0,1 (port 8189); per-GPU util/mem trace
TAG=$1; shift
D=/scratch/data/sgld_comfy/results/ltx25
$D/scripts/stop_comfy.sh
source $D/scripts/envrc.sh
export CUDA_VISIBLE_DEVICES=0,1
cd /scratch/data/sgld_comfy/ComfyUI-ltx25
nohup python main.py --listen 127.0.0.1 --port 8189 --disable-auto-launch --cache-classic --preview-method none --disable-cuda-malloc > /scratch/data/sgld_comfy/logs/comfyui_ltx25.log 2>&1 &
echo $! > $D/logs/comfyui.pid
for i in $(seq 1 60); do curl -s -m 2 127.0.0.1:8189/system_stats >/dev/null && break; sleep 3; done
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 200 > $D/logs/mg_$TAG.smi.csv &
SMI=$!
cd $D && eval "$@"
kill $SMI 2>/dev/null
cp /scratch/data/sgld_comfy/logs/comfyui_ltx25.log $D/logs/comfy_mg_$TAG.log
python3 - "$D/logs/mg_$TAG.smi.csv" <<'PY'
import sys, collections
peak=collections.defaultdict(int); busy=collections.defaultdict(list)
for line in open(sys.argv[1]):
    p=[x.strip() for x in line.split(",")]
    if len(p)<4 or not p[1].isdigit(): continue
    g=int(p[1]); peak[g]=max(peak[g],int(p[3])); busy[g].append(int(p[2]))
print("peak_MiB", dict(peak), "util>50%_samples", {g: sum(u>50 for u in v) for g,v in busy.items()})
PY
grep -o "[0-9]*/[0-9]* \[[0-9:]*<[0-9:]*, *[0-9.]*[s/it]*" $D/logs/comfy_mg_$TAG.log | grep -E "^(8/8|3/3)" | awk 'NR%2==0' | tr '\n' ' '; echo
grep "Prompt executed" $D/logs/comfy_mg_$TAG.log
echo "OOM lines: $(grep -c 'CUDACachingAllocator.*OOM\|OutOfMemoryError' $D/logs/comfy_mg_$TAG.log)"
$D/scripts/stop_comfy.sh
