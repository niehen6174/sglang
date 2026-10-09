#!/bin/bash
# Multi-GPU e2e via ComfyUI /prompt: fresh server, t2i 1024^2 25 steps cold + 2 warm,
# per-run per-GPU peak VRAM and busy samples from an nvidia-smi trace.
# Usage: mg_e2e.sh <name> '<SGLDOptions inputs json>' [extra run_workflow args for an edit run]
NAME=$1; OPTS=$2; EDIT=${3:-}
R=/scratch/data/sgld_comfy/results/qwen_image21
source $R/scripts/env_qwen21.sh
export CUDA_VISIBLE_DEVICES=0,1
D=$R/multigpu; mkdir -p $D
LOG=/scratch/data/sgld_comfy/logs/comfy_qwen21_mg_${NAME}.log
cd /scratch/data/sgld_comfy/ComfyUI-qwen21
python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --cache-classic \
  --preview-method none --disable-cuda-malloc > $LOG 2>&1 &
PID=$!
echo $PID > /scratch/data/sgld_comfy/logs/comfy_qwen21.pid
for i in $(seq 1 60); do curl -s 127.0.0.1:8188/system_stats >/dev/null && break; sleep 2; done
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 100 > $D/smi_e2e_${NAME}.csv &
SMI=$!
cd $R
OUT=$D/e2e_${NAME}.jsonl; : > $OUT; MARK=$D/e2e_${NAME}_marks.txt; : > $MARK
RW="python scripts/run_workflow.py --mode integrated --out-dir $R/images/multigpu --sgld-options $OPTS --unet ${UNET:-qwen_image_2.1_bf16.safetensors}"
for s in 1000 1001 1002; do
  echo "start t2i_s$s $(date '+%Y/%m/%d %H:%M:%S.%3N')" >> $MARK
  $RW --task t2i --runs 1 --seed $s --tag mg_${NAME}_t2i | tee -a $OUT
  echo "end t2i_s$s $(date '+%Y/%m/%d %H:%M:%S.%3N')" >> $MARK
done
if [ -n "$EDIT" ]; then
  for s in 2000 2001; do
    echo "start edit1_s$s $(date '+%Y/%m/%d %H:%M:%S.%3N')" >> $MARK
    $RW --task edit --images qi21_ref_portrait.png --runs 1 --seed $s --tag mg_${NAME}_edit1 | tee -a $OUT
    echo "end edit1_s$s $(date '+%Y/%m/%d %H:%M:%S.%3N')" >> $MARK
  done
fi
kill $SMI; kill $PID; sleep 8
grep -oE "Prompt executed in [0-9.]+ seconds|[0-9]+/25 \[[^]]*\]$|prefix K/V[^[]*" $LOG > $D/e2e_${NAME}_comfylog.txt
grep -E "Worker [0-9]+: |rank[0-9]|Rank [0-9]" $LOG | grep -v server_args | head -20 > $D/e2e_${NAME}_ranklog.txt
python3 - $D/smi_e2e_${NAME}.csv $MARK <<'PY'
import sys, datetime, collections
fmt="%Y/%m/%d %H:%M:%S.%f"
rows=[]
for l in open(sys.argv[1]):
    p=[x.strip() for x in l.split(",")]
    if len(p)!=4: continue
    try: rows.append((datetime.datetime.strptime(p[0],fmt),p[1],int(p[2]),int(p[3])))
    except ValueError: pass
marks=[l.split() for l in open(sys.argv[2])]
spans={}
for kind,name,d,t in marks: spans.setdefault(name,{})[kind]=datetime.datetime.strptime(d+" "+t,fmt)
for name,s in spans.items():
    peak=collections.defaultdict(int); busy=collections.defaultdict(int); n=collections.defaultdict(int)
    for ts,i,u,m in rows:
        if s["start"]<=ts<=s["end"]:
            peak[i]=max(peak[i],m); n[i]+=1; busy[i]+=u>50
    print(name, " | ".join(f"GPU{i} peak {peak[i]} MiB busy {busy[i]}/{n[i]}" for i in sorted(peak)))
PY
