#!/bin/bash
# DiT parity on 2 GPUs for one SGLD option set, with a per-GPU nvidia-smi trace.
# Usage: mg_parity.sh <name> '<sgld options json>'
NAME=$1; OPTS=$2
R=/scratch/data/sgld_comfy/results/qwen_image21
source $R/scripts/env_qwen21.sh
export CUDA_VISIBLE_DEVICES=0,1
mkdir -p $R/multigpu
cd /scratch/data/sgld_comfy/tmp
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 200 > $R/multigpu/smi_parity_${NAME}.csv &
SMI=$!
timeout 1200 python $R/scripts/parity_dit.py sgld --sgld-options "$OPTS" \
  --save-out /scratch/data/sgld_comfy/tmp/mg_${NAME}.pt --report $R/multigpu/parity_${NAME}.json \
  > /scratch/data/sgld_comfy/logs/qi21_mg_${NAME}.log 2>&1
RC=$?
kill $SMI
echo "exit $RC"
python3 - "$R/multigpu/smi_parity_${NAME}.csv" <<'PY'
import sys, collections
rows=[l.strip().split(", ") for l in open(sys.argv[1]) if l.count(",")==2]
peak=collections.defaultdict(int); busy=collections.defaultdict(int); n=collections.defaultdict(int)
for i,u,m in rows:
    peak[i]=max(peak[i],int(m)); n[i]+=1; busy[i]+= int(u)>50
for i in sorted(peak): print(f"GPU{i}: peak {peak[i]} MiB, samples>50% util {busy[i]}/{n[i]}")
PY
