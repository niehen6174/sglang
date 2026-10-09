#!/bin/bash
# mgpu_run.sh NAME UNET CONDS 'SGLD_OPTS json' [LORA]
NAME=$1; UNET=$2; CONDS=$3; OPTS=$4; LORA=$5
D=/scratch/data/sgld_comfy/results/ltx25
source $D/scripts/envrc.sh
export CUDA_VISIBLE_DEVICES=0,1
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 200 > $D/logs/mgpu_$NAME.smi.csv &
SMI=$!
LTX_UNET=$UNET LTX_CONDS=$CONDS SGLD_OPTS="$OPTS" MGPU_OUT=mgpu_$NAME.pt SGLD_LORA=$LORA SGLD_LORA_STRENGTH=0.5 \
  timeout 1800 python $D/scripts/mgpu_dit.py > $D/logs/mgpu_$NAME.log 2>&1
RC=$?
kill $SMI
python3 - "$D/logs/mgpu_$NAME.smi.csv" <<'PY'
import sys, collections
peak=collections.defaultdict(int); busy=collections.defaultdict(list)
for line in open(sys.argv[1]):
    p=[x.strip() for x in line.split(",")]
    if len(p)<4 or not p[1].isdigit(): continue
    g=int(p[1]); peak[g]=max(peak[g],int(p[3])); busy[g].append(int(p[2]))
print("peak_MiB", dict(peak), "util>50%_samples", {g: sum(u>50 for u in v) for g,v in busy.items()})
PY
echo "$NAME rc=$RC"; grep -E "^(stage|odd|load|set_lora)|ERROR|Error" $D/logs/mgpu_$NAME.log | grep -v "^\[" | head -12
