#!/bin/bash
cd /scratch/data/sgld_comfy/results/ltx25/scripts
IMG=/scratch/data/sgld_comfy/ComfyUI-ltx25/input/ltx_i2v_start.png
for i in 1 2; do
  for t in base branch; do
    ./native_run.sh $t t2v_${t}_t$i --perf-dump-path ../native/t2v_${t}_t$i.perf.json
    ./native_run.sh $t ti2v_${t}_t$i --image-path $IMG --perf-dump-path ../native/ti2v_${t}_t$i.perf.json
  done
done
echo TIMING_DONE
