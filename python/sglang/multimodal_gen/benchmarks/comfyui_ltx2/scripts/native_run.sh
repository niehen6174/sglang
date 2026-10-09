#!/bin/bash
# native_run.sh base|branch NAME [extra sglang generate args...]
# Plain `sglang generate` of LTX-2.5 (Diffusers layout) on GPU1; PYTHONPATH picks the tree.
WT=$1; NAME=$2; shift 2
case $WT in base) TREE=/scratch/data/sgld_comfy/wt-ltx-base;; branch) TREE=/scratch/data/sgld_comfy/wt-ltx25;; esac
source /scratch/data/sgld_comfy/env.sh
source /scratch/data/sgld_comfy/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$TREE/python
OUT=/scratch/data/sgld_comfy/results/ltx25/native
LOG=$OUT/$NAME.log
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 -lms ${SMI_MS:-200} > $OUT/$NAME.vram.csv &
SMI=$!
cd $OUT
sglang generate \
  --model-path /scratch/data/sgld_comfy/models/Lightricks--LTX-2.5-Diffusers-noenh \
  --master-port 30125 --scheduler-port 5675 \
  --prompt "A red fox trots through fresh snow in a pine forest at dawn, its breath steaming, soft golden light between the trees. Birds chirp softly." \
  --seed 42 --height 384 --width 640 --num-frames 49 --num-inference-steps 8 \
  --dit-layerwise-offload true --dit-layerwise-residency-lifetime forward --dit-layerwise-resident-layers 0.5 --cpu-offload-components text_encoder connectors vae audio_vae vocoder \
  --output-path $OUT --output-file-path $OUT/$NAME.mp4 \
  "$@" > $LOG 2>&1
RC=$?
kill $SMI
echo "$NAME rc=$RC peak_vram_mib=$(sort -n $OUT/$NAME.vram.csv | tail -1)"
