#!/bin/bash
cd /scratch/data/sgld_comfy/results/ltx25/scripts
IMG=/scratch/data/sgld_comfy/ComfyUI-ltx25/input/ltx_i2v_start.png
export SMI_MS=50
./native_run.sh branch ti2v_branch_v50 --image-path $IMG
./native_run.sh base ti2v_base_v50 --image-path $IMG
echo VRAM_DONE
