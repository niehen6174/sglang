#!/bin/bash
cd /scratch/data/sgld_comfy/results/ltx25/scripts
IMG=/scratch/data/sgld_comfy/ComfyUI-ltx25/input/ltx_i2v_start.png
./native_run.sh base t2v_base
./native_run.sh branch t2v_branch
./native_run.sh base ti2v_base --image-path $IMG
./native_run.sh branch ti2v_branch --image-path $IMG
./native_run.sh branch t2v_branch_rep
echo NATIVE_DONE
