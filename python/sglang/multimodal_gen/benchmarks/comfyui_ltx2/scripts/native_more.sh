#!/bin/bash
cd /scratch/data/sgld_comfy/results/ltx25/scripts
./native_run.sh base t2v_fp8_base --quantization fp8
./native_run.sh branch t2v_fp8_branch --quantization fp8
./native_run.sh branch t2v_fp8_branch_ign --quantization fp8 --quantization-ignored-layers proj_out audio_proj_out
./native_run.sh base t2v_fp8_base_ign --quantization fp8 --quantization-ignored-layers proj_out audio_proj_out
echo MORE_DONE
