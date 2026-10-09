#!/bin/bash
cd /scratch/data/sgld_comfy/results/ltx25/scripts
FIX='{"enhance": false, "prompt_file": "/scratch/data/sgld_comfy/results/ltx25/logs/ltx23_enhanced_prompt.txt"}'
GPU=0 ./b_session.sh F0_official_fixed "" 2 "$FIX"
GPU=0 ./b_session.sh F0_official_fixed_static "--disable-dynamic-vram" 2 "$FIX"
GPU=0 ./b_session.sh F0_official_fixed_splitattn "--use-split-cross-attention" 2 "$FIX"
GPU=0 ./b_session.sh F0_official_nolora "" 1 '{"lora": false}'
echo GPU0_DONE
