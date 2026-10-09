#!/bin/bash
cd /scratch/data/sgld_comfy/results/ltx25/scripts
FIX='"enhance": false, "prompt_file": "/scratch/data/sgld_comfy/results/ltx25/logs/ltx23_enhanced_prompt.txt"'
./session.sh R25_new "" "python scripts/run_prompt.py R25_new --sgld --repeat 2 | cut -c1-200"
./tpl_sgld_session.sh R23_new 2 "{\"server\": {\"dit_layerwise_resident_layers\": 0.8}, $FIX}"
git -C /scratch/data/sgld_comfy/wt-ltx25 checkout -q archive/comfyui-ltx2-slim && echo "on slim $(git -C /scratch/data/sgld_comfy/wt-ltx25 log --oneline -1)"
./session.sh R25_slim "" "python scripts/run_prompt.py R25_slim --sgld --repeat 1 | cut -c1-200"
git -C /scratch/data/sgld_comfy/wt-ltx25 checkout -q feat/comfyui-ltx2 && echo "back on $(git -C /scratch/data/sgld_comfy/wt-ltx25 log --oneline -1)"
echo SMOKE_DONE
