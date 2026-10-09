#!/bin/bash
cd /scratch/data/sgld_comfy/results/ltx25/scripts
FIX='"enhance": false, "prompt_file": "/scratch/data/sgld_comfy/results/ltx25/logs/ltx23_enhanced_prompt.txt"'
X=/scratch/data/sgld_comfy/wt-ltx25/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/executors/ltx_av.py
./session.sh S25_sgld "" "python scripts/run_prompt.py S25_sgld --sgld --repeat 3 | cut -c1-200"
./tpl_sgld_session.sh S23_sgld_merge 3 "{\"server\": {\"dit_layerwise_resident_layers\": 0.8}, $FIX}"
cp $X /tmp/claude-0/-scratch-data-sgld-comfy/5b083111-d623-4dc9-9585-fbd765583e0c/scratchpad/ltx_av.py.bak
sed -i 's/    lora_merge_mode = "merge"/    lora_merge_mode = "dynamic"/' $X
grep -n "lora_merge_mode =" $X
./tpl_sgld_session.sh S23_sgld_dynamic 3 "{\"server\": {\"dit_layerwise_resident_layers\": 0.8}, $FIX}"
cp /tmp/claude-0/-scratch-data-sgld-comfy/5b083111-d623-4dc9-9585-fbd765583e0c/scratchpad/ltx_av.py.bak $X
grep -n "lora_merge_mode =" $X
echo E2E2_DONE
