#!/bin/bash
cd /scratch/data/sgld_comfy/results/ltx25/scripts
FIX='{"enhance": false, "prompt_file": "/scratch/data/sgld_comfy/results/ltx25/logs/ltx23_enhanced_prompt.txt"}'
GPU=1 ./b_session.sh F_official "" 3
./tpl_sgld_session.sh F_sgld 3 '{"server": {"dit_layerwise_resident_layers": 0.8}}'
./tpl_sgld_session.sh F_sgld_fixed 2 '{"server": {"dit_layerwise_resident_layers": 0.8}, "enhance": false, "prompt_file": "/scratch/data/sgld_comfy/results/ltx25/logs/ltx23_enhanced_prompt.txt"}'
./tpl_sgld_session.sh F_sgld_nolora 1 '{"server": {"dit_layerwise_resident_layers": 0.8}, "lora": false}'
./tpl_sgld_session.sh F_sgld_native_lora 1 '{"server": {"dit_layerwise_resident_layers": 0.8}, "native_lora_on_sgld": true, "expect_error": true}'
echo GPU1_DONE
