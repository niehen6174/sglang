#!/bin/bash
# Final slimmed-branch verification on GPU1.
cd /scratch/data/sgld_comfy/results/ltx25/scripts
source envrc.sh
U23=ltx-2.3-22b-dev-fp8.safetensors
L23=ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors
LW='"master_port": 30105, "scheduler_port": 5655, "dit_layerwise_offload": true, "dit_layerwise_resident_layers": 0.8, "dit_layerwise_residency_lifetime": "permanent", "dit_offload_prefetch_size": 2'
LTX_REF_OUT=ref25_s1.pt PARITY_JSON=fin_par25_s1.json SGLD_SAVE=fin_par25_s1.pt timeout 1200 python sgld_dit.py 2>&1 | grep -E "^(ctx|t2v_|i2v_)|step|Error"
LTX_UNET=$U23 LTX_REF_OUT=ref23_s1.pt SGLD_OPTS="{$LW}" PARITY_JSON=fin_par23_s1.json SGLD_SAVE=fin_par23_s1.pt timeout 1200 python sgld_dit.py 2>&1 | grep -E "^(ctx|t2v_|i2v_)|Error"
LTX_UNET=$U23 LTX_REF_OUT=ref23_s1_lora.pt SGLD_LORA=$L23 SGLD_LORA_STRENGTH=0.5 SGLD_OPTS="{$LW}" PARITY_JSON=fin_par23_s1_lora.json SGLD_SAVE=fin_par23_s1_lora.pt timeout 1200 python sgld_dit.py 2>&1 | grep -E "^(ctx|t2v_|i2v_)|set_lora|Error"
echo DIT_DONE
FIX='"enhance": false, "prompt_file": "/scratch/data/sgld_comfy/results/ltx25/logs/ltx23_enhanced_prompt.txt"'
./session.sh F25_sgld "" "python scripts/run_prompt.py F25_sgld --sgld --repeat 3 | cut -c1-200"
./tpl_sgld_session.sh F23_sgld 3 "{\"server\": {\"dit_layerwise_resident_layers\": 0.8}, $FIX}"
./session.sh F25i_official "" "python scripts/run_i2v.py F25i_official official 2 | cut -c1-200"
./session.sh F25i_sgld "" "python scripts/run_i2v.py F25i_sgld sgld 2 | cut -c1-200"
echo VERIFY_DONE
