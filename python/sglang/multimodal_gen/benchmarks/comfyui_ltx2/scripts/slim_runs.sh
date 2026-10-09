#!/bin/bash
# Slimmed branch verification on GPU1: DiT refs, official vs SGLD e2e (2.5 T2V, 2.3 dev+LoRA), dynamic LoRA timing.
cd /scratch/data/sgld_comfy/results/ltx25/scripts
source envrc.sh
U23=ltx-2.3-22b-dev-fp8.safetensors
L23=ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors
# ComfyUI DiT references at the stage-1 shape for LTX-2.3 dev FP8, without and with the LoRA
LTX_UNET=$U23 LTX_CONDS=conds23.pt LTX_T=16 LTX_H=11 LTX_W=20 LTX_TA=121 LTX_REF_OUT=ref23_s1.pt timeout 1200 python ref_dit.py 2>&1 | grep -E "saved|Error"
LTX_UNET=$U23 LTX_CONDS=conds23.pt LTX_T=16 LTX_H=11 LTX_W=20 LTX_TA=121 LTX_LORA=$L23 LTX_LORA_STRENGTH=0.5 LTX_REF_OUT=ref23_s1_lora.pt timeout 1200 python ref_dit.py 2>&1 | grep -E "saved|Error"
LW='"master_port": 30105, "scheduler_port": 5655, "dit_layerwise_offload": true, "dit_layerwise_resident_layers": 0.8, "dit_layerwise_residency_lifetime": "permanent", "dit_offload_prefetch_size": 2'
LTX_UNET=$U23 LTX_REF_OUT=ref23_s1.pt SGLD_OPTS="{$LW}" PARITY_JSON=slim_par23_s1.json SGLD_SAVE=slim_par23_s1.pt timeout 1200 python sgld_dit.py 2>&1 | grep -E "^(ctx|t2v_|i2v_)|Error"
LTX_UNET=$U23 LTX_REF_OUT=ref23_s1_lora.pt SGLD_LORA=$L23 SGLD_LORA_STRENGTH=0.5 SGLD_OPTS="{$LW}" PARITY_JSON=slim_par23_s1_lora.json SGLD_SAVE=slim_par23_s1_lora.pt timeout 1200 python sgld_dit.py 2>&1 | grep -E "^(ctx|t2v_|i2v_)|set_lora|Error"
echo DIT_DONE
FIX='"enhance": false, "prompt_file": "/scratch/data/sgld_comfy/results/ltx25/logs/ltx23_enhanced_prompt.txt"'
./session.sh S25_official "" "python scripts/run_prompt.py S25_official --official --repeat 3 | cut -c1-200"
./session.sh S25_sgld "" "python scripts/run_prompt.py S25_sgld --sgld --repeat 3 | cut -c1-200"
./session.sh S23_official "" "python scripts/run_ltx23_tpl.py S23_official official 3 8189 '{$FIX}' | cut -c1-200"
./tpl_sgld_session.sh S23_sgld_merge 3 "{\"server\": {\"dit_layerwise_resident_layers\": 0.8}, $FIX}"
echo E2E_DONE
