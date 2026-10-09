#!/bin/bash
# Multi-GPU e2e: official two-stage templates through SGLD, configs 1gpu / sp2 / tp2 / auto.
cd /scratch/data/sgld_comfy/results/ltx25/scripts
PORTS='\"master_port\": 30105, \"scheduler_port\": 5655'
LWX="\"extra_server_args\": \"{$PORTS, \\\"dit_layerwise_resident_layers\\\": 0.8, \\\"dit_layerwise_residency_lifetime\\\": \\\"permanent\\\", \\\"dit_offload_prefetch_size\\\": 2}\""
PX="\"extra_server_args\": \"{$PORTS}\""
SP='"num_gpus": 2, "sp_degree": 2, "ulysses_degree": 2, "ring_degree": 1'
TP='"num_gpus": 2, "tp_size": 2'
AU='"num_gpus": 2, "sp_degree": 2'
# LTX-2.5 distilled INT8 T2V
for c in "1gpu:$PX" "sp2:$SP, $PX" "tp2:$TP, $PX" "auto:$AU, $PX"; do
  n=${c%%:*}; o=${c#*:}
  ./mg_session.sh l25_$n "python scripts/run_prompt.py mg25_$n --sgld --repeat 3 --sgld-options '{$o}' | cut -c1-200"
done
# LTX-2.3 dev-FP8 + distilled LoRA template (enhancer off, fixed enhanced prompt)
FIX='"enhance": false, "prompt_file": "/scratch/data/sgld_comfy/results/ltx25/logs/ltx23_enhanced_prompt.txt"'
for c in "1gpu:\"dit_layerwise_offload\": true, $LWX" "tp2:$TP, $PX" "sp2:$SP, \"dit_layerwise_offload\": true, $LWX" "auto:$AU, \"dit_layerwise_offload\": true, $LWX"; do
  n=${c%%:*}; o=${c#*:}
  ./mg_session.sh l23_$n "python scripts/run_ltx23_tpl.py mg23_$n sgld 3 8189 '{\"sgld_options\": {$o}, $FIX}' | cut -c1-200"
done
echo MG_E2E_DONE
