#!/bin/bash
cd /scratch/data/sgld_comfy/results/ltx25/scripts
PORTS='\"master_port\": 30105, \"scheduler_port\": 5655'
PX="\"extra_server_args\": \"{$PORTS}\""
LWX="\"extra_server_args\": \"{$PORTS, \\\"dit_layerwise_resident_layers\\\": 0.8, \\\"dit_layerwise_residency_lifetime\\\": \\\"permanent\\\", \\\"dit_offload_prefetch_size\\\": 2}\""
for c in "1gpu:$PX" "sp2:\"num_gpus\": 2, \"sp_degree\": 2, \"ulysses_degree\": 2, \"ring_degree\": 1, $PX" "tp2:\"num_gpus\": 2, \"tp_size\": 2, $PX" "auto:\"num_gpus\": 2, $PX"; do
  n=${c%%:*}; o=${c#*:}
  ./mg_session.sh m25_$n "python scripts/run_prompt.py m25_$n --sgld --repeat 3 --sgld-options '{$o}' | cut -c1-200"
done
FIX='"enhance": false, "prompt_file": "/scratch/data/sgld_comfy/results/ltx25/logs/ltx23_enhanced_prompt.txt"'
for c in "1gpu:\"dit_layerwise_offload\": true, $LWX" "tp2:\"num_gpus\": 2, \"tp_size\": 2, $PX"; do
  n=${c%%:*}; o=${c#*:}
  ./mg_session.sh m23_$n "python scripts/run_ltx23_tpl.py m23_$n sgld 2 8189 '{\"sgld_options\": {$o}, $FIX}' | cut -c1-200"
done
echo MG_SMOKE_DONE
