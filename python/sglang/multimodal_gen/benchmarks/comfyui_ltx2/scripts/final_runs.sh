#!/bin/bash
cd /scratch/data/sgld_comfy/results/ltx25/scripts
LW='{"sgld_options": {"dit_layerwise_offload": true, "extra_server_args": "{\"master_port\": 30105, \"scheduler_port\": 5655, \"dit_layerwise_resident_layers\": 0.75, \"dit_layerwise_residency_lifetime\": \"permanent\", \"dit_offload_prefetch_size\": 2}"}}'
./session.sh final_t2v_sgld "" "python scripts/run_prompt.py final_t2v_sgld --sgld --repeat 3 | cut -c1-220"
./session.sh final_i2v_sgld "" "python scripts/run_i2v.py final_i2v_sgld sgld 3 | cut -c1-220"
./session.sh final_ltx23_official "" "python scripts/run_ltx23.py final_ltx23_official official 3 | cut -c1-220"
./session.sh final_ltx23_sgld "" "python scripts/run_ltx23.py final_ltx23_sgld sgld 3 '$LW' | cut -c1-220"
echo ALL_FINAL_DONE
