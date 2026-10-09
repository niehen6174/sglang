#!/bin/bash
cd /scratch/data/sgld_comfy/results/ltx25/scripts
./tpl_sgld_session.sh F2_sgld_nolora 2 '{"server": {"dit_layerwise_resident_layers": 0.8}, "lora": false}'
./tpl_sgld_session.sh F2_sgld 3 '{"server": {"dit_layerwise_resident_layers": 0.8}, "run_kw": [{}, {}, {"enhance": false, "prompt": "A red fox trots through fresh snow in a pine forest at dawn, its breath steaming, soft golden light between the trees."}]}'
echo GPU1_DONE2
