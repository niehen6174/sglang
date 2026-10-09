#!/bin/bash
cd /scratch/data/sgld_comfy/results/ltx25/scripts
source envrc.sh
LTX_REF_OUT=ref25_s1.pt PARITY_JSON=fin2_par25_s1.json SGLD_SAVE=fin2_par25_s1.pt timeout 1200 python sgld_dit.py 2>&1 | grep -E "^(ctx|t2v_|i2v_)|Error"
FIX='"enhance": false, "prompt_file": "/scratch/data/sgld_comfy/results/ltx25/logs/ltx23_enhanced_prompt.txt"'
./tpl_sgld_session.sh G23_sgld 3 "{\"server\": {\"dit_layerwise_resident_layers\": 0.8}, $FIX}"
./session.sh G25i_sgld "" "python scripts/run_i2v.py G25i_sgld sgld 3 | cut -c1-200"
./session.sh G25_sgld "" "python scripts/run_prompt.py G25_sgld --sgld --repeat 3 | cut -c1-200"
echo V2_DONE
