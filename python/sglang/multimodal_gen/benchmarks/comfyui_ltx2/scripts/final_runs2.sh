#!/bin/bash
# LTX-2.5 final e2e with the official Gemma4 text encoder
cd /scratch/data/sgld_comfy/results/ltx25/scripts
./session.sh off_t2v_official "" "python scripts/run_prompt.py offte_t2v_official --official --repeat 3 | cut -c1-220"
./session.sh off_t2v_sgld "" "python scripts/run_prompt.py offte_t2v_sgld --sgld --repeat 3 | cut -c1-220"
./session.sh off_i2v_official "" "python scripts/run_i2v.py offte_i2v_official official 3 | cut -c1-220"
./session.sh off_i2v_sgld "" "python scripts/run_i2v.py offte_i2v_sgld sgld 3 | cut -c1-220"
./session.sh off_t2v_floor "--use-split-cross-attention" "python scripts/run_prompt.py offte_t2v_official_splitattn --official --repeat 2 | cut -c1-220"
echo ALL_FINAL2_DONE
