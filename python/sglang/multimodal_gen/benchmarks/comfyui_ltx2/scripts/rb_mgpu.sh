#!/bin/bash
cd /scratch/data/sgld_comfy/results/ltx25/scripts
U5=ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors
P='"master_port": 30105, "scheduler_port": 5655'
./mgpu_run.sh rb25_1gpu $U5 conds25.pt "{$P}"
./mgpu_run.sh rb25_sp2 $U5 conds25.pt "{\"num_gpus\": 2, \"sp_degree\": 2, \"ulysses_degree\": 2, \"ring_degree\": 1, $P}"
./mgpu_run.sh rb25_tp2 $U5 conds25.pt "{\"num_gpus\": 2, \"tp_size\": 2, $P}"
./mgpu_run.sh rb25_auto $U5 conds25.pt "{\"num_gpus\": 2, $P}"
echo RB_DONE
