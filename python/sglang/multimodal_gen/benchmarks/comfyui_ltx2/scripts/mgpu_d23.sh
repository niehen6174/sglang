#!/bin/bash
cd /scratch/data/sgld_comfy/results/ltx25/scripts
U=ltx-2.3-22b-dev-fp8.safetensors
L=ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors
P='"master_port": 30105, "scheduler_port": 5655'
LW="$P, \"dit_layerwise_offload\": true, \"dit_layerwise_resident_layers\": 0.8, \"dit_layerwise_residency_lifetime\": \"permanent\", \"dit_offload_prefetch_size\": 2"
SP='"num_gpus": 2, "sp_degree": 2, "ulysses_degree": 2, "ring_degree": 1'
./mgpu_run.sh e23_1gpu_lora $U conds23.pt "{$LW}" $L
./mgpu_run.sh e23_1gpu_nolora $U conds23.pt "{$LW}"
./mgpu_run.sh e23_tp2_lora $U conds23.pt "{\"num_gpus\": 2, \"tp_size\": 2, $P}" $L
./mgpu_run.sh e23_tp2_nolora $U conds23.pt "{\"num_gpus\": 2, \"tp_size\": 2, $P}"
./mgpu_run.sh e23_sp2_lora $U conds23.pt "{$SP, $LW}" $L
./mgpu_run.sh e23_sp2_nolora $U conds23.pt "{$SP, $LW}"
./mgpu_run.sh e23_sp2res_lora $U conds23.pt "{$SP, $P}" $L
./mgpu_run.sh e23_auto_lora $U conds23.pt "{\"num_gpus\": 2, \"sp_degree\": 2, $LW}" $L
U5=ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors
./mgpu_run.sh e25_1gpu $U5 conds25.pt "{$P}"
./mgpu_run.sh e25_sp2 $U5 conds25.pt "{$SP, $P}"
./mgpu_run.sh e25_tp2 $U5 conds25.pt "{\"num_gpus\": 2, \"tp_size\": 2, $P}"
./mgpu_run.sh e25_auto $U5 conds25.pt "{\"num_gpus\": 2, \"sp_degree\": 2, $P}"
echo E_DONE
