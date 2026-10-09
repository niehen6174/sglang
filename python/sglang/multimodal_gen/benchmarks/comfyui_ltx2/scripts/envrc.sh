source /scratch/data/sgld_comfy/env.sh
source /scratch/data/sgld_comfy/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/scratch/data/sgld_comfy/wt-ltx25/python:/scratch/data/sgld_comfy/ComfyUI-ltx25
export LTX_RES=/scratch/data/sgld_comfy/results/ltx25
export SGLD_OPTS='{"master_port": 30105, "scheduler_port": 5655}'
